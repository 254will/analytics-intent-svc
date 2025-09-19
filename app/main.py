import os, json, re, time, math, torch, unicodedata, difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

# --- Optional fuzzy matching (robust to typos/shortforms) ---
try:
    from rapidfuzz import process, fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

# --- T5 model for NL -> DSL (works with the finetuned checkpoint) ---
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False


# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH  = os.getenv("MODEL_PATH", "artifacts/planner_model") 
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "artifacts/schema.json")
DATA_PATH   = os.getenv("DATA_PATH", "artifacts/planner_model/embedded_data.csv")  
MAX_CAT_UNIQUE = int(os.getenv("MAX_CAT_UNIQUE", "2000"))

# =============================================================================
# SCHEMA & DATA
# =============================================================================
def load_schema(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"date_col": "date", "dimensions": [], "measures": [], "dictionary": {}}
    return json.loads(p.read_text(encoding="utf-8"))

SCHEMA = load_schema(SCHEMA_PATH)
DATE_COL = SCHEMA.get("date_col") or "date"
DIMENSIONS = list(SCHEMA.get("dimensions", []))
MEASURES   = list(SCHEMA.get("measures", []))
DICT       = dict(SCHEMA.get("dictionary", {}))
ALL_FIELDS = sorted(set([*DIMENSIONS, *MEASURES, DATE_COL]))

_df_cache: Optional[pd.DataFrame] = None
_norm_col_map: Dict[str, str] = {}  # original -> normalized
_rev_col_map: Dict[str, str] = {}   # normalized -> original
_CATEGORICALS: Dict[str, List[str]] = {}  # normalized column -> small-cardinality vocab

def _normalize_name(s: str) -> str:
    s = re.sub(r"[^\w]+", "_", str(s).strip()).strip("_")
    return re.sub(r"_+", "_", s).lower()

DATE_COL_NORM = _normalize_name(DATE_COL)
DIMENSIONS_NORM = [_normalize_name(c) for c in DIMENSIONS]
MEASURES_NORM   = [_normalize_name(c) for c in MEASURES]
ALL_FIELDS_NORM = sorted(set([*DIMENSIONS_NORM, *MEASURES_NORM, DATE_COL_NORM]))

# Map normalized field -> label text from DICT (when the DICT key matches this field)
LABELS_NORM: Dict[str, str] = {}
for raw_field, nice in (DICT or {}).items():
    nf = _normalize_name(raw_field)
    if nf in ALL_FIELDS_NORM:
        LABELS_NORM[nf] = str(nice)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    global _norm_col_map, _rev_col_map
    _norm_col_map = {c: _normalize_name(c) for c in df.columns}
    df = df.rename(columns=_norm_col_map)
    _rev_col_map = {v: k for k, v in _norm_col_map.items()}
    return df

def _build_categoricals(df: pd.DataFrame, max_card: int = None) -> None:
    """Collect per-dimension category vocab; avoid high-cardinality free text."""
    global _CATEGORICALS
    cap = MAX_CAT_UNIQUE if max_card is None else max_card
    cats = {}
    for col in df.columns:
        if col not in DIMENSIONS_NORM:     # <- only dimensions
            continue
        nunique = df[col].nunique(dropna=True)
        if pd.api.types.is_object_dtype(df[col]) or nunique <= cap:
            vals = df[col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
            vals = sorted(vals, key=lambda x: str(x).lower())
            # ignore insane vocab columns
            if len(vals) <= cap:
                cats[col] = vals
    _CATEGORICALS = cats

def load_dataframe() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    p = Path(DATA_PATH)
    if not p.exists():
        _df_cache = pd.DataFrame()
        return _df_cache
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    df = _normalize_columns(df)
    if DATE_COL_NORM in df.columns:
        df[DATE_COL_NORM] = pd.to_datetime(df[DATE_COL_NORM], errors="coerce")
    _build_categoricals(df)
    _build_value_index(df)  
    _df_cache = df
    return _df_cache


def resolve_value_to_dimension(value_norm: str, question: str) -> Optional[str]:
    _ensure_value_index(load_dataframe())
    cols = list(_VALUE_INDEX.get(value_norm, []))
    if not cols:
        return None
    if len(cols) == 1:
        return cols[0]

    # Disambiguate using alias scoring; require a minimal score to accept
    best, best_score = None, -1.0
    for col in cols:
        aliases = ALIAS_DIM.get(col, ALIAS_ALL.get(col, []))
        s = _score_match(question, col, aliases)
        if s > best_score:
            best, best_score = col, s

    # If this value appears across lots of columns and we don't have signal, skip it
    if len(cols) >= 3 and best_score < 0.40:
        return None
    return best if best_score >= 0.35 else None

# ---------- Auto-synonyms for fields (feature-agnostic) ----------
STOPWORDS = {
    "the","a","an","of","and","or","to","in","for","by","with","on","at","from",
    "patient","patients","case","cases","record","records","number","count","total",
    "had","have","has","both","and","without"
}

def _tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOPWORDS]

def _singularize_token(t: str) -> str:
    if t.endswith("ies"): return t[:-3] + "y"
    if t.endswith("s") and not t.endswith("ss"): return t[:-1]
    return t

def _acronym(tokens: List[str]) -> Optional[str]:
    if len(tokens) < 2: return None
    acr = "".join(w[0] for w in tokens if w and w[0].isalnum())
    return acr if len(acr) >= 2 else None

def _ngrams_list(tokens: List[str], nmax: int = 3) -> List[str]:
    out = []
    for n in range(1, min(nmax, len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i:i+n]))
    return out

def build_alias_index(field_names_norm: List[str],
                      labels_by_field_norm: Dict[str, str] = None) -> Dict[str, List[str]]:
    labels_by_field_norm = labels_by_field_norm or {}
    alias: Dict[str, List[str]] = {}
    for f in field_names_norm:
        phrases = set()

        # from the field name itself
        toks = [_singularize_token(t) for t in _tokens(f.replace("_", " "))]
        full = " ".join(toks)
        phrases.add(full); phrases.add(full.replace(" ", "_"))
        for p in _ngrams_list(toks, 3):
            phrases.add(p)
        acr = _acronym(toks)
        if acr: phrases.add(acr)

        # from the human-readable label/description (DICT)
        label = labels_by_field_norm.get(f)
        if label:
            ltoks = [_singularize_token(t) for t in _tokens(label)]
            if ltoks:
                lfull = " ".join(ltoks)
                phrases.add(lfull)
                for p in _ngrams_list(ltoks, 3):
                    phrases.add(p)

        alias[f] = sorted(phrases)
    return alias

# Rebuild alias indices 
ALIAS_ALL = build_alias_index(ALL_FIELDS_NORM, labels_by_field_norm=LABELS_NORM)
ALIAS_DIM = {f: ALIAS_ALL[f] for f in DIMENSIONS_NORM if f in ALIAS_ALL}
ALIAS_MEAS = {f: ALIAS_ALL[f] for f in MEASURES_NORM if f in ALIAS_ALL}

def inventory_for_prompt() -> str:
    items = []
    for f in ALL_FIELDS_NORM:
        lab = LABELS_NORM.get(f)
        items.append(f if not lab else f"{f} — {lab}")
    return ", ".join(items)
# =============================================================================
# FUZZY FIELD RESOLUTION
# =============================================================================
def best_field_match(name: str, candidates: List[str], score_cutoff: int = 75) -> Optional[str]:
    if not name or not candidates:
        return None
    low_map = {c.lower(): c for c in candidates}
    if name.lower() in low_map:
        return low_map[name.lower()]
    if HAVE_RAPIDFUZZ:
        hit = process.extractOne(name, candidates, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
        if hit:
            return hit[0]
    toks = set(re.findall(r"[a-z0-9]+", name.lower()))
    best, best_overlap = None, 0
    for c in candidates:
        ov = len(toks & set(re.findall(r"[a-z0-9]+", c.lower())))
        if ov > best_overlap:
            best, best_overlap = c, ov
    return best if best_overlap else None

def FIELD_RESOLVER(raw: str) -> Optional[str]:
    nf = _normalize_name(raw)
    return nf if nf in ALL_FIELDS_NORM else None

# =============================================================================
# HELPERS
# =============================================================================
MONTHS = {
    "january":1,"jan":1,"february":2,"feb":2,"march":3,"mar":3,"april":4,"apr":4,
    "may":5,"june":6,"jun":6,"july":7,"jul":7,"august":8,"aug":8,"september":9,"sept":9,"sep":9,
    "october":10,"oct":10,"november":11,"nov":11,"december":12,"dec":12
}
def _last_dom(y: int, m: int) -> int:
    import calendar
    return calendar.monthrange(y, m)[1]

def extract_dates(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = (question or "").strip()
    m = re.search(r'(?P<s>\d{4}-\d{2}-\d{2})\s*(to|-|until|through)\s*(?P<e>\d{4}-\d{2}-\d{2})', q, re.I)
    if m:
        return m.group("s"), m.group("e")
    m = re.search(r'from\s+([A-Za-z]{3,9})\s+(\d{4})\s+(to|until|through|-)\s+([A-Za-z]{3,9})\s+(\d{4})', q, re.I)
    if m:
        m1, y1 = MONTHS.get(m.group(1).lower()), int(m.group(2))
        m2, y2 = MONTHS.get(m.group(4).lower()), int(m.group(5))
        if m1 and m2:
            return f"{y1:04d}-{m1:02d}-01", f"{y2:04d}-{m2:02d}-{_last_dom(y2, m2):02d}"
    m = re.search(r'(between|from)\s+(\d{4})\s+(and|to|-)\s+(\d{4})', q, re.I)
    if m:
        y1, y2 = int(m.group(2)), int(m.group(4))
        return f"{y1:04d}-01-01", f"{y2:04d}-12-31"
    m = re.search(r'\b(in|year)\s+(\d{4})\b', q, re.I)
    if m:
        y = int(m.group(2))
        return f"{y:04d}-01-01", f"{y:04d}-12-31"
    m = re.search(r'\b([A-Za-z]{3,9})\s+(\d{4})\b', q)
    if m:
        mi, y = MONTHS.get(m.group(1).lower()), int(m.group(2))
        if mi:
            return f"{y:04d}-{mi:02d}-01", f"{y:04d}-{mi:02d}-{_last_dom(y, mi):02d}"
    return None, None

def parse_topk_from_question(q: str) -> Tuple[Optional[int], Optional[str]]:
    ql = (q or "").lower()
    m = re.search(r'\btop\s+(\d+)\b', ql)
    k = int(m.group(1)) if m else None
    order = None

    # DESC (avoid 'at most' false hits; no standalone 'decreasing')
    if re.search(r'(?<!at\s)\b(most)\b', ql) \
       or re.search(r'\b(highest|top|prevalent|dominant|leading|largest|biggest|greatest|max(?:imum)?|descending)\b', ql) \
       or re.search(r'\b(most\s+(?:common|frequent)|highest\s+number\s+of|largest\s+number\s+of|in\s+descending\s+order|high\s+to\s+low)\b', ql):
        order = "desc"

    # ASC (avoid 'at least' false hits; no standalone 'increasing')
    if re.search(r'(?<!at\s)\b(least)\b', ql) \
       or re.search(r'\b(lowest|fewest|smallest|min(?:imum)?|bottom|ascending|rarest)\b', ql) \
       or re.search(r'\b(least\s+(?:common|frequent)|lowest\s+number\s+of|smallest\s+number\s+of|in\s+ascending\s+order|low\s+to\s+high)\b', ql):
        order = "asc"

    return k, order

def parse_metric_from_question(question: str,
                               df: pd.DataFrame,
                               measures_norm: List[str],
                               field_resolver) -> Tuple[Optional[str], Optional[str]]:
    q = question or ""
    ql = q.lower()

    # Strong count cues
    if re.search(r"\b(number\s+of\s+cases?|cases?|case\s+count|case\s+volume|caseload|case\s+load)\b", ql):
        return "count", "*"
    if re.search(r"\b(how\s+many)\b", ql):
        return "count", "*"

    # Detect op (tentative)
    op = None
    if re.search(r"\b(avg|average|mean)\b", ql):
        op = "avg"
    elif re.search(r"\b(sum|total)\b", ql):
        op = "sum"
    elif re.search(r"\b(min|minimum|lowest)\b", ql):
        op = "min"
    elif re.search(r"\b(max|maximum|highest)\b", ql):
        op = "max"
    elif re.search(r"\b(count)\b", ql):
        return "count", "*"

    if not op:
        return None, None  # let caller decide (usually defaults to count)

    # Try to capture the noun phrase right after the op e.g., "average performance status by ..." or "mean ECOG in ..."
    m = re.search(
        r"\b(?:avg|average|mean|sum|total|min(?:imum)?|max(?:imum)?)\s+([a-z0-9 _\-]+?)(?=\s+(?:by|in|from|during|over|between|for|,|\.|$))",
        ql
    )
    candidate_field = (m.group(1).strip() if m else "")

    # 1) First: try resolving *that phrase* as a measure
    field = None
    if candidate_field:
        # try exact-phrase resolution via aliases
        cand, score = resolve_best_measure(candidate_field)
        if cand and cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]) and score >= 0.35:
            field = cand
        else:
            # try n-grams inside the captured phrase
            toks = re.findall(r"[a-z0-9]+", candidate_field)
            for n in range(min(4, len(toks)), 0, -1):
                for i in range(len(toks)-n+1):
                    span = " ".join(toks[i:i+n])
                    cand2, s2 = resolve_best_measure(span)
                    if cand2 and cand2 in df.columns and pd.api.types.is_numeric_dtype(df[cand2]) and s2 >= 0.35:
                        field = cand2
                        break
                if field:
                    break

    # 2) Second: no luck? Use the whole question with the alias resolver
    if not field:
        cand, score = resolve_best_measure(q)
        if cand and cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]) and score >= 0.40:
            field = cand

    # 3) If still nothing numeric, treat it as COUNT
    if not field:
        return "count", "*"

    return op, field

# ---------- Negation detection near a value mention ----------
NEG_CUES = {"no", "without", "absent", "lack", "lacking", "exclude", "excluding", "excluded", "except", "denies", "deny", "denied"}

def has_local_negation(question: str, value_norm: str, window: int = 4) -> bool:
    """
    Look for negation cues within a few tokens before the value phrase.
    Uses normalized tokens for robustness.
    """
    toks = _norm_text(question).split()
    vtoks = value_norm.split()
    L, n = len(toks), len(vtoks)
    for i in range(L - n + 1):
        if toks[i:i+n] == vtoks:
            prev = toks[max(0, i - window): i]
            if any(cue in prev for cue in NEG_CUES):
                return True
    return False

# ---------- Bool categories helpers ----------
BOOLEAN_TRUE_SYNS  = {"yes","Yes"}
BOOLEAN_FALSE_SYNS = {"no","No","null"}

def is_booleanish(col: str) -> bool:
    """A column is boolean-ish if its category vocab looks like yes/no or is tiny."""
    cats = [str(v).strip().lower() for v in _CATEGORICALS.get(col, [])]
    if not cats:
        return False
    if len(set(cats)) <= 3:
        return True
    return any(v in BOOLEAN_TRUE_SYNS or v in BOOLEAN_FALSE_SYNS for v in cats)

def positive_label_for(col: str) -> str:
    """Return the column's 'true' label (original casing) for filters like pain == TRUE."""
    cats = list(_CATEGORICALS.get(col, []))
    if not cats:
        return "true"
    for c in cats:
        if str(c).strip().lower() in BOOLEAN_TRUE_SYNS:
            return c
    # If no explicit true-synonym exists, pick something that doesn't look false.
    for c in cats:
        if str(c).strip().lower() not in BOOLEAN_FALSE_SYNS:
            return c
    return str(cats[0])

def negative_label_for(col: str) -> str:
    """Return the 'false' label (original casing) for negated filters like 'without bleeding'."""
    cats = list(_CATEGORICALS.get(col, []))
    if not cats:
        return "false"
    for c in cats:
        if str(c).strip().lower() in BOOLEAN_FALSE_SYNS:
            return c
    # Otherwise pick any label that's not the positive label.
    pos = positive_label_for(col)
    for c in cats:
        if c != pos:
            return c
    return str(cats[0])

def resolve_boolean_dimension_with_score(phrase: str) -> Tuple[Optional[str], float]:

    best, bscore = None, 0.0
    q = phrase or ""
    for col, aliases in ALIAS_DIM.items():
        if not is_booleanish(col):
            continue
        s = _score_match(q, col, aliases)
        if s > bscore:
            best, bscore = col, s
    return best, bscore

def resolve_boolean_dimension(phrase: str, threshold: float = 0.35) -> Optional[str]:
    best, score = resolve_boolean_dimension_with_score(phrase)
    return best if (best is not None and score >= threshold) else None

def parse_filters_from_question(question: str,
                                df: pd.DataFrame,
                                dimensions_norm: List[str],
                                field_resolver) -> List[Dict[str, Any]]:
    _ensure_value_index(df)

    q_raw = question or ""
    ql = q_raw.lower()
    qn = _norm_text(q_raw)
    filters: List[Dict[str, Any]] = []

    def add_filter(col_norm: str, op: str, value: Any):
        if col_norm not in df.columns:
            return
        if isinstance(value, str):
            vlow = value.strip().lower()
            if vlow in {"yes","Yes"}: value = "true"
            elif vlow in {"no","No","null"}: value = "false"
        filters.append({"field": col_norm, "op": op, "value": value})

    # figure out the target dimension (if any) so we can avoid folding it as a value
    target_dim = None
    # prefer explicit signals: by-phrase, then head-noun
    target_dim = parse_groupby_by_phrase_generic(question) or parse_head_dimension(question)[0]
    target_alias_norms = set()
    if target_dim:
        target_alias_norms = {_norm_text(a) for a in ALIAS_DIM.get(target_dim, [])}
        target_alias_norms.add(_norm_text(target_dim))

    # --- (A) FOLD FREE-TEXT VALUE MENTIONS ---
    mentioned_values = []
    for val_norm in _VALUE_INDEX.keys():
        if re.search(rf"\b{re.escape(val_norm)}\b", qn):
            mentioned_values.append(val_norm)
    for m in re.finditer(r'["“](.+?)["”]', q_raw):
        val_norm = _norm_text(m.group(1))
        if val_norm and val_norm not in mentioned_values and val_norm in _VALUE_INDEX:
            mentioned_values.append(val_norm)

    for val_norm in mentioned_values:
        # if this "value" equals the target dimension's alias (e.g., 'diagnosis'), skip
        if target_dim and val_norm in target_alias_norms:
            continue
        col = resolve_value_to_dimension(val_norm, q_raw)
        if not col:
            continue
        value = _VALUE_CANON.get((col, val_norm))
        if value is None:
            value = _lexical_snap(val_norm, [str(x) for x in _CATEGORICALS.get(col, [])])

        if is_booleanish(col):
            neg = has_local_negation(q_raw, val_norm)
            lbl = negative_label_for(col) if neg else positive_label_for(col)
            add_filter(col, "eq", lbl)
        else:
            neg = has_local_negation(q_raw, val_norm)
            add_filter(col, "ne" if neg else "eq", value)

    # --- (B) BOOLEAN PRESENCE from lists: "with/ had/ having ... pain, bleeding and nauesea"
    # Supports commas, '&', '/', 'and', 'plus', multiple items (2+)
    for m in re.finditer(r"\b(?:with|had|having|presenting\s+with|complain(?:ing)?\s+of|reported|reports|symptoms?\s+of)\s+([a-z0-9 _,&/\-]+)", ql):
        phrase = m.group(1).strip()
        # stop at a boundary keyword to avoid swallowing the rest of the sentence
        phrase = re.split(r"\b(?:in|on|for|during|from|over|between|within|by|when|where|who|that|which|,?\s*but)\b", phrase, maxsplit=1)[0]
        # split into terms
        terms = re.split(r"\s*(?:,|&|/|and|plus|\+)\s*", phrase)
        # keep up to a reasonable count to avoid noise
        terms = [t.strip(" .") for t in terms if t.strip(" .")]
        for term in terms[:8]:
            col = resolve_boolean_dimension_from_term(term) or FIELD_RESOLVER(term)
            if col and col in df.columns and is_booleanish(col):
                add_filter(col, "eq", positive_label_for(col))

    # --- (C) Classic rules (IN, BETWEEN, comparators) ---
    # IN lists
    for m in re.finditer(r"\b([a-z][a-z0-9 _\-]+?)\s+in\s*\(([^)]+)\)", ql):
        field = _normalize_name(m.group(1).strip())
        vals = [v.strip() for v in re.split(r"[;,]", m.group(2))]
        if field in df.columns:
            add_filter(field, "in", ",".join(vals))

    # BETWEEN numeric
    for m in re.finditer(r"\b([a-z][a-z0-9 _\-]+?)\s+between\s+([0-9\.]+)\s+and\s+([0-9\.]+)", ql):
        field = _normalize_name(m.group(1).strip())
        if field in df.columns:
            add_filter(field, "gte", m.group(2))
            add_filter(field, "lte", m.group(3))

    # Comparators
    for m in re.finditer(r"\b([a-z][a-z0-9 _\-]+?)\s*(>=|<=|>|<|=|==|equals|is)\s*([a-z0-9 _\-]+)", ql):
        field, op_raw, val = _normalize_name(m.group(1).strip()), m.group(2).strip(), m.group(3).strip()
        op_map = {"=":"eq","==":"eq","equals":"eq","is":"eq",">=":"gte","<=":"lte",">":"gt","<":"lt"}
        if field in df.columns:
            add_filter(field, op_map.get(op_raw, "eq"), val)

    # Light categorical spotting (only when field explicitly mentioned)
    for dim in dimensions_norm:
        if dim not in df.columns:
            continue
        field_mentioned = (dim.replace("_"," ") in ql) or (dim in ql)
        if not field_mentioned:
            continue
        cats = _CATEGORICALS.get(dim, [])
        for cat in cats:
            if re.search(rf"\b{re.escape(str(cat).lower())}\b", ql):
                add_filter(dim, "eq", cat)
                break

    # --- (D) Collapse equals on same column to IN; dedup everything ---
    bycol = {}
    for f in filters:
        key = (f["field"], f.get("op","eq"))
        bycol.setdefault(key, set()).add(str(f.get("value")))

    collapsed: List[Dict[str, Any]] = []
    for (field, op), vals in bycol.items():
        vals = list(vals)
        if op == "eq" and len(vals) > 1:
            collapsed.append({"field": field, "op": "in", "value": ",".join(vals)})
        else:
            for v in vals:
                collapsed.append({"field": field, "op": op, "value": v})

    # Dedup final
    out, seen = [], set()
    for f in collapsed:
        k = (f["field"], f.get("op","eq"), str(f.get("value")))
        if k not in seen:
            out.append(f); seen.add(k)
    return out

def _score_match(phrase: str, field: str, field_aliases: List[str]) -> float:
    """
    Score match quality between a free-form phrase and a field using its auto-aliases.
    Components:
      - exact alias match (1.0)
      - longest n-gram length normalized (0..1)
      - token Jaccard overlap (0..1)
    """
    q = phrase.lower().strip()
    q_toks = _tokens(q)
    if not q_toks: return 0.0

    # 1) exact alias (strong)
    if q in field_aliases:
        return 1.0

    # 2) longest alias substring length
    longest = 0
    for a in field_aliases:
        if a in q or q in a:
            longest = max(longest, len(a if len(a) < len(q) else q))
    contig = longest / max(8, len(q))  # normalize, cap minimum denom

    # 3) token overlap (Jaccard)
    f_toks = set(_tokens(field.replace("_", " ")))
    a_toks = set(it for a in field_aliases for it in _tokens(a))
    cand_toks = f_toks | a_toks
    inter = len(set(q_toks) & cand_toks)
    union = len(set(q_toks) | cand_toks)
    jacc = inter / union if union else 0.0

    # Weighted score: tune weights lightly (no domain constants)
    return 0.45 * contig + 0.55 * jacc

def resolve_best_dimension(phrase: str) -> Tuple[Optional[str], float]:
    best, bscore = None, 0.0
    for f, aliases in ALIAS_DIM.items():
        s = _score_match(phrase, f, aliases)
        if s > bscore:
            best, bscore = f, s
    return best, bscore

def resolve_best_measure(phrase: str) -> Tuple[Optional[str], float]:
    best, bscore = None, 0.0
    for f, aliases in ALIAS_MEAS.items():
        s = _score_match(phrase, f, aliases)
        if s > bscore:
            best, bscore = f, s
    return best, bscore

def parse_groupby_by_phrase_generic(question: str) -> Optional[str]:
    """
    Capture 'by <phrase>' and resolve to the best schema dimension generically.
    """
    ql = (question or "").lower()
    m = re.search(r"\bby\s+([a-z0-9 _\-]+?)(?=(?:\s+in|\s+for|\s+from|\s+during|\s+over|\s+between|[,\.]|$))", ql)
    if not m:
        return None
    phrase = m.group(1).strip()
    # try full phrase first
    cand, score = resolve_best_dimension(phrase)
    if cand and score >= 0.40:   # modest threshold; tune if needed
        return cand
    # else back off to sub-grams of the phrase
    toks = _tokens(phrase)
    for n in range(min(4, len(toks)), 0, -1):
        for i in range(len(toks)-n+1):
            span = " ".join(toks[i:i+n])
            cand, score = resolve_best_dimension(span)
            if cand and score >= 0.40:
                return cand
    return None


# ---------- Normalization helpers ----------
def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Global value index built from categorical columns ----------
_VALUE_INDEX: Optional[Dict[str, set]] = None        # value_norm -> {col_norm, ...}
_VALUE_CANON: Optional[Dict[Tuple[str, str], str]] = None  # (col_norm, value_norm) -> original value string

def _build_value_index(df: pd.DataFrame) -> None:
    """
    Build an index of all categorical values across columns so we can
    map free-text mentions to filters without hardcoding columns.
    """
    global _VALUE_INDEX, _VALUE_CANON
    _VALUE_INDEX, _VALUE_CANON = {}, {}
    for col, cats in _CATEGORICALS.items():
        for v in cats:
            key = _norm_text(v)
            _VALUE_INDEX.setdefault(key, set()).add(col)
            _VALUE_CANON[(col, key)] = v

def _ensure_value_index(df: pd.DataFrame) -> None:
    global _VALUE_INDEX
    if _VALUE_INDEX is None:
        _build_value_index(df)

def parse_head_dimension(question: str, min_score: float = 0.40) -> Tuple[Optional[str], float]:
    
    q = (question or "")
    ql = q.lower()
    # Grab the noun chunk after which/what up to a boundary token
    m = re.search(
        r"^\s*(which|what)\s+([a-z0-9 _\-]+?)(?=\s+(?:is|are|was|were|most|least|in|from|during|over|between|by|,|\?|\.|$))",
        ql
    )
    if not m:
        return None, 0.0
    phrase = m.group(2).strip()
    # Try full phrase
    best, score = resolve_best_dimension(phrase)
    if best and score >= min_score:
        return best, score
    # Back off to sub-grams (up to 4 tokens)
    toks = re.findall(r"[a-z0-9]+", phrase)
    for n in range(min(4, len(toks)), 0, -1):
        for i in range(len(toks)-n+1):
            span = " ".join(toks[i:i+n])
            cand, s = resolve_best_dimension(span)
            if cand and s >= min_score:
                return cand, s
    return None, 0.0


NUMERIC_OPS = {"avg", "mean", "sum", "min", "max"}
def _op_canonical(op: str) -> str:
    op = (op or "").lower().strip()
    return "avg" if op in {"avg", "mean"} else op

def _is_numericish(series: pd.Series, thresh: float = 0.7) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.notna().mean() >= thresh

def _apply_metric(df: pd.DataFrame, groupby: list, metric: dict) -> pd.DataFrame:
    op = metric["op"]
    field = metric["field"]

    if op == "count":
        if groupby:
            out = df.groupby(groupby, dropna=False).size().reset_index(name="metric")
        else:
            out = pd.DataFrame({"metric": [len(df)]})
        return out

    # numeric ops
    s = pd.to_numeric(df[field], errors="coerce")
    agg_map = {"avg": "mean", "sum": "sum", "min": "min", "max": "max"}
    fn = agg_map[op]

    if groupby:
        out = df[groupby].copy()
        out["__val__"] = s
        out = out.groupby(groupby, dropna=False)["__val__"].agg(fn).reset_index(name="metric")
    else:
        out = pd.DataFrame({"metric": [getattr(s, fn)()]})
    return out

def resolve_boolean_dimension_with_score(phrase: str) -> Tuple[Optional[str], float]:
    best, bscore = None, 0.0
    q = phrase or ""
    for col, aliases in ALIAS_DIM.items():
        if not is_booleanish(col):
            continue
        s = _score_match(q, col, aliases)
        if s > bscore:
            best, bscore = col, s
    return best, bscore

def resolve_boolean_dimension_from_term(term: str,
                                        min_alias_score: float = 0.35,
                                        min_fuzzy_ratio: float = 0.78) -> Optional[str]:
    """
    Resolve a free-form term (possibly misspelled) to a boolean-ish dimension.
    1) Try alias scoring (schema-driven).
    2) Fallback: fuzzy match against each alias using difflib.
    """
    # 1) Schema alias scoring
    cand, score = resolve_boolean_dimension_with_score(term)
    if cand and score >= min_alias_score:
        return cand

    # 2) Fuzzy fallback (e.g., 'nauesea' -> 'nausea')
    nt = _norm_text(term)
    best, best_ratio = None, 0.0
    for col, aliases in ALIAS_DIM.items():
        if not is_booleanish(col):
            continue
        for a in aliases:
            r = difflib.SequenceMatcher(None, nt, _norm_text(a)).ratio()
            if r > best_ratio:
                best, best_ratio = col, r
    return best if best and best_ratio >= min_fuzzy_ratio else None

# =============================================================================
# MODEL WRAPPER
# =============================================================================
INTENT_PROMPT = """
You write STRICT JSON describing analytics intent. Use only these columns:
{inventory}
Time column is "{date_col}".

Return ONLY valid JSON with keys:
{{
  "time": {{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}} | null,
  "filters": [{{"field":"<col>", "op":"eq|ne|gt|lt|gte|lte|in|contains", "value":"<literal>"}}],
  "groupby": ["<col>", ...],
  "metric": {{"op":"count|avg|sum|min|max","field":"<col|*>" }},
  "topk": {{"k":5,"order":"desc"}} | null
}}
No prose. No comments.
Question: {question}
""".strip()

DSL_PROMPT = """
You convert questions into compact DSL for analytics. Use only inventory columns.

INVENTORY:
{inventory}

Time column: "{date_col}"

DSL lines (one per line):
- TIME|start=YYYY-MM-DD|end=YYYY-MM-DD|field={date_col}
- FILTER|field=<col>|op=eq|value=<literal>
- GROUPBY|<col>[,<col2>]
- METRIC|op=(count|avg|sum|min|max)|field=<col or *>
- TOPK|k=<int>|by=metric|order=(asc|desc)

Rules:
- Use only inventory columns.
- Default metric is count if none implied.
- Include TIME if dates are mentioned.
- Output ONLY DSL, no prose.

Question: {question}
DSL:
""".strip()

class NL2Planner:
    def __init__(self, model_path: str, fields_norm: List[str], date_col_norm: str):
        if not HAVE_TRANSFORMERS:
            raise RuntimeError("Transformers not installed. `pip install transformers torch`")
        self.fields = fields_norm
        self.date_col = date_col_norm
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        )
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.device = self.model.device

    def gen_intent_json(self, question: str, max_new_tokens: int = 256) -> str:
        prompt = INTENT_PROMPT.format(inventory=inventory_for_prompt(), date_col=self.date_col, question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # deterministic
                num_beams=1
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def gen_dsl(self, question: str, max_new_tokens: int = 180) -> str:
        prompt = DSL_PROMPT.format(inventory=inventory_for_prompt(), date_col=self.date_col, question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.5,
                num_beams=1
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


# =============================================================================
# INTENT & PLAN MODELS
# =============================================================================
class TimeSpec(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None

class FilterSpec(BaseModel):
    field: str
    op: str = "eq"
    value: Any = None

class MetricSpec(BaseModel):
    op: str = "count"
    field: str = "*"

class TopKSpec(BaseModel):
    k: int = 5
    order: str = "desc"

class Intent(BaseModel):
    time: Optional[TimeSpec] = None
    filters: List[FilterSpec] = []
    groupby: List[str] = []
    metric: MetricSpec = MetricSpec()
    topk: Optional[TopKSpec] = None

class Plan(BaseModel):
    filter: List[Dict[str, Any]] = []
    time: Optional[Dict[str, Any]] = None
    groupby: List[str] = []
    metric: Dict[str, Any] = {"op": "count", "field": "*"}
    topk: Optional[Dict[str, Any]] = None

# =============================================================================
# DSL PARSER / UTILS
# =============================================================================
def _unquote(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def _is_placeholder_date(s: Optional[str]) -> bool:
    if not s: return False
    return s in ("", "0000-00-00", "None", "null")

def parse_dsl(dsl: str) -> Plan:
    plan = Plan()
    lines = [l.strip() for l in (dsl or "").splitlines() if l.strip()]
    for line in lines:
        if line.startswith("TIME|"):
            kv = dict(x.split("=", 1) for x in line.split("|")[1:] if "=" in x)
            plan.time = {
                "start": kv.get("start"),
                "end": kv.get("end"),
                "field": _normalize_name(kv.get("field") or DATE_COL),
            }
            s = plan.time.get("start"); e = plan.time.get("end")
            if _is_placeholder_date(s): plan.time["start"] = None
            if _is_placeholder_date(e): plan.time["end"] = None
            if not plan.time.get("start") and not plan.time.get("end"):
                plan.time = None
            continue

        if line.startswith("FILTER|"):
            kv = dict(x.split("=", 1) for x in line.split("|")[1:] if "=" in x)
            plan.filter.append({
                "field": kv.get("field"),
                "op": (kv.get("op") or "eq"),
                "value": _unquote(kv.get("value")),
            })
        elif line.startswith("GROUPBY|"):
            rest = line.split("|", 1)[1]
            plan.groupby = [_normalize_name(c) for c in rest.split(",") if c.strip()]
        elif line.startswith("METRIC|"):
            kv = dict(x.split("=", 1) for x in line.split("|")[1:] if "=" in x)
            plan.metric = {
                "op": (kv.get("op") or "count").lower(),
                "field": _normalize_name(kv.get("field", "*")),
            }
        elif line.startswith("TOPK|"):
            kv = dict(x.split("=", 1) for x in line.split("|")[1:] if "=" in x)
            try:
                k = int(kv.get("k", "1"))
            except Exception:
                k = 1
            plan.topk = {"k": k, "by": kv.get("by", "metric"), "order": kv.get("order", "desc")}
    return plan

def sanitize_and_fix_dsl(question: str, raw_dsl: str, date_col_norm: str, all_fields_norm: List[str]) -> str:
    """
    Clamp TIME field to the dataset date column and drop invalid fields in GROUPBY/METRIC.
    """
    lines_in = [l for l in (raw_dsl or "").splitlines() if l.strip()]
    out = []
    for line in lines_in:
        if line.startswith("TIME|"):
            # Replace field=... with date_col_norm
            line = re.sub(r"field=[^|]+", f"field={date_col_norm}", line)
            out.append(line); continue
        if line.startswith("GROUPBY|"):
            parts = line.split("|", 1)[1].split(",")
            keep = [c.strip() for c in parts if _normalize_name(c) in all_fields_norm]
            if keep:
                out.append("GROUPBY|" + ",".join(_normalize_name(c) for c in keep))
            continue
        if line.startswith("METRIC|"):
            # keep op and clamp field to known or *
            kv = dict(x.split("=", 1) for x in line.split("|")[1:] if "=" in x)
            fld = _normalize_name(kv.get("field", "*"))
            fld = fld if fld in all_fields_norm or fld == "*" else "*"
            op = kv.get("op", "count").lower()
            out.append(f"METRIC|op={op}|field={fld}")
            continue
        out.append(line)
    return "\n".join(out)

# =============================================================================
# EXECUTION (filters with numeric-aware eq + value snapping)
# =============================================================================
def _lexical_snap(value: str, options: List[str]) -> str:
    if not options:
        return value
    v = (value or "").strip()
    if not v:
        return v
    for o in options:
        if v.lower() == o.lower():
            return o
    if HAVE_RAPIDFUZZ:
        hit = process.extractOne(v, options, scorer=fuzz.WRatio, score_cutoff=80)
        if hit:
            return hit[0]
    vlow = v.lower()
    contains = [o for o in options if vlow in o.lower()]
    if contains:
        return contains[0]
    return v

def _apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]], field_resolver) -> pd.DataFrame:
    out = df
    for f in filters or []:
        raw_field = f.get("field", "")
        col = field_resolver(raw_field) or _normalize_name(raw_field)
        op  = (f.get("op") or "eq").lower()
        val = f.get("value")

        if col not in out.columns:
            continue

        # snap category value if applicable
        if op in ("eq", "==", "=") and not pd.api.types.is_numeric_dtype(out[col]):
            choices = _CATEGORICALS.get(col, [])
            if choices:
                val = _lexical_snap(str(val), choices)

        if op in ("eq", "==", "="):
            if pd.api.types.is_numeric_dtype(out[col]):
                lhs = pd.to_numeric(out[col], errors="coerce")
                rhs = pd.to_numeric(val, errors="coerce")
                out = out[lhs == rhs]
            else:
                out = out[out[col].astype(str).str.lower() == str(val).lower()]
        elif op in ("ne", "!=", "not_eq"):
            if pd.api.types.is_numeric_dtype(out[col]):
                lhs = pd.to_numeric(out[col], errors="coerce")
                rhs = pd.to_numeric(val, errors="coerce")
                out = out[lhs != rhs]
            else:
                out = out[out[col].astype(str).str.lower() != str(val).lower()]
        elif op in (">", "gt"):
            out = out[pd.to_numeric(out[col], errors="coerce") > pd.to_numeric(val, errors="coerce")]
        elif op in ("<", "lt"):
            out = out[pd.to_numeric(out[col], errors="coerce") < pd.to_numeric(val, errors="coerce")]
        elif op in (">=", "gte"):
            out = out[pd.to_numeric(out[col], errors="coerce") >= pd.to_numeric(val, errors="coerce")]
        elif op in ("<=", "lte"):
            out = out[pd.to_numeric(out[col], errors="coerce") <= pd.to_numeric(val, errors="coerce")]
        elif op in ("contains", "like"):
            out = out[out[col].astype(str).str.contains(str(val), case=False, na=False)]
        elif op == "in":
            raw_vals = [v.strip() for v in str(val).split(",")]
            if not pd.api.types.is_numeric_dtype(out[col]):
                choices = _CATEGORICALS.get(col, [])
                vals = [ _lexical_snap(v, choices).lower() for v in raw_vals ]
                out = out[out[col].astype(str).str.lower().isin(vals)]
            else:
                vals = [ pd.to_numeric(v, errors="coerce") for v in raw_vals ]
                out = out[pd.to_numeric(out[col], errors="coerce").isin(vals)]
    return out

def execute_plan(plan: Plan, df: pd.DataFrame, date_col_norm: str, field_resolver) -> Tuple[List[Dict[str, Any]], int]:
    if df.empty:
        return [], 0

    # TIME
    if plan.time and (plan.time.get("start") or plan.time.get("end")) and date_col_norm in df.columns:
        s = plan.time.get("start"); e = plan.time.get("end")
        s_dt = pd.to_datetime(s) if s else None
        e_dt = pd.to_datetime(e) if e else None
        m = pd.Series(True, index=df.index)
        if s_dt is not None: m &= df[date_col_norm] >= s_dt
        if e_dt is not None: m &= df[date_col_norm] <= e_dt
        df = df[m]

    # FILTERS
    df = _apply_filters(df, plan.filter, field_resolver)

    # GROUPBY + METRIC
    gb_cols = [c for c in (plan.groupby or []) if c in df.columns]
    metric_op = (plan.metric or {}).get("op", "count").lower()
    metric_field_raw = (plan.metric or {}).get("field", "*")
    metric_field = field_resolver(metric_field_raw) or metric_field_raw

    if not gb_cols:
        if metric_op == "count":
            rows = [{"value": int(len(df))}]
        else:
            series = pd.to_numeric(df[metric_field], errors="coerce") if metric_field in df.columns else pd.Series(dtype=float)
            if metric_op == "avg":
                rows = [{"value": float(series.mean(skipna=True)) if len(series) else 0.0}]
            elif metric_op == "sum":
                rows = [{"value": float(series.sum(skipna=True)) if len(series) else 0.0}]
            elif metric_op == "min":
                rows = [{"value": float(series.min(skipna=True)) if len(series) else math.nan}]
            elif metric_op == "max":
                rows = [{"value": float(series.max(skipna=True)) if len(series) else math.nan}]
            else:
                rows = [{"value": int(len(df))}]
    else:
        if metric_op == "count":
            g = df.groupby(gb_cols, dropna=False).size().reset_index(name="value")
        else:
            if metric_field in df.columns:
                s = pd.to_numeric(df[metric_field], errors="coerce")
                g = df.assign(__metric__=s).groupby(gb_cols, dropna=False)["__metric__"]
                if metric_op == "avg": g = g.mean().reset_index(name="value")
                elif metric_op == "sum": g = g.sum().reset_index(name="value")
                elif metric_op == "min": g = g.min().reset_index(name="value")
                elif metric_op == "max": g = g.max().reset_index(name="value")
                else: g = df.groupby(gb_cols, dropna=False).size().reset_index(name="value")
            else:
                g = df.groupby(gb_cols, dropna=False).size().reset_index(name="value")

        if plan.topk:
            order = str(plan.topk.get("order", "desc")).lower()
            asc = order.startswith("asc")
            k = int(plan.topk.get("k", 5))
            g = g.sort_values("value", ascending=asc).head(k)
        rows = g.to_dict(orient="records")

    return rows, len(df)

# =============================================================================
# ANSWER SYNTHESIS
# =============================================================================
def pretty_month_year(s: Optional[str]) -> Optional[str]:
    if not s: return None
    try:
        d = pd.to_datetime(s); return d.strftime("%B %Y")
    except Exception:
        return s

def period_text(start: Optional[str], end: Optional[str]) -> str:
    s = pretty_month_year(start); e = pretty_month_year(end)
    if s and e: return f" from {s} to {e}" if s != e else f" in {s}"
    if s: return f" since {s}"
    if e: return f" until {e}"
    return ""

def humanize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("_", " ")).strip()

def synthesize_answer(plan: Plan, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No matching records for the requested criteria."
    metric_op = (plan.metric or {}).get("op","count").lower()
    metric_field = (plan.metric or {}).get("field","")
    period = period_text(plan.time.get("start") if plan.time else None,
                         plan.time.get("end") if plan.time else None)
    def pick_metric_key(r: Dict[str, Any]) -> str:
        if "value" in r: return "value"
        for k,v in r.items():
            if isinstance(v,(int,float)): return k
        return "value"
    mk = pick_metric_key(rows[0])

    if plan.topk and len(rows) > 1:
        k = plan.topk.get("k", len(rows))
        qword = "least" if str(plan.topk.get("order","desc")).lower().startswith("asc") else "most"
        subj = humanize((plan.groupby or ["records"])[0] if plan.groupby else "records")
        items = ", ".join([f"{r.get((plan.groupby or ['label'])[0], '(unspecified)')} ({r[mk]})" for r in rows[:k]])
        return f"Top {min(k,len(rows))} {qword} {subj}{period}: {items}."
    else:
        if plan.groupby:
            key = plan.groupby[0]
            r0 = rows[0]
            lab = r0.get(key, "overall")
            val = r0.get(mk)
            if metric_op == "count":
                return f"{humanize(key)}{period}: {lab} had {int(val)} cases."
            else:
                return f"{humanize(metric_field)} for {lab}{period}: {val}."
        else:
            v = rows[0].get(mk)
            if metric_op == "count":
                return f"There were {int(v)} cases{period}."
            else:
                return f"{humanize(metric_field)}{period}: {v}."

# =============================================================================
# INTENT → PLAN → DSL + REPAIR
# =============================================================================
def _ngrams(tokens, n=3):
    for k in range(1, n+1):
        for i in range(len(tokens) - k + 1):
            yield " ".join(tokens[i:i+k])

def _singularize(word: str) -> str:
    w = word.lower()
    if w.endswith("ies"): return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss"): return w[:-1]
    return w

def infer_groupby_from_question(question: str, dimensions_norm: List[str], field_resolver) -> List[str]:
    q = re.sub(r"[^a-zA-Z0-9 ]+", " ", (question or "").lower()).strip()
    toks = [_singularize(t) for t in q.split()]
    hits, seen = [], set()
    for span in _ngrams(toks, n=3):
        cand = field_resolver(span)
        if cand and cand in dimensions_norm and cand not in seen:
            hits.append(cand); seen.add(cand)
    if not hits:
        for t in toks:
            cand = field_resolver(t)
            if cand and cand in dimensions_norm and cand not in seen:
                hits.append(cand); seen.add(cand)
    return hits

def validate_and_autofix_plan(question: str,
                              plan: Plan,
                              df: pd.DataFrame,
                              date_col_norm: str,
                              dimensions_norm: List[str],
                              measures_norm: List[str],
                              field_resolver) -> Tuple[Plan, List[str]]:

    warnings: List[str] = []

    # ---------------- TIME: prefer question's dates; otherwise clamp field ----------------
    qs, qe = extract_dates(question)
    if qs or qe:
        if (not plan.time) or (plan.time.get("start") != qs or plan.time.get("end") != qe):
            plan.time = {"start": qs, "end": qe, "field": date_col_norm}
            warnings.append(f"Time range set from question: {qs or '…'} to {qe or '…'}.")
    else:
        if plan.time:
            plan.time["field"] = date_col_norm

    # ---------------- GROUPBY: by-phrase > head-noun > generic > fallback ----------------
    gb_locked = False

    # (1) by-phrase (generic resolver you already have)
    gb_by = parse_groupby_by_phrase_generic(question)
    if gb_by:
        plan.groupby = [gb_by]
        gb_locked = True
        warnings.append(f"GROUPBY set from 'by …': '{gb_by}'.")

    # (2) 'Which/What <dimension> …' (head noun)
    if not gb_locked:
        gb_head, head_score = parse_head_dimension(question, min_score=0.40)
        if gb_head:
            plan.groupby = [gb_head]
            gb_locked = True
            warnings.append(f"GROUPBY set from question head: '{gb_head}' (score={head_score:.2f}).")

    # (3) Generic inference on full question (only if still unlocked)
    if (not plan.groupby) or any(g not in df.columns for g in (plan.groupby or [])):
        cand, score = resolve_best_dimension(question)
        if cand and cand in df.columns and score >= 0.40:
            plan.groupby = [cand]
            warnings.append(f"GROUPBY inferred generically as '{cand}' (score={score:.2f}).")
        else:
            small_cats = [c for c in DIMENSIONS_NORM if c in df.columns and df[c].nunique(dropna=True) <= 100]
            if small_cats:
                plan.groupby = [small_cats[0]]
                warnings.append(f"GROUPBY defaulted to '{small_cats[0]}' (small cardinality).")

    # (4) Never group by the date column; swap it if that happened
    if plan.groupby and any(g == date_col_norm for g in plan.groupby):
        plan.groupby = [g for g in plan.groupby if g != date_col_norm]
        cand, score = resolve_best_dimension(question)
        if cand and cand in df.columns and score >= 0.40:
            plan.groupby = [cand]
            warnings.append(f"Dropped date column from GROUPBY; set '{cand}' instead (score={score:.2f}).")

    # ---------------- METRIC: parse op + resolve field generically from measures ----------------
    op_q, field_q = parse_metric_from_question(question, df, MEASURES_NORM, FIELD_RESOLVER)
    if op_q:
        plan.metric["op"] = op_q
    if field_q and op_q and op_q != "count":
        plan.metric["field"] = field_q

    # Type safety / fallback
    op = (plan.metric or {}).get("op", "count").lower()
    fld = (plan.metric or {}).get("field", "*")

    if op != "count":
        col = FIELD_RESOLVER(fld) or fld
        if col not in df.columns:
            plan.metric = {"op": "count", "field": "*"}
            warnings.append(f"Metric field '{fld}' not found; using count.")
        else:
            cop = _op_canonical(op)
            if cop in NUMERIC_OPS:
                if not _is_numericish(df[col]):
                    plan.metric = {"op": "count", "field": "*"}
                    pretty = {"avg": "average", "sum": "sum", "min": "minimum", "max": "maximum"}[cop]
                    warnings.append(f"Requested {pretty} of '{col}' but column is not numeric; using count.")
                else:
                    # Canonicalize 'mean' -> 'avg' so your executor has a small op surface
                    plan.metric["op"] = cop
            else:
                # Unknown/unsupported op → count fallback
                plan.metric = {"op": "count", "field": "*"}
                warnings.append(f"Unknown metric op '{op}'; using count.")
    else:
        plan.metric = {"op": "count", "field": "*"}


    # ---------------- TOPK: infer/add; normalize if partially specified ----------------
    def _neutralize_at_least_most(s: str) -> str:
        s = re.sub(r"\bat\s+least\b", " ", s)
        s = re.sub(r"\bat\s+most\b", " ", s)
        return s

    def has_singular_superlative(q: str) -> bool:
        ql = _neutralize_at_least_most((q or "").lower())
        singular = bool(
            re.search(r"\b(single\s+)?(most|least)\s+(common|frequent)\b", ql) or
            re.search(r"\b(the\s+)?(most|least)\s+(prevalent|dominant|popular)\b", ql)
        )

        if re.search(r"^\s*(which|what)\b", (q or "").lower()) and re.search(r"\b(most|least)\b", (q or "").lower()):
            singular = True
        return singular

    def _has_rank_signal(q: str) -> bool:
        ql = _neutralize_at_least_most((q or "").lower())
        return bool(
            re.search(r"\btop[-\s]*\d+\b", ql) or
            re.search(r"\bbottom[-\s]*\d+\b", ql) or
            re.search(r"\b(most|highest|top|prevalent|dominant|leading|largest|biggest|greatest|max(?:imum)?)\b", ql) or
            re.search(r"\b(least|lowest|fewest|smallest|min(?:imum)?|bottom|rarest)\b", ql) or
            re.search(r"\b(in\s+descending\s+order|high\s+to\s+low)\b", ql) or
            re.search(r"\b(in\s+ascending\s+order|low\s+to\s+high)\b", ql)
        )

    qk, qorder = parse_topk_from_question(question)
    singular = has_singular_superlative(question)

    if not plan.topk:
        # Only add TOPK if we have (or can infer) a sensible GROUPBY
        if (qk is not None) or (qorder is not None) or _has_rank_signal(question) or singular:
            inferred = infer_groupby_from_question(question, dimensions_norm, field_resolver) if not plan.groupby else plan.groupby
            if plan.groupby or inferred:
                k_val = int(qk or (1 if singular else 5))
                order_val = qorder or ("asc" if re.search(r"\b(least|lowest|fewest|rarest)\b", (question or '').lower()) else "desc")
                plan.topk = {"k": k_val, "by": "metric", "order": order_val}
                if not plan.groupby and inferred:
                    plan.groupby = [inferred[0]]
                    warnings.append(f"TOPK required GROUPBY; set '{inferred[0]}'.")
                warnings.append(f"TOPK inferred from question: k={plan.topk['k']}, order={plan.topk['order']}.")
    else:
        # Normalize existing TOPK emitted by the model
        try:
            plan.topk["k"] = int(plan.topk.get("k") or qk or (1 if singular else 5))
        except Exception:
            plan.topk["k"] = 1 if singular else 5

        ord_raw = (plan.topk.get("order") or qorder or "").lower().strip()
        if ord_raw.startswith("asc") or ord_raw in {"ascending", "increasing", "low to high"}:
            plan.topk["order"] = "asc"
        elif ord_raw.startswith("desc") or ord_raw in {"descending", "decreasing", "high to low"}:
            plan.topk["order"] = "desc"
        else:
            plan.topk["order"] = ("asc" if re.search(r"\b(least|lowest|fewest|rarest)\b", (question or '').lower()) else "desc")

        if singular and plan.topk["k"] > 1:
            plan.topk["k"] = 1
            warnings.append("TOPK k forced to 1 for singular phrasing ('most/least common').")

    # Final safety: if TOPK exists but no GROUPBY (shouldn't happen due to gating), try to infer
    if plan.topk and not plan.groupby:
        inferred = infer_groupby_from_question(question, dimensions_norm, field_resolver)
        if inferred:
            plan.groupby = [inferred[0]]
            warnings.append(f"TOPK required GROUPBY; set '{inferred[0]}'.")

    # Always ensure the TIME field name is correct if time exists
    if plan.time:
        plan.time["field"] = date_col_norm


    # ---------------- FILTERS: infer from question; merge with any existing ----------------
    auto_filters = parse_filters_from_question(question, df, dimensions_norm, FIELD_RESOLVER)
    if auto_filters:
        existing = {(f["field"], f.get("op","eq"), str(f.get("value"))) for f in (plan.filter or [])}
        merged = list(plan.filter or [])
        for f in auto_filters:
            key = (f["field"], f.get("op","eq"), str(f.get("value")))
            if key not in existing:
                merged.append(f)
        if merged != (plan.filter or []):
            plan.filter = merged
            # brief readable blurb
            blurb = "; ".join([f"{f['field']} {f.get('op','eq')} {f.get('value')}" for f in auto_filters[:4]])
            if len(auto_filters) > 4: blurb += " …"
            warnings.append(f"Filters folded from values: {blurb}.")

    # ---------------- COUNT-style guard (pure counts only) ----------------
    ql = (question or "").lower()
    is_count_style = bool(re.search(r'\b(how\s+many|number\s+of|count\s+of|cases?)\b', ql))
    has_rank_signal = bool(
        re.search(r'\btop\b', ql) or
        re.search(r'(?<!at\s)\b(most|least|highest|lowest)\b', ql) or
        (plan.topk is not None)
    )
    has_by_phrase = bool(re.search(r'\bby\s+[a-z]', ql))

    # Only drop GROUPBY when it's a plain count (no rank, no "by ...")
    if is_count_style and not has_rank_signal and not has_by_phrase:
        if plan.groupby:
            plan.groupby = []
            warnings.append("Count-style question: dropped GROUPBY.")

    return plan, warnings

def intent_to_plan(intent: Intent, date_col_norm: str, dimensions_norm: List[str], field_resolver):
    p = Plan()
    if intent.time and (intent.time.start or intent.time.end):
        p.time = {"start": intent.time.start, "end": intent.time.end, "field": date_col_norm}
    for f in intent.filters or []:
        p.filter.append({"field": field_resolver(f.field) or _normalize_name(f.field),
                         "op": f.op, "value": _unquote(f.value)})
    p.groupby = [ (field_resolver(g) or _normalize_name(g)) for g in (intent.groupby or []) ]
    p.metric = {"op": intent.metric.op.lower(), "field": _normalize_name(intent.metric.field)}
    p.topk = ({"k": intent.topk.k, "by": "metric", "order": intent.topk.order.lower()} if intent.topk else None)
    return p

def plan_to_dsl(plan: Plan, date_col_norm: str) -> str:
    lines = []
    if plan.time:
        s = plan.time.get("start") or ""; e = plan.time.get("end") or ""
        lines.append(f"TIME|start={s}|end={e}|field={date_col_norm}")
    for f in plan.filter:
        lines.append(f"FILTER|field={f['field']}|op={f['op']}|value={f['value']}")
    if plan.groupby:
        lines.append("GROUPBY|" + ",".join(plan.groupby))
    if plan.metric:
        lines.append(f"METRIC|op={plan.metric.get('op','count')}|field={plan.metric.get('field','*')}")
    if plan.topk:
        o = plan.topk.get("order","desc"); k = int(plan.topk.get("k",5))
        lines.append(f"TOPK|k={k}|by=metric|order={o}")
    return "\n".join(lines)

def robust_parse_or_repair(question: str,
                           raw_dsl: str,
                           date_col_norm: str,
                           all_fields_norm: List[str],
                           df: pd.DataFrame,
                           dimensions_norm: List[str],
                           measures_norm: List[str],
                           field_resolver) -> Tuple[str, Plan, List[str]]:
    warnings = []
    dsl = sanitize_and_fix_dsl(question, raw_dsl, date_col_norm, all_fields_norm)
    plan = parse_dsl(dsl)
    plan, w = validate_and_autofix_plan(question, plan, df, date_col_norm, dimensions_norm, measures_norm, field_resolver)
    warnings.extend(w)

    # Keep DSL aligned with fixed plan
    dsl_fixed = plan_to_dsl(plan, date_col_norm)
    return dsl_fixed, plan, warnings

# =============================================================================
# API SCHEMA
# =============================================================================
class QueryIn(BaseModel):
    question: str
    max_new_tokens: Optional[int] = 180
    force_dates_from_query: Optional[bool] = True  # kept for backward-compat flag
    force_groupby_from_query: Optional[bool] = True

class QueryOut(BaseModel):
    question: str
    dsl: str
    plan: Dict[str, Any]
    rows: List[Dict[str, Any]]
    natural_answer: Optional[str] = None
    answer: Optional[str] = None
    meta: Dict[str, Any]

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="Analytics Intent/DSL Service (T5)", version="2.0")

# Serve UI if present
if Path("app/static").exists():
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

if Path("index.html").exists():
    @app.get("/", response_class=HTMLResponse)
    def root():
        return HTMLResponse(Path("index.html").read_text(encoding="utf-8"))
elif Path("app/static/index.html").exists():
    @app.get("/ui", response_class=HTMLResponse)
    def root_alt():
        return HTMLResponse(Path("app/static/index.html").read_text(encoding="utf-8"))

_t5: Optional[NL2Planner] = None
def get_planner() -> NL2Planner:
    global _t5
    if _t5 is None:
        _t5 = NL2Planner(MODEL_PATH, ALL_FIELDS_NORM, DATE_COL_NORM)
    return _t5


@app.get("/health")
def health():
    df = load_dataframe()
    return {
        "ok": True,
        "rows": int(len(df)),
        "columns": list(df.columns)[:20],
        "date_col": DATE_COL_NORM,
        "model": MODEL_PATH,
        "transformers_loaded": HAVE_TRANSFORMERS,
    }

@app.get("/schema")
def schema():
    return {
        "schema": SCHEMA,
        "normalized": {
            "date_col": DATE_COL_NORM,
            "dimensions": DIMENSIONS_NORM,
            "measures": MEASURES_NORM
        }
    }

@app.get("/columns")
def columns():
    return {"date_col": DATE_COL_NORM, "dimensions": DIMENSIONS_NORM, "measures": MEASURES_NORM}

@app.post("/query", response_model=QueryOut)
def query(inp: QueryIn):
    t0 = time.time()
    q = (inp.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question.")

    df = load_dataframe()
    warnings: List[str] = []

    try:
        planner = get_planner()
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Model load error: {ex}")

    # ---------- Primary path: INTENT JSON (deterministic) ----------
    try:
        raw_json = planner.gen_intent_json(q, max_new_tokens=256)
        intent = Intent.model_validate(json.loads(raw_json))
        plan_obj = intent_to_plan(intent, DATE_COL_NORM, DIMENSIONS_NORM, FIELD_RESOLVER)
        plan_obj, w = validate_and_autofix_plan(q, plan_obj, df, DATE_COL_NORM, DIMENSIONS_NORM, MEASURES_NORM, FIELD_RESOLVER)
        warnings.extend(w)
        dsl = plan_to_dsl(plan_obj, DATE_COL_NORM)
    except Exception:
        # ---------- Fallback: DSL generation + repair ----------
        raw_dsl = planner.gen_dsl(q, max_new_tokens=max(64, int(inp.max_new_tokens or 180)))
        dsl, plan_obj, w = robust_parse_or_repair(
            question=q,
            raw_dsl=raw_dsl,
            date_col_norm=DATE_COL_NORM,
            all_fields_norm=ALL_FIELDS_NORM,
            df=df,
            dimensions_norm=DIMENSIONS_NORM,
            measures_norm=MEASURES_NORM,
            field_resolver=FIELD_RESOLVER
        )
        warnings.extend(w)
        # Optional strong nudge from question when GROUPBY is obvious
        if getattr(inp, "force_groupby_from_query", True):
            inferred = infer_groupby_from_question(q, DIMENSIONS_NORM, FIELD_RESOLVER)
            if inferred and (not plan_obj.groupby or plan_obj.groupby[0] != inferred[0]):
                plan_obj.groupby = [inferred[0]]
                dsl = plan_to_dsl(plan_obj, DATE_COL_NORM)

    # ---------- Execute ----------
    try:
        rows, scanned = execute_plan(plan_obj, df, DATE_COL_NORM, FIELD_RESOLVER)
    except Exception as ex:
        raise HTTPException(status_code=500, detail={"error": str(ex), "dsl": dsl})

    # ---------- Answer + meta ----------
    nat = synthesize_answer(plan_obj, rows)
    meta = {
        "latency_ms": int((time.time() - t0) * 1000),
        "rowcount": len(rows),
        "scanned_rows": int(scanned)
    }
    if warnings:
        meta["warnings"] = warnings

    return QueryOut(
        question=q,
        dsl=dsl,
        plan=json.loads(plan_obj.model_dump_json()),
        rows=rows,
        natural_answer=nat,
        answer=nat,
        meta=meta
    )
