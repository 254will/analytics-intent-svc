# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates jq tini \
  && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir --prefer-binary -r /srv/requirements.txt

COPY app/ /srv/app/

ARG GH_OWNER=254will
ARG GH_REPO=analytics-intent-svc
ARG GH_TAG=v1.0.0
ARG MODEL_TAR=model-950mb.tar.gz
ARG TOKEN=github_pat11ASMJZ5Q0SL0lALSNJWRO_gUrClmwDgLZohoBig5c8LXCMZasI6tED5tXiFFk9VpI72XNJYVJM5HfoijIe

# Use BuildKit secret for token (private repos), never store it in layers.
RUN --mount=type=secret,id=gh_token,target=/run/secrets/gh_token \
    set -euo pipefail; \
    mkdir -p /srv/artifacts; \
    if [ -z "$(ls -A /srv/artifacts 2>/dev/null || true)" ]; then \
      TOKEN=""; [ -s /run/secrets/gh_token ] && TOKEN=$(cat /run/secrets/gh_token); \
      if [ -n "$TOKEN" ]; then \
        echo "🔐 Private/robust path: resolving ${GH_TAG} and ${MODEL_TAR} via GitHub API"; \
        # Get release by tag
        RELEASE_JSON=$(curl -fsSL -H "Authorization: Bearer $TOKEN" -H "X-GitHub-Api-Version: 2022-11-28" \
          "https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/releases/tags/${GH_TAG}"); \
        # Extract asset id by exact filename
        ASSET_ID=$(printf "%s" "$RELEASE_JSON" | jq -r --arg NAME "$MODEL_TAR" '.assets[] | select(.name==$NAME) | .id'); \
        if [ -z "$ASSET_ID" ] || [ "$ASSET_ID" = "null" ]; then \
          echo "❌ Could not find asset named '$MODEL_TAR' on tag '$GH_TAG' in ${GH_OWNER}/${GH_REPO}."; \
          echo "Hint: check exact filename & tag (tag_name, not release title)."; \
          exit 1; \
        fi; \
        # Download by asset id
        echo "⬇️  Downloading asset id $ASSET_ID"; \
        curl -fsSL -H "Authorization: Bearer $TOKEN" \
             -H "Accept: application/octet-stream" \
             -H "X-GitHub-Api-Version: 2022-11-28" \
             -o /tmp/model.tar.gz \
             "https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/releases/assets/${ASSET_ID}"; \
      else \
        echo "🌐 Public path: attempting direct download URL"; \
        URL="https://github.com/${GH_OWNER}/${GH_REPO}/releases/download/${GH_TAG}/${MODEL_TAR}"; \
        echo "GET $URL"; \
        curl -fL -o /tmp/model.tar.gz "$URL"; \
      fi; \
      tar -xzf /tmp/model.tar.gz -C /srv/artifacts; \
      rm -f /tmp/model.tar.gz; \
    else \
      echo "Found pre-baked artifacts → skipping download"; \
    fi

ENV PYTHONPATH=/srv \
    PORT=8080 \
    MODEL_PATH=/srv/artifacts/planner_model \
    SCHEMA_PATH=/srv/artifacts/schema.json \
    DATA_PATH=/srv/artifacts/planner_model/embedded_data.csv

EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]
