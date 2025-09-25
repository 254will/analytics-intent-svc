# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates jq tini \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Deps
COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir --prefer-binary -r /srv/requirements.txt

# App code
COPY app/ /srv/app/

# --- Artifact download from GitHub Releases (public path works on Railway) ---
ARG GH_OWNER=254will
ARG GH_REPO=analytics-intent-svc
ARG GH_TAG=v1.0.0
ARG MODEL_TAR=model-950mb.tar.gz

# Note: Railway doesn't pass BuildKit secrets; fallback public URL is used.
# Add retries + fail-fast options for reliability.
RUN set -euo pipefail; \
    mkdir -p /srv/artifacts; \
    if [ -z "$(ls -A /srv/artifacts 2>/dev/null || true)" ]; then \
      URL="https://github.com/${GH_OWNER}/${GH_REPO}/releases/download/${GH_TAG}/${MODEL_TAR}"; \
      echo "🌐 Fetching: $URL"; \
      curl -fL --retry 5 --retry-delay 2 --retry-connrefused -o /tmp/model.tar.gz "$URL"; \
      tar -xzf /tmp/model.tar.gz -C /srv/artifacts; \
      rm -f /tmp/model.tar.gz; \
    else \
      echo "Found pre-baked artifacts → skipping download"; \
    fi

# Runtime env
ENV PYTHONPATH=/srv \
    PORT=8080 \
    MODEL_PATH=/srv/artifacts/planner_model \
    SCHEMA_PATH=/srv/artifacts/schema.json \
    DATA_PATH=/srv/artifacts/planner_model/embedded_data.csv

EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]
