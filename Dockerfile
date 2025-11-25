# ============================================================
# 1) Use slim python image
# ============================================================
FROM python:3.11-slim

# ------------------------------------------------------------
# 2) Install system deps (FAISS, numpy, spaCy need these)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 3) Working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# 4) Install Python deps FIRST for Docker cache
# ------------------------------------------------------------
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# >>> Minimal required fix: install spaCy model <<<
# ------------------------------------------------------------
RUN python -m spacy download en_core_web_sm

# ------------------------------------------------------------
# 5) Copy project code ONLY (fast rebuild)
# ------------------------------------------------------------
COPY ragthrones/ ./ragthrones/
COPY cloudrun_start.sh ./cloudrun_start.sh

# ------------------------------------------------------------
# 6) NO ARTIFACTS COPIED INTO IMAGE
#    They now load from GCS at runtime.
# ------------------------------------------------------------
# (REMOVED)
# COPY ragthrones/data/artifacts ./ragthrones/data/artifacts

# Keep assets if you want logos/UI images
COPY ragthrones/assets ./ragthrones/assets

# ------------------------------------------------------------
# 7) Runtime env
# ------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# ------------------------------------------------------------
# 8) Entrypoint (no reload)
# ------------------------------------------------------------
RUN chmod +x /app/cloudrun_start.sh

CMD ["/app/cloudrun_start.sh"]