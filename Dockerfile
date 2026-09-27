FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MALLOC_ARENA_MAX=2

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY app.py ./
COPY backend ./backend
COPY models ./models

# Deterministic large-artifact fetch (pinned revision; ~21 MB).
ARG ARTIFACT_BASE=https://huggingface.co/Bini-K/sigmo-backend/resolve/v6.20260925.045900
RUN curl -fsSL -o models/v6.20260925.045900/model.ubj \
        "$ARTIFACT_BASE/models/v6.20260925.045900/model.ubj" \
    && curl -fsSL -o models/v6.20260925.045900/gate.joblib \
        "$ARTIFACT_BASE/models/v6.20260925.045900/gate.joblib"

# HF Spaces runs containers as UID 1000
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860
CMD ["python", "app.py"]
