FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt

ARG EMBEDDINGS=real
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN if [ "$EMBEDDINGS" = "real" ]; then \
      pip install --extra-index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements-ml.txt && \
      python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"; \
    fi
COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD curl -fs http://localhost:8000/health || exit 1

CMD ["gunicorn", "app.api.main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--timeout", "60", "--graceful-timeout", "30"]
