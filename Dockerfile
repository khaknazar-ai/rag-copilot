FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-docker.txt ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements-docker.txt

COPY app ./app
COPY docs/architecture.md docs/evaluation.md docs/limitations.md ./docs/
COPY docs/evaluation_results/ ./docs/evaluation_results/
COPY data/source_manifest.json ./data/source_manifest.json
COPY frontend ./frontend
COPY scripts/ask.py scripts/ingest_documents.py scripts/index_chunks.py ./scripts/
COPY scripts/init_metadata_db.py scripts/search_qdrant.py scripts/wait_for_services.py ./scripts/
COPY README.md ./README.md

EXPOSE 8000
EXPOSE 8501
EXPOSE 8502

CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
