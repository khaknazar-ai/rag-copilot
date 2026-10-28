# Consular Services RAG Copilot

A FastAPI and Streamlit RAG demo for informational questions about Kazakhstan consular services in China. It retrieves from user-provided documents through Qdrant and can generate answers with a local Ollama service. It does not replace official consular advice.

## Run locally

Requirements: Python 3.11+, Docker Compose, and Ollama with `qwen2.5:1.5b` for grounded answers (`ollama pull qwen2.5:1.5b`).

1. Copy `.env.example` to `.env` and set a local `POSTGRES_PASSWORD`.
2. Add your own `.md` or `.txt` documents under `data/raw/`. The source corpus, evaluation datasets, tests, and model artifacts are not included. Use source filenames configured in `app/retrieval/domain_intent.py`, or update that routing policy for your documents.
3. Start the services and index your documents:

   ```powershell
   docker compose up --build -d
   docker compose --profile ingest run --rm indexer
   ```

The API is at `http://localhost:8000/docs`; the chat UI is at `http://localhost:8501`; the monitoring UI is at `http://localhost:8502`. Published ports bind to loopback. Query logging is disabled by default; set `QUERY_LOG_ENABLED=true` only when you want to store raw questions locally.

## Instructions and results

- [Architecture](docs/architecture.md) and [limitations](docs/limitations.md)
- [Evaluation summary](docs/evaluation.md)

The router's offline evaluation measured 1.000 accuracy on a clean holdout of 36 questions (154 training examples and no exact normalized text overlap) and 0.617 ± 0.064 mean accuracy in three-fold cross-validation. The source policy scored 1.000 on 18 source-scored examples. These are small, domain-specific historical results, not evidence of general answer quality or production readiness, and they were not rerun for this public candidate. The classifier and evaluation datasets are excluded.

## Libraries

[FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://uvicorn.dev/), [Pydantic and Settings](https://docs.pydantic.dev/latest/), [python-dotenv](https://github.com/theskumar/python-dotenv), [Qdrant Client](https://python-client.qdrant.tech/), [Redis](https://redis.io/docs/latest/), [Psycopg 3](https://www.psycopg.org/psycopg3/docs/), [Streamlit](https://docs.streamlit.io/), [Requests](https://requests.readthedocs.io/en/stable/), [Pandas](https://pandas.pydata.org/docs/), [NumPy](https://numpy.org/doc/stable/), [OpenAI Python](https://github.com/openai/openai-python), and [tqdm](https://tqdm.github.io/) are used by the app. Optional ML paths use [scikit-learn](https://scikit-learn.org/stable/), [joblib](https://joblib.readthedocs.io/en/stable/), and [Sentence Transformers](https://www.sbert.net/); install these with the `ml` extra in `pyproject.toml`. Local services include [Qdrant](https://qdrant.tech/documentation/), [Redis](https://redis.io/docs/latest/), [PostgreSQL](https://www.postgresql.org/docs/), and [Ollama](https://docs.ollama.com/). Default dependency version ranges are in `requirements-docker.txt`.
