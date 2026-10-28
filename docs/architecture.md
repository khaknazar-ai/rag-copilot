# Architecture

## Product

Kazakhstan Consular Services RAG Copilot is a service-aware RAG assistant for Kazakhstan consular services in China.

The assistant answers with citations from a structured official-source corpus and refuses or escalates unsupported, out-of-scope, or sensitive requests.

## High-level architecture

official-source structured corpus
  -> ingestion
  -> chunking
  -> embeddings
  -> Qdrant vector index
  -> learned intent/source router
  -> source-restricted retrieval
  -> optional reranking
  -> grounded answer generation
  -> citation post-processing
  -> FastAPI API
  -> Streamlit UI
  -> Redis cache
  -> PostgreSQL metadata logging
  -> monitoring dashboard
  -> evaluation suite

## Final routing policy

The project uses one final routing policy: a supervised multilingual learned intent/source router with hard safety guardrails.

Policy:

1. Apply hard safety guardrails for fraud, bypass, false documents, or approval guarantees.
2. Run the learned intent/source classifier.
3. Route confident service intents to their allowed source files.
4. Route out-of-scope questions to refusal behavior.
5. Route unknown or unsupported questions to no-source / clarification behavior.
6. Avoid broad source search when the service intent is known.

This prevents citation leakage across consular services.

## Core services

- FastAPI backend for /chat, health checks, and ingestion routes.
- Qdrant vector database for semantic retrieval.
- Redis response cache for repeated /chat queries.
- PostgreSQL metadata DB for source, chunk, and query-event logging.
- Streamlit user UI and monitoring dashboard.
- Docker Compose deployment for API, frontend, monitoring, Qdrant, Redis, and PostgreSQL.

## Model layer

The project includes:

- local deterministic grounded generation for tests and offline development;
- strict Ollama local LLM mode;
- OpenAI-compatible client layer;
- local hashing embeddings for lightweight Docker mode;
- optional multilingual SentenceTransformer embeddings;
- optional cross-encoder reranker.
