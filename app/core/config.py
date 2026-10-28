from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Consular Services RAG Copilot"
    app_env: str = "local"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    query_log_enabled: bool = False

    qdrant_mode: str = "local"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_documents"
    qdrant_local_path: str = "data/qdrant_local"

    embedding_provider: str = "local"
    embedding_model: str = "hashing"
    embedding_dim: int = 384

    llm_provider: str = "local"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout_seconds: int = 120

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    retrieval_score_threshold: float = 0.25
    lexical_min_overlap_terms: int = 1

    reranker_provider: str = "none"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_top_n: int = 5

    cache_provider: str = "none"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl_seconds: int = 3600

    metadata_db_provider: str = "none"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_metadata"
    postgres_user: str = "rag_user"
    postgres_password: str = ""
    postgres_connect_timeout_seconds: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
