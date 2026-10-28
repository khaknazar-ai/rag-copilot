from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.core.config import settings


class BaseMetadataStore(ABC):
    @abstractmethod
    def init_schema(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_source_document(self, source: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_chunk_metadata(self, chunk: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_query_event(self, event: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def count_rows(self, table_name: str) -> int:
        raise NotImplementedError


class NoOpMetadataStore(BaseMetadataStore):
    """
    Disabled metadata store.

    Used as the default so the app can run without PostgreSQL.
    """

    def init_schema(self) -> None:
        return None

    def upsert_source_document(self, source: dict[str, Any]) -> None:
        return None

    def upsert_chunk_metadata(self, chunk: dict[str, Any]) -> None:
        return None

    def log_query_event(self, event: dict[str, Any]) -> None:
        return None

    def count_rows(self, table_name: str) -> int:
        return 0


class PostgresMetadataStore(BaseMetadataStore):
    """
    PostgreSQL-backed metadata store.

    Stores:
    - source document metadata;
    - indexed chunk metadata;
    - query events for monitoring and later dashboards.
    """

    ALLOWED_COUNT_TABLES = {
        "rag_source_documents",
        "rag_chunks",
        "rag_query_events",
    }

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        connect_timeout_seconds: int = 3,
    ) -> None:
        self.connection_kwargs = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
            "connect_timeout": connect_timeout_seconds,
        }

    def _connect(self):
        import psycopg

        return psycopg.connect(**self.connection_kwargs)

    def init_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS rag_source_documents (
            source_path TEXT PRIMARY KEY,
            title TEXT,
            source_id TEXT,
            service_category TEXT,
            jurisdiction TEXT,
            official_source BOOLEAN DEFAULT FALSE,
            source_type TEXT,
            metadata JSONB DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS rag_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_path TEXT REFERENCES rag_source_documents(source_path) ON DELETE SET NULL,
            chunk_index INTEGER,
            text_length INTEGER,
            metadata JSONB DEFAULT '{}'::jsonb,
            indexed_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS rag_query_events (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            question TEXT NOT NULL,
            top_k INTEGER,
            cache_status TEXT,
            embedding_provider TEXT,
            embedding_model TEXT,
            reranker_provider TEXT,
            reranker_model TEXT,
            llm_provider TEXT,
            answer_length INTEGER,
            citation_count INTEGER,
            latency_ms DOUBLE PRECISION,
            metadata JSONB DEFAULT '{}'::jsonb
        );

        CREATE INDEX IF NOT EXISTS idx_rag_query_events_created_at
            ON rag_query_events(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_rag_query_events_cache_status
            ON rag_query_events(cache_status);

        CREATE INDEX IF NOT EXISTS idx_rag_chunks_source_path
            ON rag_chunks(source_path);
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(ddl)

    def upsert_source_document(self, source: dict[str, Any]) -> None:
        source_path = source.get("source_path") or source.get("source") or source.get("path")

        if not source_path:
            raise ValueError("source document metadata must include source_path/source/path")

        metadata = dict(source)

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO rag_source_documents (
                        source_path,
                        title,
                        source_id,
                        service_category,
                        jurisdiction,
                        official_source,
                        source_type,
                        metadata,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (source_path)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        source_id = EXCLUDED.source_id,
                        service_category = EXCLUDED.service_category,
                        jurisdiction = EXCLUDED.jurisdiction,
                        official_source = EXCLUDED.official_source,
                        source_type = EXCLUDED.source_type,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW();
                    """,
                    (
                        source_path,
                        source.get("title"),
                        source.get("source_id"),
                        source.get("service_category"),
                        source.get("jurisdiction"),
                        bool(source.get("official_source", False)),
                        source.get("source_type"),
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )

    def upsert_chunk_metadata(self, chunk: dict[str, Any]) -> None:
        chunk_id = chunk.get("chunk_id") or chunk.get("id")

        if not chunk_id:
            raise ValueError("chunk metadata must include chunk_id/id")

        source_path = chunk.get("source")
        text = chunk.get("text", "")

        metadata = {
            key: value
            for key, value in chunk.items()
            if key not in {"text"}
        }

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO rag_chunks (
                        chunk_id,
                        source_path,
                        chunk_index,
                        text_length,
                        metadata,
                        indexed_at
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (chunk_id)
                    DO UPDATE SET
                        source_path = EXCLUDED.source_path,
                        chunk_index = EXCLUDED.chunk_index,
                        text_length = EXCLUDED.text_length,
                        metadata = EXCLUDED.metadata,
                        indexed_at = NOW();
                    """,
                    (
                        chunk_id,
                        source_path,
                        chunk.get("chunk_index"),
                        len(text),
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )

    def log_query_event(self, event: dict[str, Any]) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO rag_query_events (
                        question,
                        top_k,
                        cache_status,
                        embedding_provider,
                        embedding_model,
                        reranker_provider,
                        reranker_model,
                        llm_provider,
                        answer_length,
                        citation_count,
                        latency_ms,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                    """,
                    (
                        event["question"],
                        event.get("top_k"),
                        event.get("cache_status"),
                        event.get("embedding_provider"),
                        event.get("embedding_model"),
                        event.get("reranker_provider"),
                        event.get("reranker_model"),
                        event.get("llm_provider"),
                        event.get("answer_length"),
                        event.get("citation_count"),
                        event.get("latency_ms"),
                        json.dumps(event.get("metadata", {}), ensure_ascii=False),
                    ),
                )

    def count_rows(self, table_name: str) -> int:
        if table_name not in self.ALLOWED_COUNT_TABLES:
            raise ValueError(f"Unsupported table for count_rows: {table_name}")

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row = cursor.fetchone()

        return int(row[0])


def get_metadata_store() -> BaseMetadataStore:
    provider = settings.metadata_db_provider.lower().strip()

    if provider in {"none", "off", "disabled", ""}:
        return NoOpMetadataStore()

    if provider in {"postgres", "postgresql"}:
        return PostgresMetadataStore(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout_seconds=settings.postgres_connect_timeout_seconds,
        )

    raise ValueError(f"Unsupported METADATA_DB_PROVIDER: {settings.metadata_db_provider}")


def parse_frontmatter_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8").splitlines()

    if not lines or lines[0].strip() != "---":
        return metadata

    for line in lines[1:]:
        stripped = line.strip()

        if stripped == "---":
            break

        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        cleaned = value.strip().strip('"').strip("'")

        if cleaned.lower() == "true":
            parsed_value: Any = True
        elif cleaned.lower() == "false":
            parsed_value = False
        else:
            parsed_value = cleaned

        metadata[key.strip()] = parsed_value

    return metadata