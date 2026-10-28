from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.config import settings


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0

    return round(float(numerator) / float(denominator), 4)


def cache_hit_rate(cache_counts: dict[str, int]) -> float:
    total = sum(cache_counts.values())

    if total == 0:
        return 0.0

    return safe_ratio(cache_counts.get("HIT", 0), total)


class BaseMonitoringStore(ABC):
    @abstractmethod
    def fetch_summary(self, days: int = 7) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def fetch_cache_breakdown(self, days: int = 7) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def fetch_latency_by_cache_status(self, days: int = 7) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def fetch_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        raise NotImplementedError


class NoOpMonitoringStore(BaseMonitoringStore):
    def fetch_summary(self, days: int = 7) -> dict[str, Any]:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "avg_citation_count": 0.0,
            "cache_hit_rate": 0.0,
        }

    def fetch_cache_breakdown(self, days: int = 7) -> list[dict[str, Any]]:
        return []

    def fetch_latency_by_cache_status(self, days: int = 7) -> list[dict[str, Any]]:
        return []

    def fetch_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return []


class PostgresMonitoringStore(BaseMonitoringStore):
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

    def fetch_summary(self, days: int = 7) -> dict[str, Any]:
        query = """
        SELECT
            COUNT(*) AS total_queries,
            COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
            COALESCE(MIN(latency_ms), 0) AS min_latency_ms,
            COALESCE(MAX(latency_ms), 0) AS max_latency_ms,
            COALESCE(AVG(citation_count), 0) AS avg_citation_count,
            COALESCE(SUM(CASE WHEN cache_status = 'HIT' THEN 1 ELSE 0 END), 0) AS cache_hits
        FROM rag_query_events
        WHERE created_at >= NOW() - (%s || ' days')::interval;
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (days,))
                row = cursor.fetchone()

        total_queries = int(row[0] or 0)
        cache_hits = int(row[5] or 0)

        return {
            "total_queries": total_queries,
            "avg_latency_ms": round(float(row[1] or 0), 3),
            "min_latency_ms": round(float(row[2] or 0), 3),
            "max_latency_ms": round(float(row[3] or 0), 3),
            "avg_citation_count": round(float(row[4] or 0), 3),
            "cache_hit_rate": safe_ratio(cache_hits, total_queries),
        }

    def fetch_cache_breakdown(self, days: int = 7) -> list[dict[str, Any]]:
        query = """
        SELECT
            COALESCE(cache_status, 'UNKNOWN') AS cache_status,
            COUNT(*) AS count
        FROM rag_query_events
        WHERE created_at >= NOW() - (%s || ' days')::interval
        GROUP BY COALESCE(cache_status, 'UNKNOWN')
        ORDER BY count DESC;
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (days,))
                rows = cursor.fetchall()

        return [
            {
                "cache_status": str(row[0]),
                "count": int(row[1]),
            }
            for row in rows
        ]

    def fetch_latency_by_cache_status(self, days: int = 7) -> list[dict[str, Any]]:
        query = """
        SELECT
            COALESCE(cache_status, 'UNKNOWN') AS cache_status,
            COUNT(*) AS count,
            COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
            COALESCE(MIN(latency_ms), 0) AS min_latency_ms,
            COALESCE(MAX(latency_ms), 0) AS max_latency_ms
        FROM rag_query_events
        WHERE created_at >= NOW() - (%s || ' days')::interval
        GROUP BY COALESCE(cache_status, 'UNKNOWN')
        ORDER BY avg_latency_ms DESC;
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (days,))
                rows = cursor.fetchall()

        return [
            {
                "cache_status": str(row[0]),
                "count": int(row[1]),
                "avg_latency_ms": round(float(row[2] or 0), 3),
                "min_latency_ms": round(float(row[3] or 0), 3),
                "max_latency_ms": round(float(row[4] or 0), 3),
            }
            for row in rows
        ]

    def fetch_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        query = """
        SELECT
            id,
            created_at,
            question,
            top_k,
            cache_status,
            embedding_provider,
            embedding_model,
            reranker_provider,
            llm_provider,
            answer_length,
            citation_count,
            latency_ms
        FROM rag_query_events
        ORDER BY id DESC
        LIMIT %s;
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()

        return [
            {
                "id": int(row[0]),
                "created_at": row[1].isoformat() if row[1] else None,
                "question": row[2],
                "top_k": row[3],
                "cache_status": row[4],
                "embedding_provider": row[5],
                "embedding_model": row[6],
                "reranker_provider": row[7],
                "llm_provider": row[8],
                "answer_length": row[9],
                "citation_count": row[10],
                "latency_ms": round(float(row[11] or 0), 3),
            }
            for row in rows
        ]


def get_monitoring_store() -> BaseMonitoringStore:
    provider = settings.metadata_db_provider.lower().strip()

    if provider in {"none", "off", "disabled", ""}:
        return NoOpMonitoringStore()

    if provider in {"postgres", "postgresql"}:
        return PostgresMonitoringStore(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout_seconds=settings.postgres_connect_timeout_seconds,
        )

    raise ValueError(f"Unsupported METADATA_DB_PROVIDER: {settings.metadata_db_provider}")