import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from app.core.config import settings


class BaseCache(ABC):
    @abstractmethod
    def get_json(self, key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        raise NotImplementedError


class NoOpCache(BaseCache):
    """
    Disabled cache provider.

    Used for tests, local development, and environments where Redis is not running.
    """

    def get_json(self, key: str) -> dict[str, Any] | None:
        return None

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        return None


class RedisCache(BaseCache):
    """
    Redis-backed JSON cache.

    Intended production use:
    - cache repeated /chat responses;
    - reduce repeated embedding + retrieval + reranking work;
    - reduce future LLM API cost.
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        default_ttl_seconds: int,
    ) -> None:
        import redis

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self.default_ttl_seconds = default_ttl_seconds

    def get_json(self, key: str) -> dict[str, Any] | None:
        raw_value = self.client.get(key)

        if raw_value is None:
            return None

        return json.loads(raw_value)

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        effective_ttl = ttl_seconds or self.default_ttl_seconds

        self.client.setex(
            name=key,
            time=effective_ttl,
            value=json.dumps(value, ensure_ascii=False),
        )


def normalize_question(question: str) -> str:
    return " ".join(question.lower().strip().split())


def build_chat_cache_key(
    question: str,
    top_k: int,
    embedding_provider: str,
    embedding_model: str,
    reranker_provider: str,
    llm_provider: str,
) -> str:
    """
    Build a stable cache key for a chat request.

    The model/provider fields are part of the key because changing retrieval
    or generation settings can change the answer.
    """

    payload = {
        "question": normalize_question(question),
        "top_k": top_k,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "reranker_provider": reranker_provider,
        "llm_provider": llm_provider,
    }

    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return f"chat:{digest}"


def get_cache() -> BaseCache:
    provider = settings.cache_provider.lower().strip()

    if provider in {"none", "off", "disabled", ""}:
        return NoOpCache()

    if provider == "redis":
        return RedisCache(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            default_ttl_seconds=settings.redis_ttl_seconds,
        )

    raise ValueError(f"Unsupported CACHE_PROVIDER: {settings.cache_provider}")