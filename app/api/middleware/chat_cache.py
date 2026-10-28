import json
import time
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.cache.redis_cache import build_chat_cache_key, get_cache
from app.core.config import settings
from app.metadata.postgres_store import get_metadata_store


class ChatResponseCacheMiddleware(BaseHTTPMiddleware):
    """
    Optional Redis cache + optional PostgreSQL query event logging for POST /chat.

    Responsibilities:
    - Redis response cache:
        MISS -> run RAG pipeline -> store response
        HIT  -> return cached response
        BYPASS -> Redis failed, API still works

    - PostgreSQL metadata logging:
        Records question, cache status, retrieval mode, answer length,
        citation count, and latency for monitoring/dashboard use.

    Design choice:
    Keeping this in middleware lets /chat business logic stay focused on
    retrieval and answer generation.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method.upper() != "POST" or request.url.path != "/chat":
            return await call_next(request)

        started_at = time.perf_counter()
        body = await request.body()

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return await self._call_next_with_replayed_body(request, call_next, body)

        question = payload.get("question")
        top_k = payload.get("top_k", settings.top_k)

        if not isinstance(question, str) or not question.strip():
            return await self._call_next_with_replayed_body(request, call_next, body)

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = settings.top_k

        cache_enabled = settings.cache_provider.lower().strip() not in {
            "none",
            "off",
            "disabled",
            "",
        }

        cache_key = build_chat_cache_key(
            question=question,
            top_k=top_k,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            reranker_provider=settings.reranker_provider,
            llm_provider=settings.llm_provider,
        )

        if cache_enabled:
            try:
                cache = get_cache()
                cached_response = cache.get_json(cache_key)

                if cached_response is not None:
                    latency_ms = self._elapsed_ms(started_at)

                    self._log_query_event_safely(
                        question=question,
                        top_k=top_k,
                        cache_status="HIT",
                        latency_ms=latency_ms,
                        response_json=cached_response,
                        status_code=200,
                        cache_key=cache_key,
                    )

                    return JSONResponse(
                        content=cached_response,
                        headers={
                            "X-Cache": "HIT",
                            "X-Cache-Key": cache_key,
                        },
                    )

            except Exception:
                response = await self._call_next_with_replayed_body(request, call_next, body)
                response_body, response_json = await self._consume_response(response)

                headers = dict(response.headers)
                headers["X-Cache"] = "BYPASS"
                headers["X-Cache-Key"] = cache_key

                latency_ms = self._elapsed_ms(started_at)

                self._log_query_event_safely(
                    question=question,
                    top_k=top_k,
                    cache_status="BYPASS",
                    latency_ms=latency_ms,
                    response_json=response_json,
                    status_code=response.status_code,
                    cache_key=cache_key,
                )

                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.media_type,
                )

        response = await self._call_next_with_replayed_body(request, call_next, body)
        response_body, response_json = await self._consume_response(response)

        headers = dict(response.headers)

        if cache_enabled:
            cache_status = "MISS"
            headers["X-Cache"] = "MISS"
            headers["X-Cache-Key"] = cache_key

            if response.status_code == 200 and response_json is not None:
                try:
                    cache = get_cache()
                    cache.set_json(
                        key=cache_key,
                        value=response_json,
                        ttl_seconds=settings.redis_ttl_seconds,
                    )
                except Exception:
                    cache_status = "BYPASS"
                    headers["X-Cache"] = "BYPASS"
        else:
            cache_status = "DISABLED"
            headers["X-Cache"] = "DISABLED"

        latency_ms = self._elapsed_ms(started_at)

        self._log_query_event_safely(
            question=question,
            top_k=top_k,
            cache_status=cache_status,
            latency_ms=latency_ms,
            response_json=response_json,
            status_code=response.status_code,
            cache_key=cache_key if cache_enabled else None,
        )

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )

    async def _call_next_with_replayed_body(self, request: Request, call_next, body: bytes):
        async def receive():
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }

        replayed_request = Request(request.scope, receive)

        return await call_next(replayed_request)

    async def _consume_response(self, response) -> tuple[bytes, dict[str, Any] | None]:
        response_body = b""

        async for chunk in response.body_iterator:
            response_body += chunk

        try:
            response_json = json.loads(response_body.decode("utf-8"))
        except Exception:
            response_json = None

        return response_body, response_json

    def _elapsed_ms(self, started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 3)

    def _answer_length(self, response_json: dict[str, Any] | None) -> int | None:
        if not response_json:
            return None

        answer = response_json.get("answer")

        if not isinstance(answer, str):
            return None

        return len(answer)

    def _citation_count(self, response_json: dict[str, Any] | None) -> int | None:
        if not response_json:
            return None

        citations = response_json.get("citations")

        if isinstance(citations, list):
            return len(citations)

        return None

    def _log_query_event_safely(
        self,
        question: str,
        top_k: int,
        cache_status: str,
        latency_ms: float,
        response_json: dict[str, Any] | None,
        status_code: int,
        cache_key: str | None,
    ) -> None:
        if not settings.query_log_enabled:
            return None

        if settings.metadata_db_provider.lower().strip() in {
            "none",
            "off",
            "disabled",
            "",
        }:
            return None

        try:
            store = get_metadata_store()
            store.init_schema()

            store.log_query_event(
                {
                    "question": question,
                    "top_k": top_k,
                    "cache_status": cache_status,
                    "embedding_provider": settings.embedding_provider,
                    "embedding_model": settings.embedding_model,
                    "reranker_provider": settings.reranker_provider,
                    "reranker_model": settings.reranker_model,
                    "llm_provider": settings.llm_provider,
                    "answer_length": self._answer_length(response_json),
                    "citation_count": self._citation_count(response_json),
                    "latency_ms": latency_ms,
                    "metadata": {
                        "status_code": status_code,
                        "cache_key": cache_key,
                        "source": "chat_middleware",
                    },
                }
            )

        except Exception:
            # Fail-open: monitoring must never break the user-facing API.
            return None
