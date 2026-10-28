from abc import ABC, abstractmethod

from app.core.config import settings


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, question: str, chunks: list[dict], top_k: int) -> list[dict]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    """
    No-op reranker for default lightweight mode.
    """

    def rerank(self, question: str, chunks: list[dict], top_k: int) -> list[dict]:
        return chunks[:top_k]


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker.

    A bi-encoder embedding model retrieves a candidate pool.
    A cross-encoder then scores each (question, passage) pair more precisely.

    This is slower than vector search but often improves Top-1 relevance.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, chunks: list[dict], top_k: int) -> list[dict]:
        if not chunks:
            return []

        pairs = [
            (question, chunk.get("text", ""))
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        reranked_chunks = []

        for chunk, score in zip(chunks, scores):
            updated = dict(chunk)
            updated["rerank_score"] = float(score)
            updated["reranker_model"] = self.model_name
            reranked_chunks.append(updated)

        reranked_chunks.sort(
            key=lambda item: item.get("rerank_score", float("-inf")),
            reverse=True,
        )

        return reranked_chunks[:top_k]


def get_reranker() -> BaseReranker:
    provider = settings.reranker_provider.lower().strip()

    if provider in {"none", "off", "disabled", ""}:
        return NoOpReranker()

    if provider in {"sentence_transformers", "sentence-transformers", "cross_encoder", "cross-encoder"}:
        return CrossEncoderReranker(
            model_name=settings.reranker_model,
        )

    raise ValueError(f"Unsupported RERANKER_PROVIDER: {settings.reranker_provider}")