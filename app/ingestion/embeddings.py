import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.core.config import settings


TOKEN_PATTERN = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")


class BaseEmbeddingClient(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalHashingEmbeddingClient(BaseEmbeddingClient):
    """
    Deterministic local hashing embeddings.

    This provider is intentionally simple and reproducible.
    It is useful for tests, offline demos, and baseline comparisons.

    It is not a strong semantic embedding model.
    """

    def __init__(self, vector_size: int = 384, dim: int | None = None) -> None:
        # Backward-compatible alias:
        # older tests and callers used LocalHashingEmbeddingClient(dim=...)
        if dim is not None:
            vector_size = dim

        self.vector_size = vector_size

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    def _hash_token(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_size

        tokens = self._tokenize(text)

        if not tokens:
            return vector

        for token in tokens:
            token_hash = self._hash_token(token)
            index = token_hash % self.vector_size
            sign = 1.0 if (token_hash // self.vector_size) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            return vector

        return [value / norm for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class SentenceTransformerEmbeddingClient(BaseEmbeddingClient):
    """
    Local semantic embeddings using sentence-transformers.

    This is the stronger local provider for portfolio-quality RAG:
    - captures semantic similarity better than hashing;
    - supports multilingual questions better;
    - runs without external API keys.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """
    OpenAI embedding provider.

    Requires:
    - EMBEDDING_PROVIDER=openai
    - OPENAI_API_KEY set in .env
    """

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Set OPENAI_API_KEY in .env or use EMBEDDING_PROVIDER=local."
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def get_embedding_client() -> BaseEmbeddingClient:
    provider = settings.embedding_provider.lower().strip()

    if provider in {"local", "hashing"}:
        return LocalHashingEmbeddingClient(
            vector_size=settings.embedding_dim,
        )

    if provider in {"sentence_transformers", "sentence-transformers", "sbert"}:
        return SentenceTransformerEmbeddingClient(
            model_name=settings.embedding_model,
        )

    if provider == "openai":
        return OpenAIEmbeddingClient(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")
