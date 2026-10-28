from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.core.config import settings


class QdrantVectorStore:
    """
    Thin Qdrant wrapper for document chunk vectors.

    Supports two modes:

    1. local:
       Uses embedded Qdrant storage on disk.
       Good for development when Docker is not available.

    2. server:
       Connects to a running Qdrant service.
       Good for Docker Compose and service-based development.
    """

    def __init__(
        self,
        mode: str | None = None,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        local_path: str | None = None,
    ) -> None:
        self.mode = (mode or settings.qdrant_mode).lower().strip()
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.local_path = local_path or settings.qdrant_local_path

        if self.mode == "local":
            Path(self.local_path).parent.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=self.local_path)
        elif self.mode == "server":
            self.client = QdrantClient(host=self.host, port=self.port)
        else:
            raise ValueError(f"Unsupported QDRANT_MODE: {self.mode}")

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(collection.name == self.collection_name for collection in collections)

    def recreate_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def create_collection_if_missing(self, vector_size: int) -> None:
        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    def _to_point_id(self, chunk_id: str) -> str:
        """
        Qdrant point ids should be UUIDs or integers.

        Our chunk ids are human-readable strings, so we convert them to stable UUIDs.
        """
        return str(uuid5(NAMESPACE_URL, chunk_id))

    def upsert_chunks(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
        recreate: bool = False,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        if not chunks:
            return 0

        vector_size = len(vectors[0])

        if recreate:
            self.recreate_collection(vector_size=vector_size)
        else:
            self.create_collection_if_missing(vector_size=vector_size)

        points = []

        for chunk, vector in zip(chunks, vectors):
            payload = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "filename": chunk["filename"],
                "extension": chunk["extension"],
                "chunk_index": chunk["chunk_index"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "text": chunk["text"],
            }

            points.append(
                PointStruct(
                    id=self._to_point_id(chunk["chunk_id"]),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return len(points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """
        Search nearest chunks.

        Current qdrant-client versions use query_points().
        Some older versions had search(), so this method keeps a fallback.
        """
        if not self.collection_exists():
            return []

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            raw_results = response.points
        else:
            raw_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        matches = []

        for item in raw_results:
            payload = item.payload or {}
            matches.append(
                {
                    "score": item.score,
                    "chunk_id": payload.get("chunk_id"),
                    "source": payload.get("source"),
                    "filename": payload.get("filename"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": payload.get("text"),
                    "char_start": payload.get("char_start"),
                    "char_end": payload.get("char_end"),
                }
            )

        return matches
