from app.core.config import settings
from app.ingestion.chunk_io import load_chunks_jsonl
from app.ingestion.embeddings import get_embedding_client
from app.retrieval.vector_store import QdrantVectorStore


def main() -> None:
    chunks_path = "data/processed/chunks.jsonl"

    chunks = load_chunks_jsonl(chunks_path)

    if not chunks:
        print("No chunks found. Run python scripts/ingest_documents.py first.")
        return

    texts = [chunk["text"] for chunk in chunks]

    embedding_client = get_embedding_client()
    vectors = embedding_client.embed_texts(texts)

    vector_store = QdrantVectorStore()
    indexed_count = vector_store.upsert_chunks(
        chunks=chunks,
        vectors=vectors,
        recreate=True,
    )

    print("Qdrant indexing completed")
    print(f"Qdrant mode: {settings.qdrant_mode}")
    print(f"Indexed chunks: {indexed_count}")
    print(f"Vector size: {len(vectors[0])}")
    print(f"Collection: {vector_store.collection_name}")

    if settings.qdrant_mode == "local":
        print(f"Local path: {settings.qdrant_local_path}")


if __name__ == "__main__":
    main()