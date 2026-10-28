from app.core.config import settings
from app.ingestion.indexer import DocumentIndexer


def main() -> None:
    indexer = DocumentIndexer()

    result = indexer.index_documents(
        data_dir="data/raw",
        output_path="data/processed/chunks.jsonl",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print("Ingestion completed")
    print(f"Indexed documents: {result['indexed_documents']}")
    print(f"Indexed chunks: {result['indexed_chunks']}")
    print(f"Processed path: {result['processed_path']}")


if __name__ == "__main__":
    main()