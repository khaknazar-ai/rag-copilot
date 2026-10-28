import json
from pathlib import Path

from app.ingestion.chunking import chunk_document
from app.ingestion.loaders import load_text_files


class DocumentIndexer:
    """
    v1 indexer.

    This does not create embeddings yet.
    It creates a processed JSONL artifact that will be used by the vector DB
    indexing step in the next milestone.
    """

    def index_documents(
        self,
        data_dir: str,
        output_path: str = "data/processed/chunks.jsonl",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> dict:
        documents = load_text_files(data_dir)

        all_chunks = []

        for document in documents:
            document_chunks = chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(document_chunks)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", encoding="utf-8") as file:
            for chunk in all_chunks:
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        return {
            "indexed_documents": len(documents),
            "indexed_chunks": len(all_chunks),
            "processed_path": str(output.as_posix()),
        }