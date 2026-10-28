import json
from pathlib import Path

from app.metadata.postgres_store import get_metadata_store, parse_frontmatter_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "official_sources"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def infer_source_type(source_id: str | None) -> str:
    source_id = source_id or ""

    if source_id.startswith("gov_kz"):
        return "gov.kz"

    if source_id.startswith("adilet"):
        return "adilet.zan.kz"

    if source_id.startswith("internal"):
        return "demo policy"

    return "indexed source"


def sync_sources(store) -> int:
    count = 0

    for path in sorted(SOURCE_DIR.glob("*.md")):
        metadata = parse_frontmatter_metadata(path)
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()

        metadata["source_path"] = relative_path
        metadata["source_type"] = infer_source_type(metadata.get("source_id"))

        store.upsert_source_document(metadata)
        count += 1

    return count


def sync_chunks(store) -> int:
    if not CHUNKS_PATH.exists():
        return 0

    count = 0

    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        chunk = json.loads(line)
        store.upsert_chunk_metadata(chunk)
        count += 1

    return count


def main() -> None:
    store = get_metadata_store()
    store.init_schema()

    source_count = sync_sources(store)
    chunk_count = sync_chunks(store)

    print("Metadata DB initialized")
    print(f"Synced source documents: {source_count}")
    print(f"Synced chunks: {chunk_count}")
    print(f"rag_source_documents rows: {store.count_rows('rag_source_documents')}")
    print(f"rag_chunks rows: {store.count_rows('rag_chunks')}")
    print(f"rag_query_events rows: {store.count_rows('rag_query_events')}")


if __name__ == "__main__":
    main()