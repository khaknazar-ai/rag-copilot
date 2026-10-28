import hashlib


def chunk_text_with_offsets(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    """
    Split text into overlapping character chunks and preserve offsets.

    Character chunking is a simple v1 baseline.
    Later we can replace this with token-aware or heading-aware chunking.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    clean_text = text.strip()

    if not clean_text:
        return []

    chunks = []
    start = 0
    text_length = len(clean_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        raw_chunk = clean_text[start:end]
        chunk = raw_chunk.strip()

        if chunk:
            chunks.append(
                {
                    "text": chunk,
                    "char_start": start,
                    "char_end": end,
                }
            )

        if end == text_length:
            break

        start = end - chunk_overlap

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """
    Backward-compatible helper used by tests.

    Returns only chunk texts.
    """
    chunks = chunk_text_with_offsets(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [chunk["text"] for chunk in chunks]


def make_chunk_id(source: str, chunk_index: int, chunk_text_value: str) -> str:
    """
    Create a deterministic chunk id.

    Deterministic ids are useful because re-running ingestion on the same docs
    should produce stable identifiers.
    """
    raw = f"{source}:{chunk_index}:{chunk_text_value[:128]}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{digest}"


def chunk_document(
    document: dict,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict]:
    """
    Convert one loaded document into metadata-rich chunks.
    """
    source = document["source"]
    filename = document["filename"]

    chunks_with_offsets = chunk_text_with_offsets(
        text=document["text"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for index, chunk in enumerate(chunks_with_offsets):
        chunk_id = make_chunk_id(
            source=source,
            chunk_index=index,
            chunk_text_value=chunk["text"],
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "filename": filename,
                "extension": document["extension"],
                "chunk_index": index,
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "text": chunk["text"],
            }
        )

    return chunks