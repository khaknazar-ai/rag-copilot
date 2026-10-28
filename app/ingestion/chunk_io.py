import json
from pathlib import Path


def load_chunks_jsonl(path: str) -> list[dict]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Chunks file does not exist: {path}")

    chunks = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            chunks.append(json.loads(line))

    return chunks