from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def read_text_file(path: Path) -> str:
    """
    Read a text-like document.

    We use utf-8-sig instead of plain utf-8 because it safely removes a BOM
    if the file was created by Windows tools.
    """
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1251")


def load_text_files(data_dir: str) -> list[dict]:
    """
    Load .txt and .md files from a directory recursively.

    Returns a list of document dictionaries with text and metadata.
    """
    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = read_text_file(path).strip()

        if not text:
            continue

        documents.append(
            {
                "source": str(path.as_posix()),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "text": text,
            }
        )

    return documents