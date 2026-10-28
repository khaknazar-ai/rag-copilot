def build_context(chunks: list[dict]) -> str:
    context_blocks = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        context_blocks.append(f"[{index}] Source: {source}\n{text}")

    return "\n\n".join(context_blocks)
