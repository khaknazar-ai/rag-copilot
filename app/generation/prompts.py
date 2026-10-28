def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")

        context_blocks.append(
            f"[{index}] Source: {source}\n"
            f"{text}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""
You are a document-grounded consular services assistant.

User question:
{question}

Retrieved official-source context:
{context}

Answering rules:
1. Use only the retrieved context.
2. Do not invent requirements, fees, timelines, or legal conclusions.
3. Cite every factual claim with source markers like [1].
4. If the context is insufficient, refuse safely.
5. Do not guarantee visa approval or service outcomes.
6. Do not replace a consular officer.
7. Escalate urgent, sensitive, non-standard, or legally specific cases.

Return this structure:

## <short service title>

### Answer
- ...

### Required documents
- ...

### Processing time
- ...

### Fees
- ...

### Important notes
- ...

### When to contact the consular section
- ...

### Disclaimer
- This is informational guidance based on indexed sources. It does not replace an official consular decision.

### Sources
[1] <source>
""".strip()