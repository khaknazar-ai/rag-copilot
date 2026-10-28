import re


TOKEN_PATTERN = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")


STOPWORDS = {
    # English function words
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "which",
    "with",

    # Common auxiliary/question words
    "can",
    "could",
    "should",
    "would",
    "will",

    # Project/domain-generic terms.
    # These appear across many docs and should not be enough to prove relevance.
    "production",
    "rag",
    "copilot",
    "document",
    "documents",
    "system",
    "project",
}


def tokenize_meaningful(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
    }

    return {
        token
        for token in tokens
        if len(token) >= 2 and token not in STOPWORDS
    }


def lexical_overlap_terms(question: str, chunk_text: str) -> set[str]:
    question_terms = tokenize_meaningful(question)
    chunk_terms = tokenize_meaningful(chunk_text)

    return question_terms.intersection(chunk_terms)


def lexical_overlap_count(question: str, chunk_text: str) -> int:
    return len(lexical_overlap_terms(question, chunk_text))


def has_lexical_support(
    question: str,
    chunk_text: str,
    min_overlap_terms: int = 1,
) -> bool:
    """
    Returns True when a chunk shares meaningful terms with the question.

    This is a lightweight hybrid retrieval guard.
    It helps reject false positives from weak local embeddings.
    """
    if min_overlap_terms <= 0:
        return True

    return lexical_overlap_count(question, chunk_text) >= min_overlap_terms