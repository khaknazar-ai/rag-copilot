from app.generation.llm_client import BaseLLMClient, get_llm_client


class AnswerGenerator:
    """
    RAG answer generator.

    This component receives:
    - user question
    - retrieved chunks already filtered by Retriever

    It returns:
    - grounded answer text with citation markers

    Important:
    Retrieval relevance filtering belongs in Retriever.
    AnswerGenerator should not apply an additional vector-score threshold,
    because lexical evidence can be stronger than local hashing vector score.
    """

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self.llm_client = llm_client or get_llm_client()

    def generate_answer(self, question: str, chunks: list[dict]) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")

        return self.llm_client.generate_answer(
            question=question,
            chunks=chunks,
        )