import time

from fastapi import APIRouter

from app.core.schemas import ChatRequest, ChatResponse, Citation
from app.generation.answer_generator import AnswerGenerator
from app.observability.query_logger import log_query
from app.retrieval.retriever import Retriever

router = APIRouter()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start_time = time.perf_counter()

    retriever = Retriever()
    chunks = retriever.retrieve(
        question=request.question,
        top_k=request.top_k,
    )

    answer_generator = AnswerGenerator()
    answer = answer_generator.generate_answer(
        question=request.question,
        chunks=chunks,
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    citations = [
        Citation(
            source=chunk["source"],
            chunk_id=chunk["chunk_id"],
            text_preview=chunk["text"][:240],
        )
        for chunk in chunks
    ]

    log_query(
        {
            "question": request.question,
            "top_k": request.top_k,
            "latency_ms": latency_ms,
            "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in chunks],
            "retrieval_scores": [chunk["score"] for chunk in chunks],
            "sources": [chunk["source"] for chunk in chunks],
        }
    )

    return ChatResponse(
        answer=answer,
        citations=citations,
    )