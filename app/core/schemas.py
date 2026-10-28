from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    chunk_id: str
    text_preview: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class IngestRequest(BaseModel):
    data_dir: str = "data/raw"


class IngestResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    processed_path: str