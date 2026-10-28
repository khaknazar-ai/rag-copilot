from fastapi import APIRouter

from app.core.config import settings
from app.core.schemas import IngestRequest, IngestResponse
from app.ingestion.indexer import DocumentIndexer

router = APIRouter()


@router.post("", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    indexer = DocumentIndexer()

    result = indexer.index_documents(
        data_dir=request.data_dir,
        output_path="data/processed/chunks.jsonl",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return IngestResponse(**result)