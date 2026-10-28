from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.core.config import settings
from app.api.middleware.chat_cache import ChatResponseCacheMiddleware

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Document-grounded RAG assistant API for consular-service information.",
)
app.add_middleware(ChatResponseCacheMiddleware)

app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(ingest_router, prefix="/ingest", tags=["ingestion"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
