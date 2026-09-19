"""
Health and diagnostics router.
"""
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from src.models.schemas import HealthResponse
from src.config import get_settings, Settings

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def healthcheck(request: Request, settings: Settings = Depends(get_settings)):
    """Application and dependent services health check."""
    chroma_service = getattr(request.app.state, "chroma_service", None)
    ollama_service = getattr(request.app.state, "ollama_service", None)
    embedding_service = getattr(request.app.state, "embedding_service", None)

    chroma_healthy = chroma_service.is_healthy() if chroma_service else False
    ollama_healthy = await ollama_service.is_healthy() if ollama_service else False
    lore_count = chroma_service.count() if chroma_service else 0

    overall_status = "healthy" if chroma_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        ollama_status="connected" if ollama_healthy else "unreachable",
        chromadb_status="connected" if chroma_healthy else "unreachable",
        ollama_connected=ollama_healthy,
        chromadb_connected=chroma_healthy,
        embedding_model=settings.embedding_model_name,
        active_model=settings.ollama_model,
        lore_entries_count=lore_count
    )
