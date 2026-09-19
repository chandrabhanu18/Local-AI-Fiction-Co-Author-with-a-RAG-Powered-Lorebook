"""
Core API endpoints for Lorebook management and Story Generation.
"""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import StreamingResponse

from src.models.schemas import (
    LoreCreateRequest,
    LoreCreateResponse,
    LoreListResponse,
    LoreItem,
    GenerateRequest,
    GenerateResponse,
    ModelListResponse,
    ModelItem,
)
from src.config import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Lore & Generation"])


# ----------------------------------------------------
# Lorebook Management Endpoints
# ----------------------------------------------------
@router.post(
    "/lore",
    response_model=LoreCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new lore entry to the vector database"
)
async def add_lore(
    payload: LoreCreateRequest,
    request: Request
):
    """
    Contract Requirement 5:
    Receives lore text snippet, generates an embedding, and stores text and embedding in ChromaDB.
    Returns 201 Created with status 'success' and the created entry's unique ID.
    """
    if not payload.content or not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Lore content cannot be empty."
        )

    embedding_service = request.app.state.embedding_service
    chroma_service = request.app.state.chroma_service

    try:
        # 1. Generate embedding for lore snippet
        embedding = embedding_service.embed_text(payload.content)

        # 2. Store in ChromaDB
        lore_id = chroma_service.add_lore(
            content=payload.content.strip(),
            embedding=embedding,
            metadata=payload.metadata
        )

        logger.info(f"Lore added successfully with ID: {lore_id}")
        return LoreCreateResponse(
            status="success",
            id=lore_id
        )
    except Exception as e:
        logger.error(f"Failed to add lore entry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store lore: {str(e)}"
        )


@router.get("/lore", response_model=LoreListResponse, summary="List all lore entries")
async def list_lore(
    request: Request,
    limit: int = 100,
    offset: int = 0
):
    """Retrieve all lorebook entries stored in ChromaDB."""
    chroma_service = request.app.state.chroma_service
    items = chroma_service.get_all_lore(limit=limit, offset=offset)
    total = chroma_service.count()

    lore_items = [
        LoreItem(
            id=item["id"],
            content=item["content"],
            metadata=item.get("metadata", {})
        )
        for item in items
    ]
    return LoreListResponse(total=total, items=lore_items)


@router.delete("/lore/{lore_id}", summary="Delete a specific lore entry")
async def delete_lore_entry(
    lore_id: str,
    request: Request
):
    """Delete a lore entry by ID."""
    chroma_service = request.app.state.chroma_service
    success = chroma_service.delete_lore(lore_id)
    if not success:
        raise HTTPException(status_code=404, detail="Lore entry not found or deletion failed.")
    return {"status": "success", "message": f"Lore {lore_id} deleted."}


@router.delete("/lore", summary="Clear entire lorebook collection")
async def clear_lorebook(request: Request):
    """Clear all entries from the lorebook collection."""
    chroma_service = request.app.state.chroma_service
    success = chroma_service.clear_collection()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear lorebook.")
    return {"status": "success", "message": "Lorebook collection cleared."}


# ----------------------------------------------------
# Story Generation Endpoints
# ----------------------------------------------------
@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a story segment using RAG and local LLM"
)
async def generate_story(
    payload: GenerateRequest,
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """
    Contract Requirement 6, 7, 8:
    Receives user prompt and optional parameters (temperature, top_p, etc.).
    Embeds prompt, queries ChromaDB for top lore snippets, formats prompt with persona and context,
    and returns 200 OK with story_segment.
    """
    if not payload.prompt or not payload.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt cannot be empty."
        )

    rag_service = request.app.state.rag_service

    # Extract parameters
    temp = payload.parameters.temperature if payload.parameters else None
    top_p = payload.parameters.top_p if payload.parameters else None
    repeat_penalty = payload.parameters.repeat_penalty if payload.parameters else None
    max_tokens = payload.parameters.max_tokens if payload.parameters else settings.default_max_tokens

    try:
        result = await rag_service.generate_story_segment(
            user_prompt=payload.prompt,
            temperature=temp,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
            model=payload.model,
            top_k=payload.top_k or settings.default_top_k
        )

        return GenerateResponse(
            story_segment=result["story_segment"],
            retrieved_lore=result.get("retrieved_lore"),
            model_used=result.get("model_used")
        )
    except Exception as e:
        logger.error(f"Story generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@router.post(
    "/generate/stream",
    summary="Stream story generation token-by-token"
)
async def generate_story_stream(
    payload: GenerateRequest,
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """Stream AI story tokens in real-time."""
    if not payload.prompt or not payload.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt cannot be empty."
        )

    rag_service = request.app.state.rag_service

    temp = payload.parameters.temperature if payload.parameters else None
    top_p = payload.parameters.top_p if payload.parameters else None
    repeat_penalty = payload.parameters.repeat_penalty if payload.parameters else None
    max_tokens = payload.parameters.max_tokens if payload.parameters else settings.default_max_tokens

    async def token_generator():
        async for token in rag_service.generate_story_segment_stream(
            user_prompt=payload.prompt,
            temperature=temp,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens,
            model=payload.model,
            top_k=payload.top_k or settings.default_top_k
        ):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")


# ----------------------------------------------------
# Model & Persona Management Endpoints
# ----------------------------------------------------
@router.get("/models", response_model=ModelListResponse, summary="List available Ollama models")
async def list_models(request: Request):
    """List available LLM models in Ollama."""
    ollama_service = request.app.state.ollama_service
    raw_models = await ollama_service.list_models()
    model_items = [
        ModelItem(
            name=m.get("name", "unknown"),
            size=m.get("size"),
            modified_at=m.get("modified_at")
        )
        for m in raw_models
    ]
    return ModelListResponse(models=model_items)


@router.get("/persona", summary="Get current persona prompt")
async def get_persona(request: Request):
    """Retrieve active system persona prompt."""
    rag_service = request.app.state.rag_service
    persona_text = rag_service.get_persona_prompt()
    return {"persona": persona_text}
