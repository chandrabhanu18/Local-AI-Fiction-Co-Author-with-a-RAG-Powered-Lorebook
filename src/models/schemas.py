"""
Pydantic schemas for the API contracts.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# ----------------------------------------------------
# Lore Schemas
# ----------------------------------------------------
class LoreCreateRequest(BaseModel):
    content: str = Field(..., description="The text of the lore entry", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional key-value pairs for filtering")


class LoreCreateResponse(BaseModel):
    status: str = Field(default="success", description="Status string")
    id: str = Field(..., description="The unique ID of the stored lore entry")


class LoreItem(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class LoreListResponse(BaseModel):
    total: int
    items: List[LoreItem]


# ----------------------------------------------------
# Generation Schemas
# ----------------------------------------------------
class GenerationParameters(BaseModel):
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Randomness / creativity")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    repeat_penalty: Optional[float] = Field(default=None, ge=0.0, le=3.0, description="Repetition penalty")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096, description="Max tokens to generate")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The user's input to continue the story", min_length=1)
    parameters: Optional[GenerationParameters] = Field(default=None, description="Generation parameters")
    model: Optional[str] = Field(default=None, description="Optional override for LLM model")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of lore snippets to retrieve")
    stream: Optional[bool] = Field(default=False, description="Whether to stream response tokens")


class GenerateResponse(BaseModel):
    story_segment: str = Field(..., description="The AI-generated text")
    retrieved_lore: Optional[List[Dict[str, Any]]] = Field(default=None, description="Contextual lore retrieved by RAG")
    model_used: Optional[str] = Field(default=None, description="Model used for generation")


# ----------------------------------------------------
# Health & Diagnostic Schemas
# ----------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    chromadb_status: str
    ollama_connected: bool
    chromadb_connected: bool
    embedding_model: str
    active_model: str
    lore_entries_count: int


class ModelItem(BaseModel):
    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None


class ModelListResponse(BaseModel):
    models: List[ModelItem]
