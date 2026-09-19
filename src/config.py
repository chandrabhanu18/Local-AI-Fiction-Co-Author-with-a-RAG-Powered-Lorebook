"""
Configuration settings for Local AI Fiction Co-Author.
"""
import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # LLM Service (Ollama)
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    
    # Vector DB (ChromaDB)
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection_name: str = Field(default="fiction_lorebook", alias="CHROMA_COLLECTION_NAME")
    chroma_persist_dir: str = Field(default="./data/chroma", alias="CHROMA_PERSIST_DIR")
    
    # Embedding Model
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME")
    
    # App Settings
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8080, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Defaults
    default_top_k: int = Field(default=3, alias="DEFAULT_TOP_K")
    default_temperature: float = Field(default=0.7, alias="DEFAULT_TEMPERATURE")
    default_top_p: float = Field(default=0.9, alias="DEFAULT_TOP_P")
    default_repeat_penalty: float = Field(default=1.1, alias="DEFAULT_REPEAT_PENALTY")
    default_max_tokens: int = Field(default=512, alias="DEFAULT_MAX_TOKENS")
    
    # Persona path
    persona_prompt_path: str = Field(default="prompts/persona.md", alias="PERSONA_PROMPT_PATH")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True
    )


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached application settings."""
    return Settings()
