"""
Main FastAPI Application Entrypoint.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.config import get_settings
from src.services.embedding_service import EmbeddingService
from src.services.chroma_service import ChromaService
from src.services.ollama_service import OllamaService
from src.services.rag_service import RAGService
from src.routes.health import router as health_router
from src.routes.api import router as api_router

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fiction_coauthor")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize core services on startup and clean up on shutdown."""
    logger.info("Starting Fiction Co-Author Application...")
    
    # 1. Initialize Embedding Service
    embedding_service = EmbeddingService(model_name=settings.embedding_model_name)
    app.state.embedding_service = embedding_service
    
    # 2. Initialize ChromaDB Service
    chroma_service = ChromaService(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection_name,
        persist_dir=settings.chroma_persist_dir
    )
    app.state.chroma_service = chroma_service
    
    # 3. Initialize Ollama Service
    ollama_service = OllamaService(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_model
    )
    app.state.ollama_service = ollama_service
    
    # 4. Initialize RAG Service
    rag_service = RAGService(
        embedding_service=embedding_service,
        chroma_service=chroma_service,
        ollama_service=ollama_service,
        persona_path=settings.persona_prompt_path,
        default_top_k=settings.default_top_k
    )
    app.state.rag_service = rag_service

    logger.info("All services initialized successfully.")
    yield
    logger.info("Shutting down Fiction Co-Author Application...")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Local AI Fiction Co-Author API",
        description="A local RAG-powered fiction co-authoring assistant with persistent Lorebook",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(api_router)

    # Mount static files for the UI
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(os.path.join(static_dir, "index.html"))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True
    )
