"""
Pytest configuration and shared test fixtures.
"""
import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.services.embedding_service import EmbeddingService
from src.services.chroma_service import ChromaService
from src.services.ollama_service import OllamaService
from src.services.rag_service import RAGService


class MockOllamaService(OllamaService):
    """Mock Ollama service for predictable unit/integration testing."""
    def __init__(self):
        super().__init__(base_url="http://mock-ollama:11434", default_model="llama3.1:8b")

    async def is_healthy(self) -> bool:
        return True

    async def list_models(self):
        return [{"name": "llama3.1:8b", "size": 4700000000}]

    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = None,
        top_p: float = None,
        repeat_penalty: float = None,
        max_tokens: int = None
    ) -> str:
        temp_val = f"{temperature:.2f}" if temperature is not None else "default"
        
        # If canon context is injected, ensure keywords in the prompt are reflected in output
        if "Aethelgard" in prompt:
            return (
                f"Drawing the legendary blade Aethelgard from its scabbard, "
                f"a resonant sapphire hum filled the cold air. The ancient sword Aethelgard "
                f"illuminated the path forward. [temp={temp_val}]"
            )
        
        return (
            f"The storyteller weaves the chronicle: {prompt[:80]}. "
            f"The moonlight cast long shadows across the cobblestones. [temp={temp_val}]"
        )


@pytest.fixture(scope="session")
def test_embedding_service():
    return EmbeddingService(model_name="all-MiniLM-L6-v2")


@pytest.fixture(scope="function")
def test_chroma_service():
    # Use ephemeral in-memory chroma client for clean isolation between tests
    service = ChromaService(
        host="local_only",
        collection_name="test_fiction_lorebook",
        persist_dir="./data/test_chroma"
    )
    service.clear_collection()
    return service


@pytest.fixture(scope="function")
def test_ollama_service():
    return MockOllamaService()


@pytest.fixture(scope="function")
def test_rag_service(test_embedding_service, test_chroma_service, test_ollama_service):
    return RAGService(
        embedding_service=test_embedding_service,
        chroma_service=test_chroma_service,
        ollama_service=test_ollama_service,
        persona_path="prompts/persona.md",
        default_top_k=3
    )


@pytest.fixture(scope="function")
def client(test_embedding_service, test_chroma_service, test_ollama_service, test_rag_service):
    app = create_app()
    app.state.embedding_service = test_embedding_service
    app.state.chroma_service = test_chroma_service
    app.state.ollama_service = test_ollama_service
    app.state.rag_service = test_rag_service
    
    with TestClient(app) as test_client:
        yield test_client
