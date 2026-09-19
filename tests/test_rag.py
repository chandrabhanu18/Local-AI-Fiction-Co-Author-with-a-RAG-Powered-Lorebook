"""
Unit tests for RAG pipeline and Embedding Service.
"""
import pytest
from src.services.embedding_service import EmbeddingService
from src.services.chroma_service import ChromaService
from src.services.rag_service import RAGService


def test_embedding_generation():
    """Test text embedding and dimensionality."""
    service = EmbeddingService()
    emb = service.embed_text("A magical grimoire filled with elemental spells.")
    assert isinstance(emb, list)
    assert len(emb) == service.dimension
    assert all(isinstance(x, float) for x in emb)


def test_chroma_crud_and_search(test_chroma_service: ChromaService, test_embedding_service: EmbeddingService):
    """Test full CRUD and semantic search on ChromaDB."""
    # 1. Add lore entries
    docs = [
        "The celestial observatory of Zephyr reaches high above the cloudline.",
        "The underground forge of Karak produces impenetrable adamantine armor.",
        "Captain Donald commands the storm-skiff named Zephyr's Whisper."
    ]
    
    for doc in docs:
        emb = test_embedding_service.embed_text(doc)
        test_chroma_service.add_lore(content=doc, embedding=emb, metadata={"source": "test"})

    assert test_chroma_service.count() == 3

    # 2. Search for observatory
    query_emb = test_embedding_service.embed_text("astronomical telescope and stargazing")
    results = test_chroma_service.search_lore(query_emb, top_k=1)
    assert len(results) == 1
    assert "Zephyr" in results[0]["content"]

    # 3. Pagination
    all_lore = test_chroma_service.get_all_lore(limit=2)
    assert len(all_lore) == 2


@pytest.mark.asyncio
async def test_rag_pipeline_prompt_construction(test_rag_service: RAGService):
    """Test RAG prompt assembly with persona and context."""
    persona = test_rag_service.get_persona_prompt()
    assert len(persona) >= 100

    docs = [{"content": "The Dragon's Fang dagger is poisoned with Gorgon venom."}]
    constructed = test_rag_service.construct_llm_prompt("The rogue draws his weapon.", docs)
    
    assert "The Dragon's Fang dagger" in constructed
    assert "The rogue draws his weapon." in constructed
    assert "CANON CONTEXT" in constructed
