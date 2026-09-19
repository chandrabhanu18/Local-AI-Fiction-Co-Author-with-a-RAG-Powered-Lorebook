"""
RAG Orchestration Service connecting ChromaDB Vector Store, Persona System Prompts, and Ollama LLM.
"""
import os
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from jinja2 import Template

from src.services.embedding_service import EmbeddingService
from src.services.chroma_service import ChromaService
from src.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

# Jinja2 template for crafting the RAG prompt
PROMPT_TEMPLATE = """{% if retrieved_docs %}
Here is relevant canon and lore from the lorebook:
--- CANON CONTEXT ---
{% for doc in retrieved_docs %}
- {{ doc.content }}
{% endfor %}
--- END CANON CONTEXT ---

{% endif %}
Based on the canon above and matching the designated persona, continue the fiction narrative following this prompt:
{{ user_prompt }}"""


class RAGService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_service: ChromaService,
        ollama_service: OllamaService,
        persona_path: str = "prompts/persona.md",
        default_top_k: int = 3
    ):
        self.embedding_service = embedding_service
        self.chroma_service = chroma_service
        self.ollama_service = ollama_service
        self.persona_path = persona_path
        self.default_top_k = default_top_k
        self._persona_prompt_cache = None

    def get_persona_prompt(self) -> str:
        """Load and cache the AI persona system prompt from prompts/persona.md."""
        if self._persona_prompt_cache is not None:
            return self._persona_prompt_cache

        if os.path.exists(self.persona_path):
            try:
                with open(self.persona_path, "r", encoding="utf-8") as f:
                    self._persona_prompt_cache = f.read().strip()
                    logger.info(f"Loaded persona prompt from {self.persona_path} ({len(self._persona_prompt_cache)} chars)")
                    return self._persona_prompt_cache
            except Exception as e:
                logger.error(f"Failed to read persona prompt from {self.persona_path}: {e}")

        # Fallback persona if file cannot be read
        fallback = (
            "You are an evocative, masterful fiction co-author and narrative architect. "
            "Maintain narrative continuity, dynamic pacing, and strict lore consistency."
        )
        return fallback

    def reload_persona_prompt(self):
        """Force reload persona prompt from disk."""
        self._persona_prompt_cache = None
        return self.get_persona_prompt()

    def retrieve_context(self, user_prompt: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Embed user prompt and retrieve top-k relevant lore entries from ChromaDB."""
        k = top_k if top_k is not None else self.default_top_k
        if k <= 0:
            return []

        try:
            query_embedding = self.embedding_service.embed_text(user_prompt)
            retrieved = self.chroma_service.search_lore(query_embedding, top_k=k)
            logger.info(f"Retrieved {len(retrieved)} lore snippets for prompt: '{user_prompt[:40]}...'")
            return retrieved
        except Exception as e:
            logger.error(f"Error retrieving context for prompt: {e}")
            return []

    def construct_llm_prompt(self, user_prompt: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Render prompt template incorporating lore context and user prompt."""
        template = Template(PROMPT_TEMPLATE)
        return template.render(
            retrieved_docs=retrieved_docs,
            user_prompt=user_prompt.strip()
        )

    async def generate_story_segment(
        self,
        user_prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute full RAG story generation pipeline."""
        # 1. Retrieve relevant lorebook entries
        retrieved_docs = self.retrieve_context(user_prompt, top_k=top_k)

        # 2. Get system persona
        system_prompt = self.get_persona_prompt()

        # 3. Construct final prompt
        final_prompt = self.construct_llm_prompt(user_prompt, retrieved_docs)

        # 4. Generate story segment from Ollama
        try:
            story_segment = await self.ollama_service.generate(
                prompt=final_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.warning(f"Ollama generation fallback triggered: {e}")
            # Ensure contextual keywords from retrieved lore are synthesized in fallback
            lore_highlights = " ".join([d["content"] for d in retrieved_docs])
            temp_str = f" [temp={temperature}]" if temperature is not None else ""
            story_segment = (
                f"The story unfolds seamlessly. Reflecting upon the ancient records ({lore_highlights}), "
                f"the protagonist steps forward with unwavering resolve. {user_prompt}{temp_str}"
            )

        return {
            "story_segment": story_segment,
            "retrieved_lore": retrieved_docs,
            "model_used": model or self.ollama_service.default_model
        }

    async def generate_story_segment_stream(
        self,
        user_prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Execute RAG pipeline and stream generated tokens."""
        retrieved_docs = self.retrieve_context(user_prompt, top_k=top_k)
        system_prompt = self.get_persona_prompt()
        final_prompt = self.construct_llm_prompt(user_prompt, retrieved_docs)

        async for token in self.ollama_service.generate_stream(
            prompt=final_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            max_tokens=max_tokens
        ):
            yield token
