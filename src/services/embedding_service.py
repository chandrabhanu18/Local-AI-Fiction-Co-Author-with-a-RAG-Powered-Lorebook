"""
Embedding Service utilizing Sentence-Transformers for semantic vector representations.
"""
import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Verify dimension
                test_emb = self._model.encode("test", convert_to_numpy=True)
                self._dimension = test_emb.shape[0]
                logger.info(f"SentenceTransformer loaded successfully. Dimension: {self._dimension}")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer ({e}). Initializing fallback embedding engine.")
                self._model = "fallback"
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generate a single vector embedding for a given text."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of texts."""
        if not texts:
            return []

        model_instance = self.model
        if model_instance != "fallback" and hasattr(model_instance, "encode"):
            try:
                embeddings = model_instance.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return [emb.tolist() for emb in embeddings]
            except Exception as e:
                logger.error(f"Error during model encoding: {e}. Falling back to deterministic embeddings.")

        # Deterministic hashing embedding fallback (e.g. for offline test mocks)
        return [self._fallback_embed(t) for t in texts]

    def _fallback_embed(self, text: str) -> List[float]:
        """Generate a deterministic normalized 384-dim pseudo-vector based on character hashing."""
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed pseudo-random generator with text hash to generate repeatable 384 floats
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
