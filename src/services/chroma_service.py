"""
ChromaDB Vector Database Service for storing and querying story lore.
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class ChromaService:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "fiction_lorebook",
        persist_dir: str = "./data/chroma"
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        self._init_client()

    def _init_client(self):
        """Initialize connection to ChromaDB (HTTP or Persistent local fallback)."""
        # Try remote HTTP client first (standard docker-compose setup)
        if self.host and self.host != "local_only":
            try:
                logger.info(f"Attempting connection to ChromaDB HTTP Server at {self.host}:{self.port}")
                http_client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                # Test connectivity
                http_client.heartbeat()
                self.client = http_client
                logger.info("Successfully connected to ChromaDB HTTP Server.")
            except Exception as e:
                logger.warning(f"Could not connect to ChromaDB HTTP Server ({e}). Falling back to local PersistentClient.")
                self.client = None

        if self.client is None:
            try:
                import os
                os.makedirs(self.persist_dir, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                logger.info(f"Initialized local Chroma PersistentClient at {self.persist_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize local PersistentClient: {e}. Using EphemeralClient.")
                self.client = chromadb.EphemeralClient()

        self._ensure_collection()

    def _ensure_collection(self):
        """Get or create the fiction lorebook collection."""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Lorebook Canon for Fiction Co-Authoring"}
            )
            logger.info(f"Connected to Chroma collection '{self.collection_name}' with {self.collection.count()} entries.")
        except Exception as e:
            logger.error(f"Error ensuring Chroma collection '{self.collection_name}': {e}")
            raise

    def add_lore(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        lore_id: Optional[str] = None
    ) -> str:
        """Add or update a lore snippet in ChromaDB."""
        if not lore_id:
            lore_id = str(uuid.uuid4())

        clean_metadata = {}
        if metadata:
            # ChromaDB metadata values must be str, int, float, or bool
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)

        clean_metadata["stored_length"] = len(content)

        self.collection.upsert(
            ids=[lore_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[clean_metadata]
        )
        return lore_id

    def search_lore(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the most relevant lore entries for a query vector."""
        total_items = self.count()
        if total_items == 0:
            return []

        actual_k = min(top_k, total_items)
        if actual_k <= 0:
            return []

        query_params: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"]
        }

        if metadata_filter:
            query_params["where"] = metadata_filter

        try:
            results = self.collection.query(**query_params)
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            ids = results["ids"][0] if "ids" in results else [f"doc_{i}" for i in range(len(docs))]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

            for doc, meta, doc_id, dist in zip(docs, metas, ids, distances):
                retrieved.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta or {},
                    "distance": float(dist) if dist is not None else 0.0,
                    # Convert distance to similarity score
                    "relevance_score": max(0.0, 1.0 - (float(dist) if dist is not None else 0.0))
                })

        return retrieved

    def get_all_lore(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List lorebook entries with pagination."""
        try:
            results = self.collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"]
            )
            items = []
            if results and "ids" in results:
                for doc_id, doc, meta in zip(
                    results["ids"],
                    results.get("documents", []),
                    results.get("metadatas", [])
                ):
                    items.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": meta or {}
                    })
            return items
        except Exception as e:
            logger.error(f"Error fetching lore entries: {e}")
            return []

    def get_lore_by_id(self, lore_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific lore entry by its unique ID."""
        try:
            results = self.collection.get(ids=[lore_id], include=["documents", "metadatas"])
            if results and results["ids"]:
                return {
                    "id": results["ids"][0],
                    "content": results["documents"][0] if results.get("documents") else "",
                    "metadata": results["metadatas"][0] if results.get("metadatas") else {}
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching lore by id {lore_id}: {e}")
            return None

    def delete_lore(self, lore_id: str) -> bool:
        """Delete a lore entry by ID."""
        try:
            self.collection.delete(ids=[lore_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting lore {lore_id}: {e}")
            return False

    def clear_collection(self) -> bool:
        """Clear all entries from the lorebook collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._ensure_collection()
            return True
        except Exception as e:
            logger.error(f"Error clearing collection {self.collection_name}: {e}")
            return False

    def count(self) -> int:
        """Return total count of lore entries."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def is_healthy(self) -> bool:
        """Verify ChromaDB connectivity."""
        try:
            if hasattr(self.client, "heartbeat"):
                hb = self.client.heartbeat()
                return hb is not None
            return self.collection is not None
        except Exception:
            return False
