"""
memory/vector_store.py
========================
Checkpoint 9 — ChromaDB Vector Memory Layer (Phase 1: embedded mode)
Checkpoint 23 — Qdrant Vector Memory Layer (Phase 3: production)

Stores semantic embeddings of past queries so DMARS can recall whether
it has reasoned about a similar event before.

Backend selection is automatic, based on settings.env:
  - ENV=development (default) → ChromaDB, fully embedded, no server needed
  - ENV=production            → Qdrant, via the dedicated dmars-qdrant
                                 container (settings.qdrant_url)

The public interface (add/search/count/clear) is identical either way —
callers (memory/history.py, task_queue/tasks.py) never need to know which
backend is active.

Usage:
    from memory.vector_store import VectorStore

    store = VectorStore()
    store.add(query_id=1, question="Why did BTC spike?", metadata={...})
    results = store.search("BTC volume surge", n_results=3)
    for r in results:
        print(r["query_id"], r["distance"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Persist ChromaDB data inside the project folder (dev only)
_DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / ".chromadb"
_COLLECTION_NAME = "dmars_queries"
_QDRANT_VECTOR_NAME = "fast-bge-small-en"
_QDRANT_EMBED_MODEL = "BAAI/bge-small-en"


class VectorStore:
    """
    Semantic memory storage — ChromaDB (dev, embedded) or Qdrant (prod,
    via a running server), selected automatically from settings.env.

    Phase 1/2: Embedded ChromaDB (persists to local .chromadb/ folder).
    Phase 3:   Qdrant server (settings.qdrant_url), FastEmbed for text
               embedding (same default model Qdrant's own convenience
               API uses), cosine distance to match ChromaDB's semantics.
    """

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        self._backend = "qdrant" if settings.is_production else "chroma"
        if self._backend == "qdrant":
            self._init_qdrant()
        else:
            self._init_chroma(persist_dir)

    # =========================================================================
    # Backend init
    # =========================================================================

    def _init_chroma(self, persist_dir: Optional[Path]) -> None:
        import chromadb
        from chromadb.config import Settings

        path = persist_dir or _DEFAULT_PERSIST_DIR
        path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready | backend=chroma | collection='{_COLLECTION_NAME}' | "
            f"docs={self._collection.count()} | path={path}"
        )

    def _init_qdrant(self) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=settings.qdrant_url)
        if not self._client.collection_exists(_COLLECTION_NAME):
            self._client.create_collection(
                collection_name=_COLLECTION_NAME,
                vectors_config=self._client.get_fastembed_vector_params(),
            )
        logger.info(f"VectorStore ready | backend=qdrant | url={settings.qdrant_url}")

    # =========================================================================
    # Write
    # =========================================================================

    def add(
        self,
        query_id:  int,
        question:  str,
        metadata:  Optional[dict] = None,
    ) -> None:
        """
        Store a question embedding in the vector store.

        Args:
            query_id: The SQL query ID (used as the document/point ID).
            question: The question text (embedded automatically).
            metadata: Extra data to store alongside the embedding.
        """
        meta = dict(metadata or {})
        meta["query_id"] = query_id

        if self._backend == "qdrant":
            from qdrant_client import models
            self._client.upsert(
                collection_name=_COLLECTION_NAME,
                points=[models.PointStruct(
                    id=query_id,
                    vector={_QDRANT_VECTOR_NAME: models.Document(text=question, model=_QDRANT_EMBED_MODEL)},
                    payload={**meta, "document": question},
                )],
            )
        else:
            self._collection.upsert(
                ids=[str(query_id)],
                documents=[question],
                metadatas=[meta],
            )
        logger.info(f"VectorStore: stored query_id={query_id}")

    # =========================================================================
    # Read
    # =========================================================================

    def search(
        self,
        text:      str,
        n_results: int = 3,
    ) -> list[dict]:
        """
        Find the most semantically similar past queries.

        Args:
            text:      The query text to search for.
            n_results: Number of similar results to return.

        Returns:
            List of dicts, each with keys:
              - query_id (int)
              - document (str)  — the original question
              - distance (float) — 0.0 = identical, 2.0 = completely different
              - metadata (dict)
        """
        if self._backend == "qdrant":
            return self._search_qdrant(text, n_results)
        return self._search_chroma(text, n_results)

    def _search_chroma(self, text: str, n_results: int) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []

        actual_n = min(n_results, count)
        results  = self._collection.query(
            query_texts=[text],
            n_results=actual_n,
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "query_id": results["metadatas"][0][i].get("query_id"),
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            })
        return output

    def _search_qdrant(self, text: str, n_results: int) -> list[dict]:
        from qdrant_client import models

        response = self._client.query_points(
            collection_name=_COLLECTION_NAME,
            query=models.Document(text=text, model=_QDRANT_EMBED_MODEL),
            using=_QDRANT_VECTOR_NAME,
            limit=n_results,
        )

        output = []
        for point in response.points:
            payload = dict(point.payload or {})
            # Qdrant score is cosine *similarity* (higher = closer); ChromaDB's
            # `distance` is cosine *distance* (lower = closer) — convert so
            # callers get identical semantics regardless of backend.
            distance = max(0.0, 1.0 - point.score) * 2.0
            output.append({
                "query_id": payload.get("query_id"),
                "document": payload.get("document", ""),
                "distance": distance,
                "metadata": payload,
            })
        return output

    def count(self) -> int:
        """Return total number of stored embeddings."""
        if self._backend == "qdrant":
            return self._client.count(collection_name=_COLLECTION_NAME).count
        return self._collection.count()

    def clear(self) -> None:
        """
        Delete all documents from the collection.
        Used in tests to ensure a clean state.
        """
        if self._backend == "qdrant":
            self._client.delete_collection(_COLLECTION_NAME)
            self._client.create_collection(
                collection_name=_COLLECTION_NAME,
                vectors_config=self._client.get_fastembed_vector_params(),
            )
        else:
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        logger.info("VectorStore: collection cleared")
