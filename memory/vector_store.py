"""
memory/vector_store.py
========================
Checkpoint 9 — ChromaDB Vector Memory Layer (Phase 1: embedded mode)

Stores semantic embeddings of past queries so DMARS can recall whether
it has reasoned about a similar event before. In Phase 1, ChromaDB runs
fully embedded (no server needed). In Phase 3, this swaps to Qdrant.

Key design:
  - One ChromaDB collection: "dmars_queries"
  - Each stored document = the question text
  - Metadata: query_id, domain, confidence, timestamp
  - Similarity search uses cosine distance (ChromaDB default)

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

import chromadb

logger = logging.getLogger(__name__)

# Persist ChromaDB data inside the project folder
_DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / ".chromadb"
_COLLECTION_NAME     = "dmars_queries"


class VectorStore:
    """
    Thin wrapper around ChromaDB for semantic memory storage.

    Phase 1: Embedded ChromaDB (persists to local .chromadb/ folder).
    Phase 3: Swap client to Qdrant by changing __init__ only.
    """

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        path = persist_dir or _DEFAULT_PERSIST_DIR
        path.mkdir(parents=True, exist_ok=True)

        from chromadb.config import Settings
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready | collection='{_COLLECTION_NAME}' | "
            f"docs={self._collection.count()} | path={path}"
        )

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
            query_id: The SQLite query ID (used as the ChromaDB document ID).
            question: The question text (ChromaDB embeds this automatically).
            metadata: Extra data to store alongside the embedding.
        """
        doc_id   = str(query_id)
        meta     = metadata or {}
        meta["query_id"] = query_id

        self._collection.upsert(
            ids=[doc_id],
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

    def count(self) -> int:
        """Return total number of stored embeddings."""
        return self._collection.count()

    def clear(self) -> None:
        """
        Delete all documents from the collection.
        Used in tests to ensure a clean state.
        """
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStore: collection cleared")
