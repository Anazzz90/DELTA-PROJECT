"""
llm/cache.py
=============
Checkpoint 18 — Semantic Caching

Caches full query results keyed by question similarity, so a new question
that's semantically close enough to a previously-answered one returns
instantly instead of re-running all 5 agents (+ Meta-AI).

Design note: this checkpoint's deliverable names "GPTCache" specifically,
but the actual `gptcache` package's ChromaDB adapter constructs its client
via `chromadb.Client(legacy_settings)`, which raises `ValueError: You are
using a deprecated configuration of Chroma` against the ChromaDB version
this project already depends on (chromadb>=0.5, used by
memory/vector_store.py since Checkpoint 9 — reproduced directly, not a
DMARS bug). Pinning an old, incompatible chromadb just for gptcache would
break VectorStore, and gptcache's default embedding path pulls in a heavy
torch + transformers stack to duplicate something this project already has
working and tested. This reimplements GPTCache's actual behavior —
similarity-threshold caching over vector search — directly on top of the
same ChromaDB pattern memory/vector_store.py already uses.

Usage:
    from llm.cache import SemanticCache

    cache = SemanticCache()
    cached = cache.get(question)          # None on a miss
    if cached is None:
        result = run_pipeline(...)
        cache.set(question, result)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from config.settings import settings

logger = logging.getLogger(__name__)

_DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / ".chromadb"
_COLLECTION_NAME = "dmars_semantic_cache"


class SemanticCache:
    """
    Thin ChromaDB-backed cache: stores each completed query's full result
    dict, keyed by the question's embedding. A lookup is a "hit" when the
    closest stored question's cosine similarity meets the configured
    threshold (default: settings.cache_similarity_threshold = 0.92).

    Caches on `question` text only — a repeat question with a different
    fact_set will still hit. This matches GPTCache's own prompt-similarity
    model (and this checkpoint's test criteria, which only vary the
    question wording), but is worth knowing if fact_set can vary widely
    for the same phrasing in practice.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        path = persist_dir or _DEFAULT_PERSIST_DIR
        path.mkdir(parents=True, exist_ok=True)
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.cache_similarity_threshold
        )
        self._client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def get(self, question: str) -> Optional[dict]:
        """
        Return the cached result dict for the closest semantically similar
        question if it meets the similarity threshold, else None.
        """
        if self._collection.count() == 0:
            return None

        results = self._collection.query(query_texts=[question], n_results=1)
        if not results["ids"][0]:
            return None

        # ChromaDB cosine distance: 0.0 = identical, 2.0 = opposite.
        distance = results["distances"][0][0]
        similarity = 1.0 - (distance / 2.0)
        if similarity < self._threshold:
            logger.info(
                f"Semantic cache MISS (best similarity={similarity:.4f} < "
                f"threshold={self._threshold})"
            )
            return None

        cached_json = results["metadatas"][0][0].get("result_json")
        if not cached_json:
            return None

        result = json.loads(cached_json)
        result["cache_hit"] = True
        logger.info(f"Semantic cache HIT (similarity={similarity:.4f} >= {self._threshold})")
        return result

    def set(self, question: str, result: dict) -> None:
        """Store a completed query result, keyed by the question's embedding."""
        payload = {k: v for k, v in result.items() if k != "cache_hit"}
        doc_id = f"cache:{result.get('query_id', question)}"
        self._collection.upsert(
            ids=[doc_id],
            documents=[question],
            metadatas=[{"result_json": json.dumps(payload)}],
        )
        logger.info(f"Semantic cache: stored result for query_id={result.get('query_id')}")

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """Delete all cached entries. Used in tests for a clean state."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
