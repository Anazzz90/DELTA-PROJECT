"""
tests/unit/test_vector_store_qdrant.py
=========================================
Checkpoint 23 -- Qdrant backend for VectorStore and SemanticCache

qdrant_client is mocked throughout -- no live Qdrant server or fastembed
model download needed for these tests. Live end-to-end proof against the
real dmars-qdrant container is done separately (see checkpoint notes).

Run:
    poetry run pytest tests/unit/test_vector_store_qdrant.py -v
"""

from unittest.mock import MagicMock, patch

from config.settings import settings
from llm.cache import SemanticCache
from memory.vector_store import VectorStore


class TestVectorStoreBackendSelection:

    def test_development_uses_chroma(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "env", "development")
        store = VectorStore(persist_dir=tmp_path / "chroma")
        assert store._backend == "chroma"

    def test_production_uses_qdrant(self, monkeypatch):
        monkeypatch.setattr(settings, "env", "production")
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = True

        with patch("qdrant_client.QdrantClient", return_value=fake_client):
            store = VectorStore()

        assert store._backend == "qdrant"

    def test_production_creates_collection_if_missing(self, monkeypatch):
        monkeypatch.setattr(settings, "env", "production")
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = False

        with patch("qdrant_client.QdrantClient", return_value=fake_client):
            VectorStore()

        assert fake_client.create_collection.called


class TestVectorStoreQdrantOperations:

    def _make_store(self, monkeypatch, fake_client):
        monkeypatch.setattr(settings, "env", "production")
        fake_client.collection_exists.return_value = True
        with patch("qdrant_client.QdrantClient", return_value=fake_client):
            return VectorStore()

    def test_add_upserts_a_point(self, monkeypatch):
        fake_client = MagicMock()
        store = self._make_store(monkeypatch, fake_client)

        store.add(query_id=1, question="Why did BTC spike?", metadata={"domain": "intraday_trading"})

        assert fake_client.upsert.called
        call_kwargs = fake_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "dmars_queries"
        point = call_kwargs["points"][0]
        assert point.id == 1
        assert point.payload["query_id"] == 1
        assert point.payload["domain"] == "intraday_trading"

    def test_search_converts_score_to_chroma_style_distance(self, monkeypatch):
        fake_client = MagicMock()
        fake_point = MagicMock()
        fake_point.score = 1.0  # identical
        fake_point.payload = {"query_id": 1, "document": "Why did BTC spike?"}
        fake_response = MagicMock()
        fake_response.points = [fake_point]
        fake_client.query_points.return_value = fake_response
        store = self._make_store(monkeypatch, fake_client)

        results = store.search("BTC spike", n_results=1)

        assert len(results) == 1
        assert results[0]["query_id"] == 1
        assert results[0]["distance"] == 0.0  # score=1.0 (identical) -> distance=0.0

    def test_count_delegates_to_qdrant_count(self, monkeypatch):
        fake_client = MagicMock()
        fake_client.count.return_value = MagicMock(count=5)
        store = self._make_store(monkeypatch, fake_client)

        assert store.count() == 5
        fake_client.count.assert_called_with(collection_name="dmars_queries")

    def test_clear_recreates_the_collection(self, monkeypatch):
        fake_client = MagicMock()
        store = self._make_store(monkeypatch, fake_client)

        store.clear()

        assert fake_client.delete_collection.called
        assert fake_client.create_collection.called


class TestSemanticCacheQdrantOperations:

    def _make_cache(self, monkeypatch, fake_client):
        monkeypatch.setattr(settings, "env", "production")
        fake_client.collection_exists.return_value = True
        with patch("qdrant_client.QdrantClient", return_value=fake_client):
            return SemanticCache()

    def test_backend_is_qdrant_in_production(self, monkeypatch):
        fake_client = MagicMock()
        cache = self._make_cache(monkeypatch, fake_client)
        assert cache._backend == "qdrant"

    def test_get_applies_scope_filter(self, monkeypatch):
        import json
        fake_client = MagicMock()
        fake_point = MagicMock()
        fake_point.score = 0.95
        fake_point.payload = {"result_json": json.dumps({"query_id": 1, "system_main_driver": "test"})}
        fake_response = MagicMock()
        fake_response.points = [fake_point]
        fake_client.query_points.return_value = fake_response
        cache = self._make_cache(monkeypatch, fake_client)

        result = cache.get("Why did BTC spike?", ["data_first"])

        assert result is not None
        assert result["cache_hit"] is True
        call_kwargs = fake_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is not None

    def test_get_returns_none_below_threshold(self, monkeypatch):
        import json
        fake_client = MagicMock()
        fake_point = MagicMock()
        fake_point.score = 0.5  # below default 0.92 threshold
        fake_point.payload = {"result_json": json.dumps({"query_id": 1})}
        fake_response = MagicMock()
        fake_response.points = [fake_point]
        fake_client.query_points.return_value = fake_response
        cache = self._make_cache(monkeypatch, fake_client)

        assert cache.get("Why did BTC spike?", ["data_first"]) is None

    def test_set_upserts_with_deterministic_uuid(self, monkeypatch):
        fake_client = MagicMock()
        cache = self._make_cache(monkeypatch, fake_client)

        cache.set("Why did BTC spike?", ["data_first"], {"query_id": 1, "system_main_driver": "x"})

        assert fake_client.upsert.called
        point = fake_client.upsert.call_args.kwargs["points"][0]
        # Same doc_id must always produce the same point id (upsert semantics).
        assert point.id == SemanticCache._qdrant_point_id("cache:1")
