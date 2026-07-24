"""
tests/unit/test_cache.py
===========================
Checkpoint 18 -- Semantic Caching
Checkpoint 21 -- scope isolation fix (selected_agents/domain/meta_ai_enabled)

Uses a temp ChromaDB directory per test (via pytest's tmp_path), so no
state leaks between tests and nothing touches the real .chromadb/.

Run:
    poetry run pytest tests/unit/test_cache.py -v
"""

import pytest

from llm.cache import SemanticCache


SAMPLE_RESULT = {
    "query_id": 1,
    "question": "Why did BTC spike 8% in the last hour?",
    "system_main_driver": "Liquidation cascade",
    "system_confidence_score": 0.72,
    "total_cost_usd": 0.0003,
    "agent_results": [],
}
AGENTS = ["data_first", "skeptic"]


@pytest.fixture
def cache(tmp_path):
    return SemanticCache(persist_dir=tmp_path / "chroma", similarity_threshold=0.92)


class TestCacheMissAndHit:

    def test_miss_on_empty_cache(self, cache):
        assert cache.get("Why did BTC spike 8% in the last hour?", AGENTS) is None

    def test_hit_on_exact_repeat(self, cache):
        cache.set(SAMPLE_RESULT["question"], AGENTS, SAMPLE_RESULT)

        result = cache.get(SAMPLE_RESULT["question"], AGENTS)

        assert result is not None
        assert result["cache_hit"] is True
        assert result["system_main_driver"] == "Liquidation cascade"

    def test_hit_on_semantically_similar_rewording(self, cache):
        cache.set("Why did BTC spike 8% in the last hour?", AGENTS, SAMPLE_RESULT)

        result = cache.get("What caused BTC to jump 8% recently?", AGENTS)

        assert result is not None
        assert result["cache_hit"] is True

    def test_miss_on_unrelated_question(self, cache):
        cache.set("Why did BTC spike 8% in the last hour?", AGENTS, SAMPLE_RESULT)

        result = cache.get("What is the best recipe for chocolate cake?", AGENTS)

        assert result is None

    def test_stored_cache_hit_flag_never_persisted(self, cache):
        """Storing a result that already has cache_hit=True shouldn't poison future reads."""
        hit_result = {**SAMPLE_RESULT, "cache_hit": True}
        cache.set(SAMPLE_RESULT["question"], AGENTS, hit_result)

        result = cache.get(SAMPLE_RESULT["question"], AGENTS)

        assert result["cache_hit"] is True  # set fresh by get(), not carried over stale


class TestScopeIsolation:
    """
    Checkpoint 21 regression coverage: caught live when a single-agent
    ["contrarian"] request returned a cached 2-agent ["data_first","skeptic"]
    result for a similar-sounding question. A cache hit must match the
    exact set of selected agents (and domain/meta settings), not just the
    question text.
    """

    def test_same_question_different_agents_does_not_hit(self, cache):
        cache.set(SAMPLE_RESULT["question"], ["data_first", "skeptic"], SAMPLE_RESULT)

        result = cache.get(SAMPLE_RESULT["question"], ["contrarian"])

        assert result is None

    def test_same_question_same_agents_different_order_still_hits(self, cache):
        cache.set(SAMPLE_RESULT["question"], ["skeptic", "data_first"], SAMPLE_RESULT)

        result = cache.get(SAMPLE_RESULT["question"], ["data_first", "skeptic"])

        assert result is not None

    def test_same_question_different_domain_does_not_hit(self, cache):
        cache.set(SAMPLE_RESULT["question"], AGENTS, SAMPLE_RESULT, domain_profile="intraday_trading")

        result = cache.get(SAMPLE_RESULT["question"], AGENTS, domain_profile="macro_analysis")

        assert result is None

    def test_same_question_different_meta_ai_flag_does_not_hit(self, cache):
        cache.set(SAMPLE_RESULT["question"], AGENTS, SAMPLE_RESULT, meta_ai_enabled=False)

        result = cache.get(SAMPLE_RESULT["question"], AGENTS, meta_ai_enabled=True)

        assert result is None

    def test_two_scopes_for_the_same_question_coexist_independently(self, cache):
        cache.set(SAMPLE_RESULT["question"], ["data_first"], {**SAMPLE_RESULT, "query_id": 1})
        cache.set(SAMPLE_RESULT["question"], ["skeptic"], {**SAMPLE_RESULT, "query_id": 2})

        assert cache.get(SAMPLE_RESULT["question"], ["data_first"])["query_id"] == 1
        assert cache.get(SAMPLE_RESULT["question"], ["skeptic"])["query_id"] == 2


class TestSimilarityThreshold:

    def test_stricter_threshold_turns_hit_into_miss(self, tmp_path):
        loose_cache = SemanticCache(persist_dir=tmp_path / "chroma", similarity_threshold=0.5)
        loose_cache.set("Why did BTC spike 8% in the last hour?", AGENTS, SAMPLE_RESULT)
        assert loose_cache.get("What caused BTC to jump 8% recently?", AGENTS) is not None

        strict_cache = SemanticCache(persist_dir=tmp_path / "chroma", similarity_threshold=0.999)
        assert strict_cache.get("What caused BTC to jump 8% recently?", AGENTS) is None

    def test_default_threshold_matches_settings(self, tmp_path):
        from config.settings import settings
        cache = SemanticCache(persist_dir=tmp_path / "chroma")
        assert cache._threshold == settings.cache_similarity_threshold


class TestCacheMaintenance:

    def test_count_reflects_stored_entries(self, cache):
        assert cache.count() == 0
        cache.set("Question A", AGENTS, {**SAMPLE_RESULT, "query_id": 1})
        cache.set("Question B", AGENTS, {**SAMPLE_RESULT, "query_id": 2})
        assert cache.count() == 2

    def test_clear_resets_cache(self, cache):
        cache.set(SAMPLE_RESULT["question"], AGENTS, SAMPLE_RESULT)
        cache.clear()
        assert cache.count() == 0
        assert cache.get(SAMPLE_RESULT["question"], AGENTS) is None

    def test_upsert_same_query_id_does_not_duplicate(self, cache):
        cache.set("Question A", AGENTS, {**SAMPLE_RESULT, "query_id": 1})
        cache.set("Question A", AGENTS, {**SAMPLE_RESULT, "query_id": 1, "system_main_driver": "Updated"})
        assert cache.count() == 1
        assert cache.get("Question A", AGENTS)["system_main_driver"] == "Updated"
