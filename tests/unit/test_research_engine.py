"""
tests/unit/test_research_engine.py
====================================
Checkpoint 15a -- Unit Tests for the Firecrawl Research Engine

No live network or LLM calls are made. `_scrape_url` / `_search_web` and
`LLMRouter.call_model_direct` are mocked at the seam; only ResearchEngine's
own orchestration logic (parsing, tiering, caching, rate limiting, guards)
is under test.

Run:
    poetry run pytest tests/unit/test_research_engine.py -v
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import settings
from core.research_engine import RateLimiter, ResearchEngine
from llm.router import LLMResponse


# =============================================================================
# Helpers
# =============================================================================

def make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(agent_name="direct:test", model="deepseek-ai/DeepSeek-V3", content=content)


EXTRACTOR_OK = json.dumps({"extracted_facts": ["BTC volume up 3x", "Derivatives liquidated"]})
TRUTH_FILTER_NO_CONFLICT = json.dumps({
    "verified_claims": ["BTC volume up 3x", "Derivatives liquidated"],
    "contradictions_found": [],
    "conflict_score": 0.0,
})
TRUTH_FILTER_CONFLICT = json.dumps({
    "verified_claims": ["Price rose 5%"],
    "contradictions_found": ["Source A says price rose 5%; Source B says price fell 5% in the same window"],
    "conflict_score": 0.8,
})


@pytest.fixture
def engine(monkeypatch):
    """ResearchEngine with a fake API key so the mocked dev fallback doesn't short-circuit."""
    monkeypatch.setattr(settings, "firecrawl_api_key", "fc-fake-key-for-tests")
    monkeypatch.setattr(settings, "env", "development")
    return ResearchEngine(redis_conn=None)


# =============================================================================
# run_research — happy path, code-fence stripping, conflict detection, caching
# =============================================================================

class TestRunResearch:

    async def test_happy_path_extracts_and_tiers_facts(self, engine, monkeypatch):
        engine._scrape_url = AsyncMock(return_value={
            "data": {"markdown": "BTC volume surged.", "metadata": {"sourceURL": "https://example.com"}}
        })
        monkeypatch.setattr(
            engine.router, "call_model_direct",
            MagicMock(side_effect=[
                make_llm_response(EXTRACTOR_OK),
                make_llm_response(TRUTH_FILTER_NO_CONFLICT),
            ]),
        )

        result = await engine.run_research("https://example.com")

        assert result["extracted_facts"] == ["BTC volume up 3x", "Derivatives liquidated"]
        assert len(result["confidence_tiers"]) == 2
        assert all(t["tier"] in {"verified", "inferred", "uncertain"} for t in result["confidence_tiers"])
        assert result["conflict_score"] == 0.0
        assert result["contradictions"] == []
        assert result["cache_status"] == "miss"

    async def test_strips_markdown_code_fences_from_llm_json(self, engine, monkeypatch):
        fenced = "```json\n" + EXTRACTOR_OK + "\n```"
        engine._scrape_url = AsyncMock(return_value={
            "data": {"markdown": "content", "metadata": {}}
        })
        monkeypatch.setattr(
            engine.router, "call_model_direct",
            MagicMock(side_effect=[
                make_llm_response(fenced),
                make_llm_response(TRUTH_FILTER_NO_CONFLICT),
            ]),
        )

        result = await engine.run_research("https://example.com")

        assert result["extracted_facts"] == ["BTC volume up 3x", "Derivatives liquidated"]

    async def test_contradictory_facts_produce_positive_conflict_score(self, engine, monkeypatch):
        engine._scrape_url = AsyncMock(return_value={
            "data": {"markdown": "Source A: price rose 5%. Source B: price fell 5%.", "metadata": {}}
        })
        monkeypatch.setattr(
            engine.router, "call_model_direct",
            MagicMock(side_effect=[
                make_llm_response(json.dumps({"extracted_facts": ["Price rose 5%", "Price fell 5%"]})),
                make_llm_response(TRUTH_FILTER_CONFLICT),
            ]),
        )

        result = await engine.run_research("https://example.com")

        assert result["conflict_score"] > 0.0
        assert len(result["contradictions"]) == 1

    async def test_cache_hit_skips_scrape_entirely(self, engine):
        cached_payload = {
            "extracted_facts": ["Cached fact"],
            "confidence_tiers": [{"fact": "Cached fact", "tier": "verified"}],
            "contradictions": [],
            "conflict_score": 0.0,
            "cache_status": "miss",
            "crawl_metadata": {},
        }
        fake_redis = MagicMock()
        fake_redis.get.return_value = json.dumps(cached_payload)
        engine.redis_conn = fake_redis
        engine._scrape_url = AsyncMock(side_effect=AssertionError("should not scrape on cache hit"))

        result = await engine.run_research("https://example.com")

        assert result["cache_status"] == "hit"
        assert result["extracted_facts"] == ["Cached fact"]
        engine._scrape_url.assert_not_called()

    async def test_cache_write_on_miss(self, engine, monkeypatch):
        engine._scrape_url = AsyncMock(return_value={"data": {"markdown": "content", "metadata": {}}})
        monkeypatch.setattr(
            engine.router, "call_model_direct",
            MagicMock(side_effect=[
                make_llm_response(EXTRACTOR_OK),
                make_llm_response(TRUTH_FILTER_NO_CONFLICT),
            ]),
        )
        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        engine.redis_conn = fake_redis

        await engine.run_research("https://example.com")

        assert fake_redis.setex.called
        args, _ = fake_redis.setex.call_args
        assert args[0] == "research:https://example.com"
        assert args[1] == 86400


# =============================================================================
# Dev fallback + production guard
# =============================================================================

class TestFirecrawlKeyHandling:

    async def test_dev_fallback_used_when_no_key(self, monkeypatch):
        monkeypatch.setattr(settings, "firecrawl_api_key", "")
        monkeypatch.setattr(settings, "env", "development")

        eng = ResearchEngine(redis_conn=None)
        result = await eng._scrape_url("https://example.com")

        assert result["data"]["metadata"]["mocked"] is True

    def test_production_without_key_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "firecrawl_api_key", "")
        monkeypatch.setattr(settings, "env", "production")

        with pytest.raises(RuntimeError, match="FIRECRAWL_API_KEY"):
            ResearchEngine(redis_conn=None)

    def test_production_with_key_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(settings, "firecrawl_api_key", "fc-real-key")
        monkeypatch.setattr(settings, "env", "production")

        ResearchEngine(redis_conn=None)  # should not raise


# =============================================================================
# RateLimiter
# =============================================================================

class TestRateLimiter:

    async def test_allows_calls_up_to_limit_without_delay(self):
        limiter = RateLimiter(max_calls=3, period_seconds=10.0)
        start = time.monotonic()
        for _ in range(3):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.5

    async def test_throttles_once_limit_exceeded(self):
        limiter = RateLimiter(max_calls=2, period_seconds=0.3)
        start = time.monotonic()
        for _ in range(3):  # 3rd call must wait for the window to free up
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.25
