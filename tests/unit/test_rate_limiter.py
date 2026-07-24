"""
tests/unit/test_rate_limiter.py
==================================
Checkpoint 19 -- Per-API-Key Rate Limiting

Redis is mocked (unittest.mock) — no real Redis connection is used.

Run:
    poetry run pytest tests/unit/test_rate_limiter.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from api.middleware.rate_limiter import enforce_rate_limit
from config.settings import settings
from db.models import ApiKeyRow


def make_api_key(key_hash: str = "abc123") -> ApiKeyRow:
    return ApiKeyRow(id=1, key_hash=key_hash, key_prefix="dmars_abc", name="test", is_active=True)


class TestRateLimiter:

    async def test_allows_requests_under_the_limit(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 10)
        fake_redis = MagicMock()
        fake_redis.incr.return_value = 5

        with patch("api.middleware.rate_limiter.Redis.from_url", return_value=fake_redis):
            await enforce_rate_limit(api_key=make_api_key())  # must not raise

    async def test_blocks_once_over_the_limit(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 10)
        fake_redis = MagicMock()
        fake_redis.incr.return_value = 11

        with patch("api.middleware.rate_limiter.Redis.from_url", return_value=fake_redis):
            with pytest.raises(HTTPException) as exc:
                await enforce_rate_limit(api_key=make_api_key())
        assert exc.value.status_code == 429

    async def test_sets_expiry_only_on_first_request_in_window(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 10)
        fake_redis = MagicMock()
        fake_redis.incr.return_value = 1

        with patch("api.middleware.rate_limiter.Redis.from_url", return_value=fake_redis):
            await enforce_rate_limit(api_key=make_api_key())

        fake_redis.expire.assert_called_once_with("ratelimit:abc123", 60)

    async def test_different_keys_use_separate_buckets(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 10)
        fake_redis = MagicMock()
        fake_redis.incr.return_value = 5

        with patch("api.middleware.rate_limiter.Redis.from_url", return_value=fake_redis):
            await enforce_rate_limit(api_key=make_api_key(key_hash="key-a"))
            await enforce_rate_limit(api_key=make_api_key(key_hash="key-b"))

        called_buckets = {c.args[0] for c in fake_redis.incr.call_args_list}
        assert called_buckets == {"ratelimit:key-a", "ratelimit:key-b"}

    async def test_fails_open_when_redis_unavailable(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 10)

        with patch("api.middleware.rate_limiter.Redis.from_url", side_effect=ConnectionError("down")):
            await enforce_rate_limit(api_key=make_api_key())  # must not raise
