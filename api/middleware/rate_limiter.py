"""
api/middleware/rate_limiter.py
=================================
Checkpoint 19 — Per-API-Key Rate Limiting

Fixed 60-second window counter in Redis, keyed by the caller's API key
hash (so different keys never share a bucket). Chains off verify_api_key
as a FastAPI dependency — a request must be authenticated before it can
be rate-limited, and the limiter needs the caller's identity anyway.

Usage:
    from api.middleware.rate_limiter import enforce_rate_limit

    @router.post("/query", dependencies=[Depends(enforce_rate_limit)])
    ...
"""

from __future__ import annotations

from fastapi import Depends, HTTPException
from redis import Redis

from api.middleware.auth import verify_api_key
from config.settings import settings
from db.models import ApiKeyRow

WINDOW_SECONDS = 60


async def enforce_rate_limit(api_key: ApiKeyRow = Depends(verify_api_key)) -> None:
    """
    FastAPI dependency: increments this API key's request count for the
    current 60s window and raises 429 once it exceeds
    settings.rate_limit_per_minute.

    Fails open (does not block the request) if Redis itself is
    unreachable — a rate limiter outage shouldn't take the whole API
    down; auth (verify_api_key, which already ran) is the real gate.
    """
    try:
        redis_conn = Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        bucket_key = f"ratelimit:{api_key.key_hash}"
        count = redis_conn.incr(bucket_key)
        if count == 1:
            redis_conn.expire(bucket_key, WINDOW_SECONDS)
    except Exception:
        return  # fail open — see docstring

    if count > settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {settings.rate_limit_per_minute} requests per {WINDOW_SECONDS}s",
        )
