"""
api/middleware/auth.py
========================
Checkpoint 19 — API Key Authentication

Validates the X-API-Key header against hashed keys stored in PostgreSQL
(db.models.ApiKeyRow). Only the SHA-256 hash of each key is ever stored —
the plaintext key is shown to the caller exactly once, at creation time
(POST /auth/create-key), and never persisted or logged.

SHA-256 (not bcrypt/argon2) is deliberate: these are high-entropy random
tokens (32 bytes from secrets.token_urlsafe), not user-chosen passwords —
there's no dictionary/brute-force risk a slow hash would defend against,
and a fast deterministic hash lets lookup happen as a normal indexed
query rather than needing to fetch-then-compare every row.

Usage (as a FastAPI dependency):
    from api.middleware.auth import verify_api_key

    @router.post("/query", dependencies=[Depends(verify_api_key)])
    ...
"""

from __future__ import annotations

import datetime
import hashlib
import secrets

from fastapi import Header, HTTPException
from sqlalchemy import select, update

from config.settings import settings
from db.models import ApiKeyRow
from db.session import get_session


def hash_key(plaintext_key: str) -> str:
    """SHA-256 hex digest of an API key — this is what's stored, never the plaintext."""
    return hashlib.sha256(plaintext_key.encode("utf-8")).hexdigest()


def generate_key() -> str:
    """A new random plaintext API key, e.g. 'dmars_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'."""
    return f"dmars_{secrets.token_urlsafe(32)}"


async def verify_api_key(x_api_key: str = Header(default=None, alias="X-API-Key")) -> ApiKeyRow:
    """
    FastAPI dependency: validates X-API-Key against PostgreSQL.

    Raises:
        HTTPException(401): missing header, unknown key, or a deactivated key.

    Returns:
        The matching ApiKeyRow (available to downstream dependencies, e.g.
        the rate limiter, to key its bucket off the same identity).
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    key_hash = hash_key(x_api_key)

    async with get_session() as session:
        result = await session.execute(
            select(ApiKeyRow).where(
                ApiKeyRow.key_hash == key_hash,
                ApiKeyRow.is_active.is_(True),
            )
        )
        api_key = result.scalar_one_or_none()

        if api_key is None:
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")

        await session.execute(
            update(ApiKeyRow)
            .where(ApiKeyRow.id == api_key.id)
            .values(last_used_at=datetime.datetime.utcnow())
        )
        await session.commit()

    return api_key


async def verify_admin_secret(x_admin_secret: str = Header(default=None, alias="X-Admin-Secret")) -> None:
    """
    FastAPI dependency gating POST /auth/create-key. Separate from regular
    API keys on purpose — this is what lets someone MINT API keys, so it
    can't itself be a regular API key (that would be self-bootstrapping
    from nothing, i.e. no real gate at all).
    """
    if not settings.admin_secret:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_SECRET is not configured — key creation is disabled.",
        )
    if not x_admin_secret or not secrets.compare_digest(x_admin_secret, settings.admin_secret):
        raise HTTPException(status_code=401, detail="Invalid admin secret")
