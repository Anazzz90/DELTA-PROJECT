"""
tests/unit/test_auth.py
==========================
Checkpoint 19 -- API Key Auth + Admin Secret

Uses an in-memory SQLite database (same pattern as tests/unit/test_storage.py)
so no real Postgres connection is needed.

Run:
    poetry run pytest tests/unit/test_auth.py -v
"""

import pytest
from fastapi import HTTPException

from api.middleware.auth import generate_key, hash_key, verify_admin_secret, verify_api_key
from config.settings import settings
from db.models import ApiKeyRow, Base


@pytest.fixture
async def patched_db():
    """In-memory SQLite standing in for Postgres, same pattern as test_storage.py."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from db.session import AsyncSession

    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from db import session as session_module
    original_engine = session_module.engine
    original_session_local = session_module.AsyncSessionLocal

    session_module.engine = test_engine
    session_module.AsyncSessionLocal = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )

    yield

    session_module.engine = original_engine
    session_module.AsyncSessionLocal = original_session_local
    await test_engine.dispose()


async def _insert_key(plaintext: str, name: str = "test-key", is_active: bool = True) -> None:
    from db.session import get_session

    async with get_session() as session:
        session.add(ApiKeyRow(
            key_hash=hash_key(plaintext),
            key_prefix=plaintext[:14],
            name=name,
            is_active=is_active,
        ))
        await session.commit()


class TestKeyGeneration:

    def test_generate_key_has_dmars_prefix(self):
        assert generate_key().startswith("dmars_")

    def test_generate_key_is_unique(self):
        keys = {generate_key() for _ in range(50)}
        assert len(keys) == 50

    def test_hash_key_is_deterministic(self):
        key = "dmars_abc123"
        assert hash_key(key) == hash_key(key)

    def test_hash_key_is_64_char_hex(self):
        digest = hash_key("dmars_abc123")
        assert len(digest) == 64
        int(digest, 16)  # raises ValueError if not valid hex

    def test_different_keys_hash_differently(self):
        assert hash_key("dmars_aaa") != hash_key("dmars_bbb")


class TestVerifyApiKey:

    async def test_missing_header_raises_401(self, patched_db):
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(x_api_key=None)
        assert exc.value.status_code == 401

    async def test_unknown_key_raises_401(self, patched_db):
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(x_api_key="dmars_not-a-real-key")
        assert exc.value.status_code == 401

    async def test_valid_active_key_returns_row(self, patched_db):
        plaintext = generate_key()
        await _insert_key(plaintext, name="my-key")

        row = await verify_api_key(x_api_key=plaintext)

        assert row.name == "my-key"
        assert row.key_hash == hash_key(plaintext)

    async def test_inactive_key_raises_401(self, patched_db):
        plaintext = generate_key()
        await _insert_key(plaintext, name="revoked-key", is_active=False)

        with pytest.raises(HTTPException) as exc:
            await verify_api_key(x_api_key=plaintext)
        assert exc.value.status_code == 401

    async def test_valid_key_updates_last_used_at(self, patched_db):
        plaintext = generate_key()
        await _insert_key(plaintext)

        row = await verify_api_key(x_api_key=plaintext)
        assert row is not None
        # last_used_at is set by a separate UPDATE inside verify_api_key;
        # re-fetch to confirm it actually persisted.
        from sqlalchemy import select
        from db.session import get_session
        async with get_session() as session:
            result = await session.execute(select(ApiKeyRow).where(ApiKeyRow.key_hash == row.key_hash))
            refreshed = result.scalar_one()
        assert refreshed.last_used_at is not None


class TestVerifyAdminSecret:

    async def test_disabled_when_no_admin_secret_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "admin_secret", "")
        with pytest.raises(HTTPException) as exc:
            await verify_admin_secret(x_admin_secret="anything")
        assert exc.value.status_code == 503

    async def test_wrong_secret_raises_401(self, monkeypatch):
        monkeypatch.setattr(settings, "admin_secret", "correct-secret")
        with pytest.raises(HTTPException) as exc:
            await verify_admin_secret(x_admin_secret="wrong-secret")
        assert exc.value.status_code == 401

    async def test_missing_secret_raises_401(self, monkeypatch):
        monkeypatch.setattr(settings, "admin_secret", "correct-secret")
        with pytest.raises(HTTPException) as exc:
            await verify_admin_secret(x_admin_secret=None)
        assert exc.value.status_code == 401

    async def test_correct_secret_passes(self, monkeypatch):
        monkeypatch.setattr(settings, "admin_secret", "correct-secret")
        await verify_admin_secret(x_admin_secret="correct-secret")  # must not raise
