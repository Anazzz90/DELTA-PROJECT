"""
db/session.py
===============
Checkpoint 9 — Async Database Session Management

Provides a single async SQLAlchemy engine and session factory for the
entire DMARS application. Uses aiosqlite as the async driver.

Key design decisions:
  - Single engine instance (module-level singleton) — safe for async usage
  - AsyncSession used everywhere to avoid blocking the pipeline
  - create_all_tables() is idempotent — safe to call on every app start
  - DATABASE_URL falls back to local dmars.db if not set in .env

Usage:
    from db.session import get_session, create_all_tables

    await create_all_tables()               # once at startup

    async with get_session() as session:    # in route handlers / tasks
        session.add(my_row)
        await session.commit()
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# psycopg's async mode requires a SelectorEventLoop; Windows defaults to
# ProactorEventLoop, which raises psycopg.InterfaceError on connect. This
# must run before any event loop is created (i.e. at import time, here,
# since this module is imported before the app/worker starts its loop).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config.settings import settings
from db.models import Base

logger = logging.getLogger(__name__)

# Default to a local SQLite file alongside the project's pyproject.toml
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "dmars.db"
_DEFAULT_DB_URL  = f"sqlite+aiosqlite:///{_DEFAULT_DB_PATH}"

DATABASE_URL: str = settings.database_url or _DEFAULT_DB_URL

# Create the async engine — module-level singleton
engine = create_async_engine(
    DATABASE_URL,
    echo=False,          # Set True to see raw SQL in logs (debug only)
    future=True,
)

# Session factory — reuse this everywhere
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def create_all_tables() -> None:
    """
    Create all tables defined in db/models.py if they don't exist.
    Idempotent — safe to call on every application startup.

    PostgreSQL (Checkpoint 15+): schema is owned by Alembic
    (db/migrations/) — this is a no-op there so app startup can never
    silently create tables Alembic doesn't know about, masking a missing
    migration. Only runs create_all() for SQLite (Phase 1 dev convenience,
    which has no migration tooling of its own).
    """
    if engine.dialect.name != "sqlite":
        logger.info(
            f"Skipping create_all() for {engine.dialect.name} — schema is "
            f"managed by Alembic (run: alembic upgrade head)."
        )
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"Database tables created/verified at: {DATABASE_URL}")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager providing a database session.

    Usage:
        async with get_session() as session:
            session.add(row)
            await session.commit()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
