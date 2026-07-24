"""
run_server.py
==============
Windows-safe entry point for running the DMARS FastAPI server.

Why this exists (Checkpoint 15 -- PostgreSQL migration):
    psycopg's async driver requires a SelectorEventLoop. Windows defaults
    to ProactorEventLoop, which raises psycopg.InterfaceError on connect.

    Setting the event loop policy inside api/main.py (or anything it
    imports, e.g. db/session.py) is too late when launching via the plain
    `uvicorn api.main:app` CLI: uvicorn calls asyncio.run() -- which
    creates the loop -- *before* it imports the app string. By the time
    any app code runs, the (wrong) loop already exists and can't be
    swapped out. The policy must be set before uvicorn creates its loop,
    so it has to happen in whatever process starts uvicorn -- here.

    This only matters for PostgreSQL (via psycopg). SQLite (aiosqlite)
    and Linux/macOS deployments (Docker containers included -- Checkpoint
    26) are unaffected; `uvicorn api.main:app --reload` still works fine
    there.

Usage:
    poetry run python run_server.py
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
