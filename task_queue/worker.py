import os
import sys

# Ensure the dmars package root is on the Python path when the worker
# is launched directly (i.e., not via `poetry run`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis import Redis
from rq import Queue
from rq.timeouts import BaseDeathPenalty

# ── Windows compatibility ─────────────────────────────────────────────────────
# On Windows, os.fork() and signal.SIGALRM do not exist.
# We use SimpleWorker (no child process) and NopDeathPenalty (no SIGALRM).
# On Linux/macOS the standard Worker + default death penalty are used.

IS_WINDOWS = not hasattr(os, "fork")

if IS_WINDOWS:
    from rq.worker import SimpleWorker as Worker  # type: ignore[assignment]

    class NopDeathPenalty(BaseDeathPenalty):
        """A no-op death penalty for platforms without SIGALRM (Windows)."""

        def setup_death_penalty(self) -> None:
            pass  # nothing to do on Windows

        def cancel_death_penalty(self) -> None:
            pass  # nothing to do on Windows

    # Explicitly disable the death penalty (which uses SIGALRM) on Windows
    Worker.death_penalty_class = NopDeathPenalty
else:
    from rq import Worker  # type: ignore[assignment]
    NopDeathPenalty = None  # type: ignore[assignment,misc]

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QUEUE_NAME = "dmars-queue"


def start_worker() -> None:
    """Start an RQ worker listening on the DMARS queue."""
    redis_conn = Redis.from_url(REDIS_URL)
    queue = Queue(QUEUE_NAME, connection=redis_conn)

    print(f"[DMARS Worker] Connecting to Redis at {REDIS_URL}")
    print(f"[DMARS Worker] Listening on queue: {QUEUE_NAME}")
    print(f"[DMARS Worker] Mode: {'simple/no-fork (Windows)' if IS_WINDOWS else 'fork (Unix)'}")

    worker = Worker(queues=[queue], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    start_worker()
