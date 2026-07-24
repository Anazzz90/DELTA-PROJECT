"""
observability/tracer.py
=========================
Checkpoint 17 — LangFuse Observability

Logs every LLM call (all 5 agents + Meta-AI + the research engine's direct
calls) to LangFuse via the SDK's `start_observation(as_type="generation")`
API, called explicitly from llm/router.py after each completion — not via
LiteLLM's built-in "langfuse" callback string.

Why not LiteLLM's built-in integration: this project is pinned to
litellm==1.52.0 (see pyproject.toml — newer litellm breaks on Python 3.14).
litellm 1.52.0's langfuse callback was written against the langfuse 2.x SDK.
langfuse 2.x's generated API client uses pydantic's `v1` compat shim, which
crashes at *import time* on Python 3.14 (a pydantic/Python 3.14
incompatibility, not a DMARS bug) — and that import is triggered lazily
inside litellm's callback dispatch, so it was intermittently crashing real
agent calls rather than degrading gracefully. langfuse 4.x fixes the import
crash, but litellm 1.52.0's callback code then breaks on a different
incompatibility (`module 'langfuse' has no attribute 'version'` — 4.x
restructured the package). Both combinations are unusable together, so
tracing is done manually here instead, isolated behind try/except so a
LangFuse-side problem can never break an actual agent call.

Non-blocking by design (Checkpoint 17 criterion #5): configure_langfuse()
does an eager trial import + client construction; if that fails for *any*
reason, tracing is disabled for the rest of the process and nothing else
here runs. log_generation() independently wraps its own body in try/except
so even a transient LangFuse API problem mid-session can't break a call.

Usage:
    from observability.tracer import configure_langfuse, new_trace_id, log_generation, check_cost_alerts

    configure_langfuse()                     # once per process (LLMRouter.__init__ does this)
    trace_id = new_trace_id()                # once per pipeline run, passed to each agent
    log_generation(trace_id, agent_name, model, messages, content, ...)  # after each call
    check_cost_alerts(pipeline_result.total_cost_usd())                  # once per completed query
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_configured = False
_enabled = False
_client = None
_configure_lock = threading.Lock()


def configure_langfuse() -> bool:
    """
    Try to construct a LangFuse client. Idempotent and safe to call from
    every process that constructs an LLMRouter (API server, RQ worker,
    tests) — only actually attempts configuration once.

    Returns:
        True if tracing is active, False if disabled (no keys configured,
        or the client failed to initialize for any reason).
    """
    global _configured, _enabled, _client
    with _configure_lock:
        if _configured:
            return _enabled

        _configured = True

        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            logger.info(
                "LangFuse tracing disabled — LANGFUSE_PUBLIC_KEY/SECRET_KEY not set. "
                "System continues normally without tracing."
            )
            return False

        try:
            import langfuse

            _client = langfuse.Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            _enabled = True
            logger.info(f"LangFuse tracing enabled — host={settings.langfuse_host}")
            return True
        except Exception as e:
            # Never let a tracing setup problem break the app.
            logger.warning(f"Failed to configure LangFuse tracing (continuing without it): {e}")
            _client = None
            _enabled = False
            return False


def is_enabled() -> bool:
    """Whether LangFuse tracing is currently active."""
    return _enabled


def new_trace_id() -> str:
    """A fresh id to group one pipeline run's agent calls into a single LangFuse trace."""
    return uuid.uuid4().hex


def log_generation(
    trace_id: Optional[str],
    agent_name: str,
    model: str,
    messages: list[dict],
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: float = 0.0,
    error: Optional[str] = None,
) -> None:
    """
    Log one completed (or failed) LLM call as a LangFuse generation.

    Safe to call unconditionally — a no-op if tracing isn't enabled, and
    any LangFuse-side error is caught and logged rather than raised, so
    this can never break the agent call it's tracing.
    """
    if not _enabled or _client is None:
        return

    try:
        import langfuse.types

        trace_context = (
            langfuse.types.TraceContext(trace_id=trace_id) if trace_id else None
        )
        generation = _client.start_observation(
            trace_context=trace_context,
            name=agent_name,
            as_type="generation",
            input=messages,
            output=content if not error else None,
            model=model,
            usage_details={
                "input": prompt_tokens,
                "output": completion_tokens,
                "total": total_tokens,
            },
            cost_details={"total": cost_usd},
            metadata={"latency_ms": latency_ms},
            level="ERROR" if error else "DEFAULT",
            status_message=error,
        )
        generation.end()
    except Exception as e:
        logger.warning(f"LangFuse logging failed for {agent_name} (continuing without it): {e}")


def flush() -> None:
    """Force-send any buffered events. Mainly useful for short-lived scripts/tests."""
    if _client is not None:
        try:
            _client.flush()
        except Exception as e:
            logger.warning(f"LangFuse flush failed: {e}")


# =============================================================================
# Cost alerting
# =============================================================================
# In-memory, process-lifetime running total. Resets on restart — this is a
# dev-facing alert, not a billing system of record (that's total_cost_usd
# persisted per query in PostgreSQL via db/models.py).

_session_total_cost_usd = 0.0
_cost_lock = threading.Lock()


def check_cost_alerts(query_cost_usd: float) -> None:
    """
    Call once per completed query. Logs a warning if the query itself or
    the running session total exceeds the configured thresholds.
    """
    global _session_total_cost_usd

    if query_cost_usd > settings.per_query_cost_alert_usd:
        logger.warning(
            f"Cost alert: query cost ${query_cost_usd:.6f} exceeded "
            f"per-query threshold ${settings.per_query_cost_alert_usd:.2f}"
        )

    with _cost_lock:
        _session_total_cost_usd += query_cost_usd
        total = _session_total_cost_usd

    if total > settings.daily_cost_alert_usd:
        logger.warning(
            f"Cost alert: session spend ${total:.6f} exceeded "
            f"daily threshold ${settings.daily_cost_alert_usd:.2f}"
        )


def session_total_cost_usd() -> float:
    """Current process-lifetime running total, for inspection/tests."""
    return _session_total_cost_usd


def reset_session_cost() -> None:
    """Test-only helper — reset the running total between test cases."""
    global _session_total_cost_usd
    with _cost_lock:
        _session_total_cost_usd = 0.0


def reset_configuration() -> None:
    """Test-only helper — allow configure_langfuse() to run again in a fresh test."""
    global _configured, _enabled, _client
    with _configure_lock:
        _configured = False
        _enabled = False
        _client = None
