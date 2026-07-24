# Checkpoint 14 — Redis + RQ Task Queue
# Updated /query route: enqueues the pipeline as a background job and returns
# a job_id immediately. A new GET /query/{job_id}/status route polls for results.
#
# BACKWARD COMPATIBILITY:
#   POST /query still works exactly the same for clients that want synchronous
#   behaviour — they can poll the status URL until status == "finished".
#
# ASYNC FLOW:
#   POST /query  →  {"job_id": "...", "status": "queued", "poll_url": "/query/.../status"}
#   GET  /query/{job_id}/status  →  {"status": "started" | "finished" | "failed", ...}

import os
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from redis import Redis, ConnectionError as RedisConnectionError
from rq import Queue
from rq.job import Job, NoSuchJobError

from api.middleware.auth import verify_api_key
from api.middleware.rate_limiter import enforce_rate_limit
from api.schemas.query_schema import QueryRequest
from api.schemas.response_schema import (
    AgentOutputResponse,
    EnqueuedResponse,
    JobStatusResponse,
    MetaAIResponse,
    QueryResponse,
)
from config.settings import settings
from task_queue.tasks import AGENT_MAPPING, run_pipeline_task

router = APIRouter(tags=["Query"])

REDIS_URL = settings.redis_url
QUEUE_NAME = "dmars-queue"
JOB_TIMEOUT = 600  # 10 minutes — long enough for the full 5-agent pipeline


def _get_redis() -> Redis:
    """Return a Redis connection, raising HTTP 503 if Redis is unavailable."""
    try:
        conn = Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        conn.ping()
        return conn
    except (RedisConnectionError, Exception) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Redis is unavailable ({exc}). The task queue cannot accept jobs.",
        )


# ── POST /query — enqueue the pipeline job ────────────────────────────────────

@router.post(
    "/query",
    response_model=EnqueuedResponse,
    status_code=202,
    dependencies=[Depends(enforce_rate_limit)],
)
async def analyze_query(request: QueryRequest) -> EnqueuedResponse:
    """
    Enqueue a DMARS reasoning pipeline job.
    Returns immediately with a job_id. Poll GET /query/{job_id}/status for results.
    """
    # Basic validation (mirrors Checkpoint 13 behaviour)
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")
    if not request.fact_set:
        raise HTTPException(status_code=422, detail="At least one fact must be provided")
    if not request.selected_agents:
        raise HTTPException(status_code=422, detail="At least one agent must be selected")
    unknown_agents = [a for a in request.selected_agents if a not in AGENT_MAPPING]
    if unknown_agents:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown agent(s): {unknown_agents}. Valid agents: {list(AGENT_MAPPING.keys())}",
        )

    redis_conn = _get_redis()
    queue = Queue(QUEUE_NAME, connection=redis_conn)

    # On Windows, SIGALRM is unavailable — disable job timeout.
    # On Linux/macOS, enforce the 10-minute cap.
    timeout = -1 if not hasattr(os, "fork") else JOB_TIMEOUT

    job = queue.enqueue(
        run_pipeline_task,
        kwargs={
            "question": request.question,
            "fact_set": request.fact_set,
            "domain_profile": request.domain_profile,
            "selected_agents": request.selected_agents,
            "meta_ai_enabled": request.meta_ai_enabled,
        },
        job_timeout=timeout,
    )

    return EnqueuedResponse(
        job_id=job.id,
        status="queued",
        poll_url=f"/query/{job.id}/status",
    )


# ── GET /query/{job_id}/status — poll for result ──────────────────────────────

@router.get(
    "/query/{job_id}/status",
    response_model=JobStatusResponse,
    dependencies=[Depends(verify_api_key)],
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Poll the status of an enqueued pipeline job.
    Returns one of: queued | started | finished | failed | not_found
    When status == "finished", the full result is included in the response.
    """
    redis_conn = _get_redis()

    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        return JobStatusResponse(job_id=job_id, status="not_found")

    rq_status = job.get_status()

    # Map RQ statuses to our API statuses
    status_map = {
        "queued": "queued",
        "started": "started",
        "finished": "finished",
        "failed": "failed",
        "stopped": "failed",
        "canceled": "failed",
        "deferred": "queued",
        "scheduled": "queued",
    }
    api_status = status_map.get(str(rq_status), str(rq_status))

    # ── Job finished successfully ─────────────────────────────────────────────
    if api_status == "finished" and job.result is not None:
        raw = job.result  # dict returned by run_pipeline_task

        # Re-hydrate nested Pydantic models from plain dicts
        agent_results = [AgentOutputResponse(**a) for a in raw["agent_results"]]
        meta_ai_result = (
            MetaAIResponse(**raw["meta_ai_result"])
            if raw.get("meta_ai_result")
            else None
        )

        full_result = QueryResponse(
            query_id=raw["query_id"],
            question=raw["question"],
            system_main_driver=raw["system_main_driver"],
            system_confidence_score=raw["system_confidence_score"],
            net_bias=raw["net_bias"],
            signal_summary=raw["signal_summary"],
            decision_logic=raw["decision_logic"],
            conflict_detected=raw["conflict_detected"],
            conflict_level=raw["conflict_level"],
            conflict_type=raw["conflict_type"],
            dominant_narratives=raw["dominant_narratives"],
            agent_results=agent_results,
            failed_agents=raw.get("failed_agents", []),
            meta_ai_result=meta_ai_result,
            total_cost_usd=raw["total_cost_usd"],
            pipeline_latency_ms=raw["pipeline_latency_ms"],
            cache_hit=raw.get("cache_hit", False),
        )
        return JobStatusResponse(job_id=job_id, status="finished", result=full_result)

    # ── Job failed ────────────────────────────────────────────────────────────
    if api_status == "failed":
        error_msg = None
        if job.exc_info:
            # exc_info is the full traceback string
            error_msg = job.exc_info.strip().split("\n")[-1]
        return JobStatusResponse(job_id=job_id, status="failed", error=error_msg)

    # ── Still in progress ─────────────────────────────────────────────────────
    return JobStatusResponse(job_id=job_id, status=api_status)
