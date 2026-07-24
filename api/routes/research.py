import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from redis import Redis, ConnectionError as RedisConnectionError
from rq import Queue
from rq.job import Job, NoSuchJobError

from config.settings import settings
from task_queue.tasks import run_research_task
from api.schemas.response_schema import EnqueuedResponse

router = APIRouter(tags=["Research"])

REDIS_URL = settings.redis_url
QUEUE_NAME = "dmars-queue"
JOB_TIMEOUT = 300  # 5 minutes

class ResearchRequest(BaseModel):
    url: Optional[str] = Field(None, description="The URL to scrape and research")
    topic: Optional[str] = Field(None, description="The topic to search for and research")
    domain_profile: Optional[str] = Field("general", description="Domain context")
    auto_reason: bool = Field(False, description="Whether to automatically chain into the reasoning pipeline")

class ConfidenceTierItem(BaseModel):
    fact: str
    tier: str

class ResearchResponse(BaseModel):
    extracted_facts: List[str]
    confidence_tiers: List[ConfidenceTierItem]
    contradictions: List[str]
    conflict_score: float
    cache_status: str
    crawl_metadata: Dict[str, Any]
    research_latency_ms: Optional[float] = None
    chained_pipeline_job_id: Optional[str] = None

class ResearchJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[ResearchResponse] = None
    error: Optional[str] = None

def _get_redis() -> Redis:
    try:
        conn = Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        conn.ping()
        return conn
    except (RedisConnectionError, Exception) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Redis is unavailable ({exc}).",
        )

@router.post("/research", response_model=EnqueuedResponse, status_code=202)
async def start_research(request: ResearchRequest) -> EnqueuedResponse:
    if not request.url and not request.topic:
        raise HTTPException(status_code=422, detail="Either URL or Topic must be provided")

    if request.url and not request.url.startswith("http"):
        raise HTTPException(status_code=422, detail="URL must start with http")

    redis_conn = _get_redis()
    queue = Queue(QUEUE_NAME, connection=redis_conn)
    timeout = -1 if not hasattr(os, "fork") else JOB_TIMEOUT

    job = queue.enqueue(
        run_research_task,
        kwargs={
            "url": request.url,
            "topic": request.topic,
            "domain_profile": request.domain_profile,
            "auto_reason": request.auto_reason,
            "redis_url": REDIS_URL,
        },
        job_timeout=timeout,
    )

    return EnqueuedResponse(
        job_id=job.id,
        status="queued",
        poll_url=f"/research/{job.id}/status",
    )

@router.get("/research/{job_id}/status", response_model=ResearchJobStatusResponse)
async def get_research_status(job_id: str) -> ResearchJobStatusResponse:
    redis_conn = _get_redis()

    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        return ResearchJobStatusResponse(job_id=job_id, status="not_found")

    rq_status = job.get_status()
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

    if api_status == "finished" and job.result is not None:
        raw = job.result
        full_result = ResearchResponse(
            extracted_facts=raw.get("extracted_facts", []),
            confidence_tiers=[ConfidenceTierItem(**t) for t in raw.get("confidence_tiers", [])],
            contradictions=raw.get("contradictions", []),
            conflict_score=raw.get("conflict_score", 0.0),
            cache_status=raw.get("cache_status", "miss"),
            crawl_metadata=raw.get("crawl_metadata", {}),
            research_latency_ms=raw.get("research_latency_ms"),
            chained_pipeline_job_id=raw.get("chained_pipeline_job_id"),
        )
        return ResearchJobStatusResponse(job_id=job_id, status="finished", result=full_result)

    if api_status == "failed":
        error_msg = None
        if job.exc_info:
            error_msg = job.exc_info.strip().split("\n")[-1]
        return ResearchJobStatusResponse(job_id=job_id, status="failed", error=error_msg)

    return ResearchJobStatusResponse(job_id=job_id, status=api_status)
