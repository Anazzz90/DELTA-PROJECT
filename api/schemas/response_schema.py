from typing import Any, Optional

from pydantic import BaseModel


class AgentOutputResponse(BaseModel):
    agent_name: str
    model: str
    success: bool
    main_driver: Optional[str] = None
    confidence_score: Optional[float] = None
    final_score: Optional[float] = None
    cost_usd: float
    latency_ms: float
    error: Optional[str] = None
    extracted_facts: Optional[list[str]] = None
    ranked_hypotheses: Optional[list[str]] = None
    acknowledged_weaknesses: Optional[list[str]] = None


class MetaAIResponse(BaseModel):
    dominant_driver: str
    synthesis_conclusion: str
    recommended_action: str
    minority_views: list[str]
    final_confidence: float
    supporting_agents: list[str]
    cost_usd: float = 0.0


class QueryResponse(BaseModel):
    query_id: int
    question: str
    system_main_driver: str
    system_confidence_score: float
    net_bias: str
    signal_summary: dict[str, int]
    decision_logic: str
    conflict_detected: bool
    conflict_level: str
    conflict_type: str
    dominant_narratives: list[str]
    agent_results: list[AgentOutputResponse]
    failed_agents: list[str] = []
    meta_ai_result: Optional[MetaAIResponse] = None
    total_cost_usd: float
    pipeline_latency_ms: float
    cache_hit: bool = False


# ── Checkpoint 14 — Job Status Schema ────────────────────────────────────────

class JobStatusResponse(BaseModel):
    """Returned by GET /query/{job_id}/status"""
    job_id: str
    status: str                          # queued | started | finished | failed | not_found
    result: Optional[QueryResponse] = None
    error: Optional[str] = None


class EnqueuedResponse(BaseModel):
    """Returned immediately by POST /query when async mode is active."""
    job_id: str
    status: str = "queued"
    poll_url: str                        # e.g. /query/{job_id}/status
