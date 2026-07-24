# Checkpoint 14 — Redis + RQ Task Queue
# Task definitions for async pipeline execution.
#
# This module defines the RQ job that runs the full DMARS reasoning pipeline.
# It is called by the RQ worker process and is enqueued by the FastAPI /query route.
#
# IMPORTANT: This runs in a *synchronous* RQ worker context. Async code must be
# executed via asyncio.run() — NOT await — because RQ workers are not async.

import asyncio
import time
from typing import Any, Optional

from agents.contrarian import ContrarianAgent
from agents.data_first import DataFirstAgent
from agents.intuition import IntuitionAgent
from agents.meta_ai import MetaAIAgent
from agents.neutral_analyst import NeutralAnalyst
from agents.skeptic import SkepticAgent

from core.aggregator import Aggregator
from core.conflict_detector import ConflictDetector
from core.pipeline import Pipeline
from core.scoring_engine import AgentPerformanceTracker, ScoringEngine
from memory.history import HistoryStore
from memory.vector_store import VectorStore
from core.research_engine import ResearchEngine
from observability.tracer import check_cost_alerts
from llm.cache import SemanticCache

AGENT_MAPPING = {
    "neutral_analyst": NeutralAnalyst,
    "data_first": DataFirstAgent,
    "skeptic": SkepticAgent,
    "contrarian": ContrarianAgent,
    "intuition": IntuitionAgent,
}


def run_pipeline_task(
    question: str,
    fact_set: list[str],
    domain_profile: str,
    selected_agents: list[str],
    meta_ai_enabled: bool,
) -> dict[str, Any]:
    """
    RQ task: runs the full DMARS multi-agent pipeline synchronously.
    Returns the full structured result dict that mirrors QueryResponse.
    """
    t_start = time.perf_counter()

    # ── Semantic cache check (Checkpoint 18) ───────────────────────────────────
    cache = SemanticCache()
    cached_result = cache.get(question, selected_agents, domain_profile, meta_ai_enabled)
    if cached_result is not None:
        return cached_result

    # ── Build agent list ──────────────────────────────────────────────────────
    agents_to_run = [
        AGENT_MAPPING[name]()
        for name in selected_agents
        if name in AGENT_MAPPING
    ]

    # ── Run pipeline (async inside sync worker) ───────────────────────────────
    async def _run() -> dict[str, Any]:
        # 1. Pipeline
        pipeline = Pipeline(agents=agents_to_run)
        pipeline_result = await pipeline.run(question, fact_set, domain_profile)

        # 2. Scoring
        scoring_engine = ScoringEngine()
        scoring_results = [
            scoring_engine.score(r.output, fact_set, r.agent_name)
            for r in pipeline_result.successful_results
        ]

        # 2b. Historical performance adjustment (Checkpoint 21) — scales each
        # agent's final_score by its track record before it reaches conflict
        # detection / aggregation, both of which already weight by
        # final_score, so no changes needed there to pick this up.
        tracker = AgentPerformanceTracker()
        for sr in scoring_results:
            multiplier = await tracker.get_weight_multiplier(sr.agent_name)
            if multiplier != 1.0:
                sr.final_score = round(sr.final_score * multiplier, 4)

        # 3. Conflict & Aggregation
        conflict_detector = ConflictDetector()
        conflict_report = conflict_detector.detect(scoring_results, pipeline_result.results)
        aggregator = Aggregator()
        final_decision = aggregator.aggregate(
            scoring_results, pipeline_result.results, conflict_report
        )

        # 4. Meta-AI synthesis (optional)
        meta_response_obj = None
        if meta_ai_enabled:
            meta_agent = MetaAIAgent()
            loop = asyncio.get_event_loop()
            meta_result = await loop.run_in_executor(
                None,
                lambda: meta_agent.synthesize(
                    pipeline_result.results, question, fact_set, domain_profile
                ),
            )
            if meta_result and meta_result.success and meta_result.output:
                mo = meta_result.output
                meta_response_obj = {
                    "dominant_driver": mo.dominant_driver,
                    "synthesis_conclusion": mo.synthesis_conclusion,
                    "recommended_action": mo.recommended_action,
                    "minority_views": mo.minority_views,
                    "final_confidence": mo.final_confidence,
                    "supporting_agents": mo.supporting_agents,
                    "cost_usd": meta_result.cost_usd,
                }

        # 5. Cost alerting (Checkpoint 17) — total includes Meta-AI's synthesis
        # call, which is a separate LLM call outside the 5-agent Pipeline and
        # was previously dropped from every cost total shown to the user.
        total_cost_usd = pipeline_result.total_cost_usd() + (
            meta_response_obj["cost_usd"] if meta_response_obj else 0.0
        )
        check_cost_alerts(total_cost_usd)

        # 6. Storage
        store = HistoryStore()
        query_id = await store.save_query(question, fact_set, domain_profile)
        for r in pipeline_result.results:
            sr = next((s for s in scoring_results if s.agent_name == r.agent_name), None)
            await store.save_agent_output(query_id, r, sr)
        await store.save_final_decision(
            query_id,
            final_decision,
            conflict_report,
            total_cost_usd=total_cost_usd,
        )
        vs = VectorStore()
        vs.add(
            query_id,
            question,
            metadata={
                "domain": domain_profile,
                "confidence": final_decision.system_confidence_score,
            },
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # ── Build output dict (matches QueryResponse fields) ──────────────────
        agent_outputs = []
        for r in pipeline_result.results:
            sr = next((s for s in scoring_results if s.agent_name == r.agent_name), None)
            out = {
                "agent_name": r.agent_name,
                "model": r.model or "unknown",
                "success": r.success,
                "cost_usd": r.cost_usd,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "main_driver": None,
                "confidence_score": None,
                "final_score": None,
                "extracted_facts": None,
                "ranked_hypotheses": None,
                "acknowledged_weaknesses": None,
            }
            if r.success and r.output:
                out["main_driver"] = r.output.main_driver
                out["confidence_score"] = r.output.confidence_score
                out["extracted_facts"] = r.output.extracted_facts
                out["ranked_hypotheses"] = r.output.ranked_hypotheses
                out["acknowledged_weaknesses"] = r.output.acknowledged_weaknesses
            if sr:
                out["final_score"] = sr.final_score
            agent_outputs.append(out)

        result = {
            "query_id": query_id,
            "question": question,
            "system_main_driver": final_decision.system_main_driver,
            "system_confidence_score": final_decision.system_confidence_score,
            "net_bias": final_decision.net_bias,
            "signal_summary": final_decision.signal_summary,
            "decision_logic": final_decision.decision_logic,
            "conflict_detected": conflict_report.conflict_detected,
            "conflict_level": conflict_report.conflict_level,
            "conflict_type": conflict_report.conflict_type,
            "dominant_narratives": final_decision.dominant_narratives,
            "agent_results": agent_outputs,
            "failed_agents": pipeline_result.failed_agent_names,
            "meta_ai_result": meta_response_obj,
            "total_cost_usd": total_cost_usd,
            "pipeline_latency_ms": elapsed_ms,
            "cache_hit": False,
        }
        cache.set(question, selected_agents, result, domain_profile, meta_ai_enabled)
        return result

    return asyncio.run(_run())

def run_research_task(
    url: Optional[str],
    topic: Optional[str],
    domain_profile: str,
    auto_reason: bool,
    redis_url: str
) -> dict[str, Any]:
    """
    RQ task: runs the Firecrawl research pipeline.
    If auto_reason is True, it will automatically enqueue the main reasoning pipeline.
    """
    import os
    from redis import Redis
    from rq import Queue
    
    t_start = time.perf_counter()
    redis_conn = Redis.from_url(redis_url)

    async def _run() -> dict[str, Any]:
        engine = ResearchEngine(redis_conn=redis_conn)
        
        if url:
            result = await engine.run_research(url=url, domain_profile=domain_profile)
            source_desc = url
        else:
            result = await engine.run_topic_research(topic=topic, domain_profile=domain_profile)
            source_desc = f"topic: {topic}"
        
        # Chain into reasoning pipeline if requested
        pipeline_job_id = None
        if auto_reason and result.get("extracted_facts"):
            queue = Queue("dmars-queue", connection=redis_conn)
            timeout = -1 if not hasattr(os, "fork") else 600
            
            job = queue.enqueue(
                run_pipeline_task,
                kwargs={
                    "question": f"Analyze the findings from {source_desc}",
                    "fact_set": result["extracted_facts"],
                    "domain_profile": domain_profile,
                    "selected_agents": list(AGENT_MAPPING.keys()),
                    "meta_ai_enabled": True,
                },
                job_timeout=timeout,
            )
            pipeline_job_id = job.id
            result["chained_pipeline_job_id"] = pipeline_job_id
            
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        result["research_latency_ms"] = elapsed_ms
        return result

    return asyncio.run(_run())
