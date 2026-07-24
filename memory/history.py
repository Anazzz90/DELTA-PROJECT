"""
memory/history.py
===================
Checkpoint 9 — Query History Read/Write Interface

High-level interface for persisting and retrieving DMARS query history
from SQLite. Used by the dashboard, pipeline, and future API endpoints.

This is the only module that should talk to the database directly —
everything else goes through history.py.

Usage:
    from memory.history import HistoryStore

    store = HistoryStore()
    query_id = await store.save_query(question, fact_set, domain_profile)
    await store.save_agent_output(query_id, agent_result, scoring_result)
    await store.save_final_decision(query_id, final_decision, conflict_report, total_cost)

    history = await store.get_history(limit=20)
    for item in history:
        print(item["question"], item["system_confidence_score"])
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select

from core.aggregator import FinalDecision
from core.conflict_detector import ConflictReport
from core.schemas import AgentResult
from core.scoring_engine import ScoringResult
from db.models import AgentOutputRow, FinalDecisionRow, Query
from db.session import get_session

logger = logging.getLogger(__name__)


class HistoryStore:
    """
    Async read/write interface for DMARS query history.
    All methods return simple data types (dicts / ints) —
    callers never need to know about SQLAlchemy internals.
    """

    # =========================================================================
    # Write methods
    # =========================================================================

    async def save_query(
        self,
        question:       str,
        fact_set:       list[str],
        domain_profile: Optional[str] = None,
    ) -> int:
        """
        Save a new query to the database.
        Returns the new query's integer ID.
        """
        async with get_session() as session:
            row = Query(
                question=question,
                fact_set=fact_set,
                domain_profile=domain_profile,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            logger.info(f"Saved query id={row.id}: '{question[:60]}'")
            return row.id

    async def save_agent_output(
        self,
        query_id:       int,
        agent_result:   AgentResult,
        scoring_result: Optional[ScoringResult] = None,
    ) -> None:
        """
        Save one agent's output for a given query.
        Handles both successful and failed agent results.
        """
        output = agent_result.output if agent_result.success else None

        row = AgentOutputRow(
            query_id=query_id,
            agent_name=agent_result.agent_name,
            model=agent_result.model or "unknown",
            success=agent_result.success,
            main_driver=output.main_driver           if output else None,
            confidence_score=output.confidence_score if output else None,
            extracted_facts=output.extracted_facts   if output else None,
            possible_explanations=output.possible_explanations if output else None,
            ranked_hypotheses=output.ranked_hypotheses          if output else None,
            acknowledged_weaknesses=output.acknowledged_weaknesses if output else None,
            final_score=scoring_result.final_score if scoring_result else None,
            overconfident=scoring_result.overconfident if scoring_result else None,
            cost_usd=agent_result.cost_usd,
            latency_ms=agent_result.latency_ms,
            total_tokens=agent_result.total_tokens,
            error=agent_result.error,
        )

        async with get_session() as session:
            session.add(row)
            await session.commit()
        logger.info(f"Saved agent output: query={query_id} agent={agent_result.agent_name}")

    async def save_final_decision(
        self,
        query_id:        int,
        final_decision:  FinalDecision,
        conflict_report: Optional[ConflictReport] = None,
        total_cost_usd:  float = 0.0,
    ) -> None:
        """
        Save the aggregated final decision for a query.
        """
        row = FinalDecisionRow(
            query_id=query_id,
            system_main_driver=final_decision.system_main_driver,
            system_confidence_score=final_decision.system_confidence_score,
            signal_summary=final_decision.signal_summary,
            net_bias=final_decision.net_bias,
            decision_logic=final_decision.decision_logic,
            conflict_detected=conflict_report.conflict_detected if conflict_report else False,
            conflict_type=conflict_report.conflict_type         if conflict_report else "none",
            conflicting_agents=conflict_report.conflicting_agents if conflict_report else [],
            dominant_narratives=final_decision.dominant_narratives,
            contributing_agents=final_decision.contributing_agents,
            total_cost_usd=total_cost_usd,
        )

        async with get_session() as session:
            session.add(row)
            await session.commit()
        logger.info(f"Saved final decision for query={query_id}")

    # =========================================================================
    # Read methods
    # =========================================================================

    async def get_history(self, limit: int = 20) -> list[dict]:
        """
        Return the most recent queries with their final decisions.
        Results are sorted newest-first.

        Returns a list of plain dicts — safe for JSON serialisation
        and Streamlit display.
        """
        async with get_session() as session:
            stmt = (
                select(Query)
                .order_by(Query.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            queries = result.scalars().all()

            history = []
            for q in queries:
                # Load the final decision for this query if it exists
                fd_stmt = select(FinalDecisionRow).where(
                    FinalDecisionRow.query_id == q.id
                )
                fd_result = await session.execute(fd_stmt)
                fd = fd_result.scalar_one_or_none()

                history.append({
                    "id":               q.id,
                    "question":         q.question,
                    "fact_set":         q.fact_set,
                    "domain_profile":   q.domain_profile,
                    "created_at":       q.created_at.isoformat(),
                    "system_main_driver":       fd.system_main_driver       if fd else None,
                    "system_confidence_score":  fd.system_confidence_score  if fd else None,
                    "net_bias":                 fd.net_bias                 if fd else "Neutral",
                    "signal_summary":           fd.signal_summary           if fd else {"positive": 0, "negative": 0},
                    "decision_logic":           fd.decision_logic           if fd else None,
                    "conflict_detected":        fd.conflict_detected        if fd else None,
                    "dominant_narratives":      fd.dominant_narratives       if fd else [],
                    "total_cost_usd":           fd.total_cost_usd            if fd else 0.0,
                })

            return history

    async def get_query_detail(self, query_id: int) -> Optional[dict]:
        """
        Return full detail for one query including all agent outputs.
        Returns None if query not found.
        """
        async with get_session() as session:
            q = await session.get(Query, query_id)
            if not q:
                return None

            ao_stmt = select(AgentOutputRow).where(AgentOutputRow.query_id == query_id)
            ao_result = await session.execute(ao_stmt)
            agent_rows = ao_result.scalars().all()

            return {
                "id":             q.id,
                "question":       q.question,
                "fact_set":       q.fact_set,
                "domain_profile": q.domain_profile,
                "created_at":     q.created_at.isoformat(),
                "agent_outputs": [
                    {
                        "agent_name":     a.agent_name,
                        "model":          a.model,
                        "success":        a.success,
                        "main_driver":    a.main_driver,
                        "confidence_score": a.confidence_score,
                        "final_score":    a.final_score,
                        "cost_usd":       a.cost_usd,
                        "latency_ms":     a.latency_ms,
                        "error":          a.error,
                    }
                    for a in agent_rows
                ],
            }

    async def get_agent_performance_stats(self, agent_name: Optional[str] = None) -> dict:
        """
        Checkpoint 21 — Aggregate historical performance per agent, across
        every past query.

        Args:
            agent_name: If given, only that agent's rows are queried.
                        Otherwise stats are computed for every agent seen.

        Returns:
            dict mapping agent_name -> {
                "queries_run":     int,    total rows for this agent
                "success_rate":    float,  fraction of runs that succeeded
                "avg_confidence":  float,  mean confidence_score (successful runs)
                "avg_final_score": float,  mean final_score (successful, scored runs)
                "flagged_count":   int,    runs where the scoring engine flagged overconfidence
                "accuracy_rate":   float,  1 - (flagged_count / successful runs) — a calibration
                                           proxy, not "correctness" (there's no ground truth to
                                           grade reasoning quality against)
            }
        """
        async with get_session() as session:
            stmt = select(AgentOutputRow)
            if agent_name:
                stmt = stmt.where(AgentOutputRow.agent_name == agent_name)
            result = await session.execute(stmt)
            rows = result.scalars().all()

        grouped: dict[str, list[AgentOutputRow]] = {}
        for row in rows:
            grouped.setdefault(row.agent_name, []).append(row)

        stats: dict[str, dict] = {}
        for name, agent_rows in grouped.items():
            total = len(agent_rows)
            succeeded = [r for r in agent_rows if r.success]
            confidences = [r.confidence_score for r in succeeded if r.confidence_score is not None]
            final_scores = [r.final_score for r in succeeded if r.final_score is not None]
            flagged = sum(1 for r in succeeded if r.overconfident)

            stats[name] = {
                "queries_run":     total,
                "success_rate":    round(len(succeeded) / total, 4) if total else 0.0,
                "avg_confidence":  round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
                "avg_final_score": round(sum(final_scores) / len(final_scores), 4) if final_scores else 0.0,
                "flagged_count":   flagged,
                "accuracy_rate":   round(1 - (flagged / len(succeeded)), 4) if succeeded else 0.0,
            }

        if agent_name:
            return stats.get(agent_name, {
                "queries_run": 0, "success_rate": 0.0, "avg_confidence": 0.0,
                "avg_final_score": 0.0, "flagged_count": 0, "accuracy_rate": 0.0,
            })
        return stats
