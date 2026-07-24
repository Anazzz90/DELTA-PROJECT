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
