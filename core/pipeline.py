"""
core/pipeline.py
=================
Checkpoint 6 — Parallel Agent Pipeline

Runs multiple DMARS agents concurrently using asyncio.gather().
Returns a PipelineResult containing all agent outputs, timing data,
and a summary of which agents succeeded or failed.

The pipeline is fault-tolerant: if one agent raises an exception or
its LLM call fails, the pipeline continues with the remaining agents
and flags the failure. It never crashes the whole system because of
one agent.

Usage:
    import asyncio
    from core.pipeline import Pipeline

    pipeline = Pipeline()
    result = asyncio.run(pipeline.run(
        question="Why did BTC spike 8%?",
        fact_set=["Volume up 3x", "Derivatives liquidated", "No major news"],
        domain_profile="intraday_trading",
    ))

    for agent_result in result.results:
        print(f"{agent_result.agent_name}: {agent_result.success}")
    print(f"Total time: {result.total_latency_ms}ms")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import BaseAgent
from agents.contrarian import ContrarianAgent
from agents.data_first import DataFirstAgent
from agents.intuition import IntuitionAgent
from agents.neutral_analyst import NeutralAnalyst
from agents.skeptic import SkepticAgent
from core.schemas import AgentResult

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Result
# =============================================================================

@dataclass
class PipelineResult:
    """
    The aggregated output of one full pipeline run across all agents.
    Contains all AgentResults (success or failure) and timing metadata.
    """
    question:         str
    fact_set:         list[str]
    domain_profile:   Optional[str]
    results:          list[AgentResult]
    total_latency_ms: float
    agents_succeeded: int
    agents_failed:    int

    @property
    def success_rate(self) -> float:
        total = self.agents_succeeded + self.agents_failed
        return self.agents_succeeded / total if total > 0 else 0.0

    @property
    def successful_results(self) -> list[AgentResult]:
        return [r for r in self.results if r.success]

    @property
    def failed_results(self) -> list[AgentResult]:
        return [r for r in self.results if not r.success]

    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    def summary(self) -> str:
        lines = [
            f"Pipeline: {self.agents_succeeded}/{len(self.results)} agents succeeded",
            f"Total time: {self.total_latency_ms:.0f}ms",
            f"Total cost: ${self.total_cost_usd():.6f}",
        ]
        for r in self.results:
            status = "OK  " if r.success else "FAIL"
            driver = r.output.main_driver[:60] if r.success and r.output else r.error or "?"
            lines.append(f"  [{status}] {r.agent_name}: {driver}")
        return "\n".join(lines)


# =============================================================================
# Pipeline
# =============================================================================

class Pipeline:
    """
    Runs all registered agents in parallel via asyncio.gather().

    Fault-tolerant: if one agent fails, the rest continue.
    The pipeline never raises — it always returns a PipelineResult.
    """

    # Default 5-agent set (Checkpoint 11)
    DEFAULT_AGENTS: list[type[BaseAgent]] = [
        NeutralAnalyst,
        DataFirstAgent,
        SkepticAgent,
        ContrarianAgent,
        IntuitionAgent,
    ]

    def __init__(self, agents: Optional[list[BaseAgent]] = None) -> None:
        """
        Args:
            agents: Optional list of pre-instantiated agent objects.
                    If None, uses DEFAULT_AGENTS.
        """
        self.agents: list[BaseAgent] = agents or [
            cls() for cls in self.DEFAULT_AGENTS
        ]
        logger.info(
            f"Pipeline initialized with {len(self.agents)} agents: "
            f"{[a.name for a in self.agents]}"
        )

    async def run(
        self,
        question: str,
        fact_set: list[str],
        domain_profile: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run all agents concurrently and collect results.

        Args:
            question:       The reasoning question.
            fact_set:       List of verified fact strings.
            domain_profile: Optional domain context string.

        Returns:
            PipelineResult with all agent outputs and timing data.
        """
        logger.info(
            f"Pipeline starting | agents={[a.name for a in self.agents]} | "
            f"question='{question[:60]}...'"
        )
        start = time.perf_counter()

        # Run all agents concurrently — each in its own asyncio task
        tasks = [
            self._run_agent_safe(agent, question, fact_set, domain_profile)
            for agent in self.agents
        ]
        results: list[AgentResult] = await asyncio.gather(*tasks)

        total_ms = (time.perf_counter() - start) * 1000
        succeeded = sum(1 for r in results if r.success)
        failed    = len(results) - succeeded

        pipeline_result = PipelineResult(
            question=question,
            fact_set=fact_set,
            domain_profile=domain_profile,
            results=list(results),
            total_latency_ms=round(total_ms, 2),
            agents_succeeded=succeeded,
            agents_failed=failed,
        )

        logger.info(
            f"Pipeline complete | {succeeded}/{len(results)} succeeded | "
            f"{total_ms:.0f}ms | ${pipeline_result.total_cost_usd():.6f}"
        )
        return pipeline_result

    async def _run_agent_safe(
        self,
        agent: BaseAgent,
        question: str,
        fact_set: list[str],
        domain_profile: Optional[str],
    ) -> AgentResult:
        """
        Run a single agent inside a thread pool (LLM calls are blocking I/O).
        Catches all exceptions so one agent never kills the whole pipeline.
        """
        try:
            # agent.run() is synchronous (blocking HTTP call via LiteLLM).
            # Run it in a thread pool so asyncio.gather() can truly parallelize.
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # default ThreadPoolExecutor
                lambda: agent.run(question, fact_set, domain_profile),
            )
            return result
        except Exception as e:
            logger.error(f"[{agent.name}] Unexpected exception in pipeline: {e}")
            return AgentResult.err(
                agent_name=agent.name,
                error=f"Pipeline exception: {type(e).__name__}: {e}",
            )
