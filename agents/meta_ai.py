"""
agents/meta_ai.py
==================
Checkpoint 12 — Meta-AI Synthesis Agent

The Meta-AI is the final adjudicator. It runs AFTER all 5 agents complete
and synthesizes their structured outputs into one definitive verdict.

Model: deepseek-ai/DeepSeek-V3 (SiliconFlow) — free, highly capable.
Role:  Final synthesis — weighs all 5 agent views into one conclusion.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import yaml
from jinja2 import BaseLoader, Environment

from config.settings import settings
from core.meta_schemas import MetaAIOutput, MetaAIResult
from core.schemas import AgentResult
from llm.router import LLMRouter

logger = logging.getLogger(__name__)


class MetaAIAgent:
    """
    Checkpoint 12 — Meta-AI synthesis layer.

    Receives all 5 agent outputs and produces one calibrated final verdict
    using DeepSeek-V3 on SiliconFlow (free Chinese model, GPT-4o equivalent).

    This agent is OPTIONAL — the pipeline works with or without it.
    Call synthesize() after pipeline.run() to activate it.
    """

    name        = "meta_ai"
    description = "Final arbitration layer. Synthesizes all 5 agent outputs into one definitive verdict."

    _router = LLMRouter()

    def __init__(self) -> None:
        yaml_path = settings.prompts_dir / "meta_ai.yaml"
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        self._system_template = data["system"]
        self._user_template   = data["user"]
        self._jinja_env = Environment(loader=BaseLoader())

    # =========================================================================
    # Public API
    # =========================================================================

    def synthesize(
        self,
        agent_results: list[AgentResult],
        question: str,
        fact_set: list[str],
        domain_profile: Optional[str] = None,
    ) -> MetaAIResult:
        """
        Synthesize all successful agent outputs into one final verdict.

        Args:
            agent_results:  List of AgentResult from the 5-agent pipeline.
            question:       The original reasoning question.
            fact_set:       The verified facts provided by the user.
            domain_profile: Optional domain context.

        Returns:
            MetaAIResult — always returned, check .success for status.
        """
        successful = [r for r in agent_results if r.success and r.output]
        if not successful:
            return MetaAIResult(
                success=False,
                error="No successful agent outputs to synthesize.",
            )

        # Build the agent_outputs list for Jinja2
        agent_outputs = [
            {
                "name":             r.agent_name,
                "main_driver":      r.output.main_driver,
                "confidence_score": r.output.confidence_score,
                "top_hypothesis":   r.output.ranked_hypotheses[0] if r.output.ranked_hypotheses else "N/A",
                "weakness":         r.output.acknowledged_weaknesses[0] if r.output.acknowledged_weaknesses else "N/A",
            }
            for r in successful
        ]

        ctx = dict(
            question=question,
            fact_set=fact_set,
            domain_profile=domain_profile,
            agent_outputs=agent_outputs,
        )
        system_prompt = self._jinja_env.from_string(self._system_template).render(**ctx)
        user_prompt   = self._jinja_env.from_string(self._user_template).render(**ctx)

        logger.info(f"[meta_ai] Synthesizing {len(successful)} agent outputs via DeepSeek-V3")
        llm_response = self._router.call(
            agent_name="meta_ai",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if not llm_response.success:
            return MetaAIResult(
                success=False,
                error=f"LLM call failed: {llm_response.error}",
                model=llm_response.model,
                latency_ms=llm_response.latency_ms,
            )

        return self._parse(llm_response)

    # =========================================================================
    # Private
    # =========================================================================

    def _parse(self, llm_response) -> MetaAIResult:
        raw = llm_response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"[meta_ai] JSON parse failed: {e}")
            return MetaAIResult(
                success=False,
                error=f"JSON parse error: {e} | Raw: {raw[:300]}",
                model=llm_response.model,
                latency_ms=llm_response.latency_ms,
            )

        try:
            output = MetaAIOutput(**data)
        except Exception as e:
            logger.warning(f"[meta_ai] Pydantic validation failed: {e}")
            return MetaAIResult(
                success=False,
                error=f"Schema validation error: {e}",
                model=llm_response.model,
                latency_ms=llm_response.latency_ms,
            )

        logger.info(
            f"[meta_ai] OK | confidence={output.final_confidence:.2f} "
            f"| driver='{output.dominant_driver[:60]}'"
        )
        return MetaAIResult(
            success=True,
            output=output,
            model=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
            cost_usd=llm_response.cost_usd,
            latency_ms=llm_response.latency_ms,
        )
