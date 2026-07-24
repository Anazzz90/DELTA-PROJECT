"""
agents/base_agent.py
=====================
Checkpoint 5 — Abstract Base Agent

Every DMARS agent inherits from BaseAgent. This enforces a consistent
interface across all 5+ agents so the pipeline (core/pipeline.py) can
treat them identically regardless of which model they use.

Subclasses must implement:
    name        — unique agent identifier (e.g. "neutral_analyst")
    description — short human-readable role description

The run() method is inherited and does everything:
    1. Renders the prompt via DeltaProtocol
    2. Calls the LLM via LLMRouter
    3. Parses and validates the JSON response via Pydantic
    4. Returns a structured AgentResult
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from core.delta_protocol import DeltaProtocol
from core.schemas import AgentOutput, AgentResult
from llm.router import LLMRouter, LLMResponse

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all DMARS reasoning agents.

    Concrete agents only need to define their name and description.
    All reasoning logic (prompt rendering, LLM call, JSON validation)
    is handled here in run().
    """

    # =========================================================================
    # Subclasses must define these
    # =========================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier — must match the YAML prompt filename."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short human-readable description of the agent's cognitive role."""
        ...

    # =========================================================================
    # Shared infrastructure (injected once, shared by all agents)
    # =========================================================================

    _protocol: DeltaProtocol = DeltaProtocol()
    _router:   LLMRouter     = LLMRouter()

    # =========================================================================
    # Public API
    # =========================================================================

    def run(
        self,
        question: str,
        fact_set: list[str],
        domain_profile: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> AgentResult:
        """
        Full reasoning pipeline for this agent:
            1. Render prompt (Jinja2 + YAML)
            2. Call LLM (LiteLLM + Tenacity retry)
            3. Parse JSON response
            4. Validate with Pydantic schema
            5. Return AgentResult

        Args:
            question:       The reasoning question.
            fact_set:       List of verified facts.
            domain_profile: Optional domain context string.
            model_override: Override the default model for this agent.

        Returns:
            AgentResult — always returned, even on failure (check .success)
        """
        logger.info(f"[{self.name}] Starting run | question={question[:60]}...")

        # Step 1: Render prompt
        try:
            prompt = self._protocol.render(
                agent_name=self.name,
                question=question,
                fact_set=fact_set,
                domain_profile=domain_profile,
            )
        except Exception as e:
            return AgentResult.err(
                agent_name=self.name,
                error=f"Prompt render failed: {e}",
            )

        # Step 2: Call LLM
        llm_response: LLMResponse = self._router.call(
            agent_name=self.name,
            system_prompt=prompt.system,
            user_prompt=prompt.user,
            model_override=model_override,
        )

        if not llm_response.success:
            return AgentResult.err(
                agent_name=self.name,
                error=f"LLM call failed: {llm_response.error}",
                llm_response=llm_response,
            )

        # Step 3 + 4: Parse and validate JSON
        return self._parse_and_validate(llm_response)

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _parse_and_validate(self, llm_response: LLMResponse) -> AgentResult:
        """Parse the raw LLM text as JSON and validate it with Pydantic."""
        raw_text = llm_response.content.strip()

        # Strip markdown code fences if the LLM wrapped the JSON in ```json ... ```
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            raw_text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.warning(f"[{self.name}] JSON parse failed: {e}")
            return AgentResult.err(
                agent_name=self.name,
                error=f"JSON parse error: {e} | Raw: {raw_text[:200]}",
                llm_response=llm_response,
            )

        try:
            output = AgentOutput(**data)
        except Exception as e:
            logger.warning(f"[{self.name}] Pydantic validation failed: {e}")
            return AgentResult.err(
                agent_name=self.name,
                error=f"Schema validation error: {e}",
                llm_response=llm_response,
            )

        logger.info(
            f"[{self.name}] OK | confidence={output.confidence_score:.2f} "
            f"| driver='{output.main_driver[:60]}'"
        )
        return AgentResult.ok(
            agent_name=self.name,
            output=output,
            llm_response=llm_response,
        )
