"""
core/schemas.py
================
Checkpoint 5 — Pydantic Output Schemas

Defines the strict data contracts for agent outputs and results.
Every agent in DMARS must return a response that matches AgentOutput exactly.
If the LLM returns malformed JSON, Pydantic catches it here before it
can propagate downstream to the scoring engine or aggregator.

Schemas:
    AgentOutput  — The 6-field Delta-First JSON structure every agent must produce
    AgentResult  — Wraps AgentOutput with metadata (success, cost, latency, errors)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Agent Output Schema — the 6-field Delta-First structure
# =============================================================================

class AgentOutput(BaseModel):
    """
    Pydantic model enforcing the exact JSON schema every DMARS agent must return.

    This matches the OUTPUT FORMAT defined in every agent's system prompt YAML.
    If the LLM returns anything that doesn't fit this schema, Pydantic raises
    a ValidationError before the output ever reaches the scoring engine.
    """

    extracted_facts: list[str] = Field(
        ...,
        min_length=1,
        description="Facts the agent extracted from the input (Step 1)",
    )
    possible_explanations: list[str] = Field(
        ...,
        min_length=1,
        description="All plausible explanations the agent considered (Step 3)",
    )
    ranked_hypotheses: list[str] = Field(
        ...,
        min_length=1,
        description="Explanations ordered best-to-worst fit with facts (Step 3)",
    )
    main_driver: str = Field(
        ...,
        min_length=1,
        description="Single most likely root cause (Step 4)",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Evidence-based confidence 0.0-1.0 (not optimism)",
    )
    acknowledged_weaknesses: list[str] = Field(
        ...,
        min_length=1,
        description="Honest uncertainty statement — at least one entry required (Step 6)",
    )

    @field_validator("main_driver")
    @classmethod
    def main_driver_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("main_driver cannot be blank or whitespace")
        return v.strip()

    @field_validator("confidence_score")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence_score must be 0.0-1.0, got {v}")
        return round(v, 4)


# =============================================================================
# Agent Result — wraps AgentOutput with execution metadata
# =============================================================================

class AgentResult(BaseModel):
    """
    The full result of one agent's reasoning run.
    Always returned by BaseAgent.run(), success or failure.

    The pipeline and scoring engine work with AgentResult objects,
    never raw LLM strings.
    """

    agent_name:  str
    success:     bool
    output:      Optional[AgentOutput] = None   # None if success=False
    error:       Optional[str]         = None   # Set if success=False

    # LLM metadata (available even on failure for diagnostics)
    model:             str   = ""
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    cost_usd:          float = 0.0
    latency_ms:        float = 0.0

    # =========================================================================
    # Factory constructors
    # =========================================================================

    @classmethod
    def ok(
        cls,
        agent_name: str,
        output: AgentOutput,
        llm_response=None,
    ) -> "AgentResult":
        """Create a successful AgentResult with output and LLM metadata."""
        kwargs = dict(
            agent_name=agent_name,
            success=True,
            output=output,
        )
        if llm_response:
            kwargs.update(
                model=llm_response.model,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.total_tokens,
                cost_usd=llm_response.cost_usd,
                latency_ms=llm_response.latency_ms,
            )
        return cls(**kwargs)

    @classmethod
    def err(
        cls,
        agent_name: str,
        error: str,
        llm_response=None,
    ) -> "AgentResult":
        """Create a failed AgentResult with error info and available LLM metadata."""
        kwargs = dict(
            agent_name=agent_name,
            success=False,
            error=error,
        )
        if llm_response:
            kwargs.update(
                model=llm_response.model,
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.total_tokens,
                cost_usd=llm_response.cost_usd,
                latency_ms=llm_response.latency_ms,
            )
        return cls(**kwargs)
