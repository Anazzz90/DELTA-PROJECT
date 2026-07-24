"""
core/schemas.py  (additions for Checkpoint 12 — Meta-AI Layer)

MetaAIOutput and MetaAIResult schemas for the synthesis agent.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class MetaAIOutput(BaseModel):
    """
    Pydantic schema for the Meta-AI synthesis agent's JSON response.
    The Meta-AI receives all 5 agent outputs and produces one unified verdict.
    """

    synthesis_conclusion: str = Field(
        ..., min_length=10,
        description="Definitive paragraph synthesizing all agent views",
    )
    dominant_driver: str = Field(
        ..., min_length=1,
        description="The single most supported root cause",
    )
    supporting_agents: list[str] = Field(
        ...,
        description="Agent names that support the dominant driver",
    )
    minority_views: list[str] = Field(
        default_factory=list,
        description="Dissenting views worth preserving",
    )
    final_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Calibrated final confidence (lower if agents disagree)",
    )
    synthesis_reasoning: str = Field(
        ..., min_length=10,
        description="How agent outputs were weighted",
    )
    recommended_action: str = Field(
        ..., min_length=5,
        description="Concrete actionable takeaway",
    )

    @field_validator("final_confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"final_confidence must be 0.0-1.0, got {v}")
        return round(v, 4)


class MetaAIResult(BaseModel):
    """Full result from the Meta-AI synthesis run."""
    success:           bool
    output:            Optional[MetaAIOutput] = None
    error:             Optional[str] = None
    model:             str   = ""
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    cost_usd:          float = 0.0
    latency_ms:        float = 0.0
