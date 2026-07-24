"""
core/schemas_v2.py
===================
v2 — Adaptive Probabilistic Decision Intelligence Schemas

Extends the original DMARS schemas to support:
- Multi-horizon reasoning
- Quantitative portfolio construction
- Recursive causal nodes
- Manipulation metrics
"""

from __future__ import annotations
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Temporal Horizon Schema
# =============================================================================

class TemporalForecast(BaseModel):
    """Reasoning and confidence for a specific time horizon."""
    sentiment: str = Field(..., description="Bullish / Bearish / Neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
    primary_driver: str = Field(..., description="Main reason for this horizon's outlook")
    key_risks: List[str] = Field(default_factory=list)

class MultiHorizonReasoning(BaseModel):
    """Reasoning split across standard institutional horizons."""
    intraday: TemporalForecast
    short_term_72h: TemporalForecast
    medium_term_weekly: TemporalForecast
    long_term_quarterly: TemporalForecast
    secular_multi_year: TemporalForecast

# =============================================================================
# Portfolio & Risk Schema
# =============================================================================

class PositionSizing(BaseModel):
    """Quantitative allocation recommendations."""
    asset_name: str
    allocation_pct: float = Field(..., ge=0.0, le=100.0)
    rationale: str
    volatility_adjustment: float = Field(..., description="Multiplier applied due to vol")

class HedgeRecommendation(BaseModel):
    """Quantitative hedging strategy."""
    instrument: str
    hedge_ratio: float
    type: str = Field(..., description="Tail-risk / Delta-neutral / Correlation-play")
    trigger_condition: str

class PortfolioInstruction(BaseModel):
    """Full institutional portfolio construction block."""
    allocations: List[PositionSizing]
    hedges: List[HedgeRecommendation]
    cash_reserve_pct: float = Field(..., ge=0.0, le=100.0)
    risk_adjusted_exposure: float = Field(..., description="Net effective exposure")

# =============================================================================
# Causal & Manipulation Schema
# =============================================================================

class CausalNode(BaseModel):
    """A single node in a recursive consequence tree."""
    event: str
    probability: float = Field(..., ge=0.0, le=1.0)
    impact_magnitude: float = Field(..., ge=-1.0, le=1.0) # -1.0 (disastrous) to 1.0 (explosive growth)
    recursive_consequences: List[str] = Field(default_factory=list)

class ManipulationReport(BaseModel):
    """Metrics detecting coordinated or artificial narrative spikes."""
    coordinated_activity_score: float = Field(..., ge=0.0, le=1.0)
    bot_amplification_detected: bool
    sentiment_acceleration: float # rate of change of sentiment
    whale_trap_probability: float = Field(..., ge=0.0, le=1.0)
    reliability_attenuation_factor: float = Field(default=1.0, description="Multiplier for confidence based on reliability")

# =============================================================================
# Agent Output v2 — The Upgraded Contract
# =============================================================================

class AgentOutputV2(BaseModel):
    """
    The upgraded 2.0.0 output format for DMARS v2 agents.
    """
    # Temporal Layer
    temporal_reasoning: MultiHorizonReasoning
    
    # Reasoning Core (Legacy compatibility + Recursive extension)
    main_driver: str
    causal_tree: List[CausalNode]
    
    # Quantitative Layer
    portfolio_instructions: PortfolioInstruction
    
    # Security & Integrity Layer
    manipulation_metrics: ManipulationReport
    
    # Metadata & Confidence
    systemic_confidence: float = Field(..., ge=0.0, le=1.0)
    unresolved_contradictions: List[str] = Field(default_factory=list)
    
    @field_validator("systemic_confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)

class AgentResultV2(BaseModel):
    """Wraps AgentOutputV2 with execution metadata."""
    agent_name:  str
    success:     bool
    output:      Optional[AgentOutputV2] = None
    error:       Optional[str] = None
    
    # LLM metadata
    model:             str   = ""
    cost_usd:          float = 0.0
    latency_ms:        float = 0.0
