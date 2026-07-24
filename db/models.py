"""
db/models.py
==============
Checkpoint 9 — SQLAlchemy ORM Models

Defines the 3 core database tables:
  - Query       — one row per user question submitted to DMARS
  - AgentOutput — one row per agent response (3 per query in Phase 1)
  - FinalDecision — one row per aggregated system output

All tables use Integer primary keys for SQLite compatibility.
They will migrate cleanly to PostgreSQL in Phase 2 (Checkpoint 13).
"""

from __future__ import annotations

import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# =============================================================================
# Query — one row per DMARS question
# =============================================================================

class Query(Base):
    __tablename__ = "queries"

    id:             Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    question:       Mapped[str]      = mapped_column(Text,    nullable=False)
    fact_set:       Mapped[list]     = mapped_column(JSON,    nullable=False, default=list)
    domain_profile: Mapped[str|None] = mapped_column(String(64), nullable=True)
    created_at:     Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    # Relationships
    agent_outputs:  Mapped[list["AgentOutputRow"]]  = relationship("AgentOutputRow",  back_populates="query", cascade="all, delete-orphan")
    final_decision: Mapped["FinalDecisionRow|None"]  = relationship("FinalDecisionRow", back_populates="query", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Query id={self.id} question='{self.question[:40]}...'>"


# =============================================================================
# AgentOutput — one row per agent per query
# =============================================================================

class AgentOutputRow(Base):
    __tablename__ = "agent_outputs"

    id:                     Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id:               Mapped[int]      = mapped_column(ForeignKey("queries.id"), nullable=False)
    agent_name:             Mapped[str]      = mapped_column(String(64),  nullable=False)
    model:                  Mapped[str]      = mapped_column(String(128), nullable=False)
    success:                Mapped[bool]     = mapped_column(Boolean,     nullable=False)
    main_driver:            Mapped[str|None] = mapped_column(Text,        nullable=True)
    confidence_score:       Mapped[float|None] = mapped_column(Float,     nullable=True)
    extracted_facts:        Mapped[list|None]  = mapped_column(JSON,      nullable=True)
    possible_explanations:  Mapped[list|None]  = mapped_column(JSON,      nullable=True)
    ranked_hypotheses:      Mapped[list|None]  = mapped_column(JSON,      nullable=True)
    acknowledged_weaknesses:Mapped[list|None]  = mapped_column(JSON,      nullable=True)
    final_score:            Mapped[float|None] = mapped_column(Float,     nullable=True)
    cost_usd:               Mapped[float]    = mapped_column(Float,       nullable=False, default=0.0)
    latency_ms:             Mapped[float]    = mapped_column(Float,       nullable=False, default=0.0)
    total_tokens:           Mapped[int]      = mapped_column(Integer,     nullable=False, default=0)
    error:                  Mapped[str|None] = mapped_column(Text,        nullable=True)
    created_at:             Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    query: Mapped["Query"] = relationship("Query", back_populates="agent_outputs")

    def __repr__(self) -> str:
        return f"<AgentOutputRow agent={self.agent_name} success={self.success} score={self.final_score}>"


# =============================================================================
# FinalDecision — one row per query (aggregated system decision)
# =============================================================================

class FinalDecisionRow(Base):
    __tablename__ = "final_decisions"

    id:                      Mapped[int]       = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id:                Mapped[int]       = mapped_column(ForeignKey("queries.id"), nullable=False, unique=True)
    system_main_driver:      Mapped[str]       = mapped_column(Text,   nullable=False)
    system_confidence_score: Mapped[float]     = mapped_column(Float,  nullable=False)
    signal_summary:          Mapped[dict|None] = mapped_column(JSON,   nullable=True)
    net_bias:                Mapped[str|None]  = mapped_column(String(32), nullable=True)
    decision_logic:          Mapped[str|None]  = mapped_column(Text,   nullable=True)
    conflict_detected:       Mapped[bool]      = mapped_column(Boolean,nullable=False, default=False)
    conflict_type:           Mapped[str|None]  = mapped_column(String(32), nullable=True)
    conflicting_agents:      Mapped[list|None] = mapped_column(JSON,   nullable=True)
    dominant_narratives:     Mapped[list|None] = mapped_column(JSON,   nullable=True)
    contributing_agents:     Mapped[list|None] = mapped_column(JSON,   nullable=True)
    total_cost_usd:          Mapped[float]     = mapped_column(Float,  nullable=False, default=0.0)
    created_at:              Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    query: Mapped["Query"] = relationship("Query", back_populates="final_decision")

    def __repr__(self) -> str:
        return f"<FinalDecisionRow query_id={self.query_id} confidence={self.system_confidence_score}>"
