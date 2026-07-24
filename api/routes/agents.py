"""
api/routes/agents.py
=======================
Checkpoint 21 — GET /agents/performance now returns real historical
metrics computed from PostgreSQL, instead of a static hardcoded dict.
"""

from fastapi import APIRouter, Depends

from api.middleware.auth import verify_api_key
from llm.router import AGENT_MODEL_MAP
from memory.history import HistoryStore

router = APIRouter(tags=["Agents"], dependencies=[Depends(verify_api_key)])

AGENT_ROLES = {
    "neutral_analyst": "Balanced reasoning",
    "data_first":       "Strictly fact-bound",
    "skeptic":           "Adversarial reasoning",
    "contrarian":        "Independent logic",
    "intuition":          "Pattern recognition",
}


@router.get("/agents/performance")
async def get_agent_performance():
    """
    Per-agent metadata plus historical performance stats aggregated across
    every query ever run (Checkpoint 21): queries_run, success_rate,
    avg_confidence, avg_final_score, flagged_count, accuracy_rate.
    Agents with no history yet still appear, with zeroed stats.
    """
    stats = await HistoryStore().get_agent_performance_stats()

    return {
        agent_name: {
            "model": AGENT_MODEL_MAP.get(agent_name, "unknown"),
            "role":  AGENT_ROLES.get(agent_name, ""),
            **stats.get(agent_name, {
                "queries_run": 0, "success_rate": 0.0, "avg_confidence": 0.0,
                "avg_final_score": 0.0, "flagged_count": 0, "accuracy_rate": 0.0,
            }),
        }
        for agent_name in AGENT_ROLES
    }
