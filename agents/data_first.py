"""
agents/data_first.py
=====================
Checkpoint 6 — Data-First Agent

Strictly fact-bound reasoning agent. If it is not in the verified fact
set, it does not exist. Aggressively rejects explanations that require
information beyond the given facts.

Model: groq/llama-3.1-8b-instant
  A smaller, faster model that is excellent at strict instruction-following
  and rigid logical processing — perfect for fact-bound analysis.
  Free tier on Groq. No credits required.
"""

from agents.base_agent import BaseAgent


class DataFirstAgent(BaseAgent):
    """
    Strictly fact-bound reasoning agent.

    Cognitive role: Only uses facts explicitly present in the input.
    Aggressively rejects any explanation requiring outside knowledge.
    Protocol: Delta-First v4.4, all 6 steps.
    Model: groq/llama-3.1-8b-instant (Meta/Groq, free tier)
    """

    @property
    def name(self) -> str:
        return "data_first"

    @property
    def description(self) -> str:
        return (
            "Strictly fact-bound. If it is not in the verified facts, "
            "it does not exist."
        )
