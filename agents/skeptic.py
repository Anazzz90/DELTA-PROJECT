"""
agents/skeptic.py
==================
Checkpoint 6 — The Skeptic Agent

Adversarial reasoning agent. Actively tries to break conclusions, find
holes, challenge assumptions, and demand stronger evidence. Not cynical
— rigorous.

Model: groq/llama-3.3-70b-versatile
  A large, highly capable model with strong logical reasoning. Excellent
  at generating alternative hypotheses and stress-testing conclusions.
  Free tier on Groq. No credits required.
"""

from agents.base_agent import BaseAgent


class SkepticAgent(BaseAgent):
    """
    Adversarial reasoning agent.

    Cognitive role: Actively challenges assumptions and finds logical holes.
    Weights alternative explanations more heavily than the obvious answer.
    Protocol: Delta-First v4.4, all 6 steps, adversarial weighting.
    Model: groq/llama-3.3-70b-versatile (Meta/Groq, free tier)
    """

    @property
    def name(self) -> str:
        return "skeptic"

    @property
    def description(self) -> str:
        return (
            "Actively tries to break conclusions. Finds holes, challenges "
            "assumptions, demands stronger evidence."
        )
