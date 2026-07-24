"""
agents/intuition.py
=====================
Checkpoint 11 — Intuition Agent

Fast, heuristic-based pattern recognition. Uses Google Gemini 2.0 Flash
— a fourth completely independent AI architecture, completing the cognitive
diversity of the boardroom:

  Neutral Analyst  → OpenAI (balanced)
  Data-First       → Meta/Groq (strict logic)
  Skeptic          → Meta/Groq (adversarial)
  Contrarian       → Anthropic (counter-narrative)
  Intuition        → Google (pattern recognition)  ← this agent

The Intuition agent does NOT perform deep analysis. It makes fast,
confident, heuristic-driven judgments based on what similar situations
looked like historically.

Model: google/gemini-2.0-flash
  Gemini architecture (Google DeepMind) — extremely fast inference. Well-
  suited for pattern matching. Requires GOOGLE_API_KEY in .env.
"""

from agents.base_agent import BaseAgent


class IntuitionAgent(BaseAgent):
    """
    Fast heuristic pattern-recognition agent.

    Cognitive role: "Seen this before." Pattern-matches to known historical
    situations and provides a quick, decisive gut-read. Does not over-analyse.
    Protocol: Delta-First v4.4, all 6 steps.
    Model: google/gemini-2.0-flash (Google DeepMind)
    """

    @property
    def name(self) -> str:
        return "intuition"

    @property
    def description(self) -> str:
        return (
            "Fast heuristic judgment. Pattern-matches the situation to known "
            "historical analogues and delivers a decisive gut-read."
        )
