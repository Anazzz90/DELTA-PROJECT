"""
agents/contrarian.py
======================
Checkpoint 11 — Contrarian Agent

Actively challenges the dominant narrative. Uses Anthropic's Claude 3.5
Haiku — a completely different AI architecture from the Groq (Meta) models
used by Data-First and Skeptic, providing true cognitive diversity.

The Contrarian's job is NOT to find the most likely answer — it is to find
the strongest argument AGAINST the consensus. If all other agents say
"short squeeze", the Contrarian argues for the alternative that the majority
is missing.

Model: anthropic/claude-3-5-haiku-20241022
  Claude architecture (Anthropic) — different reasoning patterns from OpenAI
  and Meta models. Excellent at adversarial thinking and finding overlooked
  edge cases. Requires ANTHROPIC_API_KEY in .env.
"""

from agents.base_agent import BaseAgent


class ContrarianAgent(BaseAgent):
    """
    Narrative-challenging reasoning agent.

    Cognitive role: Always questions the consensus. Finds the strongest
    counter-argument and the most dangerous assumption others are making.
    Protocol: Delta-First v4.4, all 6 steps.
    Model: anthropic/claude-3-5-haiku-20241022 (Anthropic)
    """

    @property
    def name(self) -> str:
        return "contrarian"

    @property
    def description(self) -> str:
        return (
            "Challenges the dominant narrative. Finds the strongest "
            "counter-argument and assumption others are likely missing."
        )
