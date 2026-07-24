"""
agents/neutral_analyst.py
==========================
Checkpoint 5 — The Neutral Analyst Agent

The first fully working DMARS agent. Balanced, objective reasoning.
No agenda, no preferred outcome. Follows the Delta-First Protocol strictly.

Model: groq/llama-3.3-70b-versatile
  (Originally assigned gpt-4o-mini per PRD. Mapped to Groq equivalent
   as OpenAI credits are not required for Phase 1 development.)

Usage:
    from agents.neutral_analyst import NeutralAnalyst

    agent = NeutralAnalyst()
    result = agent.run(
        question="Why did BTC spike 8%?",
        fact_set=["Volume up 3x", "Derivatives liquidated", "No major news"],
        domain_profile="intraday_trading",
    )

    if result.success:
        print(result.output.main_driver)
        print(result.output.confidence_score)
    else:
        print(result.error)
"""

from agents.base_agent import BaseAgent


class NeutralAnalyst(BaseAgent):
    """
    Balanced, objective reasoning agent.

    Cognitive role: No agenda, no preferred outcome.
    Treats all explanations equally until scored by the facts.
    Protocol: Delta-First v4.4, all 6 steps.
    Model: groq/llama-3.3-70b-versatile (free tier, no credits required)
    """

    @property
    def name(self) -> str:
        return "neutral_analyst"

    @property
    def description(self) -> str:
        return (
            "Balanced, objective reasoning. No agenda. "
            "Follows Delta-First Protocol strictly."
        )
