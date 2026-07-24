"""Quick smoke test for Checkpoint 12 — Meta-AI agent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import AgentResult, AgentOutput
from agents.meta_ai import MetaAIAgent

# Create fake successful agent results to pass to Meta-AI
def make_result(name, driver, conf):
    output = AgentOutput(
        extracted_facts=["BTC volume up 3x", "Large derivatives liquidated"],
        possible_explanations=["Short squeeze", "Manipulation", "Whale buy"],
        ranked_hypotheses=["Short squeeze", "Whale buy"],
        main_driver=driver,
        confidence_score=conf,
        acknowledged_weaknesses=["Missing order book data"],
    )
    return AgentResult.ok(agent_name=name, output=output)

fake_results = [
    make_result("neutral_analyst", "Forced liquidations triggered a short squeeze", 0.75),
    make_result("data_first",      "BTC volume up 3x in 60 minutes",                0.80),
    make_result("skeptic",         "Manipulation by a large market participant",     0.40),
    make_result("contrarian",      "Deliberate whale manipulation event",            0.70),
    make_result("intuition",       "Resembles 2020 BTC short squeeze pattern",      0.85),
]

print("Initializing MetaAIAgent...")
meta = MetaAIAgent()
print("Calling synthesize()...")
result = meta.synthesize(
    agent_results=fake_results,
    question="Why did BTC spike 8% in the last hour?",
    fact_set=["BTC volume up 3x in 60min", "Large derivatives positions liquidated", "No major news"],
    domain_profile="intraday_trading",
)

if result.success:
    print("\n=== META-AI VERDICT ===")
    print(f"Dominant Driver:    {result.output.dominant_driver}")
    print(f"Final Confidence:   {result.output.final_confidence:.0%}")
    print(f"Supporting Agents:  {', '.join(result.output.supporting_agents)}")
    print(f"Synthesis:          {result.output.synthesis_conclusion[:200]}...")
    print(f"Recommended Action: {result.output.recommended_action}")
    print(f"Minority Views:     {result.output.minority_views}")
    print(f"\nModel: {result.model} | Tokens: {result.total_tokens} | {result.latency_ms:.0f}ms")
    print("\nCheckpoint 12: PASS")
else:
    print(f"\nMeta-AI FAILED: {result.error}")
    print("\nCheckpoint 12: FAIL")
