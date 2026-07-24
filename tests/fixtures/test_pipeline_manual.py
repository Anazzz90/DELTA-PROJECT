"""
Checkpoint 6 -- Manual Pipeline Test Script

This script actually makes live API calls to Groq (and OpenAI, if you have credits)
to run all 3 agents in parallel and print their distinct reasoning outputs.

Run from dmars folder:
    poetry run python tests/fixtures/test_pipeline_manual.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.pipeline import Pipeline


async def main():
    print("DEBUG: main() started")
    print("\n" + "=" * 60)
    print("DMARS -- Running 5 Agents in Parallel (Live APIs)")
    print("=" * 60)
    print("Question: Why did BTC spike 8% in the last hour?")
    print("Facts:    ['BTC volume up 3x in 60min', 'Large derivatives positions liquidated', 'No major news']")
    print("-" * 60)

    # Initialize the parallel pipeline
    pipeline = Pipeline()
    
    # Run it!
    print("Calling APIs... (watch how fast this is because they run at the same time)\n")
    result = await pipeline.run(
        question="Why did BTC spike 8% in the last hour?",
        fact_set=[
            "BTC volume up 3x in 60 minutes",
            "Large derivatives positions liquidated",
            "No major news reported in that window"
        ],
        domain_profile="intraday_trading"
    )

    # Print the summary
    print(result.summary())
    print("-" * 60)

    # Print the specific 'main_driver' conclusion from each successful agent
    for agent_result in result.successful_results:
        print(f"\n[BRAIN] {agent_result.agent_name.upper()} Conclusion:")
        print(f"Confidence: {agent_result.output.confidence_score}")
        print(f"Driver:     {agent_result.output.main_driver}")
        print(f"Weakness:   {agent_result.output.acknowledged_weaknesses[0]}")

    if result.agents_failed > 0:
        print("\n[NOTE] Some agents failed (Check your API keys in .env).")
        print("Notice how the pipeline didn't crash! It just returned the ones that worked.")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
