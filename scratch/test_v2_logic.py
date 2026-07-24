"""
scratch/test_v2_logic.py
========================
Simple test case to verify:
1. Clean formatting of conflict keywords (no more {braces}).
2. Improved sentiment detection (handling 'demand', 'supply', 'surge', 'lower').
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.aggregator import Aggregator
from core.conflict_detector import ConflictDetector
from core.schemas import AgentOutput, AgentResult
from core.scoring_engine import ScoringResult

async def run_test():
    print("Starting DMARS Logic Verification Test...\n")

    # 1. Mock Agent Outputs
    # Agent A: Bullish on Demand
    output_a = AgentOutput(
        extracted_facts=["NVDA demand up 40%"],
        possible_explanations=["AI boom driving orders"],
        ranked_hypotheses=["Demand surge"],
        main_driver="Massive AI demand driving a surge in orders",
        confidence_score=0.85,
        acknowledged_weaknesses=["Supply might not keep up"]
    )
    res_a = AgentResult.ok(agent_name="neutral_analyst", output=output_a)
    score_a = ScoringResult(
        agent_name="neutral_analyst", 
        final_score=0.9, 
        fact_consistency_score=1.0, 
        reasoning_depth_score=0.9, 
        overconfidence_penalty=0.0,
        overconfident=False
    )

    # Agent B: Bearish on Supply
    output_b = AgentOutput(
        extracted_facts=["Fab fire in Hsinchu"],
        possible_explanations=["Supply chain disruption"],
        ranked_hypotheses=["Supply shortage"],
        main_driver="Critical supply shortage and lower production volumes",
        confidence_score=0.75,
        acknowledged_weaknesses=["Demand remains high"]
    )
    res_b = AgentResult.ok(agent_name="data_first", output=output_b)
    score_b = ScoringResult(
        agent_name="data_first", 
        final_score=0.8, 
        fact_consistency_score=1.0, 
        reasoning_depth_score=0.8, 
        overconfidence_penalty=0.0,
        overconfident=False
    )

    agent_results = [res_a, res_b]
    scoring_results = [score_a, score_b]

    # 2. Run Conflict Detector
    print("Running Conflict Detector...")
    detector = ConflictDetector()
    conflict_report = detector.detect(scoring_results, agent_results)
    
    print(f"   Conflict Level: {conflict_report.conflict_level}")
    print(f"   Formatted Reason: {conflict_report.conflict_reason}")
    
    # Verify no braces in reason
    if "{" in conflict_report.conflict_reason:
        print("FAILED: Braces found in conflict reason!")
    else:
        print("PASSED: Clean string formatting verified.")

    # 3. Run Aggregator
    print("\nRunning Aggregator...")
    aggregator = Aggregator()
    final_decision = aggregator.aggregate(scoring_results, agent_results, conflict_report)

    print(f"   Net Bias: {final_decision.net_bias}")
    print(f"   Signals: +{final_decision.signal_summary['positive']} / -{final_decision.signal_summary['negative']} / {final_decision.signal_summary['neutral']}")
    
    # Verify Bias detection
    # With 'demand' and 'surge' in pos_keywords, and 'supply' and 'lower' in neg_keywords, 
    # it should be 1 pos vs 1 neg -> Neutral.
    if final_decision.net_bias == "Neutral":
        print("PASSED: Robust sentiment detection verified (1 vs 1 = Neutral).")
    else:
        print(f"FAILED: Sentiment detection error. Got {final_decision.net_bias}")

    print("\nTest Complete.")

if __name__ == "__main__":
    asyncio.run(run_test())
