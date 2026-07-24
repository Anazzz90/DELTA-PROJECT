import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env so API keys are available
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from agents.data_first import DataFirstAgent
from agents.skeptic import SkepticAgent
from agents.contrarian import ContrarianAgent
from agents.intuition import IntuitionAgent
from core.pipeline import Pipeline
from core.scoring_engine import ScoringEngine
from core.conflict_detector import ConflictDetector
from core.aggregator import Aggregator

question = "Why did BTC spike 8% in the last 60 minutes?"
fact_set = [
    "BTC trading volume increased 3x in 60 minutes",
    "Large derivatives positions were liquidated",
    "No major news was reported in this window",
]

agents   = [DataFirstAgent(), SkepticAgent(), ContrarianAgent(), IntuitionAgent()]
pipeline = Pipeline(agents=agents)

print("Running 4 agents in parallel...")
print("=" * 60)
result = asyncio.run(pipeline.run(question, fact_set, "intraday_trading"))

engine = ScoringEngine()
scores = []
for r in result.results:
    print(f"\n--- {r.agent_name.upper()} ---")
    if r.success and r.output:
        sr = engine.score(r.output, fact_set, r.agent_name)
        scores.append(sr)
        print(f"  Model      : {r.model}")
        print(f"  Latency    : {r.latency_ms:.0f}ms")
        print(f"  Main Driver: {r.output.main_driver}")
        print(f"  Confidence : {r.output.confidence_score:.0%}")
        print(f"  Score      : {sr.final_score:.3f}")
    else:
        print(f"  FAILED: {r.error[:120] if r.error else 'unknown'}")

print("\n" + "=" * 60)
detector = ConflictDetector()
conflict = detector.detect(scores, result.results)
decision = Aggregator().aggregate(scores, result.results, conflict)

print(f"Agents Succeeded : {result.agents_succeeded}/{len(agents)}")
print(f"Total Cost       : ${result.total_cost_usd():.5f}")
print(f"Conflict Type    : {conflict.conflict_type}")
print(f"System Driver    : {decision.system_main_driver}")
print(f"System Confidence: {decision.system_confidence_score:.0%}")
print(f"Top Narratives   : {decision.dominant_narratives}")
