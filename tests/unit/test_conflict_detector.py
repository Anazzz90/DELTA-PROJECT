"""
tests/unit/test_conflict_detector.py
=======================================
Checkpoint 8 -- Unit Tests for Conflict Detector + Aggregator

Coverage (maps exactly to checkpoint test criteria):
  Criterion 1 -- Agents agree -> conflict_detected = False
  Criterion 2 -- Contradictory agents -> conflict_detected = True, agents listed
  Criterion 3 -- Aggregator clusters similar hypotheses into narrative buckets
  Criterion 4 -- system_confidence_score lower when conflict detected
  Criterion 5 -- pytest passes

No API calls. Pure Python.

Run:
    poetry run pytest tests/unit/test_conflict_detector.py -v
"""

import pytest

from core.aggregator import Aggregator, FinalDecision
from core.conflict_detector import ConflictDetector, ConflictReport
from core.schemas import AgentOutput, AgentResult
from core.scoring_engine import ScoringEngine, ScoringResult


# =============================================================================
# Shared fixtures and helpers
# =============================================================================

@pytest.fixture
def detector() -> ConflictDetector:
    return ConflictDetector()


@pytest.fixture
def aggregator() -> Aggregator:
    return Aggregator()


@pytest.fixture
def engine() -> ScoringEngine:
    return ScoringEngine()


@pytest.fixture
def fact_set() -> list[str]:
    return [
        "BTC volume up 3x in 60 minutes",
        "Large derivatives positions liquidated",
        "No major news reported",
    ]


def make_output(main_driver: str, confidence: float = 0.70, hypotheses: list[str] | None = None) -> AgentOutput:
    return AgentOutput(
        extracted_facts=["BTC volume up 3x", "Derivatives liquidated"],
        possible_explanations=["Explanation A", "Explanation B"],
        ranked_hypotheses=hypotheses or ["Hypothesis 1", "Hypothesis 2"],
        main_driver=main_driver,
        confidence_score=confidence,
        acknowledged_weaknesses=["Some uncertainty remains"],
    )


def make_agent_result(name: str, driver: str, confidence: float = 0.70, hypotheses: list[str] | None = None) -> AgentResult:
    return AgentResult(
        agent_name=name,
        success=True,
        output=make_output(driver, confidence, hypotheses),
        model="test-model",
        total_tokens=200,
        cost_usd=0.0001,
        latency_ms=500.0,
    )


def make_scoring_result(name: str, score: float) -> ScoringResult:
    return ScoringResult(
        agent_name=name,
        fact_consistency_score=score,
        reasoning_depth_score=score,
        overconfidence_penalty=0.0,
        final_score=round(score, 4),
        overconfident=False,
    )


# =============================================================================
# Criterion 1 -- All agents agree -> Strong Consensus
# =============================================================================

class TestNoConflict:

    def test_agreeing_agents_return_strong_consensus(self, detector):
        scoring = [
            make_scoring_result("neutral_analyst", 0.80),
            make_scoring_result("data_first",      0.55),
            make_scoring_result("skeptic",          0.35),
        ]
        results = [
            make_agent_result("neutral_analyst", "Positive breakout from volume dynamics", confidence=0.75),
            make_agent_result("data_first",      "Positive surge and flow drove spike", confidence=0.70),
            make_agent_result("skeptic",          "Bullish buying from increased volume", confidence=0.68),
        ]
        report = detector.detect(scoring, results)
        assert report.conflict_detected is False
        assert report.conflict_level == "Strong Consensus"

    def test_no_conflict_type_is_none(self, detector):
        scoring = [
            make_scoring_result("neutral_analyst", 0.80),
            make_scoring_result("data_first",      0.55),
        ]
        results = [
            make_agent_result("neutral_analyst", "Bullish rally"),
            make_agent_result("data_first",      "Upside growth"),
        ]
        report = detector.detect(scoring, results)
        assert report.conflict_type == "none"

    def test_empty_input_returns_strong_consensus(self, detector):
        report = detector.detect([], [])
        assert report.conflict_level == "Strong Consensus"


# =============================================================================
# Criterion 2 -- Disagreement -> Soft Disagreement or High Conflict
# =============================================================================

class TestConflictClassification:

    def test_soft_confidence_disagreement(self, detector):
        """0.20 spread in confidence = Soft Disagreement."""
        scoring = [
            make_scoring_result("a", 0.80),
            make_scoring_result("b", 0.75),
        ]
        results = [
            make_agent_result("a", "Bullish rally", confidence=0.80),
            make_agent_result("b", "Bullish rally", confidence=0.55),
        ]
        report = detector.detect(scoring, results)
        assert report.conflict_detected is True
        assert report.conflict_level == "Soft Disagreement"

    def test_high_confidence_variance(self, detector):
        """0.40 spread in confidence = High Conflict."""
        scoring = [
            make_scoring_result("a", 0.80),
            make_scoring_result("b", 0.75),
        ]
        results = [
            make_agent_result("a", "Bullish rally", confidence=0.85),
            make_agent_result("b", "Bullish rally", confidence=0.40),
        ]
        report = detector.detect(scoring, results)
        assert report.conflict_level == "High Conflict"

    def test_narrative_clash_is_high_conflict(self, detector):
        """buy vs sell — direct contradiction."""
        scoring = [
            make_scoring_result("neutral_analyst", 0.80),
            make_scoring_result("skeptic",         0.30),
        ]
        results = [
            make_agent_result("neutral_analyst", "Institutional buying drove the surge"),
            make_agent_result("skeptic",         "Systematic selling caused the distribution"),
        ]
        report = detector.detect(scoring, results)
        assert report.conflict_detected is True
        assert report.conflict_level == "High Conflict"


# =============================================================================
# Criterion 3 -- Aggregator clusters and explicit decision layer
# =============================================================================

class TestAggregatorOutput:

    def test_aggregator_provides_explicit_decision_layer(self, aggregator):
        scoring = [
            make_scoring_result("neutral_analyst", 0.85),
            make_scoring_result("data_first",      0.60),
        ]
        results = [
            make_agent_result(
                "neutral_analyst", "Bullish breakout from insider buying",
                hypotheses=["Insider buying from CEO", "Accumulation by management"]
            ),
            make_agent_result(
                "data_first", "Negative margin pressure detected",
                hypotheses=["Gross margins declining", "Operating profit lower"]
            ),
        ]
        decision = aggregator.aggregate(scoring, results)
        
        assert hasattr(decision, "signal_summary")
        assert hasattr(decision, "net_bias")
        assert hasattr(decision, "decision_logic")
        assert decision.signal_summary["positive"] == 1
        assert decision.signal_summary["negative"] == 1
        assert decision.net_bias == "Neutral"
        # Should now have these clusters
        assert "insider_activity" in decision.dominant_narratives
        assert "margin_analysis" in decision.dominant_narratives

    def test_narrative_clustering_uses_reasoning_themes(self, aggregator):
        scoring = [make_scoring_result("a", 0.80)]
        results = [
            make_agent_result(
                "a", "Insider selling detected by CEO",
                hypotheses=["Insider selling due to valuation", "Margin pressure in Q3"]
            )
        ]
        decision = aggregator.aggregate(scoring, results)
        assert "insider_activity" in decision.dominant_narratives
        assert "valuation_metrics" in decision.dominant_narratives or "margin_analysis" in decision.dominant_narratives

    def test_confidence_weighted_by_quality_scores(self, aggregator):
        """Agent A (0.9 score, 0.8 conf) vs Agent B (0.1 score, 0.2 conf)."""
        scoring = [
            make_scoring_result("a", 0.90),
            make_scoring_result("b", 0.10),
        ]
        results = [
            make_agent_result("a", "High quality driver", confidence=0.80),
            make_agent_result("b", "Low quality driver", confidence=0.20),
        ]
        decision = aggregator.aggregate(scoring, results)
        
        # Simple avg would be 0.50
        # Weighted is (0.8 * 0.9 + 0.2 * 0.1) / 1.0 = 0.72 + 0.02 = 0.74
        assert abs(decision.system_confidence_score - 0.74) < 0.01

    def test_conflict_penalty_reduces_weighted_confidence(self, aggregator):
        scoring = [make_scoring_result("a", 0.80)]
        results = [make_agent_result("a", "Bullish", confidence=0.80)]
        
        report = ConflictReport(
            conflict_detected=True,
            conflict_level="High Conflict",
            conflict_type="narrative",
            conflicting_agents=["a"],
            conflict_reason="Test Conflict"
        )
        
        decision = aggregator.aggregate(scoring, results, report)
        # 0.80 - 0.15 penalty = 0.65
        assert abs(decision.system_confidence_score - 0.65) < 0.01
        assert decision.conflict_adjusted is True
