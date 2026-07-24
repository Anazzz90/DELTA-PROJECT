"""
tests/unit/test_scoring_engine.py
====================================
Checkpoint 7 -- Unit Tests for the Scoring Engine

Coverage (maps exactly to checkpoint test criteria):
  Criterion 1 -- Scoring 3 agent outputs each yields 0.0-1.0 scores
  Criterion 2 -- Agent with more hypotheses + weaknesses scores higher
  Criterion 3 -- Agent with confidence=1.0 + shallow reasoning is penalised
  Criterion 4 -- Adjusting scoring_weights.yaml changes scores
  Criterion 5 -- pytest passes

No API calls. No mocking needed. Pure Python math.

Run:
    poetry run pytest tests/unit/test_scoring_engine.py -v
"""

import copy
import pytest

from core.schemas import AgentOutput
from core.scoring_engine import ScoringEngine, ScoringResult


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def engine() -> ScoringEngine:
    return ScoringEngine()


@pytest.fixture
def fact_set() -> list[str]:
    return [
        "BTC volume up 3x in 60 minutes",
        "Large derivatives positions liquidated",
        "No major news reported in that window",
    ]


def make_output(
    extracted_facts: list[str] | None = None,
    possible_explanations: list[str] | None = None,
    ranked_hypotheses: list[str] | None = None,
    main_driver: str = "Volume surge caused forced buying",
    confidence_score: float = 0.70,
    acknowledged_weaknesses: list[str] | None = None,
) -> AgentOutput:
    return AgentOutput(
        extracted_facts=extracted_facts or ["BTC volume up 3x"],
        possible_explanations=possible_explanations or ["Short squeeze", "Institutional buy"],
        ranked_hypotheses=ranked_hypotheses or ["Short squeeze (best fit)", "Institutional buy"],
        main_driver=main_driver,
        confidence_score=confidence_score,
        acknowledged_weaknesses=acknowledged_weaknesses or ["No order book data available"],
    )


# A "deep" agent with rich reasoning
DEEP_OUTPUT = make_output(
    extracted_facts=["BTC volume up 3x in 60 minutes", "Large derivatives positions liquidated", "No major news"],
    possible_explanations=["Short squeeze", "Institutional accumulation", "Algorithm momentum", "Whale accumulation"],
    ranked_hypotheses=["Short squeeze (strongest evidence)", "Institutional accumulation", "Algorithm momentum", "Whale"],
    main_driver="Mass short squeeze triggered by volume spike",
    confidence_score=0.70,
    acknowledged_weaknesses=["No order book data", "Cannot confirm institutional identity", "Manipulation not ruled out"],
)

# A "shallow" agent with minimal reasoning
SHALLOW_OUTPUT = make_output(
    extracted_facts=["Volume up"],
    possible_explanations=["Volume caused it"],
    ranked_hypotheses=["Volume caused it"],
    main_driver="Volume",
    confidence_score=0.70,
    acknowledged_weaknesses=["Uncertain"],
)

# An "overconfident" agent — high confidence, minimal reasoning
OVERCONFIDENT_OUTPUT = make_output(
    extracted_facts=["Volume up"],
    possible_explanations=["Volume caused it"],
    ranked_hypotheses=["Volume caused it"],
    main_driver="Volume caused the spike",
    confidence_score=1.0,   # <-- claims 100% certainty
    acknowledged_weaknesses=["Maybe something else"],
)


# =============================================================================
# Criterion 1 -- Each scored agent gets a score 0.0-1.0
# =============================================================================

class TestScoreRange:

    def test_score_returns_scoring_result(self, engine, fact_set):
        result = engine.score(DEEP_OUTPUT, fact_set, "neutral_analyst")
        assert isinstance(result, ScoringResult)

    def test_final_score_is_float(self, engine, fact_set):
        result = engine.score(DEEP_OUTPUT, fact_set, "neutral_analyst")
        assert isinstance(result.final_score, float)

    def test_final_score_in_range_deep(self, engine, fact_set):
        result = engine.score(DEEP_OUTPUT, fact_set, "neutral_analyst")
        assert 0.0 <= result.final_score <= 1.0

    def test_final_score_in_range_shallow(self, engine, fact_set):
        result = engine.score(SHALLOW_OUTPUT, fact_set, "data_first")
        assert 0.0 <= result.final_score <= 1.0

    def test_final_score_in_range_overconfident(self, engine, fact_set):
        result = engine.score(OVERCONFIDENT_OUTPUT, fact_set, "skeptic")
        assert 0.0 <= result.final_score <= 1.0

    def test_all_three_agents_scored(self, engine, fact_set):
        results = engine.score_many(
            [
                (DEEP_OUTPUT,         "neutral_analyst"),
                (SHALLOW_OUTPUT,      "data_first"),
                (OVERCONFIDENT_OUTPUT,"skeptic"),
            ],
            fact_set,
        )
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.final_score <= 1.0

    def test_score_many_sorted_best_first(self, engine, fact_set):
        results = engine.score_many(
            [
                (SHALLOW_OUTPUT, "data_first"),
                (DEEP_OUTPUT, "neutral_analyst"),
            ],
            fact_set,
        )
        assert results[0].final_score >= results[1].final_score


# =============================================================================
# Criterion 2 -- Agent with more hypotheses + weaknesses scores higher
# =============================================================================

class TestReasoningDepth:

    def test_deep_agent_scores_higher_than_shallow(self, engine, fact_set):
        deep_result    = engine.score(DEEP_OUTPUT,    fact_set, "neutral_analyst")
        shallow_result = engine.score(SHALLOW_OUTPUT, fact_set, "data_first")
        assert deep_result.final_score > shallow_result.final_score, (
            f"Deep ({deep_result.final_score:.3f}) should beat "
            f"Shallow ({shallow_result.final_score:.3f})"
        )

    def test_reasoning_depth_score_is_higher_for_deep(self, engine, fact_set):
        deep_result    = engine.score(DEEP_OUTPUT,    fact_set)
        shallow_result = engine.score(SHALLOW_OUTPUT, fact_set)
        assert deep_result.reasoning_depth_score > shallow_result.reasoning_depth_score

    def test_more_weaknesses_increases_depth_score(self, engine, fact_set):
        few_weaknesses  = make_output(acknowledged_weaknesses=["One concern"])
        many_weaknesses = make_output(acknowledged_weaknesses=[
            "No order book", "No on-chain data", "Manipulation not ruled out", "Insider info unknown"
        ])
        r_few  = engine.score(few_weaknesses,  fact_set)
        r_many = engine.score(many_weaknesses, fact_set)
        assert r_many.reasoning_depth_score > r_few.reasoning_depth_score

    def test_more_explanations_increases_depth_score(self, engine, fact_set):
        few  = make_output(possible_explanations=["Only one explanation"])
        many = make_output(possible_explanations=["E1", "E2", "E3", "E4"])
        r_few  = engine.score(few,  fact_set)
        r_many = engine.score(many, fact_set)
        assert r_many.reasoning_depth_score > r_few.reasoning_depth_score

    def test_depth_score_capped_at_1_for_very_rich_reasoning(self, engine, fact_set):
        very_deep = make_output(
            possible_explanations=["E1", "E2", "E3", "E4", "E5", "E6"],
            ranked_hypotheses=["H1", "H2", "H3", "H4", "H5", "H6"],
            acknowledged_weaknesses=["W1", "W2", "W3", "W4", "W5"],
        )
        result = engine.score(very_deep, fact_set)
        assert result.reasoning_depth_score <= 1.0


# =============================================================================
# Criterion 3 -- confidence=1.0 + shallow reasoning is penalised
# =============================================================================

class TestOverconfidencePenalty:

    def test_overconfident_agent_receives_penalty(self, engine, fact_set):
        result = engine.score(OVERCONFIDENT_OUTPUT, fact_set, "skeptic")
        assert result.overconfident is True
        assert result.overconfidence_penalty > 0.0

    def test_overconfidence_flag_is_true(self, engine, fact_set):
        result = engine.score(OVERCONFIDENT_OUTPUT, fact_set)
        assert result.overconfident is True

    def test_humble_agent_receives_no_penalty(self, engine, fact_set):
        humble = make_output(confidence_score=0.50)
        result = engine.score(humble, fact_set)
        assert result.overconfidence_penalty == 0.0
        assert result.overconfident is False

    def test_high_confidence_with_deep_reasoning_no_penalty(self, engine, fact_set):
        """High confidence is fine if reasoning depth is also high."""
        justified = make_output(
            possible_explanations=["E1", "E2", "E3", "E4"],
            ranked_hypotheses=["H1", "H2", "H3", "H4"],
            acknowledged_weaknesses=["W1", "W2", "W3"],
            confidence_score=0.85,
        )
        result = engine.score(justified, fact_set)
        assert result.overconfident is False
        assert result.overconfidence_penalty == 0.0

    def test_overconfident_agent_scores_lower_than_humble_deep_agent(self, engine, fact_set):
        result_oc    = engine.score(OVERCONFIDENT_OUTPUT, fact_set, "overconfident")
        result_deep  = engine.score(DEEP_OUTPUT,          fact_set, "deep")
        assert result_deep.final_score > result_oc.final_score


# =============================================================================
# Criterion 4 -- Adjusting weights changes scores
# =============================================================================

class TestWeightConfiguration:

    def test_changing_weights_changes_final_score(self, fact_set):
        """
        Create two engines with different weights.
        Verify the final scores differ for the same input.
        """
        import yaml, io
        from core.scoring_engine import CONFIG_PATH

        # Load real config and tweak it
        with open(CONFIG_PATH) as f:
            base_config = yaml.safe_load(f)

        # Engine A: standard weights
        engine_a = ScoringEngine()

        # Engine B: heavily weight fact_consistency, discount depth
        import tempfile, os
        from pathlib import Path
        tweaked = copy.deepcopy(base_config)
        tweaked["weights"]["fact_consistency"]  = 0.80
        tweaked["weights"]["reasoning_depth"]   = 0.10
        tweaked["weights"]["overconfidence_penalty"] = 0.10

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(tweaked, tmp)
            tmp_path = Path(tmp.name)

        try:
            engine_b = ScoringEngine(config_path=tmp_path)
            result_a = engine_a.score(DEEP_OUTPUT, fact_set, "agent")
            result_b = engine_b.score(DEEP_OUTPUT, fact_set, "agent")
            # The two engines must produce different scores for the same input
            assert result_a.final_score != result_b.final_score
        finally:
            os.unlink(tmp_path)

    def test_depth_weight_zero_eliminates_depth_contribution(self, fact_set):
        """If depth weight is 0, shallow and deep agent should have equal depth contribution."""
        import tempfile, os, yaml
        from pathlib import Path
        from core.scoring_engine import CONFIG_PATH

        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        cfg["weights"]["reasoning_depth"]        = 0.0
        cfg["weights"]["fact_consistency"]       = 0.80
        cfg["weights"]["overconfidence_penalty"] = 0.20

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(cfg, tmp)
            tmp_path = Path(tmp.name)

        try:
            engine = ScoringEngine(config_path=tmp_path)
            # Both agents have same extracted_facts so fact score = same
            same_facts = make_output(extracted_facts=["BTC volume up 3x in 60 minutes"])
            deep   = make_output(
                extracted_facts=["BTC volume up 3x in 60 minutes"],
                possible_explanations=["E1", "E2", "E3", "E4"],
                acknowledged_weaknesses=["W1", "W2", "W3", "W4"],
            )
            r_same  = engine.score(same_facts, fact_set)
            r_deep  = engine.score(deep,       fact_set)
            # With depth weight=0, depth doesn't change the score
            assert abs(r_same.final_score - r_deep.final_score) < 0.01
        finally:
            os.unlink(tmp_path)


# =============================================================================
# Sub-dimension direct tests
# =============================================================================

class TestSubDimensions:

    def test_fact_consistency_returns_0_to_1(self, engine, fact_set):
        output = make_output()
        score = engine._score_fact_consistency(output, fact_set)
        assert 0.0 <= score <= 1.0

    def test_reasoning_depth_returns_0_to_1(self, engine):
        output = make_output()
        score = engine._score_reasoning_depth(output)
        assert 0.0 <= score <= 1.0

    def test_empty_fact_set_returns_neutral_score(self, engine):
        output = make_output()
        score = engine._score_fact_consistency(output, [])
        assert score == 0.0

    def test_overconfidence_returns_tuple(self, engine):
        penalty, flag = engine._calc_overconfidence_penalty(1.0, 0.1)
        assert isinstance(penalty, float)
        assert isinstance(flag, bool)

    def test_summary_string_contains_agent_name(self, engine, fact_set):
        result = engine.score(DEEP_OUTPUT, fact_set, "neutral_analyst")
        assert "neutral_analyst" in result.summary()
