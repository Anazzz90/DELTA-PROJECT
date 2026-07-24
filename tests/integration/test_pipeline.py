"""
tests/integration/test_pipeline.py
=====================================
Checkpoint 6 -- Integration Tests for the Parallel Pipeline

Coverage (maps exactly to checkpoint test criteria):
  Test 1+2 -- All 3 agents return valid JSON; ran in parallel
  Test 3   -- One agent failing does not crash the pipeline
  Test 4   -- Each agent output is independently different
  Test 5   -- pytest tests/integration/test_pipeline.py -> all pass
  Test 6   -- sample_queries.json used as input fixtures

All LLM calls are mocked. No API credits needed.

Run:
    poetry run pytest tests/integration/test_pipeline.py -v
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.data_first import DataFirstAgent
from agents.neutral_analyst import NeutralAnalyst
from agents.skeptic import SkepticAgent
from core.pipeline import Pipeline, PipelineResult
from core.schemas import AgentOutput, AgentResult
from llm.router import LLMResponse


# =============================================================================
# Helpers
# =============================================================================

SAMPLE_QUERIES_PATH = Path(__file__).parent.parent / "fixtures" / "sample_queries.json"

def make_agent_output(main_driver: str, confidence: float = 0.7) -> AgentOutput:
    return AgentOutput(
        extracted_facts=["fact A", "fact B"],
        possible_explanations=["explanation 1", "explanation 2"],
        ranked_hypotheses=["hypothesis 1 (best fit)", "hypothesis 2"],
        main_driver=main_driver,
        confidence_score=confidence,
        acknowledged_weaknesses=["missing data point X"],
    )


def make_llm_response(content: str, agent_name: str = "test", model: str = "groq/llama-3.1-8b-instant") -> LLMResponse:
    return LLMResponse(
        agent_name=agent_name,
        model=model,
        content=content,
        prompt_tokens=100,
        completion_tokens=150,
        total_tokens=250,
        cost_usd=0.000050,
        latency_ms=500.0,
    )


def make_mock_pipeline(
    neutral_driver="Liquidation cascade drove forced buying",
    data_driver="Volume surge is the only fact-supported explanation",
    skeptic_driver="Insufficient evidence to rule out market manipulation",
    neutral_confidence=0.72,
    data_confidence=0.60,
    skeptic_confidence=0.45,
):
    """
    Creates a Pipeline where all 3 agents' LLM calls are mocked.
    Each agent returns a distinct main_driver to prove isolation.
    """
    neutral_output = make_agent_output(neutral_driver, neutral_confidence)
    data_output    = make_agent_output(data_driver,    data_confidence)
    skeptic_output = make_agent_output(skeptic_driver, skeptic_confidence)

    neutral_resp = make_llm_response(neutral_output.model_dump_json(), "neutral_analyst", "gpt-4o-mini")
    data_resp    = make_llm_response(data_output.model_dump_json(),    "data_first",      "groq/llama-3.1-8b-instant")
    skeptic_resp = make_llm_response(skeptic_output.model_dump_json(), "skeptic",          "groq/llama-3.3-70b-versatile")

    neutral = NeutralAnalyst()
    data    = DataFirstAgent()
    skeptic = SkepticAgent()

    neutral._router = type("MockRouter", (), {
        "call": lambda self, **kwargs: neutral_resp
    })()
    data._router = type("MockRouter", (), {
        "call": lambda self, **kwargs: data_resp
    })()
    skeptic._router = type("MockRouter", (), {
        "call": lambda self, **kwargs: skeptic_resp
    })()

    return Pipeline(agents=[neutral, data, skeptic])


QUESTION = "Why did BTC spike 8% in the last hour?"
FACTS    = ["BTC volume up 3x", "Derivatives liquidated", "No major news"]


# =============================================================================
# Test Criterion 1+2 -- All 3 agents return valid JSON, ran in parallel
# =============================================================================

class TestPipelineBasic:

    def test_pipeline_returns_pipeline_result(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        assert isinstance(result, PipelineResult)

    def test_all_three_agents_return_results(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        assert len(result.results) == 3

    def test_all_agents_succeed(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        assert result.agents_succeeded == 3
        assert result.agents_failed == 0

    def test_all_results_are_agent_result_type(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        for r in result.results:
            assert isinstance(r, AgentResult)

    def test_all_results_have_valid_output_schema(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        for r in result.results:
            assert r.success
            assert isinstance(r.output, AgentOutput)

    def test_total_latency_recorded(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        assert result.total_latency_ms > 0

    def test_parallel_execution_faster_than_sequential(self):
        """
        Simulate each agent taking ~0.3s. Sequential would take ~0.9s.
        Parallel should take ~0.3s. We verify it's < 0.8s.
        """
        import time as time_mod

        def slow_run(self, question, fact_set, domain_profile=None):
            time_mod.sleep(0.3)
            return AgentResult(
                agent_name=self.name,
                success=True,
                output=make_agent_output(f"{self.name} driver"),
                model="test",
                total_tokens=100,
                cost_usd=0.0,
                latency_ms=300.0,
            )

        neutral = NeutralAnalyst()
        data    = DataFirstAgent()
        skeptic = SkepticAgent()
        neutral.run = lambda *a, **kw: slow_run(neutral, *a, **kw)
        data.run    = lambda *a, **kw: slow_run(data,    *a, **kw)
        skeptic.run = lambda *a, **kw: slow_run(skeptic, *a, **kw)

        pipeline = Pipeline(agents=[neutral, data, skeptic])
        start = time.perf_counter()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        elapsed = time.perf_counter() - start

        # Sequential would take 0.9s; parallel should be well under 0.8s
        assert elapsed < 0.8, (
            f"Pipeline took {elapsed:.2f}s — agents may not be running in parallel"
        )
        assert result.agents_succeeded == 3


# =============================================================================
# Test Criterion 3 -- One failing agent does not crash the pipeline
# =============================================================================

class TestPipelineFaultTolerance:

    def test_one_failed_agent_pipeline_continues(self):
        """Skeptic's LLM call fails — neutral and data_first still complete."""
        neutral_output = make_agent_output("Liquidation cascade")
        data_output    = make_agent_output("Volume surge explanation")

        neutral_resp = make_llm_response(neutral_output.model_dump_json(), "neutral_analyst")
        data_resp    = make_llm_response(data_output.model_dump_json(),    "data_first")
        failed_resp  = LLMResponse(
            agent_name="skeptic",
            model="groq/llama-3.3-70b-versatile",
            content="",
            error="RateLimitError: quota exceeded",
        )

        neutral = NeutralAnalyst()
        data    = DataFirstAgent()
        skeptic = SkepticAgent()

        neutral._router = type("R", (), {"call": lambda self, **kw: neutral_resp})()
        data._router    = type("R", (), {"call": lambda self, **kw: data_resp})()
        skeptic._router = type("R", (), {"call": lambda self, **kw: failed_resp})()

        pipeline = Pipeline(agents=[neutral, data, skeptic])
        result = asyncio.run(pipeline.run(QUESTION, FACTS))

        assert result.agents_succeeded == 2
        assert result.agents_failed == 1
        assert len(result.successful_results) == 2
        assert len(result.failed_results) == 1
        assert result.failed_results[0].agent_name == "skeptic"

    def test_failed_agent_result_has_error_message(self):
        failed_resp = LLMResponse(
            agent_name="neutral_analyst",
            model="gpt-4o-mini",
            content="",
            error="TimeoutError: request timed out",
        )
        neutral = NeutralAnalyst()
        neutral._router = type("R", (), {"call": lambda self, **kw: failed_resp})()

        pipeline = Pipeline(agents=[neutral])
        result = asyncio.run(pipeline.run(QUESTION, FACTS))

        assert result.agents_failed == 1
        assert result.results[0].error is not None
        assert "TimeoutError" in result.results[0].error or "LLM call failed" in result.results[0].error

    def test_all_agents_fail_pipeline_still_returns(self):
        """Even if every agent fails, PipelineResult is returned, never an exception."""
        failed_resp = LLMResponse(
            agent_name="test",
            model="test",
            content="",
            error="total failure",
        )

        neutral = NeutralAnalyst()
        data    = DataFirstAgent()
        skeptic = SkepticAgent()
        for a in [neutral, data, skeptic]:
            a._router = type("R", (), {"call": lambda self, **kw: failed_resp})()

        pipeline = Pipeline(agents=[neutral, data, skeptic])
        result = asyncio.run(pipeline.run(QUESTION, FACTS))

        assert isinstance(result, PipelineResult)
        assert result.agents_succeeded == 0
        assert result.agents_failed == 3


# =============================================================================
# Test Criterion 4 -- Each agent output is independently different
# =============================================================================

class TestAgentIndependence:

    def test_each_agent_has_different_main_driver(self):
        pipeline = make_mock_pipeline(
            neutral_driver="Liquidation cascade drove forced buying",
            data_driver="Volume surge is the only fact-supported explanation",
            skeptic_driver="Manipulation cannot be ruled out without order book data",
        )
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        drivers = [r.output.main_driver for r in result.successful_results]
        # All 3 must be distinct
        assert len(set(drivers)) == 3, f"Agents returned duplicate drivers: {drivers}"

    def test_each_agent_has_different_confidence_scores(self):
        pipeline = make_mock_pipeline(
            neutral_confidence=0.72,
            data_confidence=0.60,
            skeptic_confidence=0.45,
        )
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        scores = [r.output.confidence_score for r in result.successful_results]
        assert len(set(scores)) == 3, f"Agents returned identical confidence scores: {scores}"

    def test_agent_names_are_distinct(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        names = [r.agent_name for r in result.results]
        assert len(set(names)) == 3


# =============================================================================
# Test Criterion 6 -- sample_queries.json used as input fixtures
# =============================================================================

class TestSampleQueryFixtures:

    def test_sample_queries_file_exists(self):
        assert SAMPLE_QUERIES_PATH.exists(), (
            f"sample_queries.json not found at {SAMPLE_QUERIES_PATH}"
        )

    def test_sample_queries_is_valid_json(self):
        with open(SAMPLE_QUERIES_PATH) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_sample_queries_have_required_fields(self):
        with open(SAMPLE_QUERIES_PATH) as f:
            queries = json.load(f)
        for q in queries:
            assert "id"       in q, f"Missing 'id' in: {q}"
            assert "question" in q, f"Missing 'question' in: {q}"
            assert "fact_set" in q, f"Missing 'fact_set' in: {q}"
            assert isinstance(q["fact_set"], list)
            assert len(q["fact_set"]) >= 2, "fact_set must have at least 2 facts"

    @pytest.mark.parametrize("query_id", ["btc_spike", "eth_drop", "startup_churn", "factory_downtime"])
    def test_pipeline_runs_on_each_sample_query(self, query_id):
        """Pipeline runs cleanly on every fixture without crashing."""
        with open(SAMPLE_QUERIES_PATH) as f:
            queries = {q["id"]: q for q in json.load(f)}

        q = queries[query_id]
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(
            question=q["question"],
            fact_set=q["fact_set"],
            domain_profile=q.get("domain_profile"),
        ))

        assert isinstance(result, PipelineResult)
        assert result.agents_succeeded >= 1
        assert result.question == q["question"]


# =============================================================================
# Pipeline Metadata
# =============================================================================

class TestPipelineMetadata:

    def test_success_rate_is_correct(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        assert result.success_rate == 1.0

    def test_total_cost_is_sum_of_agent_costs(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        # Each mock agent returns cost_usd=0.000050; 3 agents = 0.000150
        assert abs(result.total_cost_usd() - 0.000150) < 0.000001

    def test_summary_output_contains_agent_names(self):
        pipeline = make_mock_pipeline()
        result = asyncio.run(pipeline.run(QUESTION, FACTS))
        summary = result.summary()
        assert "neutral_analyst" in summary
        assert "data_first"      in summary
        assert "skeptic"         in summary

