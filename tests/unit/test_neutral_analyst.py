"""
tests/unit/test_neutral_analyst.py
=====================================
Checkpoint 5 — Unit Tests for the Neutral Analyst Agent

Coverage (maps exactly to checkpoint test criteria):
  ✅ Criterion 1+2 — Agent returns valid JSON matching the 6-field schema
  ✅ Criterion 3   — confidence_score is between 0.0 and 1.0
  ✅ Criterion 4   — main_driver is a non-empty string
  ✅ Criterion 5   — acknowledged_weaknesses is non-empty (humility note present)
  ✅ Criterion 6   — Pydantic validation catches malformed responses

These tests mock the LLM call so no API keys or credits are needed.
The agent logic (JSON parsing, Pydantic validation, BaseAgent pipeline) 
is tested completely in isolation.

Run:
    poetry run pytest tests/unit/test_neutral_analyst.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from core.schemas import AgentOutput, AgentResult
from agents.neutral_analyst import NeutralAnalyst
from llm.router import LLMResponse


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def agent() -> NeutralAnalyst:
    return NeutralAnalyst()


@pytest.fixture
def sample_question() -> str:
    return "Why did BTC spike 8% in the last hour?"


@pytest.fixture
def sample_facts() -> list[str]:
    return [
        "BTC volume up 3x in 60 minutes",
        "Large derivatives positions liquidated",
        "No major news reported in that window",
    ]


@pytest.fixture
def valid_output_dict() -> dict:
    """A perfectly valid Delta-First protocol JSON output."""
    return {
        "extracted_facts": [
            "BTC volume up 3x in 60 minutes",
            "Large derivatives positions liquidated",
            "No major news reported",
        ],
        "possible_explanations": [
            "Short squeeze triggered by derivatives liquidations",
            "Coordinated institutional accumulation",
            "Algorithm-driven momentum buying",
        ],
        "ranked_hypotheses": [
            "Derivatives liquidations caused cascade buying (strongest fact support)",
            "Institutional accumulation (plausible but lacks direct evidence)",
            "Algorithm momentum (possible but secondary)",
        ],
        "main_driver": "Mass liquidation of short derivatives positions created a cascade of forced buying",
        "confidence_score": 0.72,
        "acknowledged_weaknesses": [
            "No order book data available to confirm liquidation cascade",
            "Cannot rule out coordinated institutional buying without on-chain data",
        ],
    }


def make_mock_llm_response(content: str) -> LLMResponse:
    """Create a mock LLMResponse with the given content string."""
    return LLMResponse(
        agent_name="neutral_analyst",
        model="groq/llama-3.3-70b-versatile",
        content=content,
        prompt_tokens=150,
        completion_tokens=200,
        total_tokens=350,
        cost_usd=0.000245,
        latency_ms=820.5,
    )


# =============================================================================
# Criterion 1+2 — Returns valid JSON matching the 6-field schema
# =============================================================================

class TestValidOutput:

    def test_returns_agent_result_type(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert isinstance(result, AgentResult)

    def test_success_is_true_on_valid_output(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.success is True
        assert result.error is None

    def test_output_matches_agent_output_schema(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert isinstance(result.output, AgentOutput)

    def test_all_six_fields_present(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        out = result.output
        assert out.extracted_facts
        assert out.possible_explanations
        assert out.ranked_hypotheses
        assert out.main_driver
        assert out.confidence_score is not None
        assert out.acknowledged_weaknesses

    def test_strips_markdown_code_fences(self, agent, sample_question, sample_facts, valid_output_dict):
        """LLMs sometimes wrap JSON in ```json ... ``` — must be handled."""
        json_str = json.dumps(valid_output_dict)
        wrapped = f"```json\n{json_str}\n```"
        mock_resp = make_mock_llm_response(wrapped)
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.success is True

    def test_llm_metadata_propagated_to_result(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.total_tokens == 350
        assert result.cost_usd == 0.000245
        assert result.latency_ms == 820.5
        assert result.model == "groq/llama-3.3-70b-versatile"


# =============================================================================
# Criterion 3 — confidence_score is between 0.0 and 1.0
# =============================================================================

class TestConfidenceScore:

    def test_confidence_score_is_float(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert isinstance(result.output.confidence_score, float)

    def test_confidence_score_within_range(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        score = result.output.confidence_score
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_boundary_confidence_scores_accepted(self, agent, sample_question, sample_facts, valid_output_dict, score):
        data = {**valid_output_dict, "confidence_score": score}
        mock_resp = make_mock_llm_response(json.dumps(data))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.success is True
        assert result.output.confidence_score == score


# =============================================================================
# Criterion 4 — main_driver is a non-empty string
# =============================================================================

class TestMainDriver:

    def test_main_driver_is_string(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert isinstance(result.output.main_driver, str)

    def test_main_driver_is_non_empty(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert len(result.output.main_driver.strip()) > 0


# =============================================================================
# Criterion 5 — acknowledged_weaknesses is non-empty
# =============================================================================

class TestAcknowledgedWeaknesses:

    def test_acknowledged_weaknesses_is_list(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert isinstance(result.output.acknowledged_weaknesses, list)

    def test_acknowledged_weaknesses_is_non_empty(self, agent, sample_question, sample_facts, valid_output_dict):
        mock_resp = make_mock_llm_response(json.dumps(valid_output_dict))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert len(result.output.acknowledged_weaknesses) >= 1


# =============================================================================
# Criterion 6 — Pydantic validation catches malformed responses
# =============================================================================

class TestSchemaValidation:

    def test_missing_field_causes_failure(self, agent, sample_question, sample_facts, valid_output_dict):
        """Remove a required field — should fail gracefully, not crash."""
        data = {k: v for k, v in valid_output_dict.items() if k != "main_driver"}
        mock_resp = make_mock_llm_response(json.dumps(data))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.success is False
        assert "validation" in result.error.lower() or "main_driver" in result.error.lower()

    def test_invalid_confidence_score_causes_failure(self, agent, sample_question, sample_facts, valid_output_dict):
        """confidence_score > 1.0 — Pydantic must reject it."""
        data = {**valid_output_dict, "confidence_score": 1.5}
        mock_resp = make_mock_llm_response(json.dumps(data))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.success is False

    def test_empty_main_driver_causes_failure(self, agent, sample_question, sample_facts, valid_output_dict):
        """Blank main_driver should fail Pydantic validation."""
        data = {**valid_output_dict, "main_driver": "   "}
        mock_resp = make_mock_llm_response(json.dumps(data))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.success is False

    def test_empty_weaknesses_list_causes_failure(self, agent, sample_question, sample_facts, valid_output_dict):
        """acknowledged_weaknesses must have at least 1 entry."""
        data = {**valid_output_dict, "acknowledged_weaknesses": []}
        mock_resp = make_mock_llm_response(json.dumps(data))
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.success is False

    def test_invalid_json_causes_failure(self, agent, sample_question, sample_facts):
        """If LLM returns non-JSON prose, it must fail gracefully."""
        mock_resp = make_mock_llm_response("I cannot answer this question.")
        with patch.object(agent._router, "call", return_value=mock_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.success is False
        assert result.error is not None

    def test_llm_failure_propagates_to_result(self, agent, sample_question, sample_facts):
        """If the LLM call itself fails, AgentResult must reflect that."""
        failed_resp = LLMResponse(
            agent_name="neutral_analyst",
            model="groq/llama-3.3-70b-versatile",
            content="",
            error="RateLimitError: quota exceeded",
        )
        with patch.object(agent._router, "call", return_value=failed_resp):
            result = agent.run(sample_question, sample_facts)
        assert result.success is False
        assert "LLM call failed" in result.error


# =============================================================================
# Agent Identity
# =============================================================================

class TestAgentIdentity:

    def test_name_is_neutral_analyst(self, agent):
        assert agent.name == "neutral_analyst"

    def test_description_is_non_empty(self, agent):
        assert agent.description
        assert len(agent.description) > 10
