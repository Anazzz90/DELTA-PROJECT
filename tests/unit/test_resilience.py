"""
tests/unit/test_resilience.py
================================
Checkpoint 16 -- Tenacity Resilience Layer (Full)

Verifies llm/resilience.py's retry + circuit-breaker behavior directly,
mocking litellm.completion so no real network calls are made.

Run:
    poetry run pytest tests/unit/test_resilience.py -v
"""

import time
from unittest.mock import MagicMock

import litellm
import pytest

from llm.resilience import call_with_retry


MESSAGES = [{"role": "user", "content": "hi"}]


class TestRetryBehavior:

    def test_succeeds_immediately_with_no_failures(self, monkeypatch):
        mock_completion = MagicMock(return_value="ok-response")
        monkeypatch.setattr(litellm, "completion", mock_completion)

        result = call_with_retry("some-model", MESSAGES, 0.3, 30)

        assert result == "ok-response"
        assert mock_completion.call_count == 1

    def test_recovers_after_transient_failures(self, monkeypatch):
        mock_completion = MagicMock(
            side_effect=[
                litellm.exceptions.RateLimitError("rate limited", llm_provider="groq", model="x"),
                litellm.exceptions.Timeout("timed out", model="x", llm_provider="groq"),
                "ok-response",
            ]
        )
        monkeypatch.setattr(litellm, "completion", mock_completion)

        result = call_with_retry("some-model", MESSAGES, 0.3, 30)

        assert result == "ok-response"
        assert mock_completion.call_count == 3

    def test_non_retryable_error_fails_immediately(self, monkeypatch):
        mock_completion = MagicMock(
            side_effect=litellm.exceptions.AuthenticationError(
                "bad key", llm_provider="groq", model="x"
            )
        )
        monkeypatch.setattr(litellm, "completion", mock_completion)

        with pytest.raises(litellm.exceptions.AuthenticationError):
            call_with_retry("some-model", MESSAGES, 0.3, 30)

        assert mock_completion.call_count == 1  # not retried

    def test_circuit_breaker_trips_after_3_retries(self, monkeypatch):
        """4 total attempts (1 initial + 3 retries), then the original exception is raised."""
        mock_completion = MagicMock(
            side_effect=litellm.exceptions.RateLimitError(
                "always rate limited", llm_provider="groq", model="x"
            )
        )
        monkeypatch.setattr(litellm, "completion", mock_completion)

        with pytest.raises(litellm.exceptions.RateLimitError):
            call_with_retry("some-model", MESSAGES, 0.3, 30)

        assert mock_completion.call_count == 4

    @pytest.mark.slow
    def test_retry_backoff_schedule_is_approximately_2_4_8_seconds(self, monkeypatch):
        """Checkpoint 16 criterion: retry timing confirmed ~2s -> 4s -> 8s (real wall clock)."""
        mock_completion = MagicMock(
            side_effect=litellm.exceptions.RateLimitError(
                "always rate limited", llm_provider="groq", model="x"
            )
        )
        monkeypatch.setattr(litellm, "completion", mock_completion)

        start = time.monotonic()
        with pytest.raises(litellm.exceptions.RateLimitError):
            call_with_retry("some-model", MESSAGES, 0.3, 30)
        elapsed = time.monotonic() - start

        # 3 waits of 2s + 4s + 8s = 14s total, allow generous scheduling slack
        assert 13.0 <= elapsed <= 20.0
