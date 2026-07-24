"""
tests/unit/test_tracer.py
============================
Checkpoint 17 -- LangFuse Observability

No real LangFuse network calls are made — the client construction and
generation logging are exercised against mocks/monkeypatched settings.

Run:
    poetry run pytest tests/unit/test_tracer.py -v
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from config.settings import settings
from observability import tracer


@pytest.fixture(autouse=True)
def _reset_tracer_state():
    """Every test starts from a clean, unconfigured, zero-cost state."""
    tracer.reset_configuration()
    tracer.reset_session_cost()
    yield
    tracer.reset_configuration()
    tracer.reset_session_cost()


class TestConfigureLangfuse:

    def test_disabled_when_no_keys_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "langfuse_public_key", "")
        monkeypatch.setattr(settings, "langfuse_secret_key", "")

        result = tracer.configure_langfuse()

        assert result is False
        assert tracer.is_enabled() is False

    def test_enabled_when_client_constructs_successfully(self, monkeypatch):
        monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
        monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
        monkeypatch.setattr(settings, "langfuse_host", "https://cloud.langfuse.com")

        result = tracer.configure_langfuse()

        assert result is True
        assert tracer.is_enabled() is True

    def test_never_raises_even_if_client_construction_fails(self, monkeypatch):
        """Non-blocking guarantee (criterion #5): a broken tracing setup can't crash the app."""
        monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
        monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")

        with patch("langfuse.Langfuse", side_effect=RuntimeError("simulated init failure")):
            result = tracer.configure_langfuse()  # must not raise

        assert result is False
        assert tracer.is_enabled() is False

    def test_second_call_is_idempotent_noop(self, monkeypatch):
        monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
        monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")

        first = tracer.configure_langfuse()
        second = tracer.configure_langfuse()

        assert first is True
        assert second is True


class TestLogGeneration:

    def test_noop_when_tracing_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "langfuse_public_key", "")
        monkeypatch.setattr(settings, "langfuse_secret_key", "")
        tracer.configure_langfuse()

        # Should not raise even though no client exists.
        tracer.log_generation(
            "trace-1", "neutral_analyst", "test-model",
            [{"role": "user", "content": "hi"}], "response text",
        )

    def test_calls_start_observation_and_end_when_enabled(self, monkeypatch):
        monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
        monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")

        fake_generation = MagicMock()
        fake_client = MagicMock()
        fake_client.start_observation.return_value = fake_generation

        with patch("langfuse.Langfuse", return_value=fake_client):
            tracer.configure_langfuse()
            tracer.log_generation(
                "trace-1", "neutral_analyst", "test-model",
                [{"role": "user", "content": "hi"}], "response text",
                prompt_tokens=10, completion_tokens=20, total_tokens=30,
                cost_usd=0.001, latency_ms=500.0,
            )

        assert fake_client.start_observation.called
        call_kwargs = fake_client.start_observation.call_args.kwargs
        assert call_kwargs["name"] == "neutral_analyst"
        assert call_kwargs["as_type"] == "generation"
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["output"] == "response text"
        assert call_kwargs["usage_details"] == {"input": 10, "output": 20, "total": 30}
        assert call_kwargs["cost_details"] == {"total": 0.001}
        fake_generation.end.assert_called_once()

    def test_failed_call_logged_with_error_level(self, monkeypatch):
        monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
        monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")

        fake_generation = MagicMock()
        fake_client = MagicMock()
        fake_client.start_observation.return_value = fake_generation

        with patch("langfuse.Langfuse", return_value=fake_client):
            tracer.configure_langfuse()
            tracer.log_generation(
                "trace-1", "skeptic", "test-model",
                [{"role": "user", "content": "hi"}], "",
                error="RateLimitError: quota exceeded",
            )

        call_kwargs = fake_client.start_observation.call_args.kwargs
        assert call_kwargs["level"] == "ERROR"
        assert call_kwargs["status_message"] == "RateLimitError: quota exceeded"
        assert call_kwargs["output"] is None

    def test_logging_error_never_propagates(self, monkeypatch):
        """A LangFuse-side failure mid-logging must never break the caller."""
        monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
        monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")

        fake_client = MagicMock()
        fake_client.start_observation.side_effect = RuntimeError("network blip")

        with patch("langfuse.Langfuse", return_value=fake_client):
            tracer.configure_langfuse()
            tracer.log_generation(
                "trace-1", "neutral_analyst", "test-model",
                [{"role": "user", "content": "hi"}], "response text",
            )  # must not raise


class TestTraceHelpers:

    def test_new_trace_id_is_unique(self):
        ids = {tracer.new_trace_id() for _ in range(100)}
        assert len(ids) == 100


class TestCostAlerts:

    def test_no_warning_when_under_thresholds(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "per_query_cost_alert_usd", 0.10)
        monkeypatch.setattr(settings, "daily_cost_alert_usd", 5.00)

        with caplog.at_level(logging.WARNING, logger="observability.tracer"):
            tracer.check_cost_alerts(0.01)

        assert "Cost alert" not in caplog.text

    def test_warns_when_single_query_exceeds_per_query_threshold(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "per_query_cost_alert_usd", 0.10)
        monkeypatch.setattr(settings, "daily_cost_alert_usd", 5.00)

        with caplog.at_level(logging.WARNING, logger="observability.tracer"):
            tracer.check_cost_alerts(0.50)

        assert "per-query threshold" in caplog.text

    def test_warns_when_running_total_exceeds_daily_threshold(self, monkeypatch, caplog):
        monkeypatch.setattr(settings, "per_query_cost_alert_usd", 10.00)  # avoid per-query trip
        monkeypatch.setattr(settings, "daily_cost_alert_usd", 1.00)

        with caplog.at_level(logging.WARNING, logger="observability.tracer"):
            tracer.check_cost_alerts(0.60)
            assert "daily threshold" not in caplog.text
            tracer.check_cost_alerts(0.60)  # running total now 1.20 > 1.00

        assert "daily threshold" in caplog.text

    def test_session_total_accumulates(self, monkeypatch):
        monkeypatch.setattr(settings, "per_query_cost_alert_usd", 10.00)
        monkeypatch.setattr(settings, "daily_cost_alert_usd", 10.00)

        tracer.check_cost_alerts(0.10)
        tracer.check_cost_alerts(0.20)

        assert tracer.session_total_cost_usd() == pytest.approx(0.30)
