"""
tests/unit/test_performance_tracker.py
=========================================
Checkpoint 21 -- Agent Performance Tracker

HistoryStore.get_agent_performance_stats() is mocked here (already
covered directly by tests/unit/test_storage.py::TestAgentPerformanceStats)
so these tests focus purely on the multiplier math.

Run:
    poetry run pytest tests/unit/test_performance_tracker.py -v
"""

from unittest.mock import AsyncMock, patch

import pytest

from core.scoring_engine import AgentPerformanceTracker


def stats(queries_run: int, accuracy_rate: float) -> dict:
    return {
        "queries_run": queries_run,
        "success_rate": 1.0,
        "avg_confidence": 0.75,
        "avg_final_score": 0.7,
        "flagged_count": round(queries_run * (1 - accuracy_rate)),
        "accuracy_rate": accuracy_rate,
    }


class TestWeightMultiplier:

    async def test_no_adjustment_below_min_query_threshold(self):
        tracker = AgentPerformanceTracker()
        # accuracy_rate=0.0 (100% overconfident) but only 2 queries -- not enough history
        mock_stats = stats(queries_run=2, accuracy_rate=0.0)
        with patch("memory.history.HistoryStore.get_agent_performance_stats", new_callable=AsyncMock, return_value=mock_stats):
            multiplier = await tracker.get_weight_multiplier("contrarian")
        assert multiplier == 1.0

    async def test_no_adjustment_when_well_calibrated(self):
        tracker = AgentPerformanceTracker()
        mock_stats = stats(queries_run=10, accuracy_rate=0.9)  # only flagged 10% of the time
        with patch("memory.history.HistoryStore.get_agent_performance_stats", new_callable=AsyncMock, return_value=mock_stats):
            multiplier = await tracker.get_weight_multiplier("neutral_analyst")
        assert multiplier == 1.0

    async def test_reduced_when_consistently_overconfident(self):
        tracker = AgentPerformanceTracker()
        # accuracy_rate=0.1 -> overconfidence_rate=0.9, well past the 0.5 threshold
        mock_stats = stats(queries_run=10, accuracy_rate=0.1)
        with patch("memory.history.HistoryStore.get_agent_performance_stats", new_callable=AsyncMock, return_value=mock_stats):
            multiplier = await tracker.get_weight_multiplier("contrarian")
        assert multiplier < 1.0
        assert multiplier >= AgentPerformanceTracker.MIN_WEIGHT_MULTIPLIER

    async def test_never_drops_below_floor(self):
        tracker = AgentPerformanceTracker()
        mock_stats = stats(queries_run=20, accuracy_rate=0.0)  # 100% overconfident
        with patch("memory.history.HistoryStore.get_agent_performance_stats", new_callable=AsyncMock, return_value=mock_stats):
            multiplier = await tracker.get_weight_multiplier("contrarian")
        assert multiplier == AgentPerformanceTracker.MIN_WEIGHT_MULTIPLIER

    async def test_get_profile_returns_full_breakdown(self):
        tracker = AgentPerformanceTracker()
        mock_stats = stats(queries_run=10, accuracy_rate=0.1)
        with patch("memory.history.HistoryStore.get_agent_performance_stats", new_callable=AsyncMock, return_value=mock_stats):
            profile = await tracker.get_profile("contrarian")

        assert profile.agent_name == "contrarian"
        assert profile.queries_run == 10
        assert profile.accuracy_rate == 0.1
        assert profile.weight_multiplier < 1.0

    async def test_multiplier_scales_between_threshold_and_floor(self):
        """Higher overconfidence rate -> lower (or equal) multiplier."""
        tracker = AgentPerformanceTracker()

        with patch("memory.history.HistoryStore.get_agent_performance_stats", new_callable=AsyncMock,
                   return_value=stats(queries_run=10, accuracy_rate=0.45)):
            mild = await tracker.get_weight_multiplier("agent_a")

        with patch("memory.history.HistoryStore.get_agent_performance_stats", new_callable=AsyncMock,
                   return_value=stats(queries_run=10, accuracy_rate=0.05)):
            severe = await tracker.get_weight_multiplier("agent_b")

        assert severe <= mild
