"""
tests/unit/test_domain_profiles.py
=====================================
Checkpoint 22 -- Domain Profiles

Pure config/math tests -- no DB, no LLM calls.

Run:
    poetry run pytest tests/unit/test_domain_profiles.py -v
"""

from core.domain_profiles import (
    get_agent_weight_multiplier,
    get_confidence_threshold,
    get_profile,
    is_valid_profile,
    list_profiles,
)

ALL_AGENTS = ["neutral_analyst", "data_first", "skeptic", "contrarian", "intuition"]


class TestProfileLookup:

    def test_list_profiles_includes_all_three(self):
        profiles = list_profiles()
        assert set(profiles) == {"intraday_trading", "macro_analysis", "general"}

    def test_is_valid_profile_true_for_known_names(self):
        assert is_valid_profile("intraday_trading") is True
        assert is_valid_profile("macro_analysis") is True
        assert is_valid_profile("general") is True

    def test_is_valid_profile_false_for_unknown_name(self):
        assert is_valid_profile("not_a_real_profile") is False

    def test_is_valid_profile_true_for_none_or_empty(self):
        """None/empty means 'use the default' -- not an error."""
        assert is_valid_profile(None) is True
        assert is_valid_profile("") is True

    def test_get_profile_falls_back_to_general_for_unknown(self):
        profile = get_profile("not_a_real_profile")
        assert profile == get_profile("general")

    def test_get_profile_falls_back_to_general_for_none(self):
        assert get_profile(None) == get_profile("general")


class TestConfidenceThresholds:

    def test_intraday_and_general_have_different_thresholds(self):
        """Criterion 1: submitting the same query on different profiles applies different thresholds."""
        intraday = get_confidence_threshold("intraday_trading")
        general = get_confidence_threshold("general")
        assert intraday != general
        assert intraday == 0.60
        assert general == 0.65

    def test_macro_analysis_threshold_is_highest(self):
        assert get_confidence_threshold("macro_analysis") == 0.70


class TestAgentWeightMultipliers:

    def test_intraday_trading_boosts_contrarian_and_skeptic(self):
        """Criterion 2: intraday_trading -> Contrarian + Skeptic weighted higher."""
        multipliers = {
            a: get_agent_weight_multiplier("intraday_trading", a, ALL_AGENTS)
            for a in ALL_AGENTS
        }
        others = [multipliers["neutral_analyst"], multipliers["data_first"], multipliers["intuition"]]
        assert multipliers["contrarian"] > max(others)
        assert multipliers["skeptic"] > max(others)

    def test_general_profile_boosts_neutral_analyst_highest(self):
        """Criterion 3: general profile -> Neutral Analyst weighted highest."""
        multipliers = {
            a: get_agent_weight_multiplier("general", a, ALL_AGENTS)
            for a in ALL_AGENTS
        }
        assert multipliers["neutral_analyst"] == max(multipliers.values())

    def test_uniform_weighting_gives_multiplier_one(self):
        """An agent weighted exactly at the uniform share gets no adjustment."""
        # In "general", data_first and skeptic are both at 0.20 = uniform share for 5 agents.
        assert get_agent_weight_multiplier("general", "data_first", ALL_AGENTS) == 1.0

    def test_unknown_profile_falls_back_to_general_weights(self):
        assert (
            get_agent_weight_multiplier("not_a_real_profile", "neutral_analyst", ALL_AGENTS)
            == get_agent_weight_multiplier("general", "neutral_analyst", ALL_AGENTS)
        )

    def test_empty_selected_agents_returns_one(self):
        assert get_agent_weight_multiplier("intraday_trading", "contrarian", []) == 1.0

    def test_multipliers_renormalize_for_a_subset_of_agents(self):
        """If only 2 agents are actually running, weights renormalize among just those two."""
        subset = ["contrarian", "intuition"]
        multiplier = get_agent_weight_multiplier("intraday_trading", "contrarian", subset)
        # contrarian (0.25) vs intuition (0.10) within just this pair -> contrarian should
        # still come out ahead of a uniform 1.0 baseline.
        assert multiplier > 1.0

    def test_agent_not_in_profile_weights_gets_no_adjustment_when_alone(self):
        multiplier = get_agent_weight_multiplier("intraday_trading", "unknown_agent", ["unknown_agent"])
        assert multiplier == 1.0
