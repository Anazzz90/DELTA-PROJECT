"""
core/domain_profiles.py
==========================
Checkpoint 22 — Domain Profiles

Loads config/domain_profiles.yaml and turns each profile's per-agent
weight proportions into a *multiplier* the pipeline can apply to a
ScoringResult.final_score — the same mechanism Checkpoint 21's
AgentPerformanceTracker uses, so core/aggregator.py and
core/conflict_detector.py (which already weight by final_score) need
zero changes to pick this up too.

Multiplier derivation: a profile's agent_weights are proportions that
sum to ~1.0 across its 5 predefined agents. For a given query, only the
*selected* agents' weights are renormalized to sum to 1.0 (a user might
not run all 5), then compared against a uniform baseline (1 / n_selected)
— an agent weighted exactly at the uniform share gets multiplier 1.0,
above it gets boosted, below it gets reduced. This is why
"intraday_trading" (contrarian/skeptic weighted 0.25 each vs 0.20/0.20/0.10
for the others) boosts contrarian and skeptic, while "general" (roughly
uniform, neutral_analyst slightly ahead at 0.25) gives neutral_analyst the
highest multiplier of the five — matching this checkpoint's test criteria.

Usage:
    from core.domain_profiles import (
        get_profile, get_agent_weight_multiplier, is_valid_profile, list_profiles,
    )

    is_valid_profile("intraday_trading")   # True
    is_valid_profile("not_a_real_profile") # False

    multiplier = get_agent_weight_multiplier("intraday_trading", "contrarian", selected_agents)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "domain_profiles.yaml"
DEFAULT_PROFILE = "general"

_cache: Optional[dict] = None


def _load_profiles() -> dict:
    global _cache
    if _cache is None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"domain_profiles.yaml not found at {CONFIG_PATH}")
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _cache = data.get("profiles", {})
    return _cache


def list_profiles() -> list[str]:
    """All valid domain profile names."""
    return list(_load_profiles().keys())


def is_valid_profile(name: Optional[str]) -> bool:
    """None/empty is valid — it means "use the default profile"."""
    if not name:
        return True
    return name in _load_profiles()


def get_profile(name: Optional[str]) -> dict:
    """Returns the named profile's config dict, falling back to DEFAULT_PROFILE."""
    profiles = _load_profiles()
    if name and name in profiles:
        return profiles[name]
    return profiles[DEFAULT_PROFILE]


def get_confidence_threshold(name: Optional[str]) -> float:
    return get_profile(name).get("confidence_threshold", 0.65)


def get_agent_weight_multiplier(
    domain_profile: Optional[str],
    agent_name: str,
    selected_agents: list[str],
) -> float:
    """
    The multiplier to apply to `agent_name`'s current-query final_score,
    given the active domain profile and which agents are actually running.
    Returns 1.0 (no adjustment) if the agent isn't in the profile's
    agent_weights or if selected_agents is empty.
    """
    if not selected_agents:
        return 1.0

    profile = get_profile(domain_profile)
    weights = profile.get("agent_weights", {})

    uniform_share = 1.0 / len(selected_agents)
    relevant = {a: weights.get(a, uniform_share) for a in selected_agents}
    total = sum(relevant.values())
    if total <= 0:
        return 1.0

    normalized_share = relevant.get(agent_name, uniform_share) / total
    return round(normalized_share / uniform_share, 4) if uniform_share > 0 else 1.0
