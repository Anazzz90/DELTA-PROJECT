"""
core/scoring_engine.py
========================
Checkpoint 7 — Weighted Scoring Engine

Evaluates the quality of each agent's AgentOutput using deterministic,
configurable math. No LLM calls — purely Python.

Scoring Dimensions:
    1. Fact Consistency (weight: 0.40)
       Measures how well the agent used the provided facts.
       - extracted_facts overlapping with input fact_set = good
       - More overlap = higher score

    2. Reasoning Depth (weight: 0.40)
       Measures how thoroughly the agent reasoned.
       - Number of possible explanations (more = better)
       - Number of ranked hypotheses (more = better)
       - Number of acknowledged weaknesses (more = better)

    3. Overconfidence Penalty (weight: 0.20)
       Detects "false certainty": when an agent claims high confidence
       but its reasoning is shallow.
       - High confidence + deep reasoning   = no penalty
       - High confidence + shallow reasoning = penalty applied

All weights and thresholds are read from config/scoring_weights.yaml,
so they can be tuned without touching this code.

Usage:
    from core.scoring_engine import ScoringEngine
    from core.schemas import AgentOutput

    engine = ScoringEngine()
    result = engine.score(output, fact_set=["BTC volume up 3x", ...])
    print(result.final_score)         # e.g. 0.74
    print(result.overconfident)       # False
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from core.schemas import AgentOutput

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "scoring_weights.yaml"


# =============================================================================
# Scoring Result — returned for every agent scored
# =============================================================================

@dataclass
class ScoringResult:
    """
    The complete scoring breakdown for one agent's output.
    All raw dimension scores are on a 0.0-1.0 scale before weighting.
    final_score is the weighted sum, clamped to 0.0-1.0.
    """
    agent_name:              str
    fact_consistency_score:  float   # Raw 0.0-1.0 before weight
    reasoning_depth_score:   float   # Raw 0.0-1.0 before weight
    overconfidence_penalty:  float   # Amount DEDUCTED (0.0 = no penalty)
    final_score:             float   # Weighted final score 0.0-1.0
    overconfident:           bool    # True if penalty was applied
    breakdown:               dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"  Agent   : {self.agent_name}",
            f"  Final   : {self.final_score:.3f}",
            f"  Fact    : {self.fact_consistency_score:.3f} (raw)",
            f"  Depth   : {self.reasoning_depth_score:.3f} (raw)",
            f"  OC Pen  : -{self.overconfidence_penalty:.3f}",
            f"  OC Flag : {'YES' if self.overconfident else 'no'}",
        ]
        return "\n".join(lines)


# =============================================================================
# Scoring Engine
# =============================================================================

class ScoringEngine:
    """
    Deterministic scoring engine for DMARS agents.

    Reads weights and thresholds from config/scoring_weights.yaml.
    No LLM calls. Fully testable offline.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config = self._load_config(config_path or CONFIG_PATH)
        self._weights     = self._config["weights"]
        self._depth_cfg   = self._config["depth_thresholds"]
        self._oc_cfg      = self._config["overconfidence"]
        logger.debug(f"ScoringEngine loaded config: {self._weights}")

    # =========================================================================
    # Public API
    # =========================================================================

    def score(
        self,
        output: AgentOutput,
        fact_set: list[str],
        agent_name: str = "unknown",
    ) -> ScoringResult:
        """
        Score a single agent's output.

        Args:
            output:     The validated AgentOutput from the agent.
            fact_set:   The original fact set provided to the pipeline.
            agent_name: Agent name for result labeling.

        Returns:
            ScoringResult with all dimension scores and final_score.
        """
        # 1. Fact Consistency
        fact_score = self._score_fact_consistency(output, fact_set)

        # 2. Reasoning Depth
        depth_score = self._score_reasoning_depth(output)

        # 3. Overconfidence Penalty
        penalty, overconfident = self._calc_overconfidence_penalty(
            output.confidence_score, depth_score
        )

        # Weighted final score
        w = self._weights
        raw = (
            fact_score  * w["fact_consistency"]
            + depth_score * w["reasoning_depth"]
            - penalty     * w["overconfidence_penalty"]
        )
        final = round(max(0.0, min(1.0, raw)), 4)

        result = ScoringResult(
            agent_name=agent_name,
            fact_consistency_score=round(fact_score, 4),
            reasoning_depth_score=round(depth_score, 4),
            overconfidence_penalty=round(penalty, 4),
            final_score=final,
            overconfident=overconfident,
            breakdown={
                "fact_score":    fact_score,
                "depth_score":   depth_score,
                "oc_penalty":    penalty,
                "weights":       dict(w),
            },
        )
        logger.info(
            f"[score] {agent_name}: final={final:.3f} | "
            f"fact={fact_score:.3f} depth={depth_score:.3f} oc_pen={penalty:.3f}"
        )
        return result

    def score_many(
        self,
        outputs: list[tuple[AgentOutput, str]],
        fact_set: list[str],
    ) -> list[ScoringResult]:
        """
        Score multiple agents. Returns results sorted best-to-worst.

        Args:
            outputs:  List of (AgentOutput, agent_name) tuples.
            fact_set: Original fact set.

        Returns:
            List of ScoringResult, sorted by final_score descending.
        """
        results = [
            self.score(output, fact_set, name)
            for output, name in outputs
        ]
        return sorted(results, key=lambda r: r.final_score, reverse=True)

    # =========================================================================
    # Scoring Dimensions (private)
    # =========================================================================

    def _score_fact_consistency(
        self,
        output: AgentOutput,
        fact_set: list[str],
    ) -> float:
        """
        How well did the agent use the provided facts?

        Strategy: count how many words from extracted_facts also appear
        in the original fact_set (case-insensitive token overlap).
        Normalised to 0.0-1.0.

        An agent that rewrites / summarises facts still scores well.
        An agent that invents facts not in the set scores lower.
        """
        if not fact_set or not output.extracted_facts:
            return 0.0

        # Build a token vocabulary from the original fact_set
        fact_tokens = set()
        for fact in fact_set:
            fact_tokens.update(fact.lower().split())

        # Remove common stop words to avoid inflating the score
        stop_words = {
            "the", "a", "an", "in", "of", "to", "and", "or", "is",
            "it", "at", "by", "on", "for", "with", "no", "not", "was",
            "that", "this", "from", "be", "are", "up", "3x", "have",
        }
        fact_tokens -= stop_words

        if not fact_tokens:
            return 0.5  # Can't measure — neutral score

        # Score each extracted_fact against the vocabulary
        scores = []
        for extracted in output.extracted_facts:
            extracted_tokens = set(extracted.lower().split()) - stop_words
            if not extracted_tokens:
                scores.append(0.0)
                continue
            overlap = fact_tokens & extracted_tokens
            scores.append(len(overlap) / len(extracted_tokens))

        return min(1.0, sum(scores) / len(scores))

    def _score_reasoning_depth(self, output: AgentOutput) -> float:
        """
        How thorough was the agent's reasoning?

        Scores 3 sub-dimensions then averages them:
          - Explanations generated (possible_explanations count)
          - Hypotheses ranked     (ranked_hypotheses count)
          - Weaknesses admitted   (acknowledged_weaknesses count)

        Each sub-dimension is capped at max_for_full_credit.
        """
        cfg = self._depth_cfg
        cap = cfg["max_for_full_credit"]

        explanation_score = min(1.0, len(output.possible_explanations)   / cap)
        hypothesis_score  = min(1.0, len(output.ranked_hypotheses)        / cap)
        weakness_score    = min(1.0, len(output.acknowledged_weaknesses)  / cap)

        return (explanation_score + hypothesis_score + weakness_score) / 3.0

    def _calc_overconfidence_penalty(
        self,
        confidence: float,
        depth_score: float,
    ) -> tuple[float, bool]:
        """
        Detect and quantify overconfidence.

        An agent is overconfident when:
          - confidence_score is above the threshold (e.g. 0.80)
          - reasoning_depth_score is below the threshold (e.g. 0.50)

        The penalty scales linearly with the gap between confidence and depth.
        Returns (penalty_amount, overconfident_flag).
        """
        cfg = self._oc_cfg
        conf_threshold  = cfg["confidence_threshold"]
        depth_threshold = cfg["depth_threshold"]
        max_penalty     = cfg["max_penalty"]

        is_overconfident = (
            confidence >= conf_threshold
            and depth_score < depth_threshold
        )

        if not is_overconfident:
            return 0.0, False

        # Penalty scales with how far confidence exceeds the depth
        gap     = confidence - depth_score
        penalty = min(max_penalty, gap * max_penalty)
        return round(penalty, 4), True

    # =========================================================================
    # Config Loader
    # =========================================================================

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(
                f"scoring_weights.yaml not found at {path}. "
                "Run from the dmars/ project root."
            )
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
