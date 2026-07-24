"""
core/conflict_detector.py
===========================
Checkpoint 8 — Conflict Detector

Analyzes multiple agent ScoringResults and AgentOutputs to detect two
types of conflict:

  1. Score Conflict (Low Differentiation)
     When agent scores are very close together, the system cannot
     clearly identify the "best" agent. This is flagged because it
     means the evidence doesn't strongly favour one interpretation.

  2. Narrative Conflict (Contradictory Main Drivers)
     When agents' main_driver strings contain semantically opposing
     keywords (e.g. "buy" vs "sell", "bullish" vs "bearish"),
     the conflict detector flags it and identifies which agents disagree.

Output:
    ConflictReport — conflict_detected, conflicting_agents, conflict_reason

Usage:
    from core.conflict_detector import ConflictDetector
    from core.scoring_engine import ScoringResult
    from core.schemas import AgentResult

    detector = ConflictDetector()
    report = detector.detect(scoring_results, agent_results)
    print(report.conflict_detected)
    print(report.conflict_reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.schemas import AgentResult
from core.scoring_engine import ScoringResult

logger = logging.getLogger(__name__)

# Keyword pairs that signal narrative opposition.
# If agent A's driver contains a word from group[0] and agent B's from group[1],
# they are in direct contradiction.
OPPOSING_KEYWORD_PAIRS: list[tuple[set[str], set[str]]] = [
    ({"buy", "buying", "accumulation", "bullish", "demand"},
     {"sell", "selling", "distribution", "bearish", "supply"}),
    ({"squeeze", "liquidation", "forced"},
     {"organic", "natural", "gradual", "sentiment"}),
    ({"manipulation", "coordinated", "artificial"},
     {"legitimate", "organic", "normal", "real"}),
    ({"institutional", "whale", "large"},
     {"retail", "small", "individual", "minor"}),
    ({"positive", "rally", "surge", "spike"},
     {"negative", "crash", "dump", "drop"}),
]

# Score similarity threshold — if the gap between the best and worst
# scoring agent is below this, flag low differentiation.
LOW_DIFFERENTIATION_THRESHOLD = 0.10


# =============================================================================
# Conflict Report
# =============================================================================

@dataclass
class ConflictReport:
    """
    The result of conflict detection across multiple agent outputs.
    Always returned — check conflict_detected to see if action is needed.
    """
    conflict_detected:  bool
    conflict_level:     str          # "Strong Consensus" | "Soft Disagreement" | "High Conflict"
    conflict_type:      str          # "none" | "score" | "narrative" | "both"
    conflicting_agents: list[str]    # Agent names involved in the conflict
    conflict_reason:    str          # Human-readable explanation
    score_spread:       float = 0.0  # Max - min score across all agents

    def summary(self) -> str:
        status = "CONFLICT DETECTED" if self.conflict_detected else "No conflict"
        lines = [
            f"  Status  : {status}",
            f"  Type    : {self.conflict_type}",
            f"  Spread  : {self.score_spread:.3f}",
            f"  Agents  : {self.conflicting_agents or 'none'}",
            f"  Reason  : {self.conflict_reason}",
        ]
        return "\n".join(lines)


# =============================================================================
# Conflict Detector
# =============================================================================

class ConflictDetector:
    """
    Detects disagreement between agents at the score level and narrative level.
    No LLM calls — fully deterministic keyword and math-based logic.
    """

    def detect(
        self,
        scoring_results: list[ScoringResult],
        agent_results: list[AgentResult],
    ) -> ConflictReport:
        """
        Run conflict detection across all agent outputs.

        Args:
            scoring_results: List of ScoringResult from ScoringEngine.
            agent_results:   List of AgentResult from the pipeline.

        Returns:
            ConflictReport describing any conflicts found.
        """
        if not scoring_results:
            return ConflictReport(
                conflict_detected=False,
                conflict_level="Strong Consensus",
                conflict_type="none",
                conflicting_agents=[],
                conflict_reason="No agents to compare.",
            )

        # Build lookup maps
        output_map = {
            r.agent_name: r.output
            for r in agent_results
            if r.success and r.output
        }

        # 1. Score Conflict (Quality Spread)
        score_conflict = self._detect_score_conflict(scoring_results)

        # 2. Confidence Conflict (Direct Agent Disagreement)
        conf_conflict = self._detect_confidence_conflict(output_map)

        # 3. Narrative Conflict (Keyword Opposition)
        narrative_conflict = self._detect_narrative_conflict(output_map)

        # Aggregate Results
        conflict_detected = (
            score_conflict["conflict"] or
            conf_conflict["conflict"] or
            narrative_conflict["conflict"]
        )

        conflicting_agents = list(set(
            score_conflict.get("agents", []) +
            conf_conflict.get("agents", []) +
            narrative_conflict.get("agents", [])
        ))

        # Determine Conflict Level (3-level system)
        # High Conflict: Direct narrative contradiction or massive spread
        if narrative_conflict["conflict"] or conf_conflict.get("extreme", False):
            conflict_level = "High Conflict"
            conflict_type = "narrative" if narrative_conflict["conflict"] else "confidence"
            reason = narrative_conflict["reason"] or conf_conflict["reason"]
        # Soft Disagreement: Quality scores don't differentiate or confidence levels vary
        elif score_conflict["conflict"] or conf_conflict["conflict"]:
            conflict_level = "Soft Disagreement"
            conflict_type = "score" if score_conflict["conflict"] else "confidence"
            reason = score_conflict["reason"] or conf_conflict["reason"]
        else:
            conflict_level = "Strong Consensus"
            conflict_type = "none"
            reason = "Agents are in agreement. No significant conflict detected."

        scores = [r.final_score for r in scoring_results]
        spread = round(max(scores) - min(scores), 4) if scores else 0.0

        report = ConflictReport(
            conflict_detected=conflict_detected,
            conflict_level=conflict_level,
            conflict_type=conflict_type,
            conflicting_agents=conflicting_agents,
            conflict_reason=reason,
            score_spread=spread,
        )
        logger.info(
            f"Conflict detection: level={conflict_level} type={conflict_type} spread={spread:.3f}"
        )
        return report

    # =========================================================================
    # Private — Detection Logic
    # =========================================================================

    def _detect_score_conflict(
        self, scoring_results: list[ScoringResult]
    ) -> dict:
        """Flag when quality scores are clustered (low differentiation)."""
        if len(scoring_results) < 2:
            return {"conflict": False, "agents": [], "reason": ""}

        scores = [r.final_score for r in scoring_results]
        spread = max(scores) - min(scores)

        if spread < LOW_DIFFERENTIATION_THRESHOLD:
            return {
                "conflict": True,
                "agents": [r.agent_name for r in scoring_results],
                "reason": f"Low quality differentiation (spread: {spread:.3f}).",
            }
        return {"conflict": False, "agents": [], "reason": ""}

    def _detect_confidence_conflict(self, output_map: dict) -> dict:
        """Flag when agents report widely varying confidence levels."""
        if len(output_map) < 2:
            return {"conflict": False, "agents": [], "reason": ""}

        confs = [out.confidence_score for out in output_map.values()]
        spread = max(confs) - min(confs)

        if spread >= 0.40: # e.g., 40% vs 80%
            return {
                "conflict": True,
                "extreme": True,
                "agents": list(output_map.keys()),
                "reason": f"High confidence variance (spread: {spread:.2f}).",
            }
        if spread >= 0.20:
            return {
                "conflict": True,
                "extreme": False,
                "agents": list(output_map.keys()),
                "reason": f"Soft confidence disagreement (spread: {spread:.2f}).",
            }
        return {"conflict": False, "agents": [], "reason": ""}

    # =========================================================================
    # Private — Narrative-level conflict
    # =========================================================================

    def _detect_narrative_conflict(
        self, output_map: dict
    ) -> dict:
        """
        Flag when two agents' main_driver strings contain opposing keywords.
        Uses the OPPOSING_KEYWORD_PAIRS vocabulary for conflict detection.
        """
        if len(output_map) < 2:
            return {"conflict": False, "agents": [], "reason": ""}

        agent_names  = list(output_map.keys())
        conflict_pairs: list[tuple[str, str, str]] = []  # (agentA, agentB, reason)

        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                name_a = agent_names[i]
                name_b = agent_names[j]
                driver_a = output_map[name_a].main_driver.lower()
                driver_b = output_map[name_b].main_driver.lower()

                tokens_a = set(driver_a.split())
                tokens_b = set(driver_b.split())

                for group_pos, group_neg in OPPOSING_KEYWORD_PAIRS:
                    a_positive = bool(tokens_a & group_pos)
                    a_negative = bool(tokens_a & group_neg)
                    b_positive = bool(tokens_b & group_pos)
                    b_negative = bool(tokens_b & group_neg)

                    if (a_positive and b_negative) or (a_negative and b_positive):
                        hit_a = tokens_a & (group_pos | group_neg)
                        hit_b = tokens_b & (group_pos | group_neg)
                        conflict_pairs.append((
                            name_a, name_b,
                            f"'{name_a}' ({', '.join(hit_a)}) contradicts '{name_b}' ({', '.join(hit_b)})"
                        ))
                        break  # one conflict per pair is enough

        if conflict_pairs:
            involved = list({n for pair in conflict_pairs for n in pair[:2]})
            reasons  = "; ".join(p[2] for p in conflict_pairs)
            return {
                "conflict": True,
                "agents":   involved,
                "reason":   f"Opposing narratives detected: {reasons}",
            }

        return {"conflict": False, "agents": [], "reason": ""}
