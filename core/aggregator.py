"""
core/aggregator.py
====================
Checkpoint 8 — Aggregator

Clusters similar hypotheses from all agents into dominant narratives,
then produces a system-level final decision:
  - system_main_driver     — the most agreed-upon explanation
  - system_confidence_score— weighted average of agent confidences,
                              reduced when conflict is present
  - dominant_narratives    — clusters of similar explanations

Narrative clustering is done with simple keyword-based grouping
(no LLM, no embeddings). Each hypothesis is classified into one of
several narrative buckets based on keyword matching.

Usage:
    from core.aggregator import Aggregator
    from core.conflict_detector import ConflictReport

    aggregator = Aggregator()
    decision = aggregator.aggregate(scoring_results, agent_results, conflict_report)
    print(decision.system_main_driver)
    print(decision.system_confidence_score)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from core.conflict_detector import ConflictReport
from core.schemas import AgentResult
from core.scoring_engine import ScoringResult

logger = logging.getLogger(__name__)

# Narrative clusters: each entry maps a cluster name to keywords.
# A hypothesis is assigned to the first cluster whose keywords it matches.
NARRATIVE_CLUSTERS: dict[str, set[str]] = {
    "insider_activity":         {"insider", "ceo", "selling", "buying", "management", "ownership"},
    "valuation_metrics":        {"valuation", "pe", "ratio", "overvalued", "undervalued", "fair", "multiple"},
    "margin_analysis":          {"margin", "margins", "profitability", "gross", "operating", "ebitda"},
    "liquidation_cascade":      {"liquidation", "liquidated", "cascade", "forced", "squeeze"},
    "institutional_activity":   {"institutional", "whale", "accumulation", "large", "coordinated"},
    "algorithmic_momentum":     {"algorithm", "algo", "momentum", "automated", "bot", "hft"},
    "retail_sentiment":         {"retail", "sentiment", "fomo", "fear", "panic", "social"},
    "macro_drivers":            {"news", "macro", "regulation", "announcement", "event", "interest", "rates"},
    "technical_signals":        {"breakout", "resistance", "support", "technical", "chart"},
    "volume_dynamics":          {"volume", "volumes", "surge", "spike", "flow"},
    "other":                     set(),  # fallback bucket
}

# Penalty applied to system_confidence_score when conflict is detected
CONFLICT_CONFIDENCE_PENALTY = 0.15


# =============================================================================
# Final Decision
# =============================================================================

@dataclass
class FinalDecision:
    """
    The system-level aggregated decision after all agents have been scored
    and conflicts resolved.
    """
    system_main_driver:      str         # The "Final Decision"
    system_confidence_score: float
    signal_summary:          dict        # Positive vs Negative signals
    net_bias:                str         # Overall bias (Bullish/Bearish/Neutral)
    decision_logic:          str         # Summary of the logic used
    dominant_narratives:     list[str]   # Top narrative clusters
    narrative_clusters:      dict        # Full cluster breakdown
    contributing_agents:     list[str]   # Agents that contributed
    conflict_adjusted:       bool        # True if confidence was penalised

    def summary(self) -> str:
        lines = [
            f"  Final Decision    : {self.system_main_driver}",
            f"  System Confidence : {self.system_confidence_score:.3f}",
            f"  Net Bias          : {self.net_bias}",
            f"  Signal Summary    : {self.signal_summary}",
            f"  Decision Logic    : {self.decision_logic}",
            f"  Dominant Narratives: {self.dominant_narratives}",
        ]
        return "\n".join(lines)


# =============================================================================
# Aggregator
# =============================================================================

class Aggregator:
    """
    Aggregates outputs from multiple agents into one final system decision.
    """

    def aggregate(
        self,
        scoring_results: list[ScoringResult],
        agent_results: list[AgentResult],
        conflict_report: Optional[ConflictReport] = None,
    ) -> FinalDecision:
        """
        Produce a system-level final decision with an explicit decision layer.
        """
        if not scoring_results:
            return FinalDecision(
                system_main_driver="Insufficient agent data.",
                system_confidence_score=0.0,
                signal_summary={"positive": 0, "negative": 0},
                net_bias="Neutral",
                decision_logic="No data available to form logic.",
                dominant_narratives=[],
                narrative_clusters={},
                contributing_agents=[],
                conflict_adjusted=False,
            )

        # Build lookup maps
        output_map = {
            r.agent_name: r.output
            for r in agent_results
            if r.success and r.output
        }

        valid_scores = [r for r in scoring_results if r.agent_name in output_map]
        if not valid_scores:
            return FinalDecision(
                system_main_driver="All agents failed — no valid outputs.",
                system_confidence_score=0.0,
                signal_summary={"positive": 0, "negative": 0},
                net_bias="Neutral",
                decision_logic="All agents failed.",
                dominant_narratives=[],
                narrative_clusters={},
                contributing_agents=[],
                conflict_adjusted=False,
            )

        # 1. System Main Driver (Highest quality score)
        best = max(valid_scores, key=lambda r: r.final_score)
        system_driver = output_map[best.agent_name].main_driver

        # 2. Weighted Confidence (Weighted by reasoning quality scores)
        total_quality_weight = sum(r.final_score for r in valid_scores)
        if total_quality_weight > 0:
            weighted_conf = sum(
                output_map[r.agent_name].confidence_score * r.final_score
                for r in valid_scores
            ) / total_quality_weight
        else:
            weighted_conf = sum(
                output_map[r.agent_name].confidence_score for r in valid_scores
            ) / len(valid_scores)

        # 3. Conflict Adjustment
        conflict_adjusted = False
        if conflict_report and conflict_report.conflict_detected:
            weighted_conf = max(0.0, weighted_conf - CONFLICT_CONFIDENCE_PENALTY)
            conflict_adjusted = True

        system_confidence = round(min(1.0, max(0.0, weighted_conf)), 4)

        # 4. Explicit Decision Layer (Signals & Bias)
        signals = {"positive": 0, "negative": 0, "neutral": 0}
        pos_keywords = {"buy", "bullish", "long", "upside", "growth", "positive", "accumulation", "undervalued", "demand", "higher", "surge", "rally", "spike"}
        neg_keywords = {"sell", "bearish", "short", "downside", "risk", "negative", "distribution", "overvalued", "supply", "lower", "crash", "dump", "drop"}

        for r in valid_scores:
            driver_lower = output_map[r.agent_name].main_driver.lower()
            tokens = set(driver_lower.split())
            has_pos = bool(tokens & pos_keywords)
            has_neg = bool(tokens & neg_keywords)
            
            if has_pos: signals["positive"] += 1
            if has_neg: signals["negative"] += 1
            if not has_pos and not has_neg:
                signals["neutral"] += 1

        if signals["positive"] > signals["negative"]:
            net_bias = "Bullish"
        elif signals["negative"] > signals["positive"]:
            net_bias = "Bearish"
        else:
            net_bias = "Neutral"

        # 5. Decision Logic
        if conflict_report and conflict_report.conflict_detected:
            logic = f"Decision weighted by quality scores but penalized for {conflict_report.conflict_level.lower()}. {conflict_report.conflict_reason}"
        else:
            logic = f"Consensus reached based on high-quality analysis from {len(valid_scores)} agents led by {best.agent_name}."

        # 6. Narrative Clustering
        all_hypotheses: list[str] = []
        for r in valid_scores:
            output = output_map[r.agent_name]
            all_hypotheses.extend(output.ranked_hypotheses)
            all_hypotheses.extend(output.possible_explanations)

        clusters = self._cluster_hypotheses(all_hypotheses)
        dominant = self._dominant_narratives(clusters)

        return FinalDecision(
            system_main_driver=system_driver,
            system_confidence_score=system_confidence,
            signal_summary=signals,
            net_bias=net_bias,
            decision_logic=logic,
            dominant_narratives=dominant,
            narrative_clusters=clusters,
            contributing_agents=[r.agent_name for r in valid_scores],
            conflict_adjusted=conflict_adjusted,
        )

    def _cluster_hypotheses(self, hypotheses: list[str]) -> dict:
        """Assign each hypothesis string to a reasoning theme cluster."""
        clusters: dict[str, list[str]] = defaultdict(list)
        for hyp in hypotheses:
            tokens = set(hyp.lower().replace(",", "").replace(".", "").split())
            assigned = False
            for cluster_name, keywords in NARRATIVE_CLUSTERS.items():
                if cluster_name == "other": continue
                if tokens & keywords:
                    clusters[cluster_name].append(hyp)
                    assigned = True
                    break
            if not assigned:
                clusters["other"].append(hyp)
        return dict(clusters)

    def _dominant_narratives(self, clusters: dict) -> list[str]:
        """Return top 3 cluster names by hypothesis count."""
        sorted_clusters = sorted(
            [(name, items) for name, items in clusters.items() if name != "other"],
            key=lambda x: len(x[1]),
            reverse=True,
        )
        return [name for name, _ in sorted_clusters[:3]]
