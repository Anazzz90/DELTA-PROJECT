"""
core/fact_validator.py
======================
Checkpoint 2 — Fact Validator

Preprocessing layer that cleans and validates the input fact set
before it reaches any agent. Enforces Step 1 of the Delta-First Protocol:
    "Lock verified facts only — no assumptions, no hindsight."

Pipeline:
    1. Normalize & deduplicate facts
    2. Reject if fact set is too sparse (< MIN_FACTS unique facts)
    3. Tier each fact: verified / inferred / uncertain
    4. Detect keyword-level contradictions between facts

Usage:
    from core.fact_validator import FactValidator

    validator = FactValidator()
    result = validator.validate([
        "BTC volume up 3x in 60 minutes",
        "Large derivatives positions liquidated",
        "No major news reported",
    ])
    print(result.facts)           # list of ValidatedFact
    print(result.contradictions)  # list of contradicting pairs
    print(result.to_plain_list()) # plain strings for prompt injection
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# =============================================================================
# Data Types
# =============================================================================

class ConfidenceTier(str, Enum):
    """Confidence level assigned to each fact based on language markers."""
    VERIFIED  = "verified"   # Stated as definitive fact
    INFERRED  = "inferred"   # Hedged language — "likely", "appears", "may"
    UNCERTAIN = "uncertain"  # Explicit doubt — "unclear", "unknown", "rumored"


@dataclass
class ValidatedFact:
    """A single fact after validation processing."""
    original:   str              # Original string from user
    normalized: str              # Lowercase, stripped version
    tier:       ConfidenceTier   # Confidence level


@dataclass
class ValidationResult:
    """Complete output of the FactValidator pipeline."""
    facts:              list[ValidatedFact]        # Deduplicated, tiered facts
    contradictions:     list[tuple[str, str]]      # Pairs of contradicting facts
    duplicates_removed: int                        # How many were removed
    is_valid:           bool
    rejection_reason:   Optional[str] = None

    def to_plain_list(self) -> list[str]:
        """Returns original fact strings — ready to inject into agent prompts."""
        return [f.original for f in self.facts]

    def summary(self) -> str:
        """Human-readable validation summary."""
        tiers = {t: 0 for t in ConfidenceTier}
        for f in self.facts:
            tiers[f.tier] += 1
        return (
            f"Facts: {len(self.facts)} unique "
            f"({tiers[ConfidenceTier.VERIFIED]} verified, "
            f"{tiers[ConfidenceTier.INFERRED]} inferred, "
            f"{tiers[ConfidenceTier.UNCERTAIN]} uncertain) | "
            f"Duplicates removed: {self.duplicates_removed} | "
            f"Contradictions: {len(self.contradictions)}"
        )


# =============================================================================
# Exceptions
# =============================================================================

class FactValidationError(ValueError):
    """
    Raised when the fact set cannot be used for reasoning.
    Catches: empty input, too sparse after deduplication.
    """
    pass


# =============================================================================
# Validator
# =============================================================================

class FactValidator:
    """
    Validates and preprocesses a list of input facts before
    they are sent to any DMARS agent.

    All checks are deterministic and require no LLM calls.
    """

    MIN_FACTS: int = 2  # Minimum unique facts required

    # ── Contradiction Keywords ────────────────────────────────────────────────
    # Each tuple represents opposing concepts.
    # If fact_A contains word_A and fact_B contains word_B (or vice versa),
    # a potential contradiction is flagged.
    CONTRADICTION_PAIRS: list[tuple[str, str]] = [
        ("increase",  "decrease"),
        ("increased", "decreased"),
        ("up",        "down"),
        ("rise",      "fall"),
        ("rising",    "falling"),
        ("rose",      "fell"),
        ("bullish",   "bearish"),
        ("buying",    "selling"),
        ("long",      "short"),
        ("high",      "low"),
        ("positive",  "negative"),
        ("gain",      "loss"),
        ("above",     "below"),
        ("more",      "less"),
        ("spike",     "drop"),
        ("spiked",    "dropped"),
        ("surge",     "plunge"),
        ("surged",    "plunged"),
        ("open",      "closed"),
        ("confirmed", "denied"),
        ("accelerate","decelerate"),
        ("stronger",  "weaker"),
        ("hot",       "cold"),
    ]

    # ── Confidence Tier Markers ───────────────────────────────────────────────
    # Checked in UNCERTAIN → INFERRED → VERIFIED order.

    UNCERTAIN_MARKERS: list[str] = [
        "unclear", "unknown", "uncertain", "unconfirmed",
        "rumored", "alleged", "disputed", "debated",
        "not confirmed", "not verified", "unverified",
        "speculation", "speculated", "questionable",
    ]

    INFERRED_MARKERS: list[str] = [
        "appears", "seems", "likely", "probably", "possibly",
        "suggests", "may", "might", "could", "reportedly",
        "expected", "approximately", "roughly", "around",
        "estimated", "indicates", "implies", "assumed",
        "believed", "perhaps", "maybe", "thought to",
    ]

    # =========================================================================
    # Public API
    # =========================================================================

    def validate(self, facts: list[str]) -> ValidationResult:
        """
        Run the full validation pipeline on a raw fact list.

        Args:
            facts: List of raw fact strings from the user.

        Returns:
            ValidationResult with cleaned facts and metadata.

        Raises:
            FactValidationError: If fact set is empty or too sparse.
        """
        if not facts:
            raise FactValidationError(
                "Fact set is empty. "
                f"Provide at least {self.MIN_FACTS} verified facts to proceed."
            )

        # Step 1 — Deduplicate
        unique_facts, duplicates_removed = self._deduplicate(facts)

        # Step 2 — Reject if too sparse
        if len(unique_facts) < self.MIN_FACTS:
            raise FactValidationError(
                f"Fact set too sparse: only {len(unique_facts)} unique fact(s) "
                f"after deduplication (minimum required: {self.MIN_FACTS}). "
                "Add more verified facts before submitting."
            )

        # Step 3 — Tier each fact
        validated_facts = [self._tier_fact(f) for f in unique_facts]

        # Step 4 — Detect contradictions
        contradictions = self._detect_contradictions(unique_facts)

        return ValidationResult(
            facts=validated_facts,
            contradictions=contradictions,
            duplicates_removed=duplicates_removed,
            is_valid=True,
            rejection_reason=None,
        )

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _normalize(self, fact: str) -> str:
        """Lowercase, strip outer whitespace, collapse internal spaces."""
        return re.sub(r"\s+", " ", fact.lower().strip())

    def _deduplicate(self, facts: list[str]) -> tuple[list[str], int]:
        """
        Remove duplicate facts based on normalized comparison.
        Preserves original casing and order of first occurrence.
        """
        seen: set[str] = set()
        unique: list[str] = []
        for fact in facts:
            key = self._normalize(fact)
            if key not in seen:
                seen.add(key)
                unique.append(fact.strip())
        return unique, len(facts) - len(unique)

    def _tier_fact(self, fact: str) -> ValidatedFact:
        """
        Assign a ConfidenceTier to a single fact based on language markers.
        UNCERTAIN is checked before INFERRED (stronger signal of doubt).
        """
        lower = fact.lower()

        # Check uncertain markers first
        for marker in self.UNCERTAIN_MARKERS:
            if marker in lower:
                return ValidatedFact(
                    original=fact,
                    normalized=self._normalize(fact),
                    tier=ConfidenceTier.UNCERTAIN,
                )

        # Then check inferred markers (word-boundary match)
        for marker in self.INFERRED_MARKERS:
            if re.search(r"\b" + re.escape(marker) + r"\b", lower):
                return ValidatedFact(
                    original=fact,
                    normalized=self._normalize(fact),
                    tier=ConfidenceTier.INFERRED,
                )

        # Default: verified fact
        return ValidatedFact(
            original=fact,
            normalized=self._normalize(fact),
            tier=ConfidenceTier.VERIFIED,
        )

    def _detect_contradictions(self, facts: list[str]) -> list[tuple[str, str]]:
        """
        Find pairs of facts that contain opposing keywords.
        Returns list of (fact_a, fact_b) contradicting pairs.
        """
        contradictions: list[tuple[str, str]] = []
        normalized = [self._normalize(f) for f in facts]

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                if self._has_contradiction(normalized[i], normalized[j]):
                    contradictions.append((facts[i], facts[j]))

        return contradictions

    def _has_contradiction(self, fact_a: str, fact_b: str) -> bool:
        """
        Returns True if fact_a and fact_b contain opposing words
        from the CONTRADICTION_PAIRS list.
        """
        for word_pos, word_neg in self.CONTRADICTION_PAIRS:
            pattern_pos = r"\b" + re.escape(word_pos) + r"\b"
            pattern_neg = r"\b" + re.escape(word_neg) + r"\b"

            a_has_pos = bool(re.search(pattern_pos, fact_a))
            a_has_neg = bool(re.search(pattern_neg, fact_a))
            b_has_pos = bool(re.search(pattern_pos, fact_b))
            b_has_neg = bool(re.search(pattern_neg, fact_b))

            if (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos):
                return True

        return False
