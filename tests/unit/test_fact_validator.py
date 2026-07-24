"""
tests/unit/test_fact_validator.py
===================================
Checkpoint 2 — Unit Tests for the Fact Validator

Coverage (maps exactly to checkpoint test criteria):
  ✅ Criterion 1 — All tests pass (pytest runs cleanly)
  ✅ Criterion 2 — Duplicate facts → duplicates removed
  ✅ Criterion 3 — Contradictory facts → contradiction flagged
  ✅ Criterion 4 — 0 or 1 fact → FactValidationError raised
  ✅ Criterion 5 — 5 clean facts → all tiered correctly, passes validation

Run:
    poetry run pytest tests/unit/test_fact_validator.py -v
"""

import pytest

from core.fact_validator import (
    ConfidenceTier,
    FactValidationError,
    FactValidator,
    ValidationResult,
)


# =============================================================================
# Fixture
# =============================================================================

@pytest.fixture
def validator() -> FactValidator:
    return FactValidator()


# =============================================================================
# Criterion 4 — Rejection: too sparse (0 or 1 fact)
# =============================================================================

class TestRejection:
    """Fact sets with fewer than 2 unique facts must be rejected."""

    def test_empty_list_raises(self, validator):
        """Empty input → FactValidationError."""
        with pytest.raises(FactValidationError, match="empty"):
            validator.validate([])

    def test_single_fact_raises(self, validator):
        """Only 1 fact → too sparse, raises."""
        with pytest.raises(FactValidationError, match="too sparse"):
            validator.validate(["BTC volume up 3x in 60 minutes"])

    def test_two_identical_facts_deduplicate_to_one_and_raise(self, validator):
        """2 identical facts → deduplicate to 1 → too sparse → raises."""
        with pytest.raises(FactValidationError, match="too sparse"):
            validator.validate([
                "BTC volume up 3x in 60 minutes",
                "BTC volume up 3x in 60 minutes",  # exact duplicate
            ])

    def test_case_insensitive_dedup_triggers_rejection(self, validator):
        """Same fact in different case → still deduplicated → raises."""
        with pytest.raises(FactValidationError, match="too sparse"):
            validator.validate([
                "BTC Volume Up 3x",
                "btc volume up 3x",
            ])

    def test_rejection_error_is_descriptive(self, validator):
        """Error message should mention the minimum required count."""
        with pytest.raises(FactValidationError, match="2"):
            validator.validate(["Only one fact here"])


# =============================================================================
# Criterion 2 — Deduplication
# =============================================================================

class TestDeduplication:
    """Duplicate facts must be detected and removed before processing."""

    def test_exact_duplicates_removed(self, validator):
        facts = [
            "BTC volume up 3x in 60 minutes",
            "BTC volume up 3x in 60 minutes",   # exact duplicate
            "Large derivatives positions liquidated",
        ]
        result = validator.validate(facts)
        assert result.duplicates_removed == 1
        assert len(result.facts) == 2

    def test_case_insensitive_dedup(self, validator):
        facts = [
            "BTC volume up 3x",
            "btc volume up 3x",          # same, different case
            "No major news reported",
        ]
        result = validator.validate(facts)
        assert result.duplicates_removed == 1
        assert len(result.facts) == 2

    def test_whitespace_normalized_dedup(self, validator):
        facts = [
            "BTC volume up 3x",
            "  BTC volume up 3x  ",      # leading/trailing whitespace
            "Price fell 8% in an hour",
        ]
        result = validator.validate(facts)
        assert result.duplicates_removed == 1
        assert len(result.facts) == 2

    def test_multiple_duplicates_all_removed(self, validator):
        facts = [
            "BTC spiked 8%",
            "BTC spiked 8%",             # dup 1
            "BTC spiked 8%",             # dup 2
            "No news was released",
        ]
        result = validator.validate(facts)
        assert result.duplicates_removed == 2
        assert len(result.facts) == 2

    def test_no_duplicates_preserved_intact(self, validator):
        facts = [
            "BTC volume up 3x in 60 minutes",
            "Large derivatives positions liquidated",
            "No major news reported",
        ]
        result = validator.validate(facts)
        assert result.duplicates_removed == 0
        assert len(result.facts) == 3

    def test_original_casing_preserved_after_dedup(self, validator):
        """The first occurrence's original casing must be kept."""
        facts = [
            "BTC Volume Up 3x",          # first → kept
            "btc volume up 3x",          # duplicate → removed
            "Market was stable",
        ]
        result = validator.validate(facts)
        originals = result.to_plain_list()
        assert "BTC Volume Up 3x" in originals


# =============================================================================
# Criterion 3 — Contradiction Detection
# =============================================================================

class TestContradictionDetection:
    """Opposing keywords across fact pairs must be flagged as contradictions."""

    def test_increase_decrease_contradiction(self, validator):
        facts = [
            "BTC price increased by 8% in the last hour",
            "BTC price decreased significantly overnight",
        ]
        result = validator.validate(facts)
        assert len(result.contradictions) > 0

    def test_bullish_bearish_contradiction(self, validator):
        facts = [
            "Market sentiment is extremely bullish",
            "Market outlook remains bearish for the week",
        ]
        result = validator.validate(facts)
        assert len(result.contradictions) > 0

    def test_spike_drop_contradiction(self, validator):
        facts = [
            "BTC spiked 12% this morning",
            "BTC dropped to weekly lows by noon",
        ]
        result = validator.validate(facts)
        assert len(result.contradictions) > 0

    def test_surge_plunge_contradiction(self, validator):
        facts = [
            "Volume surged across all exchanges",
            "Trading volume plunged after the first hour",
        ]
        result = validator.validate(facts)
        assert len(result.contradictions) > 0

    def test_no_contradiction_when_facts_agree(self, validator):
        facts = [
            "BTC volume up 3x in 60 minutes",
            "Large derivatives positions liquidated",
            "No major news reported",
        ]
        result = validator.validate(facts)
        assert result.contradictions == []

    def test_contradiction_result_contains_both_facts(self, validator):
        """Each contradiction tuple must contain the original fact strings."""
        fact_a = "Price increased sharply"
        fact_b = "Price decreased by end of day"
        result = validator.validate([fact_a, fact_b])
        assert len(result.contradictions) == 1
        pair = result.contradictions[0]
        assert fact_a in pair
        assert fact_b in pair


# =============================================================================
# Criterion 5 — Confidence Tiering (5 clean facts)
# =============================================================================

class TestConfidenceTiering:
    """Facts must be tiered into verified / inferred / uncertain correctly."""

    def test_five_clean_facts_all_verified(self, validator):
        """Checkpoint test criterion 5 — the primary standard test."""
        facts = [
            "BTC volume up 3x in 60 minutes",
            "Large derivatives positions were liquidated",
            "No major news was reported in that window",
            "Open interest dropped 15% across exchanges",
            "Bid-ask spread widened significantly during the event",
        ]
        result = validator.validate(facts)
        assert result.is_valid is True
        assert len(result.facts) == 5
        for f in result.facts:
            assert f.tier == ConfidenceTier.VERIFIED

    def test_inferred_markers_tiered_correctly(self, validator):
        cases = [
            "BTC volume likely rose due to institutional activity",
            "Price appears to have stabilized",
            "The move probably triggered stop-losses",
            "Volume may have been inflated artificially",
        ]
        for fact in cases:
            result = validator.validate([fact, "A second confirmed fact"])
            matched = next(f for f in result.facts if f.original == fact)
            assert matched.tier == ConfidenceTier.INFERRED, (
                f"Expected INFERRED for: '{fact}', got {matched.tier}"
            )

    def test_uncertain_markers_tiered_correctly(self, validator):
        cases = [
            "The cause of the move is unclear",
            "The trigger remains unknown at this time",
            "Market direction is uncertain for now",
            "The rumored buyer has not been confirmed",
        ]
        for fact in cases:
            result = validator.validate([fact, "A second confirmed fact"])
            matched = next(f for f in result.facts if f.original == fact)
            assert matched.tier == ConfidenceTier.UNCERTAIN, (
                f"Expected UNCERTAIN for: '{fact}', got {matched.tier}"
            )

    def test_uncertain_takes_priority_over_inferred(self, validator):
        """A fact with both uncertain AND inferred markers → UNCERTAIN wins."""
        fact = "It is unclear but likely that volume increased"
        result = validator.validate([fact, "Volume confirmed at 3x"])
        matched = next(f for f in result.facts if f.original == fact)
        assert matched.tier == ConfidenceTier.UNCERTAIN

    def test_mixed_tiers_in_one_batch(self, validator):
        facts = [
            "BTC volume increased 3x",               # → VERIFIED
            "Price possibly moved due to news",       # → INFERRED (possibly)
            "The trigger is unknown at this time",    # → UNCERTAIN
        ]
        result = validator.validate(facts)
        tiers = [f.tier for f in result.facts]
        assert ConfidenceTier.VERIFIED  in tiers
        assert ConfidenceTier.INFERRED  in tiers
        assert ConfidenceTier.UNCERTAIN in tiers


# =============================================================================
# Criterion 1 — General Structure and API Tests
# =============================================================================

class TestValidResultStructure:
    """Verify shape and API of the ValidationResult object."""

    def test_returns_validation_result(self, validator):
        result = validator.validate([
            "BTC volume up 3x",
            "No major news reported",
        ])
        assert isinstance(result, ValidationResult)

    def test_is_valid_true_for_good_facts(self, validator):
        result = validator.validate([
            "BTC volume up 3x",
            "No major news reported",
        ])
        assert result.is_valid is True
        assert result.rejection_reason is None

    def test_to_plain_list_returns_original_strings(self, validator):
        facts = [
            "BTC volume up 3x",
            "Market sentiment shifted",
        ]
        result = validator.validate(facts)
        assert result.to_plain_list() == facts

    def test_summary_string_is_returned(self, validator):
        result = validator.validate([
            "BTC volume up 3x",
            "No major news reported",
        ])
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Facts:" in summary
        assert "verified" in summary

    def test_normalized_field_is_lowercase(self, validator):
        result = validator.validate([
            "BTC VOLUME UP 3X",
            "No Major News Reported",
        ])
        for f in result.facts:
            assert f.normalized == f.normalized.lower()
