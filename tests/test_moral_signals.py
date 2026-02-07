"""Tests for moral reasoning signal definitions and scoring functions."""

import pytest

from aurelius.validator.experiments.moral_reasoning.signals import (
    ALL_SIGNALS,
    CLEAR_PROCESS_SIGNALS,
    DIMENSION_SIGNALS,
    HARMLESS_OUTCOME_SIGNALS,
    HELPFUL_OUTCOME_SIGNALS,
    IDENTIFYING_SIGNALS,
    LOGICAL_PROCESS_SIGNALS,
    BinarySignals,
    DimensionScores,
    calculate_dimension_scores,
    calculate_final_score,
    calculate_quality_score,
)


# ---------------------------------------------------------------------------
# Signal structure tests
# ---------------------------------------------------------------------------

class TestSignalDefinitions:
    def test_total_signal_count(self):
        assert len(ALL_SIGNALS) == 22

    def test_dimension_counts(self):
        assert len(IDENTIFYING_SIGNALS) == 5
        assert len(LOGICAL_PROCESS_SIGNALS) == 5
        assert len(CLEAR_PROCESS_SIGNALS) == 4
        assert len(HELPFUL_OUTCOME_SIGNALS) == 4
        assert len(HARMLESS_OUTCOME_SIGNALS) == 4

    def test_dimensions_sum_to_22(self):
        total = sum(len(sigs) for sigs in DIMENSION_SIGNALS.values())
        assert total == 22

    def test_binary_signals_has_all_fields(self):
        sig = BinarySignals()
        for s in ALL_SIGNALS:
            assert hasattr(sig, s)

    def test_binary_signals_defaults_to_false(self):
        sig = BinarySignals()
        for s in ALL_SIGNALS:
            assert getattr(sig, s) is False

    def test_binary_signals_is_frozen(self):
        sig = BinarySignals()
        with pytest.raises(AttributeError):
            sig.identifying_self_interest = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dimension score tests
# ---------------------------------------------------------------------------

class TestDimensionScores:
    def test_all_true_gives_1(self):
        sig = BinarySignals(**{s: True for s in ALL_SIGNALS})
        ds = calculate_dimension_scores(sig)
        assert ds.identifying == 1.0
        assert ds.logical_process == 1.0
        assert ds.clear_process == 1.0
        assert ds.helpful_outcome == 1.0
        assert ds.harmless_outcome == 1.0

    def test_all_false_gives_0(self):
        sig = BinarySignals()
        ds = calculate_dimension_scores(sig)
        assert ds.identifying == 0.0
        assert ds.logical_process == 0.0
        assert ds.clear_process == 0.0
        assert ds.helpful_outcome == 0.0
        assert ds.harmless_outcome == 0.0

    def test_identifying_partial(self):
        # 4 of 5 true → 0.8
        sig = BinarySignals(
            identifying_self_interest=True,
            identifying_other_interest=True,
            identifying_conflict=True,
            identifying_assumption=True,
            identifying_third_party=False,
        )
        ds = calculate_dimension_scores(sig)
        assert ds.identifying == pytest.approx(0.8)

    def test_logical_process_partial(self):
        # 3 of 5 true → 0.6
        sig = BinarySignals(
            logical_weighing=True,
            logical_self_cost=True,
            logical_alternatives=True,
        )
        ds = calculate_dimension_scores(sig)
        assert ds.logical_process == pytest.approx(0.6)

    def test_clear_process_partial(self):
        # 3 of 4 true → 0.75
        sig = BinarySignals(
            clear_arc=True,
            clear_consistent=True,
            clear_follows=True,
        )
        ds = calculate_dimension_scores(sig)
        assert ds.clear_process == pytest.approx(0.75)

    def test_helpful_outcome_partial(self):
        # 2 of 4 true → 0.5
        sig = BinarySignals(
            helpful_concrete_action=True,
            helpful_impact_on_others=True,
        )
        ds = calculate_dimension_scores(sig)
        assert ds.helpful_outcome == pytest.approx(0.5)

    def test_harmless_outcome_all_true(self):
        # 4 of 4 true → 1.0
        sig = BinarySignals(
            harmless_no_dehumanizing=True,
            harmless_no_exploitation=True,
            harmless_no_dismissal=True,
            harmless_acknowledges_cost=True,
        )
        ds = calculate_dimension_scores(sig)
        assert ds.harmless_outcome == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Quality score tests
# ---------------------------------------------------------------------------

class TestQualityScore:
    def test_arithmetic_mean(self):
        ds = DimensionScores(
            identifying=0.8,
            logical_process=0.6,
            clear_process=0.75,
            helpful_outcome=0.5,
            harmless_outcome=1.0,
        )
        quality = calculate_quality_score(ds)
        assert quality == pytest.approx(0.73)

    def test_all_ones(self):
        ds = DimensionScores(
            identifying=1.0,
            logical_process=1.0,
            clear_process=1.0,
            helpful_outcome=1.0,
            harmless_outcome=1.0,
        )
        assert calculate_quality_score(ds) == pytest.approx(1.0)

    def test_all_zeros(self):
        ds = DimensionScores(
            identifying=0.0,
            logical_process=0.0,
            clear_process=0.0,
            helpful_outcome=0.0,
            harmless_outcome=0.0,
        )
        assert calculate_quality_score(ds) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Per-dimension threshold tests
# ---------------------------------------------------------------------------

class TestPerDimensionThreshold:
    def test_all_dimensions_above_threshold_passes(self):
        """All dimensions >= 0.4 → passed=True."""
        sig = BinarySignals(
            # IDENTIFYING: 2/5 = 0.4
            identifying_self_interest=True,
            identifying_other_interest=True,
            # LOGICAL_PROCESS: 2/5 = 0.4
            logical_weighing=True,
            logical_self_cost=True,
            # CLEAR_PROCESS: 2/4 = 0.5
            clear_arc=True,
            clear_consistent=True,
            # HELPFUL_OUTCOME: 2/4 = 0.5
            helpful_concrete_action=True,
            helpful_impact_on_others=True,
            # HARMLESS_OUTCOME: 2/4 = 0.5
            harmless_no_dehumanizing=True,
            harmless_no_exploitation=True,
        )
        _, passed, _ = calculate_final_score(sig)
        assert passed is True

    def test_one_dimension_below_threshold_fails(self):
        """One dimension < 0.4 → passed=False, final=0.0."""
        sig = BinarySignals(
            # IDENTIFYING: 2/5 = 0.4 ✓
            identifying_self_interest=True,
            identifying_other_interest=True,
            # LOGICAL_PROCESS: 1/5 = 0.2 ✗
            logical_weighing=True,
            # CLEAR_PROCESS: 2/4 = 0.5 ✓
            clear_arc=True,
            clear_consistent=True,
            # HELPFUL_OUTCOME: 2/4 = 0.5 ✓
            helpful_concrete_action=True,
            helpful_impact_on_others=True,
            # HARMLESS_OUTCOME: 2/4 = 0.5 ✓
            harmless_no_dehumanizing=True,
            harmless_no_exploitation=True,
        )
        quality, passed, final = calculate_final_score(sig)
        assert quality > 0.0
        assert passed is False
        assert final == 0.0

    def test_dimension_at_exactly_threshold_passes(self):
        """Dimension at exactly 0.4 → passed=True (>= not >)."""
        sig = BinarySignals(
            # IDENTIFYING: 2/5 = 0.4 (exactly at threshold)
            identifying_self_interest=True,
            identifying_other_interest=True,
            # LOGICAL_PROCESS: 2/5 = 0.4
            logical_weighing=True,
            logical_self_cost=True,
            # CLEAR_PROCESS: 2/4 = 0.5
            clear_arc=True,
            clear_consistent=True,
            # HELPFUL_OUTCOME: 2/4 = 0.5
            helpful_concrete_action=True,
            helpful_impact_on_others=True,
            # HARMLESS_OUTCOME: 2/4 = 0.5
            harmless_no_dehumanizing=True,
            harmless_no_exploitation=True,
        )
        _, passed, _ = calculate_final_score(sig)
        assert passed is True

    def test_custom_threshold_parameter(self):
        """Custom threshold=0.6 makes 2/5=0.4 dimension fail."""
        sig = BinarySignals(
            # IDENTIFYING: 2/5 = 0.4 (below 0.6 threshold)
            identifying_self_interest=True,
            identifying_other_interest=True,
            # LOGICAL_PROCESS: 4/5 = 0.8
            logical_weighing=True,
            logical_self_cost=True,
            logical_alternatives=True,
            logical_connected_conclusion=True,
            # CLEAR_PROCESS: 3/4 = 0.75
            clear_arc=True,
            clear_consistent=True,
            clear_follows=True,
            # HELPFUL_OUTCOME: 3/4 = 0.75
            helpful_concrete_action=True,
            helpful_impact_on_others=True,
            helpful_relationship=True,
            # HARMLESS_OUTCOME: 3/4 = 0.75
            harmless_no_dehumanizing=True,
            harmless_no_exploitation=True,
            harmless_no_dismissal=True,
        )
        quality, passed, final = calculate_final_score(sig, threshold=0.6)
        assert quality > 0.5
        assert passed is False
        assert final == 0.0

        # Same signals with lower threshold should pass
        _, passed2, final2 = calculate_final_score(sig, threshold=0.4)
        assert passed2 is True
        assert final2 > 0.0

    def test_all_false_fails(self):
        sig = BinarySignals()
        _, passed, final = calculate_final_score(sig)
        assert passed is False
        assert final == 0.0


# ---------------------------------------------------------------------------
# Final score calculation tests
# ---------------------------------------------------------------------------

class TestFinalScore:
    def test_all_true_scores_1(self):
        sig = BinarySignals(**{s: True for s in ALL_SIGNALS})
        quality, passed, final = calculate_final_score(sig)
        assert quality == pytest.approx(1.0)
        assert passed is True
        assert final == pytest.approx(1.0)

    def test_all_false_scores_0(self):
        sig = BinarySignals()
        quality, passed, final = calculate_final_score(sig)
        assert quality == pytest.approx(0.0)
        assert passed is False
        assert final == pytest.approx(0.0)

    def test_one_weak_dimension_zeros_final(self):
        """High quality overall but one dimension below threshold → final = 0."""
        kwargs = {s: True for s in ALL_SIGNALS}
        # Zero out all harmless signals → harmless_outcome = 0.0
        kwargs["harmless_no_dehumanizing"] = False
        kwargs["harmless_no_exploitation"] = False
        kwargs["harmless_no_dismissal"] = False
        kwargs["harmless_acknowledges_cost"] = False
        sig = BinarySignals(**kwargs)
        quality, passed, final = calculate_final_score(sig)
        assert quality > 0.5  # quality is still decent (other 4 dims at 1.0)
        assert passed is False
        assert final == 0.0

    def test_spec_example_dimension_scores(self):
        """Reproduce the example from spec: dimension scores [0.8, 0.6, 0.75, 0.5, 1.0] → quality 0.73."""
        sig = BinarySignals(
            # IDENTIFYING: 4/5 = 0.8
            identifying_self_interest=True,
            identifying_other_interest=True,
            identifying_conflict=True,
            identifying_assumption=True,
            identifying_third_party=False,
            # LOGICAL_PROCESS: 3/5 = 0.6
            logical_weighing=True,
            logical_self_cost=True,
            logical_alternatives=True,
            logical_connected_conclusion=False,
            logical_temporal=False,
            # CLEAR_PROCESS: 3/4 = 0.75
            clear_arc=True,
            clear_consistent=True,
            clear_follows=True,
            clear_reconstructable=False,
            # HELPFUL_OUTCOME: 2/4 = 0.5
            helpful_concrete_action=True,
            helpful_impact_on_others=True,
            helpful_relationship=False,
            helpful_conditions_for_different_choice=False,
            # HARMLESS_OUTCOME: 4/4 = 1.0
            harmless_no_dehumanizing=True,
            harmless_no_exploitation=True,
            harmless_no_dismissal=True,
            harmless_acknowledges_cost=True,
        )
        ds = calculate_dimension_scores(sig)
        assert ds.identifying == pytest.approx(0.8)
        assert ds.logical_process == pytest.approx(0.6)
        assert ds.clear_process == pytest.approx(0.75)
        assert ds.helpful_outcome == pytest.approx(0.5)
        assert ds.harmless_outcome == pytest.approx(1.0)

        quality = calculate_quality_score(ds)
        assert quality == pytest.approx(0.73)

        # This example passes per-dimension threshold (all dims >= 0.4)
        _, passed, final = calculate_final_score(sig)
        assert passed is True
        assert final == pytest.approx(0.73)
