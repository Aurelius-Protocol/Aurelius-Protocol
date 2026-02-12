"""Tests for moral reasoning strictness modes."""

import pytest

from aurelius.validator.experiments.moral_reasoning.strictness import (
    STRICTNESS_PRESETS,
    StrictnessParams,
    resolve_strictness_params,
)


class TestStrictnessPresets:
    """Verify preset existence and values."""

    def test_all_three_presets_exist(self):
        assert "low" in STRICTNESS_PRESETS
        assert "medium" in STRICTNESS_PRESETS
        assert "high" in STRICTNESS_PRESETS

    def test_low_matches_current_defaults(self):
        """Low preset should match the original hardcoded defaults."""
        low = STRICTNESS_PRESETS["low"]
        assert low.quality_threshold == 0.4
        assert low.suspicious_high_signal_count == 20
        assert low.suspicious_min_response_length == 500
        assert low.suspicious_perfect_score_count == 22
        assert low.velocity_high_signal_threshold == 20
        assert low.velocity_flag_ratio == 0.5
        assert low.min_submissions == 1

    def test_medium_stricter_than_low(self):
        low = STRICTNESS_PRESETS["low"]
        medium = STRICTNESS_PRESETS["medium"]
        assert medium.quality_threshold > low.quality_threshold
        assert medium.suspicious_high_signal_count < low.suspicious_high_signal_count
        assert medium.suspicious_min_response_length > low.suspicious_min_response_length
        assert medium.velocity_flag_ratio < low.velocity_flag_ratio

    def test_high_stricter_than_medium(self):
        medium = STRICTNESS_PRESETS["medium"]
        high = STRICTNESS_PRESETS["high"]
        assert high.quality_threshold > medium.quality_threshold
        assert high.suspicious_high_signal_count < medium.suspicious_high_signal_count
        assert high.suspicious_min_response_length > medium.suspicious_min_response_length
        assert high.velocity_flag_ratio < medium.velocity_flag_ratio

    def test_min_submissions_always_one(self):
        for mode, preset in STRICTNESS_PRESETS.items():
            assert preset.min_submissions == 1, f"{mode} min_submissions should be 1"


class TestResolveStrictnessParams:
    """Verify resolution logic and override ordering."""

    def test_default_returns_low(self):
        result = resolve_strictness_params()
        assert result == STRICTNESS_PRESETS["low"]

    def test_select_medium(self):
        result = resolve_strictness_params("medium")
        assert result == STRICTNESS_PRESETS["medium"]

    def test_select_high(self):
        result = resolve_strictness_params("high")
        assert result == STRICTNESS_PRESETS["high"]

    def test_unknown_mode_falls_back_to_low(self):
        result = resolve_strictness_params("unknown_mode")
        assert result == STRICTNESS_PRESETS["low"]

    def test_empty_string_falls_back_to_low(self):
        result = resolve_strictness_params("")
        assert result == STRICTNESS_PRESETS["low"]

    def test_field_override_applied(self):
        result = resolve_strictness_params("low", {"quality_threshold": 0.55})
        assert result.quality_threshold == 0.55
        # Other fields remain from low preset
        assert result.suspicious_high_signal_count == 20

    def test_multiple_field_overrides(self):
        result = resolve_strictness_params("medium", {
            "quality_threshold": 0.7,
            "velocity_flag_ratio": 0.4,
        })
        assert result.quality_threshold == 0.7
        assert result.velocity_flag_ratio == 0.4
        # Remaining fields from medium
        assert result.suspicious_high_signal_count == 19

    def test_invalid_field_names_ignored(self):
        result = resolve_strictness_params("low", {"nonexistent_field": 999})
        assert result == STRICTNESS_PRESETS["low"]

    def test_none_overrides_returns_preset(self):
        result = resolve_strictness_params("high", None)
        assert result == STRICTNESS_PRESETS["high"]

    def test_empty_overrides_returns_preset(self):
        result = resolve_strictness_params("high", {})
        assert result == STRICTNESS_PRESETS["high"]

    def test_override_wins_over_preset(self):
        """Per-field override should take precedence over the mode preset."""
        # Start from high (quality_threshold=0.8), override to 0.1
        result = resolve_strictness_params("high", {"quality_threshold": 0.1})
        assert result.quality_threshold == 0.1
        # Rest stays high
        assert result.suspicious_min_response_length == 1000

    def test_result_is_frozen_dataclass(self):
        result = resolve_strictness_params("low")
        assert isinstance(result, StrictnessParams)
        with pytest.raises(AttributeError):
            result.quality_threshold = 0.99  # type: ignore[misc]
