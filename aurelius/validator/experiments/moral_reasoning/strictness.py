"""Strictness modes for the Moral Reasoning Experiment.

Configurable presets (low/medium/high) control quality thresholds,
suspicious output detection sensitivity, and velocity flagging parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrictnessParams:
    """Parameters controlled by the strictness mode."""

    quality_threshold: float = 0.4
    suspicious_high_signal_count: int = 20
    suspicious_min_response_length: int = 500
    suspicious_perfect_score_count: int = 23
    velocity_high_signal_threshold: int = 20
    velocity_flag_ratio: float = 0.5
    min_submissions: int = 1


STRICTNESS_PRESETS: dict[str, StrictnessParams] = {
    "low": StrictnessParams(
        quality_threshold=0.4,
        suspicious_high_signal_count=20,
        suspicious_min_response_length=500,
        suspicious_perfect_score_count=23,
        velocity_high_signal_threshold=20,
        velocity_flag_ratio=0.5,
        min_submissions=1,
    ),
    "medium": StrictnessParams(
        quality_threshold=0.6,
        suspicious_high_signal_count=19,
        suspicious_min_response_length=750,
        suspicious_perfect_score_count=23,
        velocity_high_signal_threshold=18,
        velocity_flag_ratio=0.35,
        min_submissions=1,
    ),
    "high": StrictnessParams(
        quality_threshold=0.8,
        suspicious_high_signal_count=18,
        suspicious_min_response_length=1000,
        suspicious_perfect_score_count=23,
        velocity_high_signal_threshold=16,
        velocity_flag_ratio=0.25,
        min_submissions=1,
    ),
}


def resolve_strictness_params(
    mode: str = "low",
    setting_overrides: dict[str, Any] | None = None,
) -> StrictnessParams:
    """Resolve strictness parameters from a mode preset + per-field overrides.

    Resolution order (highest wins):
        1. Per-field overrides from ``setting_overrides``
        2. Mode preset selected by ``mode``
        3. Fallback to "low" if mode is unknown

    Args:
        mode: One of "low", "medium", "high". Unknown values fall back to "low".
        setting_overrides: Optional dict of per-field overrides (e.g.
            ``{"quality_threshold": 0.55}``). Keys that don't match
            StrictnessParams fields are silently ignored.

    Returns:
        Resolved StrictnessParams instance.
    """
    preset = STRICTNESS_PRESETS.get(mode, STRICTNESS_PRESETS["low"])

    if not setting_overrides:
        return preset

    # Only apply overrides for fields that exist on StrictnessParams
    valid_fields = {f.name for f in preset.__dataclass_fields__.values()}
    overrides = {k: v for k, v in setting_overrides.items() if k in valid_fields}

    if not overrides:
        return preset

    # Merge: start from preset, overlay valid overrides
    merged = {f: getattr(preset, f) for f in valid_fields}
    merged.update(overrides)
    return StrictnessParams(**merged)
