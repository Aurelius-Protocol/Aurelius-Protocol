# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Closed, deterministic Verdict JSON v3 construction (lane plan R1 / frozen seam S1).

The public verdict is deliberately smaller than the referee's private diagnostics: closed base
objects carry only replay inputs, nested path/traceback/notes fields are forbidden, and every
guarded path returns a newline-inclusive 64 KiB-safe document.  This M1 capsule closes the S1
contract but does not flip a platform §12 invariant row.
"""

from __future__ import annotations

CONTENT_CODES = frozenset(
    {
        "artifact_hash_mismatch",
        "artifact_layout_invalid",
        "artifact_too_large",
        "cfg_invalid",
        "cfg_architecture_not_allowed",
        "cfg_pin_mismatch",
        "bad_shape",
        "nonfinite_weights",
        "load_error",
        "l0_out_of_band",
        "l0_out_of_band_family",
        "l0_tail_out_of_band",
        "dead_features",
        "copy_of_public",
        "copy_of_prior_submission",
        "below_random_floor",
        "below_interp_floor",
        "below_coronation_floor",
        "catastrophic_regression",
        "battery_feature_injection",
    }
)


ZERO_SHARE_PASS_CODES = frozenset(
    {
        "screen_reject",
        "ce_gap_below_epsilon",
        "ce_not_better",
        "tiebreak_incumbent_holds",
        "relative_needs_decisive_win",
    }
)


THRONE_PASS_CODES = frozenset({"decisive_ce", "tiebreak_battery_win", "first_incumbent"})


EVALUATED_PASS_CODES = ZERO_SHARE_PASS_CODES | THRONE_PASS_CODES | {"evaluation_complete"}


INFRASTRUCTURE_CODES = frozenset(
    {
        "infra_offline_miss",
        "infra_disk_guard",
        "archive_full",
        "infra_child_failure",
        "infra_broker_failure",
        "infra_oom",
        "infra_timeout",
        "infra_referee_failure",
        "infra_invalid_setup",
        "infra_invalid_output",
        # ADR-0229 D3: the same-draw no-features LLM baseline did not repeat within one epoch.
        # Epoch-level, never per challenger: nobody crowned, refund-eligible for every entry.
        "infra_battery_baseline_drift",
    }
)


def failure_class_for_reason(code: object) -> str:
    """Return the exact S8 class; unknown values are ambiguous without prefix inference."""

    if not isinstance(code, str):
        return "ambiguous"
    if code in CONTENT_CODES:
        return "content"
    if code in EVALUATED_PASS_CODES:
        return "pass"
    if code in INFRASTRUCTURE_CODES:
        return "infra"
    return "ambiguous"
