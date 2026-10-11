# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Miner-facing observations and explanations; never scoring or allocation authority."""

from __future__ import annotations
from urllib.parse import urlencode
import re
from subnet.miner.read_client import ReadUnavailable, request_json
from .referee.verdict import failure_class_for_reason

EXPLANATIONS = {
    "cfg_pin_mismatch": "Match the signed model, hook, dimensions and preprocessing; run the bundle check.",
    "bad_shape": "Export the exact tensor names, shapes and dtypes for the supported native SAE format.",
    "nonfinite_weights": "Weights contain NaN or infinity; inspect training stability and export a finite checkpoint.",
    "l0_out_of_band": "Mean active features per token is outside the signed band. Compare local sparsity measurements.",
    "l0_out_of_band_family": "At least one corpus family is outside its signed sparsity band.",
    "l0_tail_out_of_band": "High-activation tokens or windows exceed the signed ceiling.",
    "dead_features": "Too many features did not fire on this draw; inspect feature usage during training.",
    "copy_of_public": "The decoder matched the public reference archive under the existing copy rule.",
    "copy_of_prior_submission": "The decoder matched an archived submission under the existing copy rule.",
    "below_random_floor": "Recovered CE did not exceed the required random baseline plus margin.",
    "below_interp_floor": "The measured interpretability battery did not meet the applicable floor.",
    "below_coronation_floor": "Recovered CE did not meet the crown floor.",
    "catastrophic_regression": "A battery component regressed beyond the allowed margin.",
    "screen_reject": "The candidate was evaluated but did not advance past the competitive screen.",
    "ce_gap_below_epsilon": "The same-draw CE improvement is below epsilon; the incumbent holds.",
    "ce_not_better": "The candidate did not improve on the incumbent's same-draw CE.",
    "tiebreak_incumbent_holds": "The existing tie rule retains the incumbent.",
    "relative_needs_decisive_win": "The candidate did not satisfy the existing decisive-win requirement.",
    "decisive_ce": "The candidate achieved the required decisive CE improvement.",
    "first_incumbent": "The candidate satisfied the requirements to occupy the vacant slot.",
    "evaluation_complete": "Scientific evaluation completed. Consult the ladder separately for position.",
}


def explain_verdict(row: dict) -> dict:
    transcript = row.get("transcript", {})
    code = row.get("primary_code", transcript.get("primary_code"))
    classification = failure_class_for_reason(code)
    checks = transcript.get("checks", {})
    battery = checks.get("battery")
    diagnostics = transcript.get("gates", {}).get("diagnostics")
    if diagnostics is not None:
        from .diagnostics import validate_diagnostics

        diagnostics = validate_diagnostics(diagnostics)
    return {
        "code": code,
        "failure_class": classification,
        "explanation": EXPLANATIONS.get(
            code,
            "Infrastructure did not complete evaluation; this is not a proven candidate defect."
            if classification == "infra"
            else "Unrecognized outcome; keep it ambiguous and retain the result for review."
            if classification == "ambiguous"
            else "Consult the measured checks and signed slot requirements.",
        ),
        "validator_hotkey": row.get("validator_hotkey"),
        "accepted": transcript.get("accepted"),
        "ce_recovered": checks.get("ce_loss_score"),
        "diagnostics": diagnostics,
        "battery": {
            "status": "measured" if battery is not None else "not_reported",
            "values": battery,
        },
        "paired": transcript.get("paired") or None,
        "mechanism": transcript.get("mechanism") or None,
        "crowned": transcript.get("coronation", {}).get("crowned"),
        "rank": row.get("rank"),
        "weight": {"status": "not_observed", "value": None},
        "refund": {"status": "not_observed", "value": None},
    }


def fetch_replay(base_url: str, identity: dict, *, fetch=request_json) -> dict | None:
    """Read and bind every page for one attempt; a missing epoch is pending, not failed."""
    query = {
        "slot": identity["slot_id"],
        "epoch": identity["slot_epoch"],
        "content_hash": identity["content_hash"],
        "limit": 8,
    }
    seen = set()
    result = None
    row_ids = set()
    for _ in range(64):
        try:
            page = fetch(base_url.rstrip("/") + "/api/verdicts?" + urlencode(query))
        except ReadUnavailable as exc:
            if exc.status == 404 and exc.code == "epoch_not_found" and result is None:
                return None
            raise
        manifest = page.get("manifest")
        if not isinstance(manifest, dict) or (manifest.get("slot_id"), manifest.get("epoch")) != (
            identity["slot_id"],
            identity["slot_epoch"],
        ):
            raise ReadUnavailable("manifest_identity_mismatch")
        if (
            not isinstance(page.get("manifest_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", page["manifest_sha256"]) is None
        ):
            raise ReadUnavailable("invalid_manifest_digest")
        if result is None:
            result = {
                "manifest": manifest,
                "manifest_sha256": page.get("manifest_sha256"),
                "derived_ladder": page.get("derived_ladder"),
                "verdicts": [],
            }
        elif page.get("manifest_sha256") != result["manifest_sha256"]:
            raise ReadUnavailable("manifest_changed")
        rows = page.get("verdicts")
        if not isinstance(rows, list) or len(rows) > 8:
            raise ReadUnavailable("invalid_verdict_page")
        for row in rows:
            if not isinstance(row, dict):
                raise ReadUnavailable("invalid_verdict")
            if row.get("hotkey") != identity["hotkey"]:
                continue  # the same content can have multiple committing hotkeys
            if (row.get("content_hash"), row.get("artifact_sha256"), row.get("fanout_window")) != (
                identity["content_hash"],
                identity["artifact_sha256"],
                identity["fanout_window"],
            ):
                raise ReadUnavailable("verdict_identity_mismatch")
            identifier = row.get("id")
            if not isinstance(identifier, str) or identifier in row_ids:
                raise ReadUnavailable("duplicate_or_invalid_verdict")
            row_ids.add(identifier)
            result["verdicts"].append(row)
        cursor = page.get("verdict_page", {}).get("next_cursor")
        if cursor is None:
            return result
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 384 or cursor in seen:
            raise ReadUnavailable("invalid_pagination")
        seen.add(cursor)
        query["cursor"] = cursor
    raise ReadUnavailable("page_limit_exceeded")


def refresh(journal, identity: dict, base_url: str, *, fetch=request_json) -> dict:
    try:
        replay = fetch_replay(base_url, identity, fetch=fetch)
        if replay is None:
            return {"status": "waiting_for_epoch", "results": []}
        journal.record(
            identity,
            "epoch_closed",
            "public_verdict",
            {"manifest_sha256": replay["manifest_sha256"]},
        )
        commits = replay["manifest"].get("commits")
        if isinstance(commits, list) and not any(
            isinstance(commit, dict)
            and all(
                commit.get(key) == identity[key]
                for key in ("hotkey", "content_hash", "artifact_sha256")
            )
            for commit in commits
        ):
            return {
                "status": "not_in_epoch_manifest",
                "results": [],
                "ladder": replay["derived_ladder"],
            }
        reports = []
        for row in replay["verdicts"]:
            report = explain_verdict(row)
            journal.record(
                identity, "result", "public_verdict", {"id": row["id"], "report": report}
            )
            reports.append(report)
        return {
            "status": "results_available" if reports else "waiting_for_result",
            "results": reports,
            "ladder": replay["derived_ladder"],
        }
    except (ReadUnavailable, ValueError, TypeError, AttributeError, KeyError) as exc:
        code = exc.code if isinstance(exc, ReadUnavailable) else "invalid_diagnostics"
        journal.record(identity, "status_unavailable", "public_verdict", {"code": code})
        return {"status": "unavailable", "reason": code, "results": []}
