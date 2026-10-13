# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Local SAE inspection and schedule preview. No keys, network, execution or economic writes.

Uses the production byte/ABI validators; readiness describes this bundle and policy snapshot,
not a future scientific result. All independent metadata mismatches are shown together.
"""

from __future__ import annotations
import json
from pathlib import Path
from .referee.gate0 import SlotPins, artifact_digest, validate_artifact
from .referee.native_sae import native_tensor_schemas
from .referee.runtime import inspect_safetensors
from .stage import _strict_json_object, eligible_fanout_window


def slot_pins(policy: dict) -> SlotPins:
    names = (
        "d_in",
        "d_sae",
        "normalize_activations",
        "reshape_activations",
        "apply_b_dec_to_input",
        "rescale_acts_by_decoder_norm",
        "topk_k_min",
        "topk_k_max",
    )
    return SlotPins(
        **{name: policy[name] for name in names},
        dtype=policy["sae_dtype"],
        model_name=policy["model"],
        hook_name=policy["hook_id"],
        seqpos_slice=json.loads(policy["seqpos_slice_json"]),
    )


def metadata_checks(cfg: dict, policy: dict) -> list[dict]:
    """Report only fixed public field names; don't echo arbitrary cfg/metadata objects."""
    metadata = cfg.get("metadata") if isinstance(cfg.get("metadata"), dict) else {}
    pairs = [
        (name, cfg.get(name), policy[name])
        for name in (
            "d_in",
            "d_sae",
            "normalize_activations",
            "reshape_activations",
            "apply_b_dec_to_input",
        )
    ]
    pairs.append(("dtype", cfg.get("dtype"), policy["sae_dtype"]))
    for name, expected in {
        "model_name": policy["model"],
        "hook_name": policy["hook_id"],
        "seqpos_slice": json.loads(policy["seqpos_slice_json"]),
    }.items():
        pairs.append(("metadata." + name, metadata.get(name), expected))
    if cfg.get("architecture") == "topk":
        pairs.append(
            (
                "rescale_acts_by_decoder_norm",
                cfg.get("rescale_acts_by_decoder_norm"),
                policy["rescale_acts_by_decoder_norm"],
            )
        )
    rows = [
        {
            "field": name,
            "observed": actual,
            "expected": expected,
            "status": "pass" if type(actual) is type(expected) and actual == expected else "fail",
        }
        for name, actual, expected in pairs
    ]
    # ADR-0229 D1: training provenance is reported for the miner's information only. Gate 0
    # requires the field and its JSON type, never equality with the slot's own training recipe.
    for name, expected_type in (
        ("prepend_bos", bool),
        ("context_size", int),
        ("exclude_special_tokens", bool),
    ):
        actual = metadata.get(name)
        typed = type(actual) is expected_type
        rows.append(
            {
                "field": "metadata." + name,
                "observed": actual if typed else None,
                "expected": policy[name],
                "status": "info" if typed else "fail",
            }
        )
    return rows


def check_bundle(bundle: str | Path, policy: dict) -> dict:
    root = Path(bundle)
    report = {
        "schema_version": 1,
        "kind": "sae_bundle_check",
        "compatible": False,
        "slot_id": policy["_slot_id"],
        "artifact_sha256": None,
        "artifact_bytes": None,
        "checks": [],
        "tensor_status": "not_run",
        "scientific_status": "not_run",
    }
    try:
        cap = policy["artifact_max_bytes"]
        digest = artifact_digest(root, max_bytes=cap)
        report["artifact_sha256"] = digest
        # Layout/digest pass precedes metadata reads and full tensor inspection.
        cfg = _strict_json_object((root / "cfg.json").read_bytes())
        report["checks"] = metadata_checks(cfg, policy)
        report["artifact_bytes"] = sum(
            (root / name).stat().st_size for name in ("cfg.json", "sae_weights.safetensors")
        )
        if any(row["status"] == "fail" for row in report["checks"]):
            return report
        pins = slot_pins(policy)
        artifact = validate_artifact(root, pins=pins, expected_sha256=digest, max_bytes=cap)
        schema = native_tensor_schemas().get(cfg["architecture"])
        if schema is None:
            raise ValueError("unsupported native inference architecture")
        schema.cfg_validator(artifact.cfg, pins)
        infos = inspect_safetensors(
            root / "sae_weights.safetensors",
            schema=schema,
            d_in=pins.d_in,
            d_sae=pins.d_sae,
            max_bytes=cap,
        )
        artifact_digest(root, max_bytes=cap, expected_sha256=digest)
        report.update(
            compatible=True,
            tensor_status="pass",
            tensor_count=len(infos),
            tensor_abi=schema.implementation_id,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        report["error"] = {
            "code": str(getattr(exc, "code", "bundle_invalid")),
            "message": str(exc)[:500],
        }
    return report


def window_preview(
    policy: dict, *, head: int, epoch: int | None = None, window_blocks: int = 100
) -> dict:
    if type(head) is not int or head < 0 or type(window_blocks) is not int or window_blocks < 1:
        raise ValueError("head/window must be non-negative/positive integers")
    cadence = policy["cadence_blocks"]
    offset = policy["phase_offset_blocks"]
    if (
        type(cadence) is not int
        or cadence < window_blocks
        or cadence % window_blocks
        or offset % window_blocks
        or type(offset) is not int
        or offset < 0
    ):
        raise ValueError("invalid signed cadence")
    first = max(0, (2 * window_blocks - offset + cadence - 1) // cadence)
    next_epoch = max(first, (head + window_blocks + 1 - offset + cadence - 1) // cadence)
    selected = next_epoch if epoch is None else epoch
    window = eligible_fanout_window(
        slot_epoch=selected,
        cadence_blocks=cadence,
        phase_offset_blocks=offset,
        window_blocks=window_blocks,
    )
    start = window * window_blocks
    end = start + window_blocks
    return {
        "slot_id": policy["_slot_id"],
        "slot_epoch": selected,
        "fanout_window": window,
        "commit_start": start,
        "commit_end_exclusive": end,
        "head": head,
        "remaining_blocks": max(0, end - head),
        "opens_in_blocks": max(0, start - head),
        "next_eligible_epoch": next_epoch,
        "status": "expired" if head >= end else "open" if head >= start else "upcoming",
    }
