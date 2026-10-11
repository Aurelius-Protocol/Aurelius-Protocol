"""Lane-owned miner staging tool for the flat interp_sae v2 envelope (M4/N1).

The generic miner never imports this module. This tool reads the signed operator snapshot, accepts
exactly one shadow/enabled slot, authenticates the exact two-file SAE-Lens bundle with Gate 0's S4
digest, binds the declared scalar config to cfg.json and the signed slot pins, and appends one
window-scoped candidate. It executes no artifact and contains no payment or scoring logic.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any, Mapping

from subnet.lanes.interp_sae.referee.gate0 import (
    CFG_FILENAME,
    DEFAULT_ARTIFACT_MAX_BYTES,
    MAX_CFG_BYTES,
    WEIGHTS_FILENAME,
    artifact_digest,
)
from subnet.miner.config_schema import SchemaUnavailable, load_published_entries
from subnet.miner.staging import ELIGIBLE_FANOUT_WINDOW, StagingPool
from subnet.shared.artifact_ingress_ref import is_aurelius_artifact_ref, parse_aurelius_artifact_ref
from subnet.shared.windows import AUTHORITATIVE_WINDOW_BLOCKS
from subnet.shared.sn455_artifact_delivery import Sn455ArtifactAdmission, require_artifact_admission

TASK_TYPE = "interp_sae"
SCHEMA_VERSION = 2
ARTIFACT_FORMAT = "aurelius.sae-lens.v1"
ACTIVATION_RUNTIME = "transformer_lens"
HOOK_NAMESPACE = "transformer_lens"

_REVISION_RE = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SLOT_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9._-]{0,127}\Z")
_VERSION_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}\Z")
# Absolute-end anchors, not `^...$`: _HF_REF_RE and _SPARSITY_RE are asserted byte-EQUAL to the
# adapter's compiled counterparts (test_interp_sae_miner_stage.py), and the adapter's flow into a
# JSON-schema `pattern` — a SEARCH whose `$` also matches before a trailing newline — is what
# makes that equality load-bearing. Every consumer here full-matches, so this file's own verdicts
# are unchanged; the equality keeps the two engines from drifting. _REPO_RE and _COMMIT_RE are
# builder-side piece gates with NO adapter-side compiled counterpart to pin byte-for-byte. They are
# POST-conditions of the composed `_HF_REF_RE.fullmatch` that runs before them (the composed fullmatch runs first, in the same function), so
# they can only narrow it, never widen it. Their segments carry the composed grammar's {0,95} bound
# for the same reason: unbounded segments made _REPO_RE alone a superset admitting a 200-character
# org the adapter refuses — harmless while it is never consulted alone, and not worth leaving as a
# trap (security r2 P2, reviewer-python r2). _SPARSITY_RE uses [0-9],
# never \d — Python's \d is Unicode-wide, so \d would accept digits the adapter and Central
# both refuse.
_REPO_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_COMMIT_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_HF_REF_RE = re.compile(
    r"\Ahf://[A-Za-z0-9][A-Za-z0-9._-]{0,95}/"
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}@[0-9a-f]{40}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._/-]{0,1023})?\Z"
)
_SPARSITY_RE = re.compile(r"\A(?:0(?:\.[0-9]+)?|1(?:\.0+)?)\Z")
_HOOK_POINTS = frozenset({"resid_pre", "resid_post", "mlp_out", "attn_out"})
_TRAINING_ARCHITECTURES = frozenset(
    {"standard", "gated", "jumprelu", "topk", "batchtopk", "matryoshka_batchtopk"}
)
_TOPK_ARCHITECTURES = frozenset({"topk", "batchtopk", "matryoshka_batchtopk"})
_MODEL_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_SELECTABLE_STATUSES = frozenset({"shadow", "enabled"})
_S7_OPTIONAL_SUFFIXES = frozenset({"constants_provenance"})
_S7_SUFFIXES = frozenset(
    {
        "model",
        "revision",
        "runtime",
        "hook_id",
        "layer",
        "hook_point",
        "d_in",
        "d_sae",
        "model_dtype",
        "sae_dtype",
        "model_load_processing",
        "normalize_activations",
        "reshape_activations",
        "apply_b_dec_to_input",
        "rescale_acts_by_decoder_norm",
        "prepend_bos",
        "context_size",
        "exclude_special_tokens",
        "seqpos_slice_json",
        "topk_k_min",
        "topk_k_max",
        "draw_chars_per_family",
        "draw_chunk_chars",
        "artifact_max_bytes",
        "archive_per_hotkey_cap",
        "archive_total_bytes_cap",
        "slot_weight",
        "cadence_blocks",
        "phase_offset_blocks",
        "status",
        "tiebreak",
        "l0_band",
        "l0_band_per_family",
        "l0_p95_max",
        "l0_max_window",
        "dead_feature_cap",
        "random_floor",
        "random_floor_delta",
        "delta_cat_probe",
        "delta_cat_abs",
        "epsilon_rotation",
        "screen_margin_W",
        "q_decisive",
        "coronation_floor_ce",
        "pool_version",
        "pool_sha256",
        "referee_sha",
        "referee_config_sha256",
        "profile_id",
        "sae_lens_version",
        "transformer_lens_version",
        "sae_bench_version",
        "python_version",
    }
)


class StageError(ValueError):
    """The requested local staging operation is malformed or not policy-eligible."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StageError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StageError(f"{name} must be a non-negative integer")
    return value


def _bounded_string(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\0" in value:
        raise StageError(f"{name} must be a bounded non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise StageError(f"{name} must be valid UTF-8") from None
    return value


def _unit_decimal_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SPARSITY_RE.fullmatch(value) is None:
        raise StageError(f"{name} must be an App-F decimal string in [0,1]")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise StageError(f"{name} must be an App-F decimal string in [0,1]") from None
    if not parsed.is_finite() or not Decimal(0) <= parsed <= Decimal(1):
        raise StageError(f"{name} must be an App-F decimal string in [0,1]")
    return value


def _strict_json_object(data: bytes) -> dict[str, Any]:
    if len(data) > MAX_CFG_BYTES:
        raise StageError("cfg.json exceeds the Gate 0 metadata byte cap")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StageError("cfg.json contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> Any:
        raise StageError("cfg.json contains a non-finite JSON constant")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StageError("cfg.json must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise StageError("cfg.json root must be an object")
    return value


def _admitted_artifact_ref(value: Any, *, digest: str, hotkey: str | None = None) -> str:
    """Admit hf:// as before, or a shared aurelius-artifact:// locator bound to the S4 digest.

    When ``hotkey`` is given, an ingress ref must name exactly that hotkey (an equality check
    only — the binding that matters is the validator's at ``ArtifactStore.materialize``).
    """
    ref = _bounded_string(value, "artifact_ref", maximum=1024)
    if is_aurelius_artifact_ref(ref):
        parsed = parse_aurelius_artifact_ref(ref)
        if parsed.digest != digest:
            raise StageError("artifact_ref digest differs from artifact_sha256")
        if hotkey is not None and parsed.hotkey != hotkey:
            raise StageError("artifact_ref hotkey differs from the expected staging hotkey")
        return ref
    if _HF_REF_RE.fullmatch(ref) is None:
        raise StageError(
            "artifact_ref must be immutable hf://owner/repo@commit[/subdir] "
            "or an exact aurelius-artifact://hotkey/digest locator"
        )
    if hotkey is not None:
        # Never let an identity binding pass silently on a grammar it does not apply to.
        raise StageError("--hotkey applies only to aurelius-artifact:// artifact refs")
    repo, revision_path = ref[5:].split("@", 1)
    commit, separator, subdir = revision_path.partition("/")
    if _REPO_RE.fullmatch(repo) is None or _COMMIT_RE.fullmatch(commit) is None:
        raise StageError("artifact_ref must pin an owner/repo and lowercase 40-hex commit")
    if separator and (
        not subdir
        or subdir.startswith("/")
        or "\\" in subdir
        or any(part in {"", ".", ".."} for part in subdir.split("/"))
    ):
        raise StageError("artifact_ref subdir is unsafe")
    return ref


def _seqpos_slice(value: Any, name: str) -> tuple[int | None, ...] | None:
    source = _bounded_string(value, name, maximum=1024)
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as exc:
        raise StageError(f"{name} must be canonical compact JSON") from exc
    if parsed is not None:
        if not isinstance(parsed, list) or any(
            item is not None and (isinstance(item, bool) or not isinstance(item, int))
            for item in parsed
        ):
            raise StageError(f"{name} must encode null or an integer/null array")
    canonical = json.dumps(parsed, separators=(",", ":"), ensure_ascii=True)
    if source != canonical:
        raise StageError(f"{name} must be exact canonical compact JSON")
    return None if parsed is None else tuple(parsed)


def _slot_policies(entries: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    prefix = f"pertask.{TASK_TYPE}.slot."
    policies: dict[str, dict[str, Any]] = {}
    for key, value in entries.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        tail = key[len(prefix) :]
        if "." not in tail:
            raise StageError("signed slot policy key is malformed")
        slot_id, suffix = tail.rsplit(".", 1)
        if (
            _SLOT_ID_RE.fullmatch(slot_id) is None
            or suffix not in _S7_SUFFIXES | _S7_OPTIONAL_SUFFIXES
        ):
            raise StageError(f"signed slot policy key {key!r} is not in the closed S7 schema")
        policy = policies.setdefault(slot_id, {})
        if suffix in policy:
            raise StageError(f"signed slot policy duplicates {slot_id}.{suffix}")
        policy[suffix] = value
    for slot_id, policy in policies.items():
        if not _S7_SUFFIXES <= set(policy):
            missing = sorted(_S7_SUFFIXES - set(policy))
            raise StageError(f"signed slot {slot_id!r} is incomplete (missing={missing})")
        # ADR-0229: match the validator's optional provenance key and fail-closed default.
        provenance = policy.get("constants_provenance", "provisional")
        if provenance == "operator_accepted" and policy.get("profile_id") != "gh200-sn37-mainnet":
            raise StageError("operator_accepted constants require the SN37 production profile")
        if provenance not in ("provisional", "calibrated", "operator_accepted"):
            raise StageError(f"slot.{slot_id}.constants_provenance is invalid")
        if policy.get("status") == "enabled" and provenance not in {
            "calibrated",
            "operator_accepted",
        }:
            raise StageError(
                f"slot.{slot_id}.status enabled requires calibrated or operator_accepted constants"
            )
    return policies


def selectable_slot_policy(entries: Mapping[str, Any], slot_id: str) -> dict[str, Any]:
    """Select one signed shadow/enabled slot; disabled slot 2 and multi-slot policy fail closed."""

    requested = _bounded_string(slot_id, "slot_id", maximum=128)
    if _SLOT_ID_RE.fullmatch(requested) is None:
        raise StageError("slot_id is not a bounded canonical identifier")
    policies = _slot_policies(entries)
    selectable = sorted(
        key for key, policy in policies.items() if policy.get("status") in _SELECTABLE_STATUSES
    )
    if selectable != [requested]:
        raise StageError(
            f"exactly the requested sole shadow/enabled slot is required (selectable={selectable})"
        )
    policy = policies[requested]
    try:
        weight = Decimal(policy["slot_weight"])
    except (InvalidOperation, TypeError, ValueError):
        raise StageError("slot_weight must be an App-F decimal string") from None
    if not isinstance(policy["slot_weight"], str) or not weight.is_finite() or weight != Decimal(1):
        raise StageError("the sole selectable slot must have slot_weight exactly 1")
    d8 = {
        field: Decimal(_unit_decimal_string(policy[field], f"slot.{requested}.{field}"))
        for field in ("random_floor", "random_floor_delta", "delta_cat_probe", "delta_cat_abs")
    }
    if d8["random_floor"] + d8["random_floor_delta"] > Decimal(1):
        raise StageError(f"slot.{requested}.random_floor + random_floor_delta must not exceed 1")
    selected = dict(policy)
    selected["_slot_id"] = requested
    return selected


def eligible_fanout_window(
    *, slot_epoch: int, cadence_blocks: int, phase_offset_blocks: int, window_blocks: int
) -> int:
    """Derive the one commit fanout window [B_e-2W, B_e-W) from signed cadence."""

    epoch = _nonnegative_int(slot_epoch, "slot_epoch")
    cadence = _positive_int(cadence_blocks, "cadence_blocks")
    offset = _nonnegative_int(phase_offset_blocks, "phase_offset_blocks")
    width = _positive_int(window_blocks, "window_blocks")
    boundary = epoch * cadence + offset
    commit_start = boundary - 2 * width
    if commit_start < 0 or commit_start % width:
        raise StageError("slot cadence does not produce one aligned non-negative commit window")
    return commit_start // width


def _bundle_cfg(bundle_dir: str | Path, *, max_bytes: int) -> tuple[dict[str, Any], str, int]:
    """Read a Gate-0-authenticated two-file bundle without executing or loading the artifact."""

    root = Path(bundle_dir)
    digest = artifact_digest(root, max_bytes=max_bytes)
    try:
        cfg_bytes = (root / CFG_FILENAME).read_bytes()
        cfg = _strict_json_object(cfg_bytes)
        size_bytes = (root / CFG_FILENAME).stat().st_size + (root / WEIGHTS_FILENAME).stat().st_size
    except OSError as exc:
        raise StageError("bundle changed while reading cfg.json") from exc
    # A second no-follow S4 pass detects any ordinary local mutation between digest and cfg parse.
    artifact_digest(root, max_bytes=max_bytes, expected_sha256=digest)
    return cfg, digest, size_bytes


def build_candidate(
    bundle_dir: str | Path,
    *,
    artifact_ref: str | None = None,
    slot_id: str,
    slot_epoch: int,
    sparsity_target: str,
    policy: Mapping[str, Any],
    window_blocks: int = AUTHORITATIVE_WINDOW_BLOCKS,
    local_artifact_admission: Sn455ArtifactAdmission | None = None,
    miner_hotkey: str | None = None,
    ref_hotkey: str | None = None,
) -> dict[str, Any]:
    """Build the exact flat v2 candidate from cfg.json, S4 bytes, and one signed slot policy.

    ``ref_hotkey`` (optional, non-local mode only) refuses an ingress ref naming another hotkey.
    """

    if local_artifact_admission is None:
        if miner_hotkey is not None:
            raise StageError("miner_hotkey requires explicit local artifact admission")
        # Optional operator identity for the ingress-locator equality check (never a payment
        # fact): an aurelius-artifact:// artifact_ref must name exactly this hotkey.
        expected_hotkey = (
            None if ref_hotkey is None else _bounded_string(ref_hotkey, "ref_hotkey", maximum=128)
        )
    else:
        if artifact_ref is not None:
            raise StageError("local artifact_ref is derived from bytes, not supplied by the caller")
        if ref_hotkey is not None:
            raise StageError("ref_hotkey applies only to the hf:// or aurelius-artifact:// path")
        try:
            require_artifact_admission(local_artifact_admission)
        except ValueError as exc:
            raise StageError(str(exc)) from exc
    if not isinstance(policy, Mapping):
        raise StageError("slot policy must be an object")
    if policy.get("_slot_id") != slot_id or _SLOT_ID_RE.fullmatch(slot_id) is None:
        raise StageError("slot_id does not match the selected signed slot")
    if policy.get("status") not in _SELECTABLE_STATUSES:
        raise StageError("slot policy must be shadow or enabled before staging")
    if policy.get("runtime") != ACTIVATION_RUNTIME:
        raise StageError("slot runtime must be exactly transformer_lens")
    revision = policy.get("revision")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise StageError("slot revision must be immutable lowercase 40/64-hex")
    runtime_version = policy.get("transformer_lens_version")
    if not isinstance(runtime_version, str) or _VERSION_RE.fullmatch(runtime_version) is None:
        raise StageError("slot transformer_lens_version must be a concrete pinned token")
    model = _bounded_string(policy.get("model"), "slot.model")
    hook_id = _bounded_string(policy.get("hook_id"), "slot.hook_id")
    layer = _nonnegative_int(policy.get("layer"), "slot.layer")
    hook_point = policy.get("hook_point")
    if hook_point not in _HOOK_POINTS:
        raise StageError(f"slot.hook_point must be one of {sorted(_HOOK_POINTS)}")
    d_sae = _positive_int(policy.get("d_sae"), "slot.d_sae")
    artifact_max_bytes = _positive_int(policy.get("artifact_max_bytes"), "slot.artifact_max_bytes")
    if artifact_max_bytes > DEFAULT_ARTIFACT_MAX_BYTES:
        raise StageError("slot.artifact_max_bytes exceeds the four-GiB outer cap")
    if policy.get("normalize_activations") != "none":
        raise StageError("slot.normalize_activations must be exactly none")
    if policy.get("model_dtype") not in _MODEL_DTYPES:
        raise StageError("slot.model_dtype is invalid")
    if policy.get("sae_dtype") not in _MODEL_DTYPES:
        raise StageError("slot.sae_dtype is invalid")
    model_load_processing = policy.get("model_load_processing")
    if model_load_processing not in {"none", "default"}:
        raise StageError("slot.model_load_processing is invalid")
    for name in ("apply_b_dec_to_input", "rescale_acts_by_decoder_norm", "prepend_bos"):
        if not isinstance(policy.get(name), bool):
            raise StageError(f"slot.{name} must be boolean")

    try:
        sparsity = Decimal(sparsity_target)
    except (InvalidOperation, TypeError, ValueError):
        raise StageError("sparsity_target must be an App-F decimal string in [0,1]") from None
    if (
        not isinstance(sparsity_target, str)
        or _SPARSITY_RE.fullmatch(sparsity_target) is None
        or not sparsity.is_finite()
        or not Decimal(0) <= sparsity <= Decimal(1)
    ):
        raise StageError("sparsity_target must be an App-F decimal string in [0,1]")

    cfg, digest, size_bytes = _bundle_cfg(bundle_dir, max_bytes=artifact_max_bytes)
    architecture = cfg.get("architecture")
    if architecture not in _TRAINING_ARCHITECTURES:
        raise StageError("cfg architecture is outside the production Gate 0 allowlist")
    cfg_d_sae = _positive_int(cfg.get("d_sae"), "cfg.d_sae")
    cfg_d_in = _positive_int(cfg.get("d_in"), "cfg.d_in")
    cfg_dtype = cfg.get("dtype")
    cfg_normalize = cfg.get("normalize_activations")
    cfg_reshape = cfg.get("reshape_activations")
    if cfg_reshape is not None and not isinstance(cfg_reshape, (bool, str)):
        raise StageError("cfg.reshape_activations must be boolean, string, or null")
    cfg_apply_b_dec = cfg.get("apply_b_dec_to_input")
    metadata = cfg.get("metadata")
    if not isinstance(metadata, dict):
        raise StageError("cfg.metadata must be an object")
    cfg_model = _bounded_string(metadata.get("model_name"), "cfg.metadata.model_name")
    cfg_hook = _bounded_string(metadata.get("hook_name"), "cfg.metadata.hook_name")
    prepend_bos = metadata.get("prepend_bos")
    if not isinstance(prepend_bos, bool):
        raise StageError("cfg.metadata.prepend_bos must be boolean")
    # ADR-0229 D1: context_size and exclude_special_tokens are training provenance — typed,
    # never bound to the slot (prepend_bos is still declared in the S3 config, unbound).
    _positive_int(metadata.get("context_size"), "cfg.metadata.context_size")
    if not isinstance(metadata.get("exclude_special_tokens"), bool):
        raise StageError("cfg.metadata.exclude_special_tokens must be boolean")
    cfg_seqpos = metadata.get("seqpos_slice")
    if cfg_seqpos is not None:
        if not isinstance(cfg_seqpos, list) or any(
            item is not None and (isinstance(item, bool) or not isinstance(item, int))
            for item in cfg_seqpos
        ):
            raise StageError("cfg.metadata.seqpos_slice must be null or an integer/null array")
        cfg_seqpos = tuple(cfg_seqpos)
    policy_seqpos = _seqpos_slice(policy.get("seqpos_slice_json"), "slot.seqpos_slice_json")
    if (
        cfg_model != model
        or cfg_hook != hook_id
        or cfg_d_in != policy.get("d_in")
        or cfg_d_sae != d_sae
        or cfg_dtype != policy["sae_dtype"]
        or cfg_normalize != policy["normalize_activations"]
        or cfg_reshape != policy.get("reshape_activations")
        or cfg_apply_b_dec is not policy.get("apply_b_dec_to_input")
        or cfg_seqpos != policy_seqpos
    ):
        # The local import keeps the helper reusable without changing the generic miner boundary.
        from .miner_checks import metadata_checks

        mismatches = [
            f"{row['field']}: expected {row['expected']!r}, got {row['observed']!r}"
            for row in metadata_checks(cfg, dict(policy))
            if row["status"] == "fail"
        ]
        raise StageError("cfg.json does not match the signed slot pins: " + "; ".join(mismatches))

    k: int | None = None
    if architecture in _TOPK_ARCHITECTURES:
        k = _positive_int(cfg.get("k"), "cfg.k")
        minimum = _positive_int(policy.get("topk_k_min"), "slot.topk_k_min")
        maximum = _positive_int(policy.get("topk_k_max"), "slot.topk_k_max")
        if not minimum <= k <= maximum:
            raise StageError("cfg.k is outside the signed TopK band")
        rescale_decoder = cfg.get("rescale_acts_by_decoder_norm")
        if not isinstance(rescale_decoder, bool):
            raise StageError("cfg.rescale_acts_by_decoder_norm must be boolean")
        if rescale_decoder is not policy.get("rescale_acts_by_decoder_norm"):
            raise StageError("cfg decoder-norm rescaling differs from the signed slot pin")
    elif "k" in cfg:
        raise StageError("cfg.k is forbidden for a non-TopK architecture")

    inference_architecture = (
        "jumprelu" if architecture in {"batchtopk", "matryoshka_batchtopk"} else architecture
    )
    if local_artifact_admission is None:
        selected_ref = _admitted_artifact_ref(artifact_ref, digest=digest, hotkey=expected_hotkey)
    else:
        try:
            selected_ref = local_artifact_admission.ref_for(miner_hotkey, digest)
            export = local_artifact_admission.require_export(miner_hotkey, digest)
            if sum(obj.size_bytes for obj in export.objects) != size_bytes:
                raise ValueError("bundle size differs from the admitted local export")
        except ValueError as exc:
            raise StageError(str(exc)) from exc
    config: dict[str, Any] = {
        "slot_id": slot_id,
        "slot_epoch": _nonnegative_int(slot_epoch, "slot_epoch"),
        "base_model": model,
        "base_model_revision": revision,
        "activation_runtime": ACTIVATION_RUNTIME,
        "activation_runtime_version": runtime_version,
        "hook_namespace": HOOK_NAMESPACE,
        "site_layer": layer,
        "site_hook_point": hook_point,
        "site_hook_id": hook_id,
        "training_architecture": architecture,
        "inference_architecture": inference_architecture,
        "width": d_sae,
        "sparsity_target": sparsity_target,
        "normalize_activations": cfg_normalize,
        "dtype": cfg_dtype,
        "model_load_processing": model_load_processing,
        "prepend_bos": prepend_bos,
        "artifact_ref": selected_ref,
        "artifact_sha256": digest,
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_size_bytes": size_bytes,
    }
    if k is not None:
        config["k"] = k
    window = eligible_fanout_window(
        slot_epoch=slot_epoch,
        cadence_blocks=policy.get("cadence_blocks"),
        phase_offset_blocks=policy.get("phase_offset_blocks"),
        window_blocks=window_blocks,
    )
    return {
        "task_type": TASK_TYPE,
        "schema_version": SCHEMA_VERSION,
        "config": config,
        ELIGIBLE_FANOUT_WINDOW: window,
    }


def stage_candidate(*args, pool: StagingPool | None = None, **kwargs) -> dict[str, Any]:
    candidate = build_candidate(*args, **kwargs)
    (pool or StagingPool()).add(candidate)
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aurelius-sae-stage")
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--artifact-ref",
        required=True,
        help=(
            "immutable hf://owner/repo@commit[/subdir] locator, or "
            "aurelius-artifact://<hotkey>/<sha256> whose digest equals the bundle's S4 digest"
        ),
    )
    parser.add_argument(
        "--hotkey",
        help=(
            "expected miner SS58 hotkey; when set, an aurelius-artifact:// --artifact-ref "
            "must name exactly this hotkey (equality only)"
        ),
    )
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--slot-epoch", required=True, type=int)
    parser.add_argument("--sparsity-target", required=True)
    parser.add_argument("--pool")
    args = parser.parse_args(argv)
    try:
        entries = load_published_entries()
        policy = selectable_slot_policy(entries, args.slot_id)
        candidate = stage_candidate(
            args.bundle,
            artifact_ref=args.artifact_ref,
            slot_id=args.slot_id,
            slot_epoch=args.slot_epoch,
            sparsity_target=args.sparsity_target,
            policy=policy,
            window_blocks=entries.get("window_blocks"),
            ref_hotkey=args.hotkey,
            pool=StagingPool(args.pool) if args.pool else None,
        )
    except (OSError, SchemaUnavailable, StageError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "staged": True,
                "slot_id": candidate["config"]["slot_id"],
                "slot_epoch": candidate["config"]["slot_epoch"],
                "artifact_sha256": candidate["config"]["artifact_sha256"],
                ELIGIBLE_FANOUT_WINDOW: candidate[ELIGIBLE_FANOUT_WINDOW],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
