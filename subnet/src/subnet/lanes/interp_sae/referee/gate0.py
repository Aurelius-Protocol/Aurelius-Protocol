# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Cheap, dependency-free Gate 0 checks for the v5-native SAE referee.

Contract: R2/S4 authenticates the exact two-file SAE-Lens bundle and validates every signed slot
pin before a caller may construct an SAE.  R4 aggregates already-computed draw observations; it
does not collect activations, import tensor/model libraries, or access the network.  This capsule
closes the lane-plan §1.4 attacks 5.4--5.6; no §12 invariant row is flipped at M1.
"""

from __future__ import annotations
import hashlib
import hmac
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

BUNDLE_DOMAIN = b"AURELIUS-SAE-LENS-BUNDLE-v1\0"


CFG_FILENAME = "cfg.json"


WEIGHTS_FILENAME = "sae_weights.safetensors"


DEFAULT_ARTIFACT_MAX_BYTES = 4 * 1024 * 1024 * 1024


MAX_CFG_BYTES = 64 * 1024


_READ_CHUNK_BYTES = 1024 * 1024


_MAX_DETAIL_CHARS = 256


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


_ALLOWED_ARCHITECTURES = frozenset(
    {"standard", "gated", "jumprelu", "topk", "batchtopk", "matryoshka_batchtopk"}
)


_TOPK_ARCHITECTURES = frozenset({"topk", "batchtopk", "matryoshka_batchtopk"})


class Gate0Reason(StrEnum):
    """Stable content reason codes in their production precedence order."""

    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    ARTIFACT_LAYOUT_INVALID = "artifact_layout_invalid"
    ARTIFACT_TOO_LARGE = "artifact_too_large"
    CFG_INVALID = "cfg_invalid"
    CFG_ARCHITECTURE_NOT_ALLOWED = "cfg_architecture_not_allowed"
    CFG_PIN_MISMATCH = "cfg_pin_mismatch"
    L0_OUT_OF_BAND = "l0_out_of_band"
    L0_OUT_OF_BAND_FAMILY = "l0_out_of_band_family"
    L0_TAIL_OUT_OF_BAND = "l0_tail_out_of_band"
    DEAD_FEATURES = "dead_features"


ARTIFACT_HASH_MISMATCH = Gate0Reason.ARTIFACT_HASH_MISMATCH


ARTIFACT_LAYOUT_INVALID = Gate0Reason.ARTIFACT_LAYOUT_INVALID


ARTIFACT_TOO_LARGE = Gate0Reason.ARTIFACT_TOO_LARGE


CFG_INVALID = Gate0Reason.CFG_INVALID


CFG_ARCHITECTURE_NOT_ALLOWED = Gate0Reason.CFG_ARCHITECTURE_NOT_ALLOWED


CFG_PIN_MISMATCH = Gate0Reason.CFG_PIN_MISMATCH


class Gate0Rejected(ValueError):
    """A miner-controlled Gate 0 input failed with a stable, typed reason code.

    ``pin_mismatches`` (ADR-0229, diagnostics v2) names each compared cfg field that differed as
    ``(field, observed, expected)`` bounded JSON strings; it is advisory feedback, never the code.
    """

    def __init__(
        self,
        code: Gate0Reason,
        detail: str,
        *,
        pin_mismatches: tuple[tuple[str, str, str], ...] = (),
    ) -> None:
        if not isinstance(code, Gate0Reason):
            raise TypeError("Gate0Rejected code must be Gate0Reason")
        bounded = " ".join(str(detail).split()) or "rejected"
        self.code = code
        self.detail = bounded[:_MAX_DETAIL_CHARS]
        self.pin_mismatches = tuple(pin_mismatches)
        super().__init__(f"{code.value}: {self.detail}")


_PIN_FIELD_NAMES = (
    "d_in",
    "d_sae",
    "dtype",
    "normalize_activations",
    "reshape_activations",
    "apply_b_dec_to_input",
    "hook_name",
    "model_name",
    "seqpos_slice",
)


_MAX_PIN_REPR_BYTES = 64


def _bounded_repr(value: Any) -> str:
    """Render a compared scalar/list as bounded canonical JSON (never arbitrary cfg objects)."""

    try:
        text = json.dumps(
            list(value) if isinstance(value, tuple) else value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        text = "<unrepresentable>"
    encoded = text.encode("ascii", "replace")[:_MAX_PIN_REPR_BYTES]
    return encoded.decode("ascii", "replace")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_string(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\0" in value:
        raise ValueError(f"{name} must be a non-empty string (max {maximum})")
    return value


def _json_seqpos(value: Any, name: str) -> tuple[int | None, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be null or an integer/null sequence")
    result: list[int | None] = []
    for item in value:
        if item is not None and (isinstance(item, bool) or not isinstance(item, int)):
            raise ValueError(f"{name} must be null or an integer/null sequence")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class SlotPins:
    """The signed slot values Gate 0 must match exactly before SAE construction."""

    d_in: int
    d_sae: int
    dtype: str
    normalize_activations: str
    reshape_activations: bool | str | None
    apply_b_dec_to_input: bool
    rescale_acts_by_decoder_norm: bool
    hook_name: str
    model_name: str
    seqpos_slice: tuple[int | None, ...] | list[int | None] | None
    topk_k_min: int
    topk_k_max: int

    def __post_init__(self) -> None:
        _positive_integer(self.d_in, "d_in")
        _positive_integer(self.d_sae, "d_sae")
        _nonempty_string(self.dtype, "dtype", maximum=32)
        if self.normalize_activations != "none":
            raise ValueError("normalize_activations slot pin must be 'none'")
        if self.reshape_activations is not None and not isinstance(
            self.reshape_activations, (bool, str)
        ):
            raise ValueError("reshape_activations must be bool, string, or null")
        if isinstance(self.reshape_activations, str) and not self.reshape_activations:
            raise ValueError("reshape_activations string must not be empty")
        if not isinstance(self.apply_b_dec_to_input, bool):
            raise ValueError("apply_b_dec_to_input must be boolean")
        if not isinstance(self.rescale_acts_by_decoder_norm, bool):
            raise ValueError("rescale_acts_by_decoder_norm must be boolean")
        _nonempty_string(self.hook_name, "hook_name")
        _nonempty_string(self.model_name, "model_name")
        object.__setattr__(
            self,
            "seqpos_slice",
            _json_seqpos(self.seqpos_slice, "seqpos_slice"),
        )
        _positive_integer(self.topk_k_min, "topk_k_min")
        _positive_integer(self.topk_k_max, "topk_k_max")
        if self.topk_k_min > self.topk_k_max:
            raise ValueError("topk_k_min must not exceed topk_k_max")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SlotPins:
        expected = {
            "d_in",
            "d_sae",
            "dtype",
            "normalize_activations",
            "reshape_activations",
            "apply_b_dec_to_input",
            "rescale_acts_by_decoder_norm",
            "hook_name",
            "model_name",
            "seqpos_slice",
            "topk_k_min",
            "topk_k_max",
        }
        if set(value) != expected:
            raise ValueError("slot pins require the complete closed R2 field set")
        return cls(**dict(value))


@dataclass(frozen=True)
class ValidatedArtifact:
    """Gate-clean identity and config; deliberately contains no loaded SAE object."""

    sha256: str
    size_bytes: int
    cfg: Mapping[str, Any]


@dataclass(frozen=True)
class _FileSnapshot:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _BundleRead:
    cfg: bytes
    sha256: str
    size_bytes: int


def _snapshot_from_stat(value: os.stat_result) -> _FileSnapshot:
    return _FileSnapshot(
        device=value.st_dev,
        inode=value.st_ino,
        size=value.st_size,
        modified_ns=value.st_mtime_ns,
        changed_ns=value.st_ctime_ns,
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _same_snapshot(left: _FileSnapshot, right: _FileSnapshot) -> bool:
    return left == right


@contextmanager
def _open_bundle_directory(package_ref: Path | str):
    try:
        package = Path(package_ref)
        before = package.lstat()
    except (OSError, TypeError, ValueError) as exc:
        raise Gate0Rejected(
            ARTIFACT_LAYOUT_INVALID, "bundle directory cannot be inspected"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise Gate0Rejected(
            ARTIFACT_LAYOUT_INVALID,
            "bundle path must be a real directory, not a symlink",
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(package, flags)
    except OSError as exc:
        raise Gate0Rejected(
            ARTIFACT_LAYOUT_INVALID,
            "bundle directory cannot be opened without following links",
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not _same_identity(before, opened) or not stat.S_ISDIR(opened.st_mode):
            raise Gate0Rejected(ARTIFACT_LAYOUT_INVALID, "bundle directory changed while opening")
        yield descriptor
    finally:
        os.close(descriptor)


def _list_entries(descriptor: int) -> set[str]:
    try:
        return set(os.listdir(descriptor))
    except OSError as exc:
        raise Gate0Rejected(
            ARTIFACT_LAYOUT_INVALID, "bundle directory cannot be enumerated"
        ) from exc


def _require_exact_entries(descriptor: int) -> None:
    entries = _list_entries(descriptor)
    if entries != {CFG_FILENAME, WEIGHTS_FILENAME}:
        raise Gate0Rejected(
            ARTIFACT_LAYOUT_INVALID,
            "bundle must contain exactly cfg.json and sae_weights.safetensors",
        )


def _regular_snapshot(descriptor: int, name: str) -> _FileSnapshot:
    try:
        value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except OSError as exc:
        raise Gate0Rejected(
            ARTIFACT_LAYOUT_INVALID, "required bundle file cannot be inspected"
        ) from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise Gate0Rejected(ARTIFACT_LAYOUT_INVALID, "bundle entries must be regular files")
    if value.st_size < 0:
        raise Gate0Rejected(ARTIFACT_LAYOUT_INVALID, "bundle entry has an invalid size")
    return _snapshot_from_stat(value)


def _read_snapshot(
    descriptor: int,
    name: str,
    expected: _FileSnapshot,
    *,
    oversize_code: Gate0Reason = ARTIFACT_LAYOUT_INVALID,
    max_bytes: int | None = None,
    on_chunk: Callable[[bytes], None] | None = None,
    collect: bool = True,
) -> bytes:
    if max_bytes is not None and expected.size > max_bytes:
        raise Gate0Rejected(oversize_code, "bounded bundle file exceeds its byte limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(name, flags, dir_fd=descriptor)
    except OSError as exc:
        raise Gate0Rejected(
            ARTIFACT_LAYOUT_INVALID,
            "bundle entry cannot be opened without following links",
        ) from exc
    try:
        opened_stat = os.fstat(file_descriptor)
        opened = _snapshot_from_stat(opened_stat)
        if opened != expected or not stat.S_ISREG(opened_stat.st_mode):
            raise Gate0Rejected(ARTIFACT_LAYOUT_INVALID, "bundle entry changed while opening")
        chunks: list[bytes] | None = [] if collect else None
        total = 0
        while total <= expected.size:
            request = min(_READ_CHUNK_BYTES, expected.size + 1 - total)
            chunk = os.read(file_descriptor, request)
            if not chunk:
                break
            if chunks is not None:
                chunks.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
            total += len(chunk)
        final = _snapshot_from_stat(os.fstat(file_descriptor))
        if final != expected or total != expected.size:
            raise Gate0Rejected(ARTIFACT_LAYOUT_INVALID, "bundle entry changed while reading")
        return b"".join(chunks) if chunks is not None else b""
    except OSError as exc:
        raise Gate0Rejected(ARTIFACT_LAYOUT_INVALID, "bundle entry cannot be read safely") from exc
    finally:
        os.close(file_descriptor)


def _validate_limit(max_bytes: Any) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    return max_bytes


def _read_two_file_bundle(
    package_ref: Path | str, *, max_bytes: int = DEFAULT_ARTIFACT_MAX_BYTES
) -> _BundleRead:
    limit = _validate_limit(max_bytes)
    with _open_bundle_directory(package_ref) as descriptor:
        _require_exact_entries(descriptor)
        cfg_snapshot = _regular_snapshot(descriptor, CFG_FILENAME)
        weights_snapshot = _regular_snapshot(descriptor, WEIGHTS_FILENAME)
        size_bytes = cfg_snapshot.size + weights_snapshot.size
        if size_bytes > limit:
            raise Gate0Rejected(ARTIFACT_TOO_LARGE, "bundle exceeds the slot artifact byte cap")

        cfg = _read_snapshot(
            descriptor,
            CFG_FILENAME,
            cfg_snapshot,
            oversize_code=ARTIFACT_TOO_LARGE,
            max_bytes=MAX_CFG_BYTES,
        )
        digest = hashlib.sha256()
        digest.update(BUNDLE_DOMAIN)
        digest.update(len(cfg).to_bytes(8, "big"))
        digest.update(cfg)
        digest.update(weights_snapshot.size.to_bytes(8, "big"))
        _read_snapshot(
            descriptor,
            WEIGHTS_FILENAME,
            weights_snapshot,
            on_chunk=digest.update,
            collect=False,
        )
        if not _same_snapshot(cfg_snapshot, _regular_snapshot(descriptor, CFG_FILENAME)):
            raise Gate0Rejected(ARTIFACT_LAYOUT_INVALID, "cfg.json changed while reading bundle")
        if not _same_snapshot(weights_snapshot, _regular_snapshot(descriptor, WEIGHTS_FILENAME)):
            raise Gate0Rejected(
                ARTIFACT_LAYOUT_INVALID,
                "sae_weights.safetensors changed while reading bundle",
            )
        _require_exact_entries(descriptor)
    return _BundleRead(cfg=cfg, sha256=digest.hexdigest(), size_bytes=size_bytes)


def _validate_expected_digest(expected: str | None) -> None:
    if expected is None:
        return
    if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
        raise Gate0Rejected(
            ARTIFACT_HASH_MISMATCH,
            "declared artifact digest is not lowercase SHA-256",
        )


def _check_expected_digest(actual: str, expected: str | None) -> None:
    _validate_expected_digest(expected)
    if expected is None:
        return
    if not hmac.compare_digest(actual, expected):
        raise Gate0Rejected(ARTIFACT_HASH_MISMATCH, "bundle digest does not match its commitment")


def artifact_digest(
    package_ref: Path | str,
    *,
    max_bytes: int = DEFAULT_ARTIFACT_MAX_BYTES,
    expected_sha256: str | None = None,
) -> str:
    """Return S4 identity after enforcing the closed two-regular-file layout and byte cap."""

    _validate_expected_digest(expected_sha256)
    bundle = _read_two_file_bundle(package_ref, max_bytes=max_bytes)
    _check_expected_digest(bundle.sha256, expected_sha256)
    return bundle.sha256


class _CfgSyntaxError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _CfgSyntaxError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> Any:
    raise _CfgSyntaxError("non-finite JSON constant")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _CfgSyntaxError("non-finite JSON number")
    return parsed


def _parse_cfg_bytes(data: bytes) -> dict[str, Any]:
    if len(data) > MAX_CFG_BYTES:
        raise Gate0Rejected(CFG_INVALID, "cfg.json exceeds the metadata byte cap")
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_json_float,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _CfgSyntaxError,
        RecursionError,
        ValueError,
    ) as exc:
        raise Gate0Rejected(CFG_INVALID, "cfg.json is not strict finite unique-key JSON") from exc
    if not isinstance(value, dict):
        raise Gate0Rejected(CFG_INVALID, "cfg.json root must be an object")
    return value


def _cfg_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Gate0Rejected(CFG_INVALID, f"{name} must be a positive integer")
    return value


def _cfg_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or "\0" in value:
        raise Gate0Rejected(CFG_INVALID, f"{name} must be a bounded non-empty string")
    return value


def _cfg_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise Gate0Rejected(CFG_INVALID, f"{name} must be boolean")
    return value


def _coerce_pins(pins: SlotPins | Mapping[str, Any]) -> SlotPins:
    if isinstance(pins, SlotPins):
        return pins
    if isinstance(pins, Mapping):
        return SlotPins.from_mapping(pins)
    raise TypeError("pins must be SlotPins or a complete mapping")


def _check_cfg_value(cfg: dict[str, Any], pins: SlotPins) -> dict[str, Any]:
    required = {
        "architecture",
        "d_in",
        "d_sae",
        "dtype",
        "normalize_activations",
        "reshape_activations",
        "apply_b_dec_to_input",
        "metadata",
    }
    if not required.issubset(cfg):
        raise Gate0Rejected(CFG_INVALID, "cfg.json is missing required Gate 0 fields")

    architecture = _cfg_string(cfg["architecture"], "architecture")
    if architecture not in _ALLOWED_ARCHITECTURES:
        raise Gate0Rejected(
            CFG_ARCHITECTURE_NOT_ALLOWED,
            "SAE architecture is outside the referee allowlist",
        )

    d_in = _cfg_integer(cfg["d_in"], "d_in")
    d_sae = _cfg_integer(cfg["d_sae"], "d_sae")
    dtype = _cfg_string(cfg["dtype"], "dtype")
    normalize = _cfg_string(cfg["normalize_activations"], "normalize_activations")
    reshape = cfg["reshape_activations"]
    if reshape is not None and not isinstance(reshape, (bool, str)):
        raise Gate0Rejected(CFG_INVALID, "reshape_activations has an invalid JSON type")
    if isinstance(reshape, str) and not reshape:
        raise Gate0Rejected(CFG_INVALID, "reshape_activations string must not be empty")
    apply_b_dec_to_input = _cfg_boolean(cfg["apply_b_dec_to_input"], "apply_b_dec_to_input")

    metadata = cfg["metadata"]
    if not isinstance(metadata, dict):
        raise Gate0Rejected(CFG_INVALID, "metadata must be an object")
    metadata_fields = {
        "hook_name",
        "model_name",
        "prepend_bos",
        "context_size",
        "exclude_special_tokens",
        "seqpos_slice",
    }
    if not metadata_fields.issubset(metadata):
        raise Gate0Rejected(CFG_INVALID, "metadata is missing required Gate 0 fields")
    hook_name = _cfg_string(metadata["hook_name"], "metadata.hook_name")
    model_name = _cfg_string(metadata["model_name"], "metadata.model_name")
    # ADR-0229 D1: prepend_bos, context_size and exclude_special_tokens are training
    # provenance. Gate 0 requires their presence and JSON type (cfg_invalid otherwise) but no
    # longer compares them to the signed slot: they do not alter the referee's forward pass,
    # and pinning them turned the slot into a single-recipe competition (gap analysis G2/G21).
    _cfg_boolean(metadata["prepend_bos"], "metadata.prepend_bos")
    _cfg_integer(metadata["context_size"], "metadata.context_size")
    _cfg_boolean(metadata["exclude_special_tokens"], "metadata.exclude_special_tokens")
    try:
        seqpos_slice = _json_seqpos(metadata["seqpos_slice"], "metadata.seqpos_slice")
    except ValueError as exc:
        raise Gate0Rejected(CFG_INVALID, "metadata.seqpos_slice has an invalid shape") from exc

    actual_pins = (
        d_in,
        d_sae,
        dtype,
        normalize,
        reshape,
        apply_b_dec_to_input,
        hook_name,
        model_name,
        seqpos_slice,
    )
    expected_pins = (
        pins.d_in,
        pins.d_sae,
        pins.dtype,
        pins.normalize_activations,
        pins.reshape_activations,
        pins.apply_b_dec_to_input,
        pins.hook_name,
        pins.model_name,
        pins.seqpos_slice,
    )
    if actual_pins != expected_pins or normalize != "none":
        mismatches = tuple(
            (name, _bounded_repr(actual), _bounded_repr(expected))
            for name, actual, expected in zip(_PIN_FIELD_NAMES, actual_pins, expected_pins)
            if actual != expected
        )
        raise Gate0Rejected(
            CFG_PIN_MISMATCH,
            "cfg.json does not match the signed slot pins",
            pin_mismatches=mismatches,
        )

    if architecture in _TOPK_ARCHITECTURES:
        if "k" not in cfg or "rescale_acts_by_decoder_norm" not in cfg:
            raise Gate0Rejected(
                CFG_INVALID,
                "TopK cfg.json must declare k and rescale_acts_by_decoder_norm",
            )
        k = _cfg_integer(cfg["k"], "k")
        if not pins.topk_k_min <= k <= pins.topk_k_max:
            raise Gate0Rejected(
                CFG_PIN_MISMATCH,
                "TopK k is outside the signed slot band",
                pin_mismatches=(
                    ("k", _bounded_repr(k), _bounded_repr([pins.topk_k_min, pins.topk_k_max])),
                ),
            )
        rescale = _cfg_boolean(
            cfg["rescale_acts_by_decoder_norm"],
            "rescale_acts_by_decoder_norm",
        )
        if rescale is not pins.rescale_acts_by_decoder_norm:
            raise Gate0Rejected(
                CFG_PIN_MISMATCH,
                "TopK decoder-norm rescaling differs from the signed slot pin",
                pin_mismatches=(
                    (
                        "rescale_acts_by_decoder_norm",
                        _bounded_repr(rescale),
                        _bounded_repr(pins.rescale_acts_by_decoder_norm),
                    ),
                ),
            )
    return cfg


def check_cfg_bytes(data: bytes, pins: SlotPins | Mapping[str, Any]) -> dict[str, Any]:
    """Apply the exact production Gate-0 cfg parser/checker to already-bounded bytes.

    Artifact admission owns a descriptor-backed, no-follow reader and therefore must not turn its
    authenticated bytes back into a host path merely to reuse Gate 0.  This public bytes entrypoint
    shares the same size, duplicate-key, finite-JSON, architecture, pin, and TopK checks as
    :func:`check_cfg`; it adds no alternate cfg semantics.

    The accepted architecture-specific SAE-Lens extension-key ABI remains an M6 image/fixture gate.
    This helper intentionally neither guesses that ABI nor claims the current common-field checker
    is a closed serializer for all six architectures.
    """

    if type(data) is not bytes:
        raise Gate0Rejected(CFG_INVALID, "cfg.json bytes must be an exact bytes value")
    if len(data) > MAX_CFG_BYTES:
        raise Gate0Rejected(CFG_INVALID, "cfg.json exceeds its byte cap")
    return _check_cfg_value(_parse_cfg_bytes(data), _coerce_pins(pins))


def validate_artifact(
    package_ref: Path | str,
    *,
    pins: SlotPins | Mapping[str, Any],
    expected_sha256: str,
    max_bytes: int = DEFAULT_ARTIFACT_MAX_BYTES,
) -> ValidatedArtifact:
    """Authenticate layout/bytes, then cfg pins, returning only data needed before SAE load."""

    _validate_expected_digest(expected_sha256)
    bundle = _read_two_file_bundle(package_ref, max_bytes=max_bytes)
    _check_expected_digest(bundle.sha256, expected_sha256)
    cfg = _check_cfg_value(_parse_cfg_bytes(bundle.cfg), _coerce_pins(pins))
    return ValidatedArtifact(
        sha256=bundle.sha256,
        size_bytes=bundle.size_bytes,
        cfg=MappingProxyType(cfg),
    )
