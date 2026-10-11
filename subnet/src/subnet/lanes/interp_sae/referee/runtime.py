# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Trusted, offline SAE scoring runtime for the v5 referee (R7/S7/S10).

The module intentionally has no import-time ML dependency.  Miner bundles are inspected as the
documented safetensors byte format before a trusted, image-pinned factory may construct an SAE;
pickle, dynamic imports, downloads, and caller-selected paths are not part of this contract.

SAE-Lens does not publish one stable tensor-key ABI for all six architectures accepted by Gate 0.
Consequently this capsule does not guess one.  The evaluator image must bind an exact
``TensorSchema`` and ``SAEFactory`` for every architecture it enables.  An unbound architecture
fails closed with ``load_error``.  This keeps the M2 runtime usable without silently turning a
library implementation detail into a consensus rule; M6 owns the version-pinned bindings.
"""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from types import MappingProxyType
from typing import Any
from .gate0 import DEFAULT_ARTIFACT_MAX_BYTES, SlotPins

BAD_SHAPE = "bad_shape"


NONFINITE_WEIGHTS = "nonfinite_weights"


LOAD_ERROR = "load_error"


INFRA_OFFLINE_MISS = "infra_offline_miss"


_RUNTIME_CODES = frozenset({BAD_SHAPE, NONFINITE_WEIGHTS, LOAD_ERROR, INFRA_OFFLINE_MISS})


_TENSOR_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")


_IMPLEMENTATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$")


_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024


_READ_CHUNK_BYTES = 1024 * 1024


_DTYPE_WIDTH = MappingProxyType(
    {
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
    }
)


class RuntimeFailure(RuntimeError):
    """A stable child-runtime failure with no public reflection of its detail."""

    def __init__(self, code: str, operator_detail: str = "") -> None:
        if code not in _RUNTIME_CODES:
            raise ValueError("runtime failure code is outside the closed allowlist")
        self.code = code
        super().__init__(operator_detail)


@dataclass(frozen=True, slots=True)
class TensorField:
    """One exact tensor in an image-pinned SAE-Lens serialization ABI."""

    shape: tuple[int | str, ...]
    dtypes: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("tensor shape contract must be a non-empty tuple")
        for dimension in self.shape:
            if isinstance(dimension, bool) or not (
                isinstance(dimension, int) and dimension > 0 or dimension in {"d_in", "d_sae"}
            ):
                raise ValueError("tensor dimensions must be positive integers, d_in, or d_sae")
        if not self.dtypes or not self.dtypes.issubset(_DTYPE_WIDTH):
            raise ValueError("tensor dtype contract must use supported floating dtypes")


@dataclass(frozen=True, slots=True)
class TensorSchema:
    """Exact key/shape/dtype ABI for one pinned SAE-Lens architecture implementation."""

    architecture: str
    fields: Mapping[str, TensorField]
    implementation_id: str
    cfg_validator: Callable[[Mapping[str, Any], SlotPins], None]

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, str) or not self.architecture:
            raise ValueError("architecture must be a non-empty string")
        if (
            not isinstance(self.implementation_id, str)
            or _IMPLEMENTATION_ID_RE.fullmatch(self.implementation_id) is None
        ):
            raise ValueError("implementation_id must be a safe bounded identifier")
        if not callable(self.cfg_validator):
            raise ValueError("cfg_validator must be trusted pinned code")
        fields = dict(self.fields)
        if not fields or any(
            not isinstance(name, str)
            or _TENSOR_NAME_RE.fullmatch(name) is None
            or not isinstance(field, TensorField)
            for name, field in fields.items()
        ):
            raise ValueError("tensor schema fields must be an exact named TensorField mapping")
        object.__setattr__(self, "fields", MappingProxyType(fields))


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    data_sha256: str


def inspect_safetensors(
    path: str | Path,
    *,
    schema: TensorSchema,
    d_in: int,
    d_sae: int,
    max_bytes: int = DEFAULT_ARTIFACT_MAX_BYTES,
) -> Mapping[str, TensorInfo]:
    """Validate exact safetensor keys/shapes/dtypes and scan every float for finiteness.

    This is a small parser for the documented safetensors container, not a tensor constructor.  It
    reads an already-opened ``O_NOFOLLOW`` descriptor, validates all byte ranges, and never imports
    NumPy, torch, safetensors, or SAE-Lens.
    """

    if not isinstance(schema, TensorSchema):
        raise TypeError("schema must be TensorSchema")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (d_in, d_sae)
    ):
        raise ValueError("d_in and d_sae must be positive integers")
    limit = _artifact_byte_limit(max_bytes)
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = source.lstat()
        if source.is_symlink() or not source.is_file():
            raise RuntimeFailure(LOAD_ERROR, "weights must be a regular file")
        descriptor = os.open(source, flags)
    except RuntimeFailure:
        raise
    except OSError as exc:
        raise RuntimeFailure(LOAD_ERROR, "weights cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if not _same_stat(before, opened) or opened.st_size < 10 or opened.st_size > limit:
            raise RuntimeFailure(LOAD_ERROR, "weights changed while opening")
        infos, _data_start = _inspect_safetensors_descriptor(
            descriptor,
            opened,
            schema=schema,
            d_in=d_in,
            d_sae=d_sae,
        )
        return infos
    except RuntimeFailure:
        raise
    except (OSError, OverflowError, RecursionError, ValueError) as exc:
        raise RuntimeFailure(LOAD_ERROR, "weights are not a valid pinned safetensors file") from exc
    finally:
        os.close(descriptor)


def _inspect_safetensors_descriptor(
    descriptor: int,
    opened: os.stat_result,
    *,
    schema: TensorSchema,
    d_in: int,
    d_sae: int,
) -> tuple[Mapping[str, TensorInfo], int]:
    """Inspect one already-authenticated descriptor and retain per-tensor byte commitments."""

    if opened.st_size < 10:
        raise RuntimeFailure(LOAD_ERROR, "safetensors file is too small")
    prefix = _pread_exact(descriptor, 8, 0)
    header_size = int.from_bytes(prefix, "little", signed=False)
    if not 2 <= header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
        raise RuntimeFailure(LOAD_ERROR, "safetensors header length is invalid")
    data_start = 8 + header_size
    if data_start > opened.st_size:
        raise RuntimeFailure(LOAD_ERROR, "safetensors header exceeds the file")
    header_bytes = _pread_exact(descriptor, header_size, 8)
    header = _strict_json_object(header_bytes, label="safetensors header")
    tensor_header = {key: value for key, value in header.items() if key != "__metadata__"}
    metadata = header.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise RuntimeFailure(LOAD_ERROR, "safetensors metadata is invalid")
    if set(tensor_header) != set(schema.fields):
        raise RuntimeFailure(BAD_SHAPE, "safetensor keys do not match the pinned ABI")

    infos: dict[str, TensorInfo] = {}
    ranges: list[tuple[int, int, str, str, tuple[int, ...]]] = []
    dimensions = {"d_in": d_in, "d_sae": d_sae}
    data_bytes = opened.st_size - data_start
    for name, field in schema.fields.items():
        raw = tensor_header[name]
        if not isinstance(raw, dict) or set(raw) != {"dtype", "shape", "data_offsets"}:
            raise RuntimeFailure(LOAD_ERROR, "tensor header has unknown or missing fields")
        dtype = raw["dtype"]
        shape_raw = raw["shape"]
        offsets = raw["data_offsets"]
        if dtype not in field.dtypes:
            raise RuntimeFailure(BAD_SHAPE, "tensor dtype differs from the pinned ABI")
        if not isinstance(shape_raw, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape_raw
        ):
            raise RuntimeFailure(BAD_SHAPE, "tensor shape is invalid")
        expected_shape = tuple(
            dimensions[value] if isinstance(value, str) else value for value in field.shape
        )
        shape = tuple(shape_raw)
        if shape != expected_shape:
            raise RuntimeFailure(BAD_SHAPE, "tensor shape differs from the pinned ABI")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise RuntimeFailure(LOAD_ERROR, "tensor offsets are invalid")
        start, end = offsets
        expected_bytes = math.prod(shape) * _DTYPE_WIDTH[dtype]
        if not 0 <= start < end <= data_bytes or end - start != expected_bytes:
            raise RuntimeFailure(LOAD_ERROR, "tensor byte range is invalid")
        ranges.append((start, end, name, dtype, shape))

    cursor = 0
    for start, end, name, dtype, shape in sorted(ranges):
        if start != cursor:
            raise RuntimeFailure(LOAD_ERROR, "tensor ranges are overlapping or non-contiguous")
        data_sha256 = _scan_finite(descriptor, data_start + start, end - start, dtype)
        infos[name] = TensorInfo(name, dtype, shape, start, end, data_sha256)
        cursor = end
    if cursor != data_bytes:
        raise RuntimeFailure(LOAD_ERROR, "safetensors data has unclaimed trailing bytes")
    if not _same_stat(opened, os.fstat(descriptor)):
        raise RuntimeFailure(LOAD_ERROR, "weights changed while being inspected")
    return MappingProxyType(infos), data_start


def _scan_finite(descriptor: int, offset: int, size: int, dtype: str) -> str:
    width = _DTYPE_WIDTH[dtype]
    cursor = 0
    request_size = _READ_CHUNK_BYTES - (_READ_CHUNK_BYTES % width)
    digest = hashlib.sha256()
    while cursor < size:
        request = min(request_size, size - cursor)
        data = _pread_exact(descriptor, request, offset + cursor)
        if _contains_nonfinite(data, dtype):
            raise RuntimeFailure(NONFINITE_WEIGHTS, "safetensor contains NaN or infinity")
        digest.update(data)
        cursor += request
    return digest.hexdigest()


def _contains_nonfinite(data: bytes, dtype: str) -> bool:
    if dtype == "F16":
        return any((value & 0x7C00) == 0x7C00 for (value,) in struct.iter_unpack("<H", data))
    if dtype == "BF16":
        return any((value & 0x7F80) == 0x7F80 for (value,) in struct.iter_unpack("<H", data))
    if dtype == "F32":
        return any(
            (value & 0x7F800000) == 0x7F800000 for (value,) in struct.iter_unpack("<I", data)
        )
    if dtype == "F64":
        return any(
            (value & 0x7FF0000000000000) == 0x7FF0000000000000
            for (value,) in struct.iter_unpack("<Q", data)
        )
    raise RuntimeFailure(BAD_SHAPE, "tensor dtype is unsupported")


def _pread_exact(descriptor: int, count: int, offset: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        chunk = os.pread(descriptor, count - len(result), offset + len(result))
        if not chunk:
            raise RuntimeFailure(LOAD_ERROR, "unexpected end of file")
        result.extend(chunk)
    return bytes(result)


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    class DuplicateKey(ValueError):
        pass

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise DuplicateKey
            value[key] = item
        return value

    def reject_constant(_value: str) -> Any:
        raise DuplicateKey

    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=reject_constant,
            parse_float=lambda raw: _finite_float(raw),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateKey,
        ValueError,
        RecursionError,
    ) as exc:
        raise RuntimeFailure(LOAD_ERROR, f"{label} is not strict finite unique-key JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeFailure(LOAD_ERROR, f"{label} root must be an object")
    return value


def _finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON float")
    return value


def _artifact_byte_limit(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= DEFAULT_ARTIFACT_MAX_BYTES
    ):
        raise RuntimeFailure(
            LOAD_ERROR,
            "artifact byte limit must be positive and no larger than the four-GiB outer ceiling",
        )
    return value


def _same_stat(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_mode,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_mode,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )
