# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Descriptor-bound readers for retained operator evidence.

Security-sensitive terminal decisions must not validate one pathname incarnation and hash a
different one.  This module opens the pathname once with ``O_NOFOLLOW``, validates the opened
inode, reads a bounded payload, parses strict JSON, and returns both the object and SHA-256 of the
same bytes.  Callers opt in to canonical-byte enforcement only for producer contracts that promise
canonical JSON; checked-in or third-party JSON may still use the same inode-safe reader without
that additional assertion.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
import stat
from typing import Any

DEFAULT_MAX_BYTES = 1_048_576


class EvidenceReadError(RuntimeError):
    """A retained evidence pathname cannot safely authorize a decision."""


@dataclass(frozen=True)
class JsonEvidence:
    """One parsed object and digest derived from the same already-opened inode."""

    value: dict[str, Any]
    sha256: str
    size: int
    mode: int
    uid: int


@dataclass(frozen=True)
class BytesEvidence:
    """Bounded bytes and digest from one safely opened inode."""

    payload: bytes
    sha256: str
    size: int
    mode: int
    uid: int


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical newline-terminated JSON form used by operator producers."""

    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceReadError("evidence cannot be represented as canonical JSON") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def read_evidence_bytes(
    path: str | os.PathLike[str],
    *,
    label: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    require_mode: int | None = None,
    require_owner_uid: int | None = None,
) -> BytesEvidence:
    """Read bounded bytes and hash them without reopening their pathname.

    Symlinks, hardlinks, non-regular files, and empty/oversized files fail closed. A pathname
    replacement after ``open`` cannot alter either the returned bytes or their digest.
    """

    if type(max_bytes) is not int or max_bytes < 1:
        raise EvidenceReadError(f"{label} has an invalid reader byte bound")
    if require_mode is not None and (
        type(require_mode) is not int or require_mode < 0 or require_mode > 0o777
    ):
        raise EvidenceReadError(f"{label} has an invalid required file mode")
    if require_owner_uid is not None and (
        type(require_owner_uid) is not int or require_owner_uid < 0
    ):
        raise EvidenceReadError(f"{label} has an invalid required owner uid")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise EvidenceReadError(f"{label} cannot be opened safely: O_NOFOLLOW is unavailable")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= max_bytes
            or (require_mode is not None and stat.S_IMODE(before.st_mode) != require_mode)
            or (require_owner_uid is not None and before.st_uid != require_owner_uid)
        ):
            raise EvidenceReadError(f"{label} has an unsafe opened-inode shape")
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(descriptor, min(65_536, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        # A pathname rename changes ctime without changing the already-opened file's bytes.  That
        # is precisely the race this descriptor-bound reader is designed to tolerate safely, so
        # ctime is not a content-stability signal. Link count, size, and mtime still catch unlink
        # and in-place mutation while the descriptor is live.
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
        )
        if (
            total > max_bytes
            or len(payload) != before.st_size
            or any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        ):
            raise EvidenceReadError(f"{label} changed while its opened inode was read")
    except EvidenceReadError:
        raise
    except OSError as exc:
        raise EvidenceReadError(f"{label} is unavailable or unsafe") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return BytesEvidence(
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        mode=stat.S_IMODE(before.st_mode),
        uid=before.st_uid,
    )


def read_json_evidence(
    path: str | os.PathLike[str],
    *,
    label: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    require_canonical: bool = False,
    require_mode: int | None = None,
    require_owner_uid: int | None = None,
) -> JsonEvidence:
    """Read one strict JSON object and hash it without reopening its pathname.

    In addition to :func:`read_evidence_bytes`, malformed UTF-8, duplicate keys, non-finite JSON
    constants, and non-object roots fail closed.
    """

    evidence = read_evidence_bytes(
        path,
        label=label,
        max_bytes=max_bytes,
        require_mode=require_mode,
        require_owner_uid=require_owner_uid,
    )
    payload = evidence.payload

    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvidenceReadError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceReadError(f"{label} must contain one JSON object")
    if require_canonical and payload != canonical_json_bytes(value):
        raise EvidenceReadError(f"{label} is not canonical JSON")
    return JsonEvidence(
        value=value,
        sha256=evidence.sha256,
        size=evidence.size,
        mode=evidence.mode,
        uid=evidence.uid,
    )
