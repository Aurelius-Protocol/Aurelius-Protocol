r"""work_id signing: authorship + freshness (spec §3.3), REAL sr25519.

CONTRACT (capsule header)
- Owns (§12): 1 (jointly: nothing here debits anything — signing proves authorship only; the
  single spend is the server-side terminal charge).
- Guarantees: each signed-byte construction is versioned and domain-separated. V1 is retained as
  the committed fixture decoder; production authorization uses `aurelius/work_id/v2\n`, which also
  binds the validator audience, request id, and chain-derived fanout window (ADR-0058). A work_id
  signature can never be replayed as any other protocol message (e.g. a session challenge). The
  byte constructions are mirrored in Elixir
  (`CentralApi.Shared.WorkId`); the server reconstructs the EXACT bytes and re-verifies at the
  point of economic effect (§8.3). Verifying over different bytes than were signed is a
  confused-deputy bug (§17).
- sr25519 via `py-sr25519-bindings` (schnorrkel, substrate signing context) — the same primitive
  the future Elixir NIF library verifies. Until that library lands, the Elixir side runs a
  shape-validating always-valid stub (see divergence register); the verify vectors in
  `fixtures/signatures/sr25519_vectors.json` are the acceptance suite for the real library.

Field rules keep the `|`-joined format unambiguous: content_hash and nonce are fixed-width
lowercase hex, the timestamp is a bare integer, and the ss58 hotkey is base58 (no `|` possible).
"""

from __future__ import annotations

import hashlib
import re

import sr25519

WORK_ID_DOMAIN = b"aurelius/work_id/v1\n"
WORK_ID_V2_DOMAIN = b"aurelius/work_id/v2\n"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SS58 = re.compile(r"^[1-9A-HJ-NP-Za-km-z]+$")  # base58 alphabet — cannot contain '|'


def keypair_from_seed(seed: bytes) -> tuple[bytes, bytes]:
    """Return (public_key, secret_key) for a 32-byte mini secret seed."""
    if not isinstance(seed, bytes) or len(seed) != 32:
        raise ValueError("seed must be exactly 32 bytes")
    return sr25519.pair_from_seed(seed)


def sign(message: bytes, keypair: tuple[bytes, bytes]) -> bytes:
    """sr25519-sign `message` with (public_key, secret_key). Non-deterministic by design."""
    public_key, secret_key = keypair
    return sr25519.sign((public_key, secret_key), message)


def verify(signature: bytes, message: bytes, public_key: bytes) -> bool:
    """Return True iff `signature` is a valid sr25519 signature over `message`.

    Fail-closed: malformed inputs (wrong lengths, wrong types) return False rather than raising —
    a caller must never mistake a crash for a pass.
    """
    if not isinstance(signature, bytes) or len(signature) != 64:
        return False
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        return False
    if not isinstance(message, bytes):
        return False
    try:
        return bool(sr25519.verify(signature, message, public_key))
    except Exception:
        return False


def work_id_signed_bytes(content_hash: str, hotkey_ss58: str, timestamp: int, nonce: str) -> bytes:
    """The EXACT byte string a miner's hotkey signs for one unit of work (versioned, v1)."""
    if not isinstance(content_hash, str) or not _HEX64.fullmatch(content_hash):
        raise ValueError("content_hash must be 64 lowercase hex chars")
    if (
        not isinstance(hotkey_ss58, str)
        or not 1 <= len(hotkey_ss58) <= 64
        or not _SS58.fullmatch(hotkey_ss58)
    ):
        raise ValueError("hotkey_ss58 must be a base58 SS58 address")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError("timestamp must be a non-negative integer")
    if not isinstance(nonce, str) or not _HEX64.fullmatch(nonce):
        raise ValueError("nonce must be 64 lowercase hex chars (32 random bytes)")
    return WORK_ID_DOMAIN + f"{content_hash}|{hotkey_ss58}|{timestamp}|{nonce}".encode("utf-8")


def work_id_v2_signed_bytes(
    *,
    content_hash: str,
    hotkey_ss58: str,
    validator_hotkey_ss58: str,
    request_id: str,
    fanout_window: int,
    timestamp: int,
    nonce: str,
) -> bytes:
    """Exact v2 authorization bytes (ADR-0058).

    V2 is deliberately keyword-only: the two SS58 identities and three integer/hex identifiers
    are security-sensitive and a positional transposition must be visually difficult. Every
    delimiter-bearing field has a closed alphabet/fixed width.
    """
    if not isinstance(content_hash, str) or not _HEX64.fullmatch(content_hash):
        raise ValueError("content_hash must be 64 lowercase hex chars")
    if (
        not isinstance(hotkey_ss58, str)
        or not 1 <= len(hotkey_ss58) <= 64
        or not _SS58.fullmatch(hotkey_ss58)
    ):
        raise ValueError("hotkey_ss58 must be a base58 SS58 address")
    if (
        not isinstance(validator_hotkey_ss58, str)
        or not 1 <= len(validator_hotkey_ss58) <= 64
        or not _SS58.fullmatch(validator_hotkey_ss58)
    ):
        raise ValueError("validator_hotkey_ss58 must be a base58 SS58 address")
    if not isinstance(request_id, str) or not _HEX64.fullmatch(request_id):
        raise ValueError("request_id must be 64 lowercase hex chars")
    if isinstance(fanout_window, bool) or not isinstance(fanout_window, int) or fanout_window < 0:
        raise ValueError("fanout_window must be a non-negative integer")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError("timestamp must be a non-negative integer")
    if not isinstance(nonce, str) or not _HEX64.fullmatch(nonce):
        raise ValueError("nonce must be 64 lowercase hex chars (32 random bytes)")
    return WORK_ID_V2_DOMAIN + (
        f"{content_hash}|{hotkey_ss58}|{validator_hotkey_ss58}|{request_id}|"
        f"{fanout_window}|{timestamp}|{nonce}"
    ).encode("utf-8")


ARTIFACT_UPLOAD_DOMAIN = b"aurelius/artifact_upload/v1\n"

_SLOT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def artifact_upload_signed_bytes(
    *,
    hotkey_ss58: str,
    slot_id: str,
    digest: str,
    size: int,
    nonce: str,
    timestamp: int,
) -> bytes:
    """Exact artifact-upload authorization bytes (upload ingress).

    Same field discipline as `work_id_v2_signed_bytes`: every delimiter-bearing field has a
    closed alphabet/fixed width so the `|`-joined format stays unambiguous, and the distinct
    domain prefix keeps an upload signature from replaying as any other protocol message.
    `hotkey_ss58` is only shape-checked here (base58, 1..64); canonical SS58-42 form is
    enforced by the ref grammar (`artifact_ingress_ref`) and by the ingress, which decodes it.
    """
    if (
        not isinstance(hotkey_ss58, str)
        or not 1 <= len(hotkey_ss58) <= 64
        or not _SS58.fullmatch(hotkey_ss58)
    ):
        raise ValueError("hotkey_ss58 must be a base58 SS58 address")
    if not isinstance(slot_id, str) or "|" in slot_id or _SLOT_ID.fullmatch(slot_id) is None:
        raise ValueError("slot_id must match ^[a-z0-9][a-z0-9._-]{0,63}$ (and contain no '|')")
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise ValueError("digest must be 64 lowercase hex chars")
    if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= 2**40:
        raise ValueError("size must be a non-negative integer <= 2**40")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError("timestamp must be a non-negative integer")
    if not isinstance(nonce, str) or not _HEX64.fullmatch(nonce):
        raise ValueError("nonce must be 64 lowercase hex chars (32 random bytes)")
    return ARTIFACT_UPLOAD_DOMAIN + (
        f"{hotkey_ss58}|{slot_id}|{digest}|{size}|{timestamp}|{nonce}"
    ).encode("utf-8")


def derive_work_id(content_hash: str, hotkey_ss58: str, timestamp: int, nonce: str) -> str:
    """The opaque work_id token: lowercase hex sha256 of the signed bytes (server-recomputable)."""
    return hashlib.sha256(
        work_id_signed_bytes(content_hash, hotkey_ss58, timestamp, nonce)
    ).hexdigest()


def derive_work_id_v2(**fields) -> str:
    """Opaque lowercase-hex token for the exact v2 authorization bytes."""
    return hashlib.sha256(work_id_v2_signed_bytes(**fields)).hexdigest()


def sign_work_id(
    keypair: tuple[bytes, bytes],
    *,
    content_hash: str,
    hotkey_ss58: str,
    timestamp: int,
    nonce: str,
) -> bytes:
    return sign(work_id_signed_bytes(content_hash, hotkey_ss58, timestamp, nonce), keypair)


def sign_work_id_v2(keypair: tuple[bytes, bytes], **fields) -> bytes:
    return sign(work_id_v2_signed_bytes(**fields), keypair)


def verify_work_id(
    public_key: bytes,
    signature: bytes,
    *,
    content_hash: str,
    hotkey_ss58: str,
    timestamp: int,
    nonce: str,
) -> bool:
    """Application-level re-verification of authorship (§3.1 defense in depth). Fail-closed."""
    try:
        message = work_id_signed_bytes(content_hash, hotkey_ss58, timestamp, nonce)
    except (TypeError, ValueError):
        # fail-closed: any out-of-shape field means "not verified", never a crash
        return False
    return verify(signature, message, public_key)


def verify_work_id_v2(public_key: bytes, signature: bytes, **fields) -> bool:
    """Fail-closed application verification of the audience/window-bound v2 authorization."""
    try:
        message = work_id_v2_signed_bytes(**fields)
    except (TypeError, ValueError):
        return False
    return verify(signature, message, public_key)


def sign_artifact_upload(keypair: tuple[bytes, bytes], **fields) -> bytes:
    return sign(artifact_upload_signed_bytes(**fields), keypair)


def verify_artifact_upload(public_key: bytes, signature: bytes, **fields) -> bool:
    """Fail-closed application verification of artifact-upload authorization.

    Callers MUST pass ``ss58_decode(fields["hotkey_ss58"])`` as ``public_key`` — the signature
    is only meaningful when the verifying key is the one the message names.
    """
    try:
        message = artifact_upload_signed_bytes(**fields)
    except (TypeError, ValueError):
        return False
    return verify(signature, message, public_key)


def is_fresh(timestamp: int, *, now: int, max_past: int, max_future: int) -> bool:
    """Freshness check against a caller-supplied anchor (§3.3) — no wall-clock in here.

    `now` is injected (the validator's cycle-start anchor / the frozen testlab Clock), keeping
    this pure and deterministic under test.
    """
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        return False
    return (now - max_past) <= timestamp <= (now + max_future)
