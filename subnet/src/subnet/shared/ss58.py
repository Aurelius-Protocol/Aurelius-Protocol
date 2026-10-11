# Miner dependency subset extracted from Aurelius v5 (02375403).
"""SS58 address codec (minimal, substrate-standard).

CONTRACT (capsule header)
- Guarantees: `ss58_decode(ss58_encode(pk)) == pk` for 32-byte account ids under simple (< 64)
  prefixes; checksum verified on decode (fail-closed `ValueError` on tamper/garbage).
- Scope: deliberately tiny — 32-byte account ids, single-byte prefix (Bittensor uses the generic
  substrate prefix 42). The full multi-byte-prefix scheme is not needed anywhere in Aurelius.

Checksum per the SS58 registry: first 2 bytes of blake2b-512(b"SS58PRE" || prefix || pubkey).
"""

from __future__ import annotations
import hashlib
import base58

SS58_PREFIX = 42  # generic substrate; what Bittensor hotkeys/coldkeys use


_CHECKSUM_PREAMBLE = b"SS58PRE"


def _checksum(payload: bytes) -> bytes:
    return hashlib.blake2b(_CHECKSUM_PREAMBLE + payload, digest_size=64).digest()[:2]


def ss58_encode(public_key: bytes, prefix: int = SS58_PREFIX) -> str:
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError("ss58_encode requires a 32-byte public key")
    if not 0 <= prefix < 64:
        raise ValueError("only simple (< 64) SS58 prefixes are supported")
    payload = bytes([prefix]) + public_key
    return base58.b58encode(payload + _checksum(payload)).decode("ascii")


def ss58_decode(address: str, expected_prefix: int | None = SS58_PREFIX) -> bytes:
    """Return the 32-byte public key; raise ValueError on bad checksum/length/prefix."""
    try:
        raw = base58.b58decode(address)
    except ValueError as e:
        raise ValueError(f"invalid ss58 address (base58): {e}") from None
    if len(raw) != 1 + 32 + 2:
        raise ValueError(f"invalid ss58 address length: {len(raw)}")
    prefix, public_key, checksum = raw[0], raw[1:33], raw[33:]
    if prefix >= 64:
        raise ValueError("only simple (< 64) SS58 prefixes are supported")
    if checksum != _checksum(raw[:33]):
        raise ValueError("invalid ss58 checksum")
    if expected_prefix is not None and prefix != expected_prefix:
        raise ValueError(f"ss58 prefix mismatch: got {prefix}, expected {expected_prefix}")
    return public_key
