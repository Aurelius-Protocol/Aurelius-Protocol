# Miner dependency subset extracted from Aurelius v5 (02375403).
r"""Pure verification of a signed operator config snapshot (§9.2), milestone 1.8.

CONTRACT (capsule header)
- Recompute the App-F canonical hash of the body and check the operator's Ed25519 signature over
  `b"aurelius/config_snapshot/v1\n" ‖ hash|version`. Signature + hash ONLY — no clamps, no safety
  floor, no spec parsing (those are validator concerns). Returns the verified body or raises
  `SnapshotInvalid(rule, ...)` naming the failed check.
- SHARED so the validator's `config_client` AND the miner's config reader run the IDENTICAL check
  from ONE implementation: a snapshot the miner trusts is byte-for-byte one the validator trusts.
  `subnet.shared` may not import anything else in-repo (contract `shared-independent`); the crypto
  deps (cryptography for Ed25519, the App-F encoder) are the same ones both callers already use.
"""

from __future__ import annotations
import hashlib
import json
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from subnet.shared.canonicalization import (
    CanonicalizationError,
    canonical_bytes,
    parse_canonical_json,
)

SNAPSHOT_SIGNATURE_DOMAIN = b"aurelius/config_snapshot/v1\n"


class SnapshotInvalid(Exception):
    """The snapshot failed verification. `rule` names the failed check (shape / canonical_hash /
    signature / canonicalization) so callers can log + fail-open on the exact cause."""

    def __init__(self, rule: str, message: str):
        self.rule = rule
        super().__init__(f"{rule}: {message}")


def canonical_hash(body: dict) -> str:
    """The snapshot's identity hash: SHA-256 of the App-F canonical encoding of the body. Round-trip
    through the strict parser first so the value domain matches what the encoder accepts (rejecting,
    e.g., floats a sloppy source might smuggle in)."""
    try:
        strict_body = parse_canonical_json(json.dumps(body))
        return hashlib.sha256(canonical_bytes(strict_body)).hexdigest()
    except (CanonicalizationError, TypeError, ValueError) as e:
        raise SnapshotInvalid("canonicalization", str(e)) from None


def null_keys_present(value, path="") -> dict[str, bool]:
    """JSON-pointer commitment to nulls, which App-F intentionally drops from maps.

    The manifest itself contains only true values, so App-F covers every listed path.
    It is additive to the snapshot envelope; content/charge canonicalization is unchanged.
    """
    if value is None:
        return {path: True}
    found = {}
    if isinstance(value, dict):
        children = value.items()
    elif isinstance(value, list):
        children = enumerate(value)
    else:
        return found
    for key, child in children:
        segment = str(key).replace("~", "~0").replace("/", "~1")
        found.update(null_keys_present(child, path + "/" + segment))
    return found


def verify_snapshot(doc: dict, operator_pubkey_hex: str) -> dict:
    """Return the verified `body`, or raise `SnapshotInvalid` naming the rule. The caller supplies
    the ONE pinned operator pubkey; there is no trust-on-first-use."""
    for field in ("version", "body", "canonical_hash", "signature_hex"):
        if field not in doc:
            raise SnapshotInvalid("shape", f"snapshot missing field {field!r}")

    version = doc["version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise SnapshotInvalid("shape", "version must be an integer")

    body = doc["body"]
    if not isinstance(body, dict):
        raise SnapshotInvalid("shape", "snapshot body must be an object")
    recomputed = canonical_hash(body)
    if "null_keys_present" in body:
        payload = {k: v for k, v in body.items() if k != "null_keys_present"}
        manifest = body["null_keys_present"]
        if (
            not isinstance(manifest, dict)
            or any(value is not True for value in manifest.values())
            or manifest != null_keys_present(payload)
        ):
            raise SnapshotInvalid("null_keys_present", "snapshot null-key commitment differs")
    if recomputed != doc["canonical_hash"]:
        raise SnapshotInvalid("canonical_hash", "body does not hash to the claimed canonical_hash")

    preimage = SNAPSHOT_SIGNATURE_DOMAIN + f"{recomputed}|{version}".encode()
    try:
        pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(operator_pubkey_hex))
        pubkey.verify(bytes.fromhex(doc["signature_hex"]), preimage)
    except (InvalidSignature, ValueError) as e:
        raise SnapshotInvalid("signature", f"operator signature invalid: {e}") from None

    return body
