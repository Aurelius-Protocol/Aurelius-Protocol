r"""Immutable gamma funding intent wire identity (gamma plan §8.2) — pure representation.

CONTRACT (capsule header)
- Mirrors the Elixir twin (`GammaFundingIntentV1` in `CentralApi.Shared`) EXACTLY: identical
  acceptance set, identical App-F canonical bytes, identical 32-byte payment reference. Both
  languages are pinned to `subnet/tests/fixtures/gamma_funding_vectors.json`.
- STDLIB-ONLY: stdlib plus the in-package `subnet.shared.canonicalization` /
  `subnet.shared.gamma_policy` modules. Ed25519 verification is the repo's existing convention
  (`cryptography` Ed25519 over hex pubkeys, fail-closed bool) imported LAZILY inside
  `verify_signature` so importing this module never adds a third-party dependency; the SS58-42
  account check is a minimal in-module codec mirroring `subnet.shared.ss58`'s exact checksum
  contract without importing `base58`.
- Fail-CLOSED: every violation raises `GammaIntentRefused` with a stable machine-readable
  `.kind`. Wire TEXT must be decoded with `parse_canonical_json` first (duplicate keys reject
  there); `from_map` takes the parsed dict and rejects unknown fields, non-canonical decimals,
  malformed accounts and every ordering violation.
- The quote is immutable after issue: `build` derives the policy-bound fields from the selected
  `GammaPurchasePolicy` and cross-checks pack existence, amount equality, policy identity and
  `(valid_until_block - issued_at_finalized_block) == quote_lifetime_blocks` — it never
  invents amounts, blocks, nonces or policy values (plan §8.2).

Acceptance contract both languages must implement identically:
- identifier fields are NONEMPTY: `intent_id` 1..64 of `[A-Za-z0-9._-]`; `pack_id` 1..64 of
  `[a-z0-9-]`; `purchase_policy_version` 1..128 of `[A-Za-z0-9._-]`; `service_id` printable
  ASCII (0x20–0x7E) 1..64; `client_request_id` printable ASCII 1..128.
- `chain_id` = `bittensor-mainnet:` + 64 lowercase hex (no `0x`); `genesis_hash`,
  `purchase_policy_digest`, `config_hash` and `server_nonce` are `0x` + 64 lowercase hex.
- the five account fields (`beneficiary_coldkey`, `beneficiary_hotkey`, `funding_coldkey`,
  `gamma_api_hotkey`, `authorized_spender`) are SS58-42 (prefix 42, 32-byte account id,
  blake2b checksum) — anything else refuses with `invalid_ss58_account`.
- `payer_mode` must equal `self_spend_direct_v1`; `netuid` int 1..65535; `gamma_atoms` /
  `work_tokens` canonical decimal strings with the policy bounds (1..2^64-1 / 1..2^63-1).
- validity blocks are u32 with `0 <= issued_at_finalized_block <= valid_from_block <=
  valid_until_block <= 2^32-1`.
- V1 invariants (refuse, never repair): `funding_coldkey == beneficiary_coldkey`;
  `authorized_spender == funding_coldkey`; `gamma_api_hotkey` not in
  `{beneficiary_hotkey, beneficiary_coldkey, funding_coldkey}`.

Reference (EXACT plan §8.2 formula):
`reference = SHA256(UTF8("aurelius:wt-gamma-purchase:v1") || 0x00 || canonical_bytes(body))`,
hex with the `0x` prefix. The signature payload is
`"aurelius/gamma_funding_intent/v1\\n" || 0x00 || canonical_bytes(body) || reference_raw_32`.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass

from subnet.shared.canonicalization import canonical_bytes as _appf_canonical_bytes
from subnet.shared.gamma_policy import (
    PAYER_MODE,
    U32_MAX,
    GAMMA_ATOMS_MAX,
    NETUID_MAX,
    WORK_TOKENS_MAX,
    GammaPolicyRefused,
    GammaPurchasePolicy,
    canonical_decimal_amount,
    validate_selection,
)

SCHEMA = "aurelius/gamma_funding_intent/v1"
REFERENCE_DOMAIN = b"aurelius:wt-gamma-purchase:v1"
SIGNATURE_DOMAIN = b"aurelius/gamma_funding_intent/v1\n"

_SERVICE_ID = re.compile(r"[\x20-\x7e]{1,64}")
_CHAIN_ID = re.compile(r"bittensor-mainnet:[0-9a-f]{64}")
_HEX0X64 = re.compile(r"0x[0-9a-f]{64}")
_INTENT_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_POLICY_VERSION = re.compile(r"[A-Za-z0-9._-]{1,128}")
_PACK_ID = re.compile(r"[a-z0-9-]{1,64}")
_CLIENT_REQUEST_ID = re.compile(r"[\x20-\x7e]{1,128}")

_FIELDS = frozenset(
    {
        "schema",
        "intent_id",
        "service_id",
        "chain_id",
        "genesis_hash",
        "netuid",
        "beneficiary_coldkey",
        "beneficiary_hotkey",
        "funding_coldkey",
        "gamma_api_hotkey",
        "authorized_spender",
        "payer_mode",
        "pack_id",
        "gamma_atoms",
        "work_tokens",
        "purchase_policy_version",
        "purchase_policy_digest",
        "config_hash",
        "issued_at_finalized_block",
        "valid_from_block",
        "valid_until_block",
        "server_nonce",
        "client_request_id",
    }
)

# fields `build` derives from the selected policy (assigns may repeat them only verbatim)
_POLICY_DERIVED = (
    "service_id",
    "chain_id",
    "genesis_hash",
    "netuid",
    "payer_mode",
    "purchase_policy_version",
    "purchase_policy_digest",
)
_ASSIGNS_FIELDS = _FIELDS - frozenset(_POLICY_DERIVED) - {"schema", "pack_id"}

_SS58_PREFIX = 42
_SS58_PRE = b"SS58PRE"
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {ch: i for i, ch in enumerate(_B58_ALPHABET)}


class GammaIntentRefused(ValueError):
    """Fail-closed gamma funding-intent rejection. `kind` is a stable, machine-readable label."""

    def __init__(self, kind: str = "invalid_funding_intent"):
        self.kind = kind
        super().__init__(kind)


def _require(condition, kind: str = "invalid_funding_intent") -> None:
    if not condition:
        raise GammaIntentRefused(kind)


def _match(value, pattern, kind: str = "invalid_funding_intent") -> None:
    _require(type(value) is str and pattern.fullmatch(value) is not None, kind)


def _integer(value, minimum: int, maximum: int, kind: str = "invalid_funding_intent") -> None:
    _require(type(value) is int and minimum <= value <= maximum, kind)


# --- minimal dep-free SS58-42 codec (mirrors subnet.shared.ss58's exact contract) ------------


def _b58decode(text: str):
    """Base58 decode to bytes, or None when `text` leaves the base58 alphabet."""
    num = 0
    for ch in text:
        digit = _B58_INDEX.get(ch)
        if digit is None:
            return None
        num = num * 58 + digit
    body = num.to_bytes((num.bit_length() + 7) // 8, "big")
    pad = len(text) - len(text.lstrip("1"))
    return b"\x00" * pad + body


def _b58encode(raw: bytes) -> str:
    num = int.from_bytes(raw, "big")
    digits = []
    while num:
        num, rem = divmod(num, 58)
        digits.append(_B58_ALPHABET[rem])
    pad = len(raw) - len(raw.lstrip(b"\x00"))
    return "1" * pad + "".join(reversed(digits))


def ss58_encode_account(public_key: bytes, prefix: int = _SS58_PREFIX) -> str:
    """SS58-encode a 32-byte account id (simple prefixes only).

    Checksum per the SS58 registry, identical to `subnet.shared.ss58.ss58_encode`: first 2
    bytes of blake2b-512(b"SS58PRE" || prefix || pubkey). Dep-free so tests and the golden
    vector generator can produce synthetic addresses without the `base58` package.
    """
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError("ss58_encode_account requires a 32-byte public key")
    if not 0 <= prefix < 64:
        raise ValueError("only simple (< 64) SS58 prefixes are supported")
    payload = bytes([prefix]) + public_key
    checksum = hashlib.blake2b(_SS58_PRE + payload, digest_size=64).digest()[:2]
    return _b58encode(payload + checksum)


def ss58_decode_account(address: str, expected_prefix: int = _SS58_PREFIX) -> bytes:
    """Return the 32-byte public key of an SS58-42 account id; raise ValueError on any bad
    alphabet, length, prefix or checksum (same rejections as `subnet.shared.ss58.ss58_decode`)."""
    if not isinstance(address, str):
        raise ValueError("ss58 address must be a string")
    raw = _b58decode(address)
    if raw is None or len(raw) != 1 + 32 + 2:
        raise ValueError(f"invalid ss58 address length: {len(address)} chars")
    prefix, public_key, checksum = raw[0], raw[1:33], raw[33:]
    if prefix >= 64:
        raise ValueError("only simple (< 64) SS58 prefixes are supported")
    if checksum != hashlib.blake2b(_SS58_PRE + raw[:33], digest_size=64).digest()[:2]:
        raise ValueError("invalid ss58 checksum")
    if prefix != expected_prefix:
        raise ValueError(f"ss58 prefix mismatch: got {prefix}, expected {expected_prefix}")
    return public_key


def _decimal(value, maximum: int) -> int:
    """Canonical decimal conversion with the INTENT refusal type (the shared helper in
    `gamma_policy` raises its own module's exception; a cross-module leak would surface the
    wrong refusal class to callers of this module)."""
    try:
        return canonical_decimal_amount(value, maximum)
    except GammaPolicyRefused:
        raise GammaIntentRefused("invalid_funding_intent") from None


def _account(body, name: str) -> None:
    try:
        ss58_decode_account(body[name])
    except ValueError:
        raise GammaIntentRefused("invalid_ss58_account") from None


# --- wire validation -------------------------------------------------------------------------


def _validate(body) -> None:
    _require(type(body) is dict and frozenset(body) == _FIELDS)
    _require(body["schema"] == SCHEMA)
    _match(body["intent_id"], _INTENT_ID)
    _match(body["service_id"], _SERVICE_ID)
    _match(body["chain_id"], _CHAIN_ID)
    _match(body["genesis_hash"], _HEX0X64)
    _integer(body["netuid"], 1, NETUID_MAX)
    for name in (
        "beneficiary_coldkey",
        "beneficiary_hotkey",
        "funding_coldkey",
        "gamma_api_hotkey",
        "authorized_spender",
    ):
        _account(body, name)
    _require(body["payer_mode"] == PAYER_MODE)
    _match(body["pack_id"], _PACK_ID)
    _decimal(body["gamma_atoms"], GAMMA_ATOMS_MAX)
    _decimal(body["work_tokens"], WORK_TOKENS_MAX)
    _match(body["purchase_policy_version"], _POLICY_VERSION)
    _match(body["purchase_policy_digest"], _HEX0X64)
    _match(body["config_hash"], _HEX0X64)
    _integer(body["issued_at_finalized_block"], 0, U32_MAX)
    _integer(body["valid_from_block"], 0, U32_MAX)
    _integer(body["valid_until_block"], 0, U32_MAX)
    _require(
        body["valid_from_block"] >= body["issued_at_finalized_block"],
        "valid_from_before_issued",
    )
    _require(body["valid_until_block"] >= body["valid_from_block"], "valid_until_before_valid_from")
    _match(body["server_nonce"], _HEX0X64)
    _match(body["client_request_id"], _CLIENT_REQUEST_ID)
    # V1 invariants (plan §8.2): own-account payment, own-account spend, distinct API hotkey.
    _require(body["funding_coldkey"] == body["beneficiary_coldkey"], "funding_coldkey_mismatch")
    _require(body["authorized_spender"] == body["funding_coldkey"], "authorized_spender_mismatch")
    _require(
        body["gamma_api_hotkey"]
        not in {
            body["beneficiary_hotkey"],
            body["beneficiary_coldkey"],
            body["funding_coldkey"],
        },
        "gamma_api_hotkey_not_distinct",
    )


def canonical_bytes(body) -> bytes:
    """App-F canonical bytes of the closed intent body (everything except reference/signature)."""
    return _appf_canonical_bytes(body)


def reference(body) -> str:
    """The EXACT 32-byte payment reference (plan §8.2), `0x`-prefixed hex:
    sha256("aurelius:wt-gamma-purchase:v1" || 0x00 || canonical_bytes(body))."""
    _validate(body)
    return "0x" + hashlib.sha256(REFERENCE_DOMAIN + b"\x00" + canonical_bytes(body)).hexdigest()


def signature_payload(body) -> bytes:
    """The exact bytes the quote/intent envelope signs (domain tag, 0x00 separator, canonical
    body, then the RAW 32 reference bytes so body and reference are bound together)."""
    _validate(body)
    encoded = canonical_bytes(body)
    reference_raw = hashlib.sha256(REFERENCE_DOMAIN + b"\x00" + encoded).digest()
    return SIGNATURE_DOMAIN + b"\x00" + encoded + reference_raw


@dataclass(frozen=True)
class GammaFundingIntent:
    """One validated immutable funding-intent wire body (properties read the wire dict)."""

    body: dict

    @property
    def intent_id(self) -> str:
        return self.body["intent_id"]

    @property
    def reference_hex(self) -> str:
        return reference(self.body)

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.body)

    def reference(self) -> str:
        return reference(self.body)

    def signature_payload(self) -> bytes:
        return signature_payload(self.body)


def from_map(d) -> GammaFundingIntent:
    """Strict parse of an already-decoded intent map (unknown fields rejected; V1 invariants
    enforced). Returns an isolated immutable copy."""
    _validate(d)
    return GammaFundingIntent(body=deepcopy(d))


def build(policy, pack_id, **assigns) -> GammaFundingIntent:
    """Construct one policy-bound intent (the server-side quote path, plan §8.2/§8.3).

    `policy` is a `GammaPurchasePolicy` or None. Refusal kinds, in check order:
    `purchase_policy_unselected` (None policy — the typed-unavailable seam),
    `pack_id_invalid`, `pack_unknown` (via `gamma_policy.validate_selection`),
    `invalid_funding_intent` (unknown/missing assigns, malformed amounts or blocks),
    `pack_amount_mismatch` (assigns amounts != the selected pack),
    `purchase_policy_mismatch` (a policy-derived field repeated with a different value),
    `quote_lifetime_mismatch` (valid_until - issued_at != quote_lifetime_blocks),
    then every `from_map` kind. `pack_id` comes from the positional argument (never assigns).
    The policy-derived fields (`service_id`, `chain_id`, `genesis_hash`, `netuid`,
    `payer_mode`, `purchase_policy_version`, `purchase_policy_digest`) always come from the
    policy; assigns may repeat them only verbatim. Nothing is defaulted or invented here.
    """
    _require(policy is None or isinstance(policy, GammaPurchasePolicy))
    selected = validate_selection(policy, pack_id)
    unknown = frozenset(assigns) - _ASSIGNS_FIELDS - frozenset(_POLICY_DERIVED)
    missing = _ASSIGNS_FIELDS - frozenset(assigns)
    _require(not unknown and not missing)
    derived = {
        "service_id": policy.service_id,
        "chain_id": policy.chain_id,
        "genesis_hash": policy.genesis_hash,
        "netuid": policy.netuid,
        "payer_mode": policy.payer_mode,
        "purchase_policy_version": policy.policy_version,
        "purchase_policy_digest": policy.digest(),
    }
    body = {
        "schema": SCHEMA,
        "pack_id": pack_id,
        # sorted so the constructed body (and any JSON echoing it) is byte-deterministic
        # across processes — set iteration order is hash-seed dependent
        **{k: assigns[k] for k in sorted(_ASSIGNS_FIELDS)},
        **derived,
    }
    _decimal(body["gamma_atoms"], GAMMA_ATOMS_MAX)
    _decimal(body["work_tokens"], WORK_TOKENS_MAX)
    _require(
        int(body["gamma_atoms"]) == selected["gamma_atoms"]
        and int(body["work_tokens"]) == selected["work_tokens"],
        "pack_amount_mismatch",
    )
    _require(
        all(assigns.get(k, derived[k]) == derived[k] for k in _POLICY_DERIVED),
        "purchase_policy_mismatch",
    )
    issued = body["issued_at_finalized_block"]
    until = body["valid_until_block"]
    _integer(issued, 0, U32_MAX)
    _integer(until, 0, U32_MAX)
    _require(until - issued == policy.quote_lifetime_blocks, "quote_lifetime_mismatch")
    return from_map(body)


def verify_signature(payload, signature, pubkey_hex) -> bool:
    """Fail-closed Ed25519 verification of a signature over `signature_payload` bytes.

    Same convention as the repo's other Ed25519 verify sites (`cryptography` Ed25519 over a
    hex pubkey); `cryptography` is imported lazily so this module stays importable everywhere
    the dep-free core runs. Malformed inputs return False, never raise (mirrors
    `subnet.shared.signing.verify`).
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not isinstance(payload, bytes):
        return False
    try:
        if isinstance(signature, str):
            signature = bytes.fromhex(signature)
        if not isinstance(signature, bytes) or not isinstance(pubkey_hex, str):
            return False
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex)).verify(signature, payload)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False
