r"""Gamma funding receipt wire identity (gamma plan §9 "Receipt") — pure representation.

CONTRACT (capsule header)
- Mirrors the Elixir twin (`GammaFundingReceiptV1` in `CentralApi.Shared`) EXACTLY: identical
  acceptance set, identical App-F canonical bytes, identical signature domain. Both languages
  are pinned to `subnet/tests/fixtures/gamma_funding_vectors.json`.
- STDLIB-ONLY: stdlib plus the in-package `canonicalization` / `gamma_policy` / `gamma_intent`
  modules; Ed25519 verification follows the repo convention with a lazy `cryptography` import
  (same as `gamma_intent.verify_signature`).
- Fail-CLOSED: every violation raises `GammaFundingReceiptRefused` with a stable
  machine-readable `.kind`. The receipt is EVIDENCE of one verified settlement — this module
  validates representation and the physical-event identity only; it never recomputes chain
  facts, policy math or credits (plan §9/§10).
- The `deposit_key` field must equal the EXACT deployed physical-event identity built by the
  current codebase (`Deposits.Intake`, `subnet.miner.funding`, `chain_sidecar.alpha_evidence`):
  `chain_id || block_hash || extrinsic_index || event_index` joined with U+2016 (‖). Reusing
  that format — not a gamma-specific one — keeps one physical on-chain event to one credit,
  across funding providers (plan §9).

Acceptance contract both languages must implement identically:
- `schema` must equal `aurelius/gamma_funding_receipt/v1`; `funding_method` must equal `gamma`.
- identifier fields are NONEMPTY: `intent_id` 1..64 of `[A-Za-z0-9._-]`; `pack_id` 1..64 of
  `[a-z0-9-]`; `purchase_policy_version` 1..128 of `[A-Za-z0-9._-]`; `verification_profile`
  1..128 and `signing_key_id` 1..64 of any string (length-bounded only).
- `reference`, `purchase_policy_digest`, `config_hash`, `genesis_hash`, `finalized_block_hash`,
  `tx_hash` are `0x` + 64 lowercase hex; `chain_id` is `bittensor-mainnet:` + 64 lowercase hex.
- the five account fields (`beneficiary_coldkey`, `beneficiary_hotkey`, `funding_coldkey`,
  `gamma_api_hotkey`, `authorized_spender`) are SS58-42 (prefix 42, 32-byte account id,
  blake2b checksum); anything else refuses with `invalid_ss58_account`.
- `netuid` int 1..65535; `finalized_block_number` any int (bool excluded); `extrinsic_index`
  and `event_index` ints >= 0; `ledger_transaction_id` int >= 1; `gamma_atoms` / `work_tokens`
  canonical decimal strings with the wire bounds (1..2^64-1 / 1..2^63-1).
- `deposit_key` must equal `deposit_key(chain_id, finalized_block_hash, extrinsic_index,
  event_index)` or the receipt refuses with `deposit_key_mismatch`.

Signature payload: `"aurelius/gamma_funding_receipt/v1\\n" || 0x00 || canonical_bytes(body)`
(the receipt body alone — unlike the intent payload it binds no separate reference bytes).
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass

from subnet.shared.canonicalization import canonical_bytes as _appf_canonical_bytes
from subnet.shared.gamma_intent import ss58_decode_account
from subnet.shared.gamma_policy import (
    GAMMA_ATOMS_MAX,
    NETUID_MAX,
    WORK_TOKENS_MAX,
    GammaPolicyRefused,
    canonical_decimal_amount,
)

SCHEMA = "aurelius/gamma_funding_receipt/v1"
FUNDING_METHOD = "gamma"
SIGNATURE_DOMAIN = b"aurelius/gamma_funding_receipt/v1\n"
DEPOSIT_KEY_SEPARATOR = "‖"  # U+2016 — the deployed deposit_key join, unchanged by gamma

_CHAIN_ID = re.compile(r"bittensor-mainnet:[0-9a-f]{64}")
_HEX0X64 = re.compile(r"0x[0-9a-f]{64}")
_INTENT_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_POLICY_VERSION = re.compile(r"[A-Za-z0-9._-]{1,128}")
_PACK_ID = re.compile(r"[a-z0-9-]{1,64}")

_FIELDS = frozenset(
    {
        "schema",
        "funding_method",
        "intent_id",
        "reference",
        "pack_id",
        "purchase_policy_version",
        "purchase_policy_digest",
        "config_hash",
        "beneficiary_coldkey",
        "beneficiary_hotkey",
        "chain_id",
        "genesis_hash",
        "netuid",
        "finalized_block_hash",
        "finalized_block_number",
        "extrinsic_index",
        "event_index",
        "tx_hash",
        "funding_coldkey",
        "gamma_api_hotkey",
        "authorized_spender",
        "gamma_atoms",
        "work_tokens",
        "deposit_key",
        "ledger_transaction_id",
        "verification_profile",
        "signing_key_id",
    }
)

_ACCOUNT_FIELDS = (
    "beneficiary_coldkey",
    "beneficiary_hotkey",
    "funding_coldkey",
    "gamma_api_hotkey",
    "authorized_spender",
)


class GammaFundingReceiptRefused(ValueError):
    """Fail-closed gamma funding-receipt rejection. `kind` is stable and machine-readable."""

    def __init__(self, kind: str = "invalid_funding_receipt"):
        self.kind = kind
        super().__init__(kind)


def _require(condition, kind: str = "invalid_funding_receipt") -> None:
    if not condition:
        raise GammaFundingReceiptRefused(kind)


def _match(value, pattern, kind: str = "invalid_funding_receipt") -> None:
    _require(type(value) is str and pattern.fullmatch(value) is not None, kind)


def _integer(value, minimum: int, kind: str = "invalid_funding_receipt") -> None:
    _require(type(value) is int and value >= minimum, kind)


def _decimal(value, maximum: int) -> int:
    """Canonical decimal conversion with the RECEIPT refusal type (the shared helper in
    `gamma_policy` raises its own module's exception; no cross-module leak)."""
    try:
        return canonical_decimal_amount(value, maximum)
    except GammaPolicyRefused:
        raise GammaFundingReceiptRefused("invalid_funding_receipt") from None


def deposit_key(chain_id: str, block_hash: str, extrinsic_index: int, event_index: int) -> str:
    """The EXACT deployed physical-event identity (unchanged by gamma, plan §9):
    `chain_id ‖ block_hash ‖ extrinsic_index ‖ event_index` with U+2016 joins — byte-identical
    to `Deposits.Intake`, `subnet.miner.funding` and `chain_sidecar.alpha_evidence`."""
    return (
        f"{chain_id}{DEPOSIT_KEY_SEPARATOR}{block_hash}"
        f"{DEPOSIT_KEY_SEPARATOR}{extrinsic_index}{DEPOSIT_KEY_SEPARATOR}{event_index}"
    )


def _validate(body) -> None:
    _require(type(body) is dict and frozenset(body) == _FIELDS)
    _require(body["schema"] == SCHEMA)
    _require(body["funding_method"] == FUNDING_METHOD)
    _match(body["intent_id"], _INTENT_ID)
    _match(body["reference"], _HEX0X64)
    _match(body["pack_id"], _PACK_ID)
    _match(body["purchase_policy_version"], _POLICY_VERSION)
    _match(body["purchase_policy_digest"], _HEX0X64)
    _match(body["config_hash"], _HEX0X64)
    for name in _ACCOUNT_FIELDS:
        try:
            ss58_decode_account(body[name])
        except ValueError:
            raise GammaFundingReceiptRefused("invalid_ss58_account") from None
    _match(body["chain_id"], _CHAIN_ID)
    _match(body["genesis_hash"], _HEX0X64)
    _require(type(body["netuid"]) is int and 1 <= body["netuid"] <= NETUID_MAX)
    _match(body["finalized_block_hash"], _HEX0X64)
    _require(type(body["finalized_block_number"]) is int)
    _integer(body["extrinsic_index"], 0)
    _integer(body["event_index"], 0)
    _match(body["tx_hash"], _HEX0X64)
    _decimal(body["gamma_atoms"], GAMMA_ATOMS_MAX)
    _decimal(body["work_tokens"], WORK_TOKENS_MAX)
    _require(type(body["deposit_key"]) is str)
    expected_key = deposit_key(
        body["chain_id"],
        body["finalized_block_hash"],
        body["extrinsic_index"],
        body["event_index"],
    )
    _require(body["deposit_key"] == expected_key, "deposit_key_mismatch")
    _integer(body["ledger_transaction_id"], 1)
    _require(
        type(body["verification_profile"]) is str and 1 <= len(body["verification_profile"]) <= 128
    )
    _require(type(body["signing_key_id"]) is str and 1 <= len(body["signing_key_id"]) <= 64)


def canonical_bytes(body) -> bytes:
    """App-F canonical bytes of the receipt body (the exact bytes the signature covers)."""
    return _appf_canonical_bytes(body)


def signature_payload(body) -> bytes:
    """The exact bytes the receipt signature covers (domain tag, 0x00, canonical body)."""
    _validate(body)
    return SIGNATURE_DOMAIN + b"\x00" + canonical_bytes(body)


@dataclass(frozen=True)
class GammaFundingReceipt:
    """One validated immutable funding-receipt wire body (properties read the wire dict)."""

    body: dict

    @property
    def intent_id(self) -> str:
        return self.body["intent_id"]

    @property
    def reference_hex(self) -> str:
        return self.body["reference"]

    @property
    def deposit_key(self) -> str:
        return self.body["deposit_key"]

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.body)

    def signature_payload(self) -> bytes:
        return signature_payload(self.body)


def from_map(d) -> GammaFundingReceipt:
    """Strict parse of an already-decoded receipt map (unknown fields rejected; the
    physical-event identity re-derived and compared). Returns an isolated immutable copy."""
    _validate(d)
    return GammaFundingReceipt(body=deepcopy(d))


def verify_signature(payload, signature, pubkey_hex) -> bool:
    """Fail-closed Ed25519 verification of a signature over `signature_payload` bytes.

    Same convention as `gamma_intent.verify_signature` and the repo's other Ed25519 sites:
    `cryptography` is imported lazily; malformed inputs return False, never raise.
    """
    from subnet.shared.gamma_intent import verify_signature as _intent_verify

    return _intent_verify(payload, signature, pubkey_hex)
