r"""Immutable gamma purchase policy wire identity (gamma plan §8.1) — pure representation.

CONTRACT (capsule header)
- Mirrors the Elixir twin (`GammaPurchasePolicyV1` in `CentralApi.Shared`) EXACTLY: identical
  acceptance set, identical App-F canonical bytes, identical digest domain. Both languages are
  pinned to the same golden vectors (G1), shipped here as
  `subnet/tests/fixtures/gamma_funding_vectors.json`.
- STDLIB-ONLY: no third-party dependency is added — stdlib plus the in-package, golden-pinned
  `subnet.shared.canonicalization` codec. Wire TEXT must be decoded with `parse_canonical_json`
  first (duplicate object keys reject there, fail-closed); `from_map` takes the already-parsed
  dict and rejects unknown fields, non-canonical decimals and every out-of-range value.
- Fail-CLOSED: every violation raises `GammaPolicyRefused` with a stable machine-readable
  `.kind`; nothing is coerced, defaulted or silently repaired. Amounts are integer-only: the
  wire carries canonical decimal STRINGS (`^[1-9][0-9]*$`), converted once to bounded ints
  (plan §8.1 — zero, negative, float, exponent, malformed and overflowed amounts all reject).
- NO economic values live here: pack prices, limits and heights arrive inside the signed
  document. `validate_selection` / `active_at` are pure predicates over supplied input, and a
  `None` policy is the typed-unavailable seam `purchase_policy_unselected` — never a bypass.

Acceptance contract both languages must implement identically (gamma plan §8.1):
- identifier fields are NONEMPTY: `service_id` printable ASCII (0x20–0x7E) 1..64;
  `policy_version` 1..128 of `[A-Za-z0-9._-]`; `packs` keys 1..64 of `[a-z0-9-]`;
  `runtime_profile` 1..128 and `signing_key_id` 1..64 of any string (length-bounded only).
  An explicit `retirement_block: null` is REJECTED — absence is the only "not scheduled" form
  (App F drops null keys before hashing, so a wire null would be a meaningless second spelling).
- `chain_id` is the deployed mainnet chain-id shape `bittensor-mainnet:` + 64 lowercase hex
  (no `0x`, matching the pinned `MAINNET_CHAIN_ID`); `genesis_hash` / `governing_config_hash`
  are `0x` + 64 lowercase hex.
- `netuid` int 1..65535; block fields u32 (`0..2^32-1`); `quote_lifetime_blocks` 1..2^32-1;
  `retirement_block` (when present) must be `>= activation_block` and u32;
  `max_packs_per_intent` must equal 1; `per_account_packs_per_day` / `global_packs_per_day`
  are ints >= 1; `payer_mode` must equal `self_spend_direct_v1`.
- per pack: exactly `{gamma_atoms, work_tokens}` with `gamma_atoms` 1..2^64-1 and
  `work_tokens` 1..2^63-1 as canonical decimal strings.

Digest: `sha256("aurelius:gamma-purchase-policy:v1" || 0x00 || canonical_bytes(body))`,
hex with the `0x` prefix.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass

from subnet.shared.canonicalization import canonical_bytes as _appf_canonical_bytes

SCHEMA = "aurelius/gamma_purchase_policy/v1"
PAYER_MODE = "self_spend_direct_v1"
DIGEST_DOMAIN = b"aurelius:gamma-purchase-policy:v1"

U32_MAX = 2**32 - 1
NETUID_MAX = 65535
GAMMA_ATOMS_MAX = 2**64 - 1
WORK_TOKENS_MAX = 2**63 - 1

_SERVICE_ID = re.compile(r"[\x20-\x7e]{1,64}")
_CHAIN_ID = re.compile(r"bittensor-mainnet:[0-9a-f]{64}")
_HEX0X64 = re.compile(r"0x[0-9a-f]{64}")
_POLICY_VERSION = re.compile(r"[A-Za-z0-9._-]{1,128}")
_PACK_ID = re.compile(r"[a-z0-9-]{1,64}")
_CANONICAL_DECIMAL = re.compile(r"[1-9][0-9]*")

_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "service_id",
        "chain_id",
        "genesis_hash",
        "netuid",
        "policy_version",
        "packs",
        "activation_block",
        "quote_lifetime_blocks",
        "max_packs_per_intent",
        "per_account_packs_per_day",
        "global_packs_per_day",
        "payer_mode",
        "runtime_profile",
        "governing_config_hash",
        "signing_key_id",
    }
)
_OPTIONAL_FIELDS = frozenset({"retirement_block"})


class GammaPolicyRefused(ValueError):
    """Fail-closed gamma purchase-policy rejection. `kind` is a stable, machine-readable label."""

    def __init__(self, kind: str = "invalid_purchase_policy"):
        self.kind = kind
        super().__init__(kind)


def _require(condition, kind: str = "invalid_purchase_policy") -> None:
    if not condition:
        raise GammaPolicyRefused(kind)


def canonical_decimal_amount(value, maximum: int) -> int:
    """Convert one canonical decimal wire string to its bounded int (the single conversion,
    plan §8.1). Rejects everything that is not `^[1-9][0-9]*$` or is outside `1..maximum`."""
    _require(
        isinstance(value, str) and _CANONICAL_DECIMAL.fullmatch(value) is not None,
    )
    amount = int(value)
    _require(1 <= amount <= maximum)
    return amount


def _integer(value, minimum: int, maximum: int, kind: str = "invalid_purchase_policy") -> None:
    _require(type(value) is int and minimum <= value <= maximum, kind)


def _match(value, pattern, kind: str = "invalid_purchase_policy") -> None:
    _require(type(value) is str and pattern.fullmatch(value) is not None, kind)


def _packs(value) -> dict:
    _require(type(value) is dict and len(value) >= 1)
    packs: dict = {}
    for pack_id, pack in value.items():
        _match(pack_id, _PACK_ID)
        _require(type(pack) is dict and set(pack) == {"gamma_atoms", "work_tokens"})
        packs[pack_id] = {
            "gamma_atoms": canonical_decimal_amount(pack["gamma_atoms"], GAMMA_ATOMS_MAX),
            "work_tokens": canonical_decimal_amount(pack["work_tokens"], WORK_TOKENS_MAX),
        }
    return packs


def _validate(body) -> dict:
    """Validate the wire shape and return the parsed packs (ints); raise `GammaPolicyRefused`."""
    _require(type(body) is dict)
    present = frozenset(body)
    _require(
        _REQUIRED_FIELDS <= present and present <= _REQUIRED_FIELDS | _OPTIONAL_FIELDS,
    )
    _require(body["schema"] == SCHEMA)
    _match(body["service_id"], _SERVICE_ID)
    _match(body["chain_id"], _CHAIN_ID)
    _match(body["genesis_hash"], _HEX0X64)
    _integer(body["netuid"], 1, NETUID_MAX)
    _match(body["policy_version"], _POLICY_VERSION)
    packs = _packs(body["packs"])
    _integer(body["activation_block"], 0, U32_MAX)
    if "retirement_block" in body:
        retirement = body["retirement_block"]
        _require(type(retirement) is int and 0 <= retirement <= U32_MAX)
        _require(retirement >= body["activation_block"], "retirement_before_activation")
    _integer(body["quote_lifetime_blocks"], 1, U32_MAX)
    _require(
        type(body["max_packs_per_intent"]) is int and body["max_packs_per_intent"] == 1,
    )
    _require(
        type(body["per_account_packs_per_day"]) is int and body["per_account_packs_per_day"] >= 1
    )
    _require(type(body["global_packs_per_day"]) is int and body["global_packs_per_day"] >= 1)
    _require(body["payer_mode"] == PAYER_MODE)
    _require(type(body["runtime_profile"]) is str and 1 <= len(body["runtime_profile"]) <= 128)
    _match(body["governing_config_hash"], _HEX0X64)
    _require(type(body["signing_key_id"]) is str and 1 <= len(body["signing_key_id"]) <= 64)
    return packs


def canonical_bytes(body) -> bytes:
    """App-F canonical bytes of the policy body (the exact bytes the digest commits to)."""
    return _appf_canonical_bytes(body)


def digest(body) -> str:
    """Domain-separated canonical digest, `0x`-prefixed hex (validates the body first)."""
    _validate(body)
    return "0x" + hashlib.sha256(DIGEST_DOMAIN + b"\x00" + canonical_bytes(body)).hexdigest()


@dataclass(frozen=True)
class GammaPurchasePolicy:
    """One validated immutable purchase-policy wire body.

    `body` is the deep-copied wire dict (canonical-decimal strings preserved exactly); the
    accessor properties re-validate nothing — `from_map` already enforced the contract.
    """

    body: dict

    @property
    def service_id(self) -> str:
        return self.body["service_id"]

    @property
    def chain_id(self) -> str:
        return self.body["chain_id"]

    @property
    def genesis_hash(self) -> str:
        return self.body["genesis_hash"]

    @property
    def netuid(self) -> int:
        return self.body["netuid"]

    @property
    def policy_version(self) -> str:
        return self.body["policy_version"]

    @property
    def activation_block(self) -> int:
        return self.body["activation_block"]

    @property
    def retirement_block(self) -> int | None:
        """Finalized height at/below which the policy stays active; None = not scheduled."""
        return self.body.get("retirement_block")

    @property
    def quote_lifetime_blocks(self) -> int:
        return self.body["quote_lifetime_blocks"]

    @property
    def payer_mode(self) -> str:
        return self.body["payer_mode"]

    @property
    def runtime_profile(self) -> str:
        return self.body["runtime_profile"]

    @property
    def governing_config_hash(self) -> str:
        return self.body["governing_config_hash"]

    @property
    def signing_key_id(self) -> str:
        return self.body["signing_key_id"]

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.body)

    def digest(self) -> str:
        """`0x`-prefixed sha256 over `DIGEST_DOMAIN || 0x00 || canonical_bytes`."""
        return "0x" + hashlib.sha256(DIGEST_DOMAIN + b"\x00" + self.canonical_bytes()).hexdigest()

    def packs(self) -> dict:
        """Validated pack view: `{pack_id: {"gamma_atoms": int, "work_tokens": int}}`."""
        return _packs(self.body["packs"])

    def pack(self, pack_id) -> dict:
        """Exact pack amounts as ints; `pack_id_invalid` for a malformed id, `pack_unknown`
        for a well-formed id the policy does not offer."""
        _match(pack_id, _PACK_ID, "pack_id_invalid")
        packs = self.packs()
        _require(pack_id in packs, "pack_unknown")
        return packs[pack_id]

    def active_at(self, block) -> bool:
        """Pure activity predicate, INCLUSIVE at both boundaries:
        `activation_block <= block` and (no retirement scheduled or `block <= retirement_block`).
        Block heights are u32, mirroring the schema's block domain."""
        _integer(block, 0, U32_MAX, "invalid_block")
        if block < self.activation_block:
            return False
        retirement = self.retirement_block
        return retirement is None or block <= retirement


def from_map(d) -> GammaPurchasePolicy:
    """Strict parse of an already-decoded policy map (unknown fields rejected; non-canonical
    decimals rejected; see the module contract). Returns an isolated immutable copy."""
    _validate(d)
    return GammaPurchasePolicy(body=deepcopy(d))


def validate_selection(policy, pack_id) -> dict:
    """Typed seam over "may this pack be purchased under this policy?".

    A `None` policy refuses with `purchase_policy_unselected` (the typed-unavailable seam — the
    caller must surface the missing policy, never fall back to invented terms). A non-None
    policy refuses with `pack_id_invalid` / `pack_unknown` exactly like `GammaPurchasePolicy.pack`.
    Returns the selected pack's exact amounts on success.
    """
    _require(policy is not None, "purchase_policy_unselected")
    return policy.pack(pack_id)
