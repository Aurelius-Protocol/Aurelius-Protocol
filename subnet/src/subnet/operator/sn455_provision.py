# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Guarded, manual SN455 provisioning for the real Bittensor 11 wire proof.

This module owns no production economics.  It provisions one testnet-only burn identity and one
optional miner-2 TAO refill behind explicit operator acknowledgements.  Every chain mutation is
claimed with an owner-only ``O_EXCL`` sentinel before signing.  Mutations use SDK11's public
``plan.call -> prepare_call -> sign -> submit_signature`` flow, with each fallible boundary made
durable before the next one is crossed.  A claimed submission is scan-only until its exact mortal
era is fully finalized; at most one freshly prepared second attempt is authorized by an exact
zero-match or unsigned-expiry proof.

The orchestration is deliberately backend-driven so its safety contract is hermetically testable.
``BittensorV11ProvisionBackend`` is the only class that imports the optional SDK, and it uses only
v11 intents/generated runtime descriptors plus the public prepared-extrinsic API.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any
from subnet.shared.ss58 import ss58_decode

EXPECTED_ENDPOINT = "wss://test.finney.opentensor.ai:443"


EXPECTED_CHAIN = "Bittensor"


EXPECTED_GENESIS_HASH = "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"


RUNTIME452_BURN_GENERATION = "bittensor-11-sn455-runtime452-e2e-burn-v1"


RUNTIME452_BURN_TOPOLOGY_SHA256 = "6a275587daed74c5f9939fa2dcf91482dd222a09b6dbc6b4b09ce142ad88270e"


EXPECTED_SOURCE_WALLET_ROWS = (
    (
        "deposit-address",
        "deposit_address",
        None,
        "5CnsmMifhnd5LVH2Lq9DbokVkwxXTwueTmmKjanNeUUN5Yxt",
        "default",
        "5GnEb8t6V1SQ31h6VN3vvAqbzj2N664GqQLX1nH4Ddbqhd9D",
    ),
    (
        "deposit-treasury",
        "deposit_treasury",
        None,
        "5H3WLvU7u2RKxNdpDCBuQc9nnJ33dv6eoS84NuQv7u3fUMHM",
        "default",
        "5Eq93sfzs61oz41P4mF3pKXP6JXFZDUdsmUoL8GGTLZGnjqN",
    ),
    (
        "miner-2",
        "miner",
        16,
        "5DLvnYrcWAUtLmQNoB6uZKG6vNcmAYHkjESTUgZp1xLmGbRA",
        "default",
        "5Df2NGWgGhKHYBWfivty3AaL42bnKXSeFGcjCCdULdXk9yAe",
    ),
)


BURN_FUND_RAO = 100_000_000  # exactly 0.10 test TAO


BURN_REGISTRATION_CAP_RAO = 10_000_000  # at most 0.01 test TAO


MAX_EXTRINSIC_FEE_RAO = 10_000_000  # at most 0.01 test TAO per mutation


SOURCE_FREE_RESERVE_RAO = 100_000_000


BURN_FREE_RESERVE_RAO = 20_000_000


CHAIN_HASH = re.compile(r"^0x[0-9a-fA-F]{64}$")


class ProvisionError(RuntimeError):
    """The bounded provisioning contract is not satisfied."""


@dataclass(frozen=True)
class ReceiptEvidence:
    response_type: str
    success: bool
    message: str
    block_hash: str
    block_number: int
    extrinsic_index: int
    fee_rao: int
    fee_source: str = "sdk_response"

    def document(self) -> dict[str, Any]:
        value = {
            "response_type": self.response_type,
            "success": self.success,
            "message": self.message,
            "block_hash": self.block_hash,
            "block_number": self.block_number,
            "extrinsic_index": self.extrinsic_index,
            "reported_actual_fee_tao_rao": self.fee_rao,
            "fee_source": self.fee_source,
        }
        if self.fee_source == "sdk_response":
            value["sdk_reported_actual_fee_tao_rao"] = self.fee_rao
        return value


def _value(value: Any) -> Any:
    value = getattr(value, "value", value)
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return value


def _integer(value: Any, *, label: str) -> int:
    value = _value(value)
    if type(value) is not int or value < 0:
        raise ProvisionError(f"{label} must be a non-negative integer")
    return value


def _ss58(value: Any, *, label: str) -> str:
    value = _value(value)
    if not isinstance(value, str):
        raise ProvisionError(f"{label} must be an exact public SS58 address")
    try:
        ss58_decode(value)
    except ValueError as exc:
        raise ProvisionError(
            f"{label} is not a checksum-valid Bittensor SS58 address: {exc}"
        ) from None
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ProvisionError("provisioning evidence is not canonical JSON") from exc
