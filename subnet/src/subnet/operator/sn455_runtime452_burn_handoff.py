# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Wallet-free finalization of the dedicated runtime-452 SN455 testnet burn identity.

The provisioning controller owns signing.  This module reopens only its fixed, owner-only public
evidence and independently re-observes both finalized extrinsics plus the latest identity state.
It emits one immutable public handoff; it has no wallet or signing API.
"""

from __future__ import annotations
import hashlib
import re
from typing import Any, Mapping
from subnet.operator import sn455_provision as provision

SCHEMA_VERSION = 2


STATUS = "finalized_runtime452_e2e_burn_handoff"


HANDOFF_NAME = "sn455-runtime452-burn-handoff-v2.json"


PUBLIC_ENV_NAME = "sn455-burn.env"


PRIVATE_FIELD_NAMES = frozenset(
    {"mnemonic", "seed", "seed_hex", "private_key", "password", "token"}
)


HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


DEPOSIT_ADDRESS_COLDKEY_SS58 = next(
    coldkey
    for _name, role, _uid, coldkey, _hotkey_name, _hotkey in provision.EXPECTED_SOURCE_WALLET_ROWS
    if role == "deposit_address"
)


class Runtime452BurnHandoffError(provision.ProvisionError):
    """The bounded public provisioning evidence cannot authorize a main-generation handoff."""


def _public_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            nested for child in value.values() for nested in _public_keys(child)
        }
    if isinstance(value, (list, tuple)):
        return {nested for child in value for nested in _public_keys(child)}
    return set()


def _receipt(value: object) -> provision.ReceiptEvidence:
    if not isinstance(value, Mapping):
        raise Runtime452BurnHandoffError("resolved mutation receipt is not an object")
    try:
        receipt = provision.ReceiptEvidence(
            response_type=str(value["response_type"]),
            success=value["success"],
            message=str(value["message"]),
            block_hash=str(value["block_hash"]),
            block_number=provision._integer(value["block_number"], label="receipt block"),
            extrinsic_index=provision._integer(
                value["extrinsic_index"], label="receipt extrinsic index"
            ),
            fee_rao=provision._integer(
                value["reported_actual_fee_tao_rao"], label="receipt actual fee"
            ),
            fee_source=str(value["fee_source"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Runtime452BurnHandoffError("resolved mutation receipt is malformed") from exc
    if (
        receipt.success is not True
        or provision.CHAIN_HASH.fullmatch(receipt.block_hash) is None
        or receipt.response_type == ""
        or receipt.document() != dict(value)
    ):
        raise Runtime452BurnHandoffError("resolved mutation receipt is noncanonical")
    return receipt


def _exact_mapping(value: object, names: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != names:
        raise Runtime452BurnHandoffError(f"{label} field set differs")
    return value


def _digest(value: object, *, label: str) -> str:
    text = str(value)
    if type(value) is not str or HEX_DIGEST.fullmatch(text) is None:
        raise Runtime452BurnHandoffError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _nonnegative(value: object, *, label: str) -> int:
    number = provision._integer(value, label=label)
    if number < 0:
        raise Runtime452BurnHandoffError(f"{label} is negative")
    return number


def _load_non_evaluator(
    value: object,
    *,
    uid: int,
    hotkey: str,
    coldkey: str,
    block_number: int,
) -> Mapping[str, Any]:
    proof = _exact_mapping(
        value,
        {
            "block_number",
            "uid",
            "hotkey_ss58",
            "coldkey_ss58",
            "active",
            "validator_permit",
            "axon",
            "axon_storage_present",
            "weights",
            "last_update",
            "block_at_registration",
            "rank_raw_present",
            "trust_raw_present",
            "pruning_score_raw_present",
            "rank",
            "trust",
            "consensus",
            "incentive",
            "dividends",
            "pruning_score",
            "emission_rao",
            "alpha_stake_rao",
            "tao_stake_rao",
            "total_stake_rao",
            "parent_keys",
            "child_keys",
            "runtime_api_storage_crosschecked",
        },
        label="runtime452 burn non-evaluator proof",
    )
    last_update = _nonnegative(proof["last_update"], label="burn last update")
    registered = _nonnegative(proof["block_at_registration"], label="burn registration block")
    raw_presence = (
        proof["rank_raw_present"],
        proof["trust_raw_present"],
        proof["pruning_score_raw_present"],
    )
    scores = tuple(
        proof[name]
        for name in (
            "rank",
            "trust",
            "consensus",
            "incentive",
            "dividends",
            "pruning_score",
        )
    )
    if (
        proof["block_number"] != block_number
        or proof["uid"] != uid
        or proof["hotkey_ss58"] != hotkey
        or proof["coldkey_ss58"] != coldkey
        or proof["active"] is not False
        or proof["validator_permit"] is not False
        or proof["axon"] is not None
        or proof["axon_storage_present"] is not False
        or proof["weights"] != []
        or not all(type(item) is bool for item in raw_presence)
        or any(type(score) is not float or score != 0.0 for score in scores)
        or any(
            proof[name] != 0
            for name in (
                "emission_rao",
                "alpha_stake_rao",
                "tao_stake_rao",
                "total_stake_rao",
            )
        )
        or proof["parent_keys"] != []
        or proof["child_keys"] != []
        or proof["runtime_api_storage_crosschecked"] is not True
        or registered > block_number
        or last_update > block_number
    ):
        raise Runtime452BurnHandoffError("runtime452 burn identity is not a non-evaluator")
    return proof


def _load_fee(
    value: object,
    *,
    receipt: provision.ReceiptEvidence,
    signer: str,
    fee_cap: int,
) -> Mapping[str, Any]:
    source_fee_name = (
        "sdk_reported_actual_fee_tao_rao"
        if receipt.fee_source == "sdk_response"
        else "recovered_event_actual_fee_tao_rao"
    )
    fee = _exact_mapping(
        value,
        {
            "planned_fee_tao_rao",
            "finalized_actual_fee_tao_rao",
            "finalized_tip_tao_rao",
            "fee_cap_tao_rao",
            "fee_payer_ss58",
            source_fee_name,
        },
        label="burn handoff finalized fee",
    )
    planned = _nonnegative(fee["planned_fee_tao_rao"], label="planned fee")
    finalized = _nonnegative(fee["finalized_actual_fee_tao_rao"], label="finalized fee")
    if (
        planned > fee_cap
        or finalized != receipt.fee_rao
        or fee[source_fee_name] != receipt.fee_rao
        or fee["finalized_tip_tao_rao"] != 0
        or fee["fee_cap_tao_rao"] != fee_cap
        or fee["fee_payer_ss58"] != signer
    ):
        raise Runtime452BurnHandoffError("burn handoff finalized fee facts differ")
    return fee


def _load_mutation(
    value: object,
    *,
    step: str,
    expected_call: dict[str, Any],
    maximum_spend: int,
    uid: int,
    hotkey: str,
    coldkey: str,
) -> tuple[Mapping[str, Any], provision.ReceiptEvidence]:
    mutation = _exact_mapping(
        value,
        {
            "step",
            "call",
            "maximum_spend_tao_rao",
            "fee_cap_tao_rao",
            "receipt",
            "terminal_receipt_sha256",
            "finalized_observation",
            "prepared_sha256",
            "payload_sha256",
            "signature_sha256",
        },
        label=f"burn handoff {step}",
    )
    fee_cap = provision.MAX_EXTRINSIC_FEE_RAO
    if (
        mutation["step"] != step
        or mutation["call"] != expected_call
        or mutation["maximum_spend_tao_rao"] != maximum_spend
        or mutation["fee_cap_tao_rao"] != fee_cap
    ):
        raise Runtime452BurnHandoffError(f"burn handoff {step} scope differs")
    for name in (
        "terminal_receipt_sha256",
        "prepared_sha256",
        "payload_sha256",
        "signature_sha256",
    ):
        _digest(mutation[name], label=f"burn handoff {step} {name}")

    receipt = _receipt(mutation["receipt"])
    observation_names = {
        "block_hash",
        "block_number",
        "extrinsic_index",
        "extrinsic_hash",
        "success_event_index",
        "call",
        "signer_coldkey_ss58",
        "fee",
    }
    if step == "fund-burn":
        observation_names |= {
            "source_free_after_tao_rao",
            "destination_free_after_tao_rao",
            "transferred_tao_rao",
            "source_reserve_tao_rao",
            "exact_source_fee_debit_proven",
        }
    else:
        observation_names |= {
            "uid",
            "hotkey_ss58",
            "coldkey_ss58",
            "total_hotkey_alpha_rao",
            "coldkey_free_after_tao_rao",
            "registration_non_fee_debit_tao_rao",
            "source_reserve_tao_rao",
            "exact_fee_component_proven",
        }
    observation = _exact_mapping(
        mutation["finalized_observation"],
        observation_names,
        label=f"burn handoff {step} finalized observation",
    )
    finalized_call = {
        "module": expected_call["module"],
        "function": expected_call["function"],
        "args": expected_call["args"],
    }
    if (
        observation["block_hash"] != receipt.block_hash
        or observation["block_number"] != receipt.block_number
        or observation["extrinsic_index"] != receipt.extrinsic_index
        or provision.CHAIN_HASH.fullmatch(str(observation["extrinsic_hash"])) is None
        or _nonnegative(observation["success_event_index"], label="success event index") < 0
        or observation["call"] != finalized_call
        or observation["signer_coldkey_ss58"] != expected_call["signer"]
    ):
        raise Runtime452BurnHandoffError(f"burn handoff {step} finalized call differs")
    _load_fee(
        observation["fee"],
        receipt=receipt,
        signer=expected_call["signer"],
        fee_cap=fee_cap,
    )
    if step == "fund-burn":
        if (
            observation["transferred_tao_rao"] != provision.BURN_FUND_RAO
            or observation["source_reserve_tao_rao"] != provision.SOURCE_FREE_RESERVE_RAO
            or observation["exact_source_fee_debit_proven"] is not True
        ):
            raise Runtime452BurnHandoffError("burn handoff funding effect differs")
        _nonnegative(observation["source_free_after_tao_rao"], label="funding source balance")
        destination_after = _nonnegative(
            observation["destination_free_after_tao_rao"],
            label="funding destination balance",
        )
        if destination_after < provision.BURN_FUND_RAO:
            raise Runtime452BurnHandoffError("burn handoff funding destination balance differs")
    elif (
        observation["uid"] != uid
        or observation["hotkey_ss58"] != hotkey
        or observation["coldkey_ss58"] != coldkey
        or observation["total_hotkey_alpha_rao"] != 0
        or observation["source_reserve_tao_rao"] != provision.BURN_FREE_RESERVE_RAO
        or observation["exact_fee_component_proven"] is not True
        or _nonnegative(
            observation["registration_non_fee_debit_tao_rao"],
            label="registration non-fee debit",
        )
        > provision.BURN_REGISTRATION_CAP_RAO
    ):
        raise Runtime452BurnHandoffError("burn handoff registration effect differs")
    if step == "register-burn":
        _nonnegative(
            observation["coldkey_free_after_tao_rao"],
            label="registration coldkey balance",
        )
    return mutation, receipt


def validate_handoff(
    value: object,
    *,
    source_generation: str = provision.RUNTIME452_BURN_GENERATION,
    runtime_spec_version: int = 452,
    topology_manifest_sha256: str = provision.RUNTIME452_BURN_TOPOLOGY_SHA256,
) -> tuple[dict[str, Any], str]:
    """Validate one already-decoded public handoff without network or wallet access.

    The generation-specific values are compiled per profile (ADR-0199): the host generation
    consumes the runtime452 burn handoff at spec 452, the k8s generation its own at 453. The
    defaults are the host values, so every existing caller is byte-identical.
    """

    _exact_mapping(
        value,
        {
            "schema_version",
            "status",
            "source_generation",
            "network",
            "burn_identity",
            "funding",
            "registration",
            "non_evaluator",
            "latest_finalized",
            "public_env",
            "private_material_output",
            "terminal_e2e_authorized",
        },
        label="runtime452 public burn handoff",
    )
    assert isinstance(value, dict)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != STATUS
        or value.get("source_generation") != source_generation
        or value.get("private_material_output") is not False
        or value.get("terminal_e2e_authorized") is not False
        or _public_keys(value) & PRIVATE_FIELD_NAMES
    ):
        raise Runtime452BurnHandoffError("runtime452 public burn handoff posture differs")
    identity = value.get("burn_identity")
    network = value.get("network")
    latest = value.get("latest_finalized")
    if not all(isinstance(item, Mapping) for item in (identity, network, latest)):
        raise Runtime452BurnHandoffError("runtime452 public burn handoff shape differs")
    assert isinstance(identity, Mapping)
    assert isinstance(network, Mapping)
    assert isinstance(latest, Mapping)
    _exact_mapping(
        identity,
        {
            "AURELIUS_BURN_UID",
            "AURELIUS_BURN_HOTKEY_SS58",
            "AURELIUS_BURN_COLDKEY_SS58",
            "outside_all_topology_identities",
            "total_hotkey_alpha_rao",
        },
        label="runtime452 burn identity",
    )
    _exact_mapping(
        network,
        {
            "endpoint",
            "chain",
            "genesis_hash",
            "runtime_spec_version",
            "runtime_transaction_version",
            "netuid",
            "topology_manifest_sha256",
        },
        label="runtime452 burn network",
    )
    _exact_mapping(
        latest,
        {
            "block_hash",
            "block_number",
            "commit_reveal_weights_enabled",
            "uid_mapping_reproved",
            "owner_mapping_reproved",
            "zero_alpha_stake_reproved",
            "non_evaluator_reproved",
        },
        label="runtime452 latest finalized state",
    )
    uid = provision._integer(identity.get("AURELIUS_BURN_UID"), label="burn handoff UID")
    hotkey = provision._ss58(identity.get("AURELIUS_BURN_HOTKEY_SS58"), label="burn hotkey")
    coldkey = provision._ss58(identity.get("AURELIUS_BURN_COLDKEY_SS58"), label="burn coldkey")
    if (
        uid == 200
        or hotkey == coldkey
        or identity.get("outside_all_topology_identities") is not True
        or identity.get("total_hotkey_alpha_rao") != 0
        or network
        != {
            "endpoint": provision.EXPECTED_ENDPOINT,
            "chain": provision.EXPECTED_CHAIN,
            "genesis_hash": provision.EXPECTED_GENESIS_HASH,
            "runtime_spec_version": runtime_spec_version,
            "runtime_transaction_version": 1,
            "netuid": 455,
            "topology_manifest_sha256": topology_manifest_sha256,
        }
        or latest.get("commit_reveal_weights_enabled") is not True
        or latest.get("uid_mapping_reproved") is not True
        or latest.get("owner_mapping_reproved") is not True
        or latest.get("zero_alpha_stake_reproved") is not True
        or latest.get("non_evaluator_reproved") is not True
    ):
        raise Runtime452BurnHandoffError("runtime452 public burn handoff facts differ")

    _load_non_evaluator(
        value["non_evaluator"],
        uid=uid,
        hotkey=hotkey,
        coldkey=coldkey,
        block_number=_nonnegative(latest["block_number"], label="latest finalized block"),
    )

    funding_call = {
        "module": "Balances",
        "function": "transfer_keep_alive",
        "args": {"dest": coldkey, "value": provision.BURN_FUND_RAO},
        "signer": DEPOSIT_ADDRESS_COLDKEY_SS58,
    }
    registration_call = {
        "module": "SubtensorModule",
        "function": "register_limit",
        "args": {
            "netuid": 455,
            "hotkey": hotkey,
            "limit_price": provision.BURN_REGISTRATION_CAP_RAO,
        },
        "signer": coldkey,
    }
    _funding, funding_receipt = _load_mutation(
        value["funding"],
        step="fund-burn",
        expected_call=funding_call,
        maximum_spend=provision.BURN_FUND_RAO,
        uid=uid,
        hotkey=hotkey,
        coldkey=coldkey,
    )
    _registration, registration_receipt = _load_mutation(
        value["registration"],
        step="register-burn",
        expected_call=registration_call,
        maximum_spend=provision.BURN_REGISTRATION_CAP_RAO,
        uid=uid,
        hotkey=hotkey,
        coldkey=coldkey,
    )
    if (
        (funding_receipt.block_number, funding_receipt.extrinsic_index)
        >= (registration_receipt.block_number, registration_receipt.extrinsic_index)
        or provision.CHAIN_HASH.fullmatch(str(latest["block_hash"])) is None
        or _nonnegative(latest["block_number"], label="latest finalized block")
        < registration_receipt.block_number
    ):
        raise Runtime452BurnHandoffError("runtime452 burn finality order differs")

    public_env = _exact_mapping(
        value["public_env"],
        {"path", "sha256", "private_material"},
        label="runtime452 public burn env",
    )
    expected_env = (
        f"AURELIUS_BURN_UID={uid}\n"
        f"AURELIUS_BURN_HOTKEY_SS58={hotkey}\n"
        f"AURELIUS_BURN_COLDKEY_SS58={coldkey}\n"
    ).encode()
    if (
        public_env["path"] != PUBLIC_ENV_NAME
        or public_env["sha256"] != hashlib.sha256(expected_env).hexdigest()
        or public_env["private_material"] is not False
    ):
        raise Runtime452BurnHandoffError("runtime452 public burn env facts differ")
    payload = provision._json_bytes(value)
    return value, hashlib.sha256(payload).hexdigest()
