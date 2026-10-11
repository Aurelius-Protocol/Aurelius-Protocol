# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Typed runtime-452 testnet burn destination selected from exact public evidence.

Production keeps the specification's fixed UID 200.  The one-validator SN455 testnet successor may
use its freshly registered, unstaked burn UID only by consuming both the immutable successor marker
and the exact finalized public handoff it names.  No integer, hotkey, or coldkey environment value
is an authority.
"""

from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import stat
from typing import Any, Mapping
from subnet.operator import sn455_provision as provision
from subnet.operator import sn455_runtime452_burn_handoff as handoff_contract

GENERATION = "bittensor-11-sn455-runtime452-gh200-one-validator-v1"


BURN_GENERATION = "bittensor-11-sn455-runtime452-e2e-burn-v1"


GENERATION_ACKNOWLEDGEMENT = (
    "I_ACKNOWLEDGE_STARTING_BITTENSOR_11_SN455_RUNTIME452_GH200_ONE_VALIDATOR_V1"
)


PROFILE_PATH = "infra/testlab/env/gh200-sn455-one-validator-runtime452-profile.json"


PROFILE_SHA256 = "675db82a883234a553cfdcf6665041bf3e427113d10ab2aa086e67c8b2725cc4"


TOPOLOGY_PATH = "infra/testlab/env/gh200-sn455-one-validator-runtime452-topology.json"


TOPOLOGY_SHA256 = "6a275587daed74c5f9939fa2dcf91482dd222a09b6dbc6b4b09ce142ad88270e"


GENESIS_HASH = "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"


TOPOLOGY_IDENTITIES = frozenset(
    {
        "5G66vWmh3C6dUFBRzAMthLP8qeYzsg15bC4RaWkvSbwacSw3",
        "5CnmfdG2HmzXFrdpdfLq1jMJjhqxVWucxrcowZNE2RoZRzgs",
        "5D2auQJV8Ktvs8oVKBFbK8qEt7usiF1pJpXzran4mfo27H9c",
        "5DnvYrsPtPksiz68EjQsTBE5ZJxrF581eNbgi3wvcMcmZskn",
        "5DLvnYrcWAUtLmQNoB6uZKG6vNcmAYHkjESTUgZp1xLmGbRA",
        "5Df2NGWgGhKHYBWfivty3AaL42bnKXSeFGcjCCdULdXk9yAe",
        "5EhQjRDajCzmHW2aYkikVZeZThSsnJ1v9LZ59XJentAMTdsN",
        "5CkYMTPdifgqzfzYRLTfyhpN37zwGtX2mSycxPqNiqztbFuv",
        "5DqAMWRDnPNZRCo5pkn6mGDH9cm6tyYCR9CgHB8k3HM2x8CK",
        "5F1hDD7jveZrr95Kb2sJF27vzZm2348UGYo6HhpssRDZtiFD",
        "5DsnV3xbgnXySBRjsTo4z5HQaNzhjKsmftcTfNZueJ4mfnde",
        "5FKdtpKLGBGkwCEHXmmL4bmXiBqARzUZwR6GTsET79AXuAko",
        "5H6NbHnHs1wKTVJ2qwVD3H5DWZz162RJEmTPHzD59aqkQCUF",
        "5GUM1PMSibggJowqRoPj43CsdtdeSrr6cs278LaKKuMu7CAS",
        "5CnsmMifhnd5LVH2Lq9DbokVkwxXTwueTmmKjanNeUUN5Yxt",
        "5GnEb8t6V1SQ31h6VN3vvAqbzj2N664GqQLX1nH4Ddbqhd9D",
        "5H3WLvU7u2RKxNdpDCBuQc9nnJ33dv6eoS84NuQv7u3fUMHM",
        "5Eq93sfzs61oz41P4mF3pKXP6JXFZDUdsmUoL8GGTLZGnjqN",
    }
)


GENERATION_FILE_ENV = "AURELIUS_WEIGHT_BURN_GENERATION_FILE"


HANDOFF_FILE_ENV = "AURELIUS_WEIGHT_BURN_HANDOFF_FILE"


PROFILE_ENV = "AURELIUS_WEIGHT_AUTHORITY_PROFILE"


PROFILE_ID = "gh200-sn455-one-validator-runtime452"


PLUMBING_PROFILE_ID = "gh200-sn455-one-validator-runtime452-synthetic-plumbing"


PLUMBING_GENERATION = "bittensor-11-sn455-runtime452-gh200-one-validator-synthetic-plumbing-v1"


PLUMBING_GENERATION_ACKNOWLEDGEMENT = (
    "I_ACKNOWLEDGE_STARTING_BITTENSOR_11_SN455_RUNTIME452_GH200_SYNTHETIC_PLUMBING_V1"
)


PLUMBING_PROFILE_PATH = (
    "infra/testlab/env/gh200-sn455-one-validator-runtime452-plumbing-profile.json"
)


PLUMBING_PROFILE_SHA256 = "56a6ee9310e779d27a188dbee6a2c276452598c94166f91c25982a415b56bb87"


K8S_PROFILE_ID = "k8s-sn455-one-validator-runtime453"


K8S_GENERATION = "bittensor-11-sn455-k8s-one-validator-v1"


K8S_GENERATION_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_STARTING_BITTENSOR_11_SN455_K8S_ONE_VALIDATOR_V1"


K8S_PROFILE_PATH = "infra/testlab/env/gh200-sn455-one-validator-k8s-profile.json"


K8S_PROFILE_SHA256 = "23362ec2437678d0d2d1e7d45957ce52603ca9818c689f96835bcb1980ddd94a"


K8S_TOPOLOGY_PATH = "infra/testlab/env/gh200-sn455-one-validator-k8s-topology.json"


K8S_TOPOLOGY_SHA256 = "01306534b2a0ba3427ce4c1ca2efade034d6f1ed890d94584d73771faaa72d60"


K8S_RUNTIME_SPEC_VERSION = 453


K8S_BURN_GENERATION = "bittensor-11-sn455-k8s-e2e-burn-v1"


K8S_SOURCE_ACTOR = {
    "axon_publishers": [],
    "weight_setter": "validator-0",
}


K8S_AUTHORIZED_ACTIONS = [
    "SubtensorModule.commit_timelocked_mechanism_weights",
]


K8S_ACTION_BOUNDS = {
    "score_derived_weight_cycle": {
        "acknowledgement": "I_ACKNOWLEDGE_ONE_SN455_K8S_SCORE_DERIVED_WEIGHT_CYCLE",
        "logical_cycle_count": 1,
        "maximum_signed_extrinsics": 1,
        "actor": "validator-0",
        "commit_reveal_required": True,
        "score_derived_vector_required": True,
        "finalized_burn_handoff_required": True,
        "active_reveal_observation_required": True,
        "plain_set_weights_authorized": False,
        "commit_reveal_disable_authorized": False,
        "admin_mutation_authorized": False,
    },
}


K8S_CONSUMED_DEPOSITS = [
    {
        "deposit_key": "bittensor-testnet:8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
        "‖0xdd90e1ac3849d3f0fa1fdcb6edcfe28647b4790e535c1c76b6fbea855d229fff‖6‖84",
        "block_number": 7_928_258,
        "extrinsic_index": 6,
        "event_index": 84,
    },
    {
        "deposit_key": "bittensor-testnet:8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
        "‖0xfa3f1462a32c6cea30d933274ead155fed06a993831cc5da38a5c482121a3aee‖10‖106",
        "block_number": 7_928_572,
        "extrinsic_index": 10,
        "event_index": 106,
    },
    {
        "deposit_key": "bittensor-testnet:8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
        "‖0xfbf5dd3b96ed518a3e23f1b59120fedc107a3eb6fc1c877062e02b0c911ff6e0‖6‖78",
        "block_number": 7_928_777,
        "extrinsic_index": 6,
        "event_index": 78,
    },
]


MAX_DOCUMENT_BYTES = 1_048_576


HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


SOURCE_ACTOR = {
    "axon_publishers": [f"miner-{index}" for index in range(4)],
    "weight_setter": "validator-0",
}


AUTHORIZED_ACTIONS = [
    "SubtensorModule.serve_axon",
    "SubtensorModule.commit_timelocked_mechanism_weights",
]


ACTION_BOUNDS = {
    "serve_axon": {
        "acknowledgement": "I_ACKNOWLEDGE_UP_TO_FOUR_SN455_AXON_ENDPOINT_UPDATES",
        "logical_action_count": 4,
        "maximum_signed_extrinsics": 4,
        "endpoints": [
            {"actor": f"miner-{index}", "ip": "192.222.51.32", "port": 8090 + index}
            for index in range(4)
        ],
        "already_exact_endpoint_signs_nothing": True,
    },
    "score_derived_weight_cycle": {
        "acknowledgement": "I_ACKNOWLEDGE_ONE_SN455_RUNTIME452_SCORE_DERIVED_WEIGHT_CYCLE",
        "logical_cycle_count": 1,
        "maximum_signed_extrinsics": 1,
        "actor": "validator-0",
        "commit_reveal_required": True,
        "score_derived_vector_required": True,
        "finalized_burn_handoff_required": True,
        "active_reveal_observation_required": True,
        "plain_set_weights_authorized": False,
        "commit_reveal_disable_authorized": False,
        "admin_mutation_authorized": False,
    },
}


CONSUMED_DEPOSIT = {
    "status": "consumed_finalized_read_only",
    "generation": "bittensor-11-sn455-runtime450-alpha-v1",
    "registry_directory_name": ".aurelius-sn455-runtime450-alpha-v1",
    "receipt_file": "finalized.json",
    "receipt_sha256": "eafb7f1e96c3f0e4dd59be5fd0d4e00374a5887c552815c6d8cd053b27e5ad67",
    "extrinsic_hash": "0x37b921961233e8ed62af52fcee339a1061df986f04ca43eb02c937b4d3fc9079",
    "block_number": 7_878_924,
    "block_hash": "0xa5306adfcd08388766e880c38a4d690b81c9236286cb964f7e1188349b5efe64",
    "extrinsic_index": 6,
    "amount_alpha_rao": 2_000_000_000,
    "resend_authorized": False,
    "central_credit_proven_by_this_binding": False,
    "e2e_proven_by_this_binding": False,
}


class BurnAuthorityError(ValueError):
    """The supplied evidence cannot select a non-production burn destination."""


@dataclass(frozen=True, slots=True)
class Runtime452BurnAuthority:
    """Exact public evidence for the one permitted dynamic testnet burn UID."""

    uid: int
    hotkey_ss58: str
    coldkey_ss58: str
    netuid: int
    runtime_spec_version: int
    generation: str
    profile_sha256: str
    topology_sha256: str
    generation_marker_sha256: str
    handoff_sha256: str
    latest_finalized_block_number: int
    latest_finalized_block_hash: str
    profile_id: str = PROFILE_ID

    def __post_init__(self) -> None:
        if (
            type(self.uid) is not int
            or not 0 <= self.uid < 1 << 16
            or self.uid == 200
            or self.netuid != 455
            or self.profile_id not in PROFILE_SPECS
            or self.runtime_spec_version != PROFILE_SPECS[self.profile_id]["runtime_spec_version"]
            or self.generation != PROFILE_SPECS[self.profile_id]["generation"]
            or self.profile_sha256 != PROFILE_SPECS[self.profile_id]["profile_sha256"]
            or self.topology_sha256 != PROFILE_SPECS[self.profile_id]["topology_sha256"]
            or any(
                HEX_DIGEST.fullmatch(value) is None
                for value in (self.generation_marker_sha256, self.handoff_sha256)
            )
            or type(self.latest_finalized_block_number) is not int
            or self.latest_finalized_block_number < 1
            or provision.CHAIN_HASH.fullmatch(self.latest_finalized_block_hash) is None
        ):
            raise BurnAuthorityError("runtime452 burn authority fields differ")
        try:
            hotkey = provision._ss58(self.hotkey_ss58, label="runtime452 burn authority hotkey")
            coldkey = provision._ss58(self.coldkey_ss58, label="runtime452 burn authority coldkey")
        except provision.ProvisionError as exc:
            raise BurnAuthorityError("runtime452 burn authority identity is malformed") from exc
        if hotkey == coldkey:
            raise BurnAuthorityError("runtime452 burn authority identities alias")
        if hotkey in TOPOLOGY_IDENTITIES or coldkey in TOPOLOGY_IDENTITIES:
            raise BurnAuthorityError("runtime452 burn authority aliases a topology identity")


PROFILE_SPECS = {
    PROFILE_ID: {
        "generation": GENERATION,
        "acknowledgement": GENERATION_ACKNOWLEDGEMENT,
        "profile_schema_version": 3,
        "profile_path": PROFILE_PATH,
        "profile_sha256": PROFILE_SHA256,
        "runtime_spec_version": 452,
        "topology_path": TOPOLOGY_PATH,
        "topology_sha256": TOPOLOGY_SHA256,
        "source_actor": SOURCE_ACTOR,
        "authorized_actions": AUTHORIZED_ACTIONS,
        "action_bounds": ACTION_BOUNDS,
        "consumed_deposit": CONSUMED_DEPOSIT,
        "burn_generation": BURN_GENERATION,
        "registry_directory_name": ".aurelius-sn455-runtime452-e2e-burn-v1",
    },
    PLUMBING_PROFILE_ID: {
        "generation": PLUMBING_GENERATION,
        "acknowledgement": PLUMBING_GENERATION_ACKNOWLEDGEMENT,
        "profile_schema_version": 1,
        "profile_path": PLUMBING_PROFILE_PATH,
        "profile_sha256": PLUMBING_PROFILE_SHA256,
        "runtime_spec_version": 452,
        "topology_path": TOPOLOGY_PATH,
        "topology_sha256": TOPOLOGY_SHA256,
        "source_actor": SOURCE_ACTOR,
        "authorized_actions": AUTHORIZED_ACTIONS,
        "action_bounds": ACTION_BOUNDS,
        "consumed_deposit": CONSUMED_DEPOSIT,
        "burn_generation": BURN_GENERATION,
        "registry_directory_name": ".aurelius-sn455-runtime452-e2e-burn-v1",
    },
    K8S_PROFILE_ID: {
        "generation": K8S_GENERATION,
        "acknowledgement": K8S_GENERATION_ACKNOWLEDGEMENT,
        "profile_schema_version": 1,
        "profile_path": K8S_PROFILE_PATH,
        "profile_sha256": K8S_PROFILE_SHA256,
        "runtime_spec_version": K8S_RUNTIME_SPEC_VERSION,
        "topology_path": K8S_TOPOLOGY_PATH,
        "topology_sha256": K8S_TOPOLOGY_SHA256,
        "source_actor": K8S_SOURCE_ACTOR,
        "authorized_actions": K8S_AUTHORIZED_ACTIONS,
        "action_bounds": K8S_ACTION_BOUNDS,
        "consumed_deposit": K8S_CONSUMED_DEPOSITS,
        "burn_generation": K8S_BURN_GENERATION,
        "registry_directory_name": ".aurelius-sn455-k8s-e2e-burn-v1",
    },
}


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BurnAuthorityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise BurnAuthorityError(f"invalid JSON constant: {value}")


def _canonical(value: object) -> bytes:
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
        ).encode()
    except (TypeError, ValueError) as exc:
        raise BurnAuthorityError("burn authority evidence is not canonical JSON") from exc


def _held_document(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_absolute():
        raise BurnAuthorityError(f"{label} path must be absolute")
    flags = os.O_RDONLY
    for optional in ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC"):
        flags |= getattr(os, optional, 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid not in {0, os.getuid()}
            or before.st_nlink != 1
            or before.st_mode & 0o022
            or not 1 <= before.st_size <= MAX_DOCUMENT_BYTES
        ):
            raise BurnAuthorityError(f"{label} has unsafe metadata")
        payload = bytearray()
        while len(payload) <= MAX_DOCUMENT_BYTES:
            chunk = os.read(descriptor, min(65_536, MAX_DOCUMENT_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        stable = (
            "st_dev",
            "st_ino",
            "st_uid",
            "st_gid",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            len(payload) != before.st_size
            or any(getattr(before, field) != getattr(after, field) for field in stable)
            or any(getattr(before, field) != getattr(current, field) for field in stable)
        ):
            raise BurnAuthorityError(f"{label} changed during its held-inode read")
        raw = bytes(payload)
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except BurnAuthorityError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise BurnAuthorityError(f"{label} is unavailable or malformed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if type(value) is not dict or raw != _canonical(value):
        raise BurnAuthorityError(f"{label} is not one canonical JSON object")
    return value, raw


def _exact(value: object, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise BurnAuthorityError(f"{label} field set differs")
    return value


def load_runtime452_burn_authority(
    generation_path: Path,
    handoff_path: Path,
) -> Runtime452BurnAuthority:
    """Join exact immutable successor and handoff bytes into one typed destination."""

    marker, marker_raw = _held_document(generation_path, label="burn generation marker")
    handoff_value, handoff_raw = _held_document(handoff_path, label="burn public handoff")

    # The marker's generation selects the compiled profile; every other marker fact is checked
    # against THAT profile's compiled scope (ADR-0199: the host and k8s generations differ in
    # chain scope, actors, and deposit evidence, not in discipline).
    generation = marker.get("generation")
    matching_specs = [
        (profile_id, spec)
        for profile_id, spec in PROFILE_SPECS.items()
        if spec["generation"] == generation
    ]
    if len(matching_specs) != 1:
        raise BurnAuthorityError("burn generation marker selects no compiled profile")
    profile_id, profile_spec = matching_specs[0]
    try:
        handoff, handoff_sha256 = handoff_contract.validate_handoff(
            handoff_value,
            source_generation=profile_spec["burn_generation"],
            runtime_spec_version=profile_spec["runtime_spec_version"],
            topology_manifest_sha256=profile_spec["topology_sha256"],
        )
    except handoff_contract.Runtime452BurnHandoffError as exc:
        raise BurnAuthorityError("burn public handoff validation failed") from exc
    if (
        handoff_raw != _canonical(handoff)
        or hashlib.sha256(handoff_raw).hexdigest() != handoff_sha256
    ):
        raise BurnAuthorityError("burn public handoff bytes differ")

    _exact(
        marker,
        {
            "schema_version",
            "status",
            "generation",
            "acknowledgement",
            "ceremony",
            "profile",
            "topology",
            "source_actor",
            "authorized_actions",
            "action_bounds",
            "consumed_deposit",
            "consumed_burn_handoff",
            "mutation_authorized",
            "terminal_status_authorized",
            "action_claim_api_implemented",
            "signing_wire_implemented",
        },
        label="burn generation marker",
    )
    if (
        marker["schema_version"] != 1
        or marker["status"] != "authority_declared_after_finalized_burn_handoff"
        or marker["generation"] != profile_spec["generation"]
        or marker["acknowledgement"] != profile_spec["acknowledgement"]
        or marker["ceremony"] != "agent_executed_under_operator_delegation"
        or marker["profile"]
        != {
            "schema_version": profile_spec["profile_schema_version"],
            "path": profile_spec["profile_path"],
            "sha256": profile_spec["profile_sha256"],
        }
        or marker["topology"]
        != {
            "path": profile_spec["topology_path"],
            "sha256": profile_spec["topology_sha256"],
            "runtime_spec_version": profile_spec["runtime_spec_version"],
            "transaction_version": 1,
            "netuid": 455,
        }
        or marker["source_actor"] != profile_spec["source_actor"]
        or marker["authorized_actions"] != profile_spec["authorized_actions"]
        or marker["action_bounds"] != profile_spec["action_bounds"]
        or marker["consumed_deposit"] != profile_spec["consumed_deposit"]
        or marker["mutation_authorized"] is not True
        or marker["terminal_status_authorized"] is not False
        or marker["action_claim_api_implemented"] is not True
        or marker["signing_wire_implemented"] is not True
    ):
        raise BurnAuthorityError("burn generation marker facts differ")

    identity = handoff["burn_identity"]
    consumed = marker["consumed_burn_handoff"]
    expected_consumed = {
        "status": "consumed_finalized_public_read_only",
        "source_generation": profile_spec["burn_generation"],
        "registry_directory_name": profile_spec["registry_directory_name"],
        "handoff_file": handoff_contract.HANDOFF_NAME,
        "handoff_sha256": handoff_sha256,
        "AURELIUS_BURN_UID": identity["AURELIUS_BURN_UID"],
        "AURELIUS_BURN_HOTKEY_SS58": identity["AURELIUS_BURN_HOTKEY_SS58"],
        "AURELIUS_BURN_COLDKEY_SS58": identity["AURELIUS_BURN_COLDKEY_SS58"],
        "total_hotkey_alpha_rao": 0,
        "outside_all_topology_identities": True,
        "funding_terminal_receipt_sha256": handoff["funding"]["terminal_receipt_sha256"],
        "registration_terminal_receipt_sha256": handoff["registration"]["terminal_receipt_sha256"],
        "non_evaluator": handoff["non_evaluator"],
        "latest_finalized": handoff["latest_finalized"],
        "private_material_consumed": False,
        "terminal_e2e_proven_by_this_binding": False,
    }
    if consumed != expected_consumed:
        raise BurnAuthorityError("burn generation marker does not consume the exact handoff")
    latest = handoff["latest_finalized"]
    return Runtime452BurnAuthority(
        uid=identity["AURELIUS_BURN_UID"],
        hotkey_ss58=identity["AURELIUS_BURN_HOTKEY_SS58"],
        coldkey_ss58=identity["AURELIUS_BURN_COLDKEY_SS58"],
        netuid=455,
        runtime_spec_version=profile_spec["runtime_spec_version"],
        generation=str(profile_spec["generation"]),
        profile_sha256=str(profile_spec["profile_sha256"]),
        topology_sha256=profile_spec["topology_sha256"],
        generation_marker_sha256=hashlib.sha256(marker_raw).hexdigest(),
        handoff_sha256=handoff_sha256,
        latest_finalized_block_number=latest["block_number"],
        latest_finalized_block_hash=latest["block_hash"],
        profile_id=profile_id,
    )
