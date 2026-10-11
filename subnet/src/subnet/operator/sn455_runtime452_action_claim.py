# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Consume one immutable action claim for the GH200 runtime452 SN455 generation.

The owner-side generation registry claims five logical actions before any runtime process opens a
wallet: four exact miner endpoint publications and one validator weight cycle.  A claim is useful
only when joined to the same exact generation marker and finalized burn handoff.  This module has
no wallet, signing, RPC, submission, Docker, or SSH capability; it turns those three public files
into a typed, actor-specific authority that mutation owners can require before their own durable
prepared-extrinsic journal is entered.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping
from subnet.operator import burn_authority as generation_authority

PROFILE_ID = "gh200-sn455-one-validator-runtime452"


PLUMBING_PROFILE_ID = "gh200-sn455-one-validator-runtime452-synthetic-plumbing"


K8S_PROFILE_ID = generation_authority.K8S_PROFILE_ID


CHAIN_ENDPOINT = "wss://test.finney.opentensor.ai:443"


CHAIN_NAME = "Bittensor"


GENESIS_HASH = generation_authority.GENESIS_HASH


NETUID = 455


TRANSACTION_VERSION = 1


PROFILE_RUNTIME_SPECS = {
    PROFILE_ID: 452,
    PLUMBING_PROFILE_ID: 452,
    K8S_PROFILE_ID: generation_authority.K8S_RUNTIME_SPEC_VERSION,
}


CLAIM_FILE_ENV = "AURELIUS_SN455_RUNTIME452_ACTION_CLAIM_FILE"


GENERATION_FILE_ENV = generation_authority.GENERATION_FILE_ENV


HANDOFF_FILE_ENV = generation_authority.HANDOFF_FILE_ENV


SERVE_AXON_ACTION = "SubtensorModule.serve_axon"


WEIGHT_ACTION = "SubtensorModule.commit_timelocked_mechanism_weights"


PRIVATE_FIELD_NAMES = frozenset(
    {"mnemonic", "seed", "seed_hex", "private_key", "password", "token"}
)


ACTION_SIGNERS = {
    "miner-0": {
        "coldkey_ss58": "5G66vWmh3C6dUFBRzAMthLP8qeYzsg15bC4RaWkvSbwacSw3",
        "hotkey_ss58": "5CnmfdG2HmzXFrdpdfLq1jMJjhqxVWucxrcowZNE2RoZRzgs",
    },
    "miner-1": {
        "coldkey_ss58": "5D2auQJV8Ktvs8oVKBFbK8qEt7usiF1pJpXzran4mfo27H9c",
        "hotkey_ss58": "5DnvYrsPtPksiz68EjQsTBE5ZJxrF581eNbgi3wvcMcmZskn",
    },
    "miner-2": {
        "coldkey_ss58": "5DLvnYrcWAUtLmQNoB6uZKG6vNcmAYHkjESTUgZp1xLmGbRA",
        "hotkey_ss58": "5Df2NGWgGhKHYBWfivty3AaL42bnKXSeFGcjCCdULdXk9yAe",
    },
    "miner-3": {
        "coldkey_ss58": "5EhQjRDajCzmHW2aYkikVZeZThSsnJ1v9LZ59XJentAMTdsN",
        "hotkey_ss58": "5CkYMTPdifgqzfzYRLTfyhpN37zwGtX2mSycxPqNiqztbFuv",
    },
    "validator-0": {
        "coldkey_ss58": "5DqAMWRDnPNZRCo5pkn6mGDH9cm6tyYCR9CgHB8k3HM2x8CK",
        "hotkey_ss58": "5F1hDD7jveZrr95Kb2sJF27vzZm2348UGYo6HhpssRDZtiFD",
    },
}


AXON_ENDPOINTS = {
    f"miner-{index}": {"ip": "192.222.51.32", "port": 8090 + index} for index in range(4)
}


ACTION_JOURNAL_ROOT = "/var/lib/aurelius-gh200-sn455-one-validator-runtime452/action-journals"


JOURNAL_RECORDER_SOCKET_ROOT = f"{ACTION_JOURNAL_ROOT}/recorders"


AXON_JOURNAL_PATHS = {
    actor: f"{ACTION_JOURNAL_ROOT}/{actor}-serve-axon" for actor in AXON_ENDPOINTS
}


WEIGHT_JOURNAL_PATH = f"{ACTION_JOURNAL_ROOT}/validator-0-weight-cycle-1"


JOURNAL_INSTALLATION_MARKER = ".runtime452-action-journal-installation.json"


JOURNAL_RECORDER_PROTOCOL = "aurelius-runtime452-action-recorder-v1"


ACTION_RUNTIME_IDS = {
    "miner-0": {"uid": 45510, "gid": 45510},
    "miner-1": {"uid": 45511, "gid": 45511},
    "miner-2": {"uid": 45512, "gid": 45512},
    "miner-3": {"uid": 45513, "gid": 45513},
    "validator-0": {"uid": 45520, "gid": 45520},
}


JOURNAL_RECORDER_SOCKET_PATHS = {
    actor: f"{JOURNAL_RECORDER_SOCKET_ROOT}/{actor}.sock" for actor in ACTION_RUNTIME_IDS
}


PLUMBING_ACTION_JOURNAL_ROOT = (
    "/var/lib/aurelius-gh200-sn455-runtime452-synthetic-plumbing-v1/action-journals"
)


K8S_ACTION_JOURNAL_ROOT = "/var/lib/aurelius/validator/action-journals"


K8S_ACTION_RUNTIME_IDS = {"validator-0": {"uid": 10002, "gid": 10002}}


ACTION_PROFILE_SPECS = {
    PROFILE_ID: {
        "generation": generation_authority.GENERATION,
        "journal_root": ACTION_JOURNAL_ROOT,
    },
    PLUMBING_PROFILE_ID: {
        "generation": generation_authority.PLUMBING_GENERATION,
        "journal_root": PLUMBING_ACTION_JOURNAL_ROOT,
    },
    K8S_PROFILE_ID: {
        "generation": generation_authority.K8S_GENERATION,
        "journal_root": K8S_ACTION_JOURNAL_ROOT,
        # The k8s generation authorizes ONLY the weight cycle: the miners serve their own
        # axons, so there is no ServeAxon budget and no miner actor in this profile.
        "actions": (WEIGHT_ACTION,),
        "runtime_ids": K8S_ACTION_RUNTIME_IDS,
    },
}


SELECTED_PROFILE_ID = os.environ.get("AURELIUS_ENV", PROFILE_ID)


if SELECTED_PROFILE_ID not in ACTION_PROFILE_SPECS:
    SELECTED_PROFILE_ID = PROFILE_ID


if SELECTED_PROFILE_ID == PLUMBING_PROFILE_ID:
    ACTION_JOURNAL_ROOT = PLUMBING_ACTION_JOURNAL_ROOT
    JOURNAL_RECORDER_SOCKET_ROOT = f"{ACTION_JOURNAL_ROOT}/recorders"
    AXON_JOURNAL_PATHS = {
        actor: f"{ACTION_JOURNAL_ROOT}/{actor}-serve-axon" for actor in AXON_ENDPOINTS
    }
    WEIGHT_JOURNAL_PATH = f"{ACTION_JOURNAL_ROOT}/validator-0-weight-cycle-1"
    JOURNAL_RECORDER_SOCKET_PATHS = {
        actor: f"{JOURNAL_RECORDER_SOCKET_ROOT}/{actor}.sock" for actor in ACTION_RUNTIME_IDS
    }


def _runtime_ids(profile_id: str) -> dict[str, dict[str, int]]:
    ids = ACTION_PROFILE_SPECS.get(profile_id, {}).get("runtime_ids")
    return ids if isinstance(ids, dict) else ACTION_RUNTIME_IDS


def _journal_paths(profile_id: str) -> tuple[dict[str, str], str, dict[str, str]]:
    if profile_id == PROFILE_ID:
        return AXON_JOURNAL_PATHS, WEIGHT_JOURNAL_PATH, JOURNAL_RECORDER_SOCKET_PATHS
    try:
        root = str(ACTION_PROFILE_SPECS[profile_id]["journal_root"])
    except KeyError as exc:
        raise Runtime452ActionClaimError("runtime452 action profile is not compiled") from exc
    runtime_ids = _runtime_ids(profile_id)
    sockets = f"{root}/recorders"
    return (
        {actor: f"{root}/{actor}-serve-axon" for actor in AXON_ENDPOINTS},
        f"{root}/validator-0-weight-cycle-1",
        {actor: f"{sockets}/{actor}.sock" for actor in runtime_ids},
    )


class Runtime452ActionClaimError(ValueError):
    """The public evidence cannot authorize the requested runtime452 action."""


@dataclass(frozen=True, slots=True)
class Runtime452ActionClaim:
    action: str
    actor: str
    signer_hotkey_ss58: str
    signer_coldkey_ss58: str
    endpoint_ip: str | None
    endpoint_port: int | None
    claim_index: int
    claim_sha256: str
    generation_marker_sha256: str
    burn_handoff_sha256: str
    burn_uid: int
    chain_endpoint: str
    chain_name: str
    genesis_hash: str
    netuid: int
    runtime_spec_version: int
    transaction_version: int
    journal_path: str
    journal_installation_marker: str
    journal_recorder_socket: str
    journal_runtime_uid: int
    journal_runtime_gid: int
    profile_id: str = PROFILE_ID

    def __post_init__(self) -> None:
        signer = ACTION_SIGNERS.get(self.actor)
        expected_endpoint = AXON_ENDPOINTS.get(self.actor)
        axon_journal_paths, weight_journal_path, recorder_paths = _journal_paths(self.profile_id)
        if (
            signer is None
            or self.action not in {SERVE_AXON_ACTION, WEIGHT_ACTION}
            or self.signer_hotkey_ss58 != signer["hotkey_ss58"]
            or self.signer_coldkey_ss58 != signer["coldkey_ss58"]
            or type(self.claim_index) is not int
            or not 1 <= self.claim_index <= 5
            or generation_authority.HEX_DIGEST.fullmatch(self.claim_sha256) is None
            or generation_authority.HEX_DIGEST.fullmatch(self.generation_marker_sha256) is None
            or generation_authority.HEX_DIGEST.fullmatch(self.burn_handoff_sha256) is None
            or type(self.burn_uid) is not int
            or not 0 <= self.burn_uid < 1 << 16
            or self.burn_uid == 200
            or self.chain_endpoint != CHAIN_ENDPOINT
            or self.chain_name != CHAIN_NAME
            or self.genesis_hash != GENESIS_HASH
            or self.netuid != NETUID
            or self.runtime_spec_version != PROFILE_RUNTIME_SPECS[self.profile_id]
            or self.transaction_version != TRANSACTION_VERSION
            or self.journal_path
            != (
                axon_journal_paths.get(self.actor)
                if self.action == SERVE_AXON_ACTION
                else weight_journal_path
            )
            or self.journal_installation_marker != JOURNAL_INSTALLATION_MARKER
            or self.journal_recorder_socket != recorder_paths.get(self.actor)
            or {
                "uid": self.journal_runtime_uid,
                "gid": self.journal_runtime_gid,
            }
            != _runtime_ids(self.profile_id).get(self.actor)
        ):
            raise Runtime452ActionClaimError("runtime452 action claim fields differ")
        if self.action == SERVE_AXON_ACTION:
            if (
                expected_endpoint is None
                or self.claim_index != list(AXON_ENDPOINTS).index(self.actor) + 1
                or self.endpoint_ip != expected_endpoint["ip"]
                or self.endpoint_port != expected_endpoint["port"]
            ):
                raise Runtime452ActionClaimError("runtime452 Axon claim fields differ")
        elif (
            self.actor != "validator-0"
            or self.claim_index != (1 if self.profile_id == K8S_PROFILE_ID else 5)
            or self.endpoint_ip is not None
            or self.endpoint_port is not None
        ):
            raise Runtime452ActionClaimError("runtime452 weight claim fields differ")


def _public_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key).casefold() for key in value} | {
            nested for child in value.values() for nested in _public_keys(child)
        }
    if isinstance(value, (list, tuple)):
        return {nested for child in value for nested in _public_keys(child)}
    return set()


def _expected_scope(action: str, actor: str, profile_id: str = PROFILE_ID) -> dict[str, Any]:
    axon_journal_paths, weight_journal_path, recorder_paths = _journal_paths(profile_id)
    if profile_id == K8S_PROFILE_ID:
        # The k8s generation has ONE logical action: the validator's weight cycle. The miners
        # serve their own axons outside this authority (ADR-0199).
        if action != WEIGHT_ACTION or actor != "validator-0":
            raise Runtime452ActionClaimError("action or actor is outside the runtime452 generation")
        return {
            "status": "declared_not_claimed_not_ready_to_sign",
            "action": action,
            "actor": actor,
            "logical_cycle": 1,
            "commit_reveal_required": True,
            "score_derived_vector_required": True,
            "finalized_burn_handoff_required": True,
            "active_reveal_observation_required": True,
            "journal_path": weight_journal_path,
            "journal_installation_marker": JOURNAL_INSTALLATION_MARKER,
            "journal_recorder_socket": recorder_paths[actor],
            "journal_recorder_protocol": JOURNAL_RECORDER_PROTOCOL,
            "journal_runtime_identity": K8S_ACTION_RUNTIME_IDS[actor],
        }
    if action == SERVE_AXON_ACTION and actor in AXON_ENDPOINTS:
        return {
            "status": "declared_not_claimed_not_ready_to_sign",
            "action": action,
            "actor": actor,
            "endpoint": AXON_ENDPOINTS[actor],
            "journal_path": axon_journal_paths[actor],
            "journal_installation_marker": JOURNAL_INSTALLATION_MARKER,
            "journal_recorder_socket": recorder_paths[actor],
            "journal_recorder_protocol": JOURNAL_RECORDER_PROTOCOL,
            "journal_runtime_identity": ACTION_RUNTIME_IDS[actor],
            "logical_action_index": list(AXON_ENDPOINTS).index(actor) + 1,
            "logical_action_count": 4,
        }
    if action == WEIGHT_ACTION and actor == "validator-0":
        return {
            "status": "declared_not_claimed_not_ready_to_sign",
            "action": action,
            "actor": actor,
            "logical_cycle": 1,
            "commit_reveal_required": True,
            "score_derived_vector_required": True,
            "finalized_burn_handoff_required": True,
            "active_reveal_observation_required": True,
            "journal_path": weight_journal_path,
            "journal_installation_marker": JOURNAL_INSTALLATION_MARKER,
            "journal_recorder_socket": recorder_paths[actor],
            "journal_recorder_protocol": JOURNAL_RECORDER_PROTOCOL,
            "journal_runtime_identity": ACTION_RUNTIME_IDS[actor],
        }
    raise Runtime452ActionClaimError("action or actor is outside the runtime452 generation")


def _claim_name(action: str, actor: str, profile_id: str = PROFILE_ID) -> str:
    if action == WEIGHT_ACTION and actor == "validator-0":
        return "action-claim-weight-cycle-1.json"
    if profile_id == K8S_PROFILE_ID:
        raise Runtime452ActionClaimError("action or actor is outside the runtime452 generation")
    if action == SERVE_AXON_ACTION and actor in AXON_ENDPOINTS:
        return f"action-claim-serve-axon-{actor}.json"
    raise Runtime452ActionClaimError("action or actor is outside the runtime452 generation")


def load_runtime452_action_claim(
    claim_path: Path,
    generation_path: Path,
    handoff_path: Path,
    *,
    expected_action: str,
    expected_actor: str,
) -> Runtime452ActionClaim:
    """Join an exact action claim to the exact generation/burn evidence pair."""

    try:
        authority = generation_authority.load_runtime452_burn_authority(
            generation_path,
            handoff_path,
        )
        value, raw = generation_authority._held_document(
            claim_path,
            label="runtime452 action claim",
        )
    except generation_authority.BurnAuthorityError as exc:
        raise Runtime452ActionClaimError("runtime452 action evidence is invalid") from exc
    # The claim filename is profile-scoped (the k8s generation names no ServeAxon claims), so
    # it is checked against the profile the marker+handoff actually selected.
    if claim_path.name != _claim_name(expected_action, expected_actor, authority.profile_id):
        raise Runtime452ActionClaimError("runtime452 action claim filename differs")
    expected_signer = ACTION_SIGNERS[expected_actor]
    expected_scope = _expected_scope(expected_action, expected_actor, authority.profile_id)
    k8s_generation = authority.profile_id == K8S_PROFILE_ID
    expected = {
        "schema_version": 1,
        "status": "claimed_before_signing",
        "generation": authority.generation,
        "generation_marker_sha256": authority.generation_marker_sha256,
        "burn_handoff_sha256": authority.handoff_sha256,
        "action": expected_action,
        "actor": expected_actor,
        "signer": expected_signer,
        "scope": expected_scope,
        "claim_index": (
            expected_scope["logical_action_index"]
            if expected_action == SERVE_AXON_ACTION
            else 1
            if k8s_generation
            else 5
        ),
        "claim_count": 1 if k8s_generation else 5,
        "maximum_signed_extrinsics": 1,
        "already_exact_signs_nothing": expected_action == SERVE_AXON_ACTION,
        "terminal_status_authorized": False,
        "private_material_consumed": False,
    }
    if value != expected or _public_keys(value) & PRIVATE_FIELD_NAMES:
        raise Runtime452ActionClaimError("runtime452 action claim differs from compiled scope")
    endpoint = expected_scope.get("endpoint")
    return Runtime452ActionClaim(
        action=expected_action,
        actor=expected_actor,
        signer_hotkey_ss58=expected_signer["hotkey_ss58"],
        signer_coldkey_ss58=expected_signer["coldkey_ss58"],
        endpoint_ip=endpoint["ip"] if isinstance(endpoint, dict) else None,
        endpoint_port=endpoint["port"] if isinstance(endpoint, dict) else None,
        claim_index=expected["claim_index"],
        claim_sha256=hashlib.sha256(raw).hexdigest(),
        generation_marker_sha256=authority.generation_marker_sha256,
        burn_handoff_sha256=authority.handoff_sha256,
        burn_uid=authority.uid,
        chain_endpoint=CHAIN_ENDPOINT,
        chain_name=CHAIN_NAME,
        genesis_hash=GENESIS_HASH,
        netuid=authority.netuid,
        runtime_spec_version=authority.runtime_spec_version,
        transaction_version=TRANSACTION_VERSION,
        journal_path=expected_scope["journal_path"],
        journal_installation_marker=expected_scope["journal_installation_marker"],
        journal_recorder_socket=expected_scope["journal_recorder_socket"],
        journal_runtime_uid=expected_scope["journal_runtime_identity"]["uid"],
        journal_runtime_gid=expected_scope["journal_runtime_identity"]["gid"],
        profile_id=authority.profile_id,
    )


def runtime452_action_claim_from_env(
    environ: Mapping[str, str],
    *,
    expected_action: str,
    expected_actor: str,
) -> Runtime452ActionClaim | None:
    """Require all three public evidence files only in the exact selected authority profile."""

    names = (CLAIM_FILE_ENV, GENERATION_FILE_ENV, HANDOFF_FILE_ENV)
    selected_profile = environ.get(generation_authority.PROFILE_ENV)
    selected = selected_profile in ACTION_PROFILE_SPECS
    present = [name for name in names if name in environ]
    if not selected:
        if present:
            raise Runtime452ActionClaimError(
                "runtime452 action evidence is forbidden outside its exact profile"
            )
        return None
    values = [environ.get(name, "") for name in names]
    if not all(values):
        raise Runtime452ActionClaimError(
            "runtime452 profile requires claim, generation, and handoff evidence"
        )
    claim = load_runtime452_action_claim(
        Path(values[0]),
        Path(values[1]),
        Path(values[2]),
        expected_action=expected_action,
        expected_actor=expected_actor,
    )
    if claim.profile_id != selected_profile:
        raise Runtime452ActionClaimError(
            "runtime452 action evidence belongs to a different compiled profile"
        )
    return claim
