"""Bittensor v11 miner endpoint: publish on-chain, serve receiver-bound btauth/1 HTTP.

CONTRACT (capsule header)
- The envelope is BYTE-IDENTICAL to plain-http mode: the same `WorkIdMinter.envelope`
  and mints an exact v2 authorization bound to request ID, fanout window, and the authenticated
  btauth caller hotkey. A caller whose declared validator audience differs from that identity is
  rejected before candidate selection; the coldkey is never present.
- `external_ip` is ALWAYS explicit (env `AURELIUS_AXON_EXTERNAL_IP`, default the container's own
  address) — never the SDK's IP-echo autodiscovery (no network egress from the testlab). The bind
  port and advertised NAT port are separate: `AURELIUS_AXON_EXTERNAL_PORT` selects only the
  on-chain endpoint and is range/shape checked before publication.
- Candidate admission requires a finalized validator-permit view before any public listener opens.
  The declared sender is a bounded negative-membership prefilter only; btauth then verifies and
  commits the nonce, and the returned caller is identity- and permit-rechecked. Refreshes pin both
  independent finality and the SDK metagraph read to one block; drift keeps the LAST-GOOD set.
- Every btauth deployment requires a durable `AURELIUS_AXON_PUBLISH_STATE_DIR`: it writes one
  durable intent, submits at most once, and may recover only from an exact finalized-chain tuple.
  An ambiguous result is never retried across process or Compose-project restarts.
- No HTTP socket opens until durable publication/recovery and the first finalized permit sync.
  The :8092 sidecar then answers ready; `/v1/query` is available only in the exact local
  testlab/localnet/observed-genesis posture. Both surfaces have bounded concurrency/read deadlines.
- SDK pin: bittensor 11.0.0. V11 provides no Axon/Dendrite/Synapse server; the application owns
  HTTP and uses ``bittensor.http_auth`` for raw-body authentication and replay protection.
- NO priority_fn: stake-ordered priority is economics-adjacent (STOP discipline).
- bittensor imports stay lazy (inside functions): the node image imports this module bt-free.
"""

from __future__ import annotations

import json
import hashlib
import ipaddress
import os
import re
import secrets
import socket
import stat
import threading
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from subnet.axon_protocol import (
    CANDIDATE_SOURCE_INLINE,
    MAX_CONFIG_JSON_BYTES,
    PROTOCOL_VERSION,
)
from subnet.bittensor_v11 import (
    DurableNonceStore,
    FinalizedRpc,
    open_subtensor,
    parse_published_endpoint,
    require_bittensor_v11,
)
from subnet.miner.candidates import candidate_for_request
from subnet.miner.config_schema import ServingConfig
from subnet.miner.serve import (
    ServingStats,
    WorkIdMinter,
    QueryAuthorizationError,
    caller_rate_limiter,
    load_bittensor_wallet,
    make_server,
    minter_from_bittensor_wallet,
    plain_query_allowed,
    start_status_line,
    validate_miner_startup_posture,
)
from subnet.operator.evidence_reader import (
    EvidenceReadError,
    canonical_json_bytes,
    read_json_evidence,
)
from subnet.operator.sn455_runtime452_action_claim import (
    ACTION_SIGNERS as RUNTIME452_ACTION_SIGNERS,
    AXON_ENDPOINTS as RUNTIME452_AXON_ENDPOINTS,
    CLAIM_FILE_ENV as RUNTIME452_CLAIM_FILE_ENV,
    GENERATION_FILE_ENV as RUNTIME452_GENERATION_FILE_ENV,
    HANDOFF_FILE_ENV as RUNTIME452_HANDOFF_FILE_ENV,
    PLUMBING_PROFILE_ID as RUNTIME452_PLUMBING_PROFILE_ID,
    PROFILE_ID as RUNTIME452_PROFILE_ID,
    SERVE_AXON_ACTION as RUNTIME452_SERVE_AXON_ACTION,
    Runtime452ActionClaim,
    Runtime452ActionClaimError,
    runtime452_action_claim_from_env,
)
from subnet.operator.runtime452_action_journal import (
    Runtime452ActionJournalCustody,
    Runtime452ActionJournalError,
)
from subnet.operator.prepared_extrinsic import (
    MAX_PREPARED_ATTEMPTS,
    decoded_scan_candidate,
    prepared_evidence,
    reconstruct_unsigned,
    scan_resolution_evidence,
    signature_evidence,
    submission_claim_evidence,
    validate_era_chain_anchor,
    validate_next_attempt_authority,
    validate_prepared_evidence,
    validate_reprepared_attempt,
    validate_scan_resolution_evidence,
    validate_signature_evidence,
    validate_submission_claim_evidence,
    validate_unsigned_dict,
)
from subnet.shared.canonicalization import parse_canonical_json
from subnet.shared.network_policy import AxonEndpointPolicy


RUNTIME452_PROFILE_IDS = frozenset({RUNTIME452_PROFILE_ID, RUNTIME452_PLUMBING_PROFILE_ID})


class RegisteredSet:
    """The blacklist's last-good validator-permit view. ``None`` means never synchronized."""

    def __init__(self):
        self._lock = threading.Lock()
        self._hotkeys: Optional[frozenset] = None

    @property
    def hotkeys(self) -> Optional[frozenset]:
        with self._lock:
            return self._hotkeys

    @hotkeys.setter
    def hotkeys(self, value: Optional[frozenset]) -> None:
        if value is not None and (
            type(value) is not frozenset
            or any(not isinstance(hotkey, str) or not hotkey for hotkey in value)
        ):
            raise ValueError("validator-permit view must be a frozenset of non-empty hotkeys")
        with self._lock:
            self._hotkeys = value


class AxonPublishError(RuntimeError):
    """The guarded external-chain Axon publication cannot be proven unambiguous."""


class _AxonCoordinateMismatch(AxonPublishError):
    """The SDK named a canonical extrinsic that is provably not our durable signature."""


class _AxonScanPending(AxonPublishError):
    """A claimed submission must remain scan-only until finalized evidence catches up."""


_TESTNET_AXON_SERVE_ACK = "I_ACKNOWLEDGE_UP_TO_FOUR_SN455_AXON_ENDPOINT_UPDATES"
_DEFAULT_MAX_FEE_TAO = Decimal("0.01")
_MAX_MAX_FEE_TAO = Decimal("0.01")
_RAO_PER_TAO = 1_000_000_000
_EXTRINSIC_PERIOD_BLOCKS = 64
_MAX_PUBLISH_EVIDENCE_BYTES = 1_048_576
_MAX_AUTH_HEADERS = 128
_MAX_AUTH_HOTKEY_BYTES = 128
AXON_TERMINAL_EVIDENCE_SCHEMA_VERSION = 2
AXON_TERMINAL_EVIDENCE_STATUSES = frozenset(
    {
        "already_exact_no_submission",
        "finalized_exact",
        "recovered_from_prepared_era",
    }
)


def _runtime452_axon_claim_preflight(
    environ,
    *,
    hotkey_name: str,
    netuid: int,
    external_ip: str,
    external_port: int,
) -> Runtime452ActionClaim | None:
    """Resolve the exact one-shot Axon authority before wallet or network access."""

    selected = environ.get("AURELIUS_ENV") in RUNTIME452_PROFILE_IDS
    evidence_names = (
        RUNTIME452_CLAIM_FILE_ENV,
        RUNTIME452_GENERATION_FILE_ENV,
        RUNTIME452_HANDOFF_FILE_ENV,
    )
    if not selected:
        if any(name in environ for name in evidence_names):
            raise AxonPublishError(
                "runtime452 Axon action evidence is forbidden outside its exact profile"
            )
        return None
    actor = environ.get("AURELIUS_BT_WALLET")
    if (
        actor not in RUNTIME452_AXON_ENDPOINTS
        or hotkey_name != actor
        or environ.get("AURELIUS_BT_HOTKEY") != "default"
    ):
        raise AxonPublishError("runtime452 Axon process identity differs from its exact actor")
    if "AURELIUS_TESTNET_AXON_SERVE_ACK" in environ:
        raise AxonPublishError(
            "raw Axon acknowledgement is forbidden in the runtime452 evidence profile"
        )
    try:
        claim = runtime452_action_claim_from_env(
            environ,
            expected_action=RUNTIME452_SERVE_AXON_ACTION,
            expected_actor=actor,
        )
    except Runtime452ActionClaimError as exc:
        raise AxonPublishError("runtime452 Axon action claim is invalid") from exc
    if (
        claim is None
        or netuid != claim.netuid
        or external_ip != claim.endpoint_ip
        or external_port != claim.endpoint_port
        or environ.get("SUBSTRATE_URL") != claim.chain_endpoint
        or environ.get("AURELIUS_AXON_EXPECTED_CHAIN_NAME") != claim.chain_name
        or environ.get("AURELIUS_AXON_EXPECTED_GENESIS_HASH") != claim.genesis_hash
        or environ.get("AURELIUS_RUNTIME_SPEC_PIN") != str(claim.runtime_spec_version)
        or environ.get("AURELIUS_RUNTIME_TRANSACTION_VERSION_PIN") != str(claim.transaction_version)
        or environ.get("AURELIUS_AXON_PUBLISH_STATE_DIR") != claim.journal_path
    ):
        raise AxonPublishError("runtime452 Axon process coordinates differ from its action claim")
    return claim


def _require_runtime452_axon_wallet(
    claim: Runtime452ActionClaim | None,
    wallet,
) -> None:
    """Bind the already-validated public claim to the wallet before any RPC is opened."""

    if claim is None:
        return
    signer = RUNTIME452_ACTION_SIGNERS.get(claim.actor)
    if (
        signer is None
        or str(wallet.hotkey.ss58_address) != claim.signer_hotkey_ss58
        or str(wallet.coldkeypub.ss58_address) != claim.signer_coldkey_ss58
        or signer
        != {
            "hotkey_ss58": claim.signer_hotkey_ss58,
            "coldkey_ss58": claim.signer_coldkey_ss58,
        }
    ):
        raise AxonPublishError("runtime452 Axon wallet differs from its action claim")


_RUNTIME452_AXON_ATTEMPT_KINDS = (
    "authority",
    "signature",
    "submission-claim",
    "sdk-result",
    "scan-resolution",
)


def _axon_prepared_attempt_limit(
    state_root: Path | Runtime452ActionJournalCustody,
    claim: Runtime452ActionClaim | None,
) -> int:
    """Make the runtime452 action's one-signature bound structural in the recovery loop."""

    if claim is None:
        return MAX_PREPARED_ATTEMPTS
    if not isinstance(state_root, Runtime452ActionJournalCustody):
        raise AxonPublishError("runtime452 Axon attempt bound lacks installed journal custody")
    if any(state_root.exists(f"{kind}-2.json") for kind in _RUNTIME452_AXON_ATTEMPT_KINDS):
        raise AxonPublishError("runtime452 Axon journal exceeds its one signed extrinsic claim")
    return 1


def finalized_axon_evidence_document(
    *,
    status: str,
    identity: dict[str, Any],
    observation: dict[str, Any],
    submission_attempts: int,
    prepared_attempt: int,
    causal_extrinsic_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build the sole current terminal contract consumed by Product and resume."""

    if status not in AXON_TERMINAL_EVIDENCE_STATUSES:
        raise AxonPublishError("unknown current Axon terminal status")
    if type(submission_attempts) is not int or type(prepared_attempt) is not int:
        raise AxonPublishError("Axon terminal attempts must be exact integers")
    if status == "already_exact_no_submission":
        exact_lineage = (
            submission_attempts == 0
            and prepared_attempt == 0
            and causal_extrinsic_evidence == {"status": "not_required"}
        )
    else:
        exact_lineage = (
            type(prepared_attempt) is int
            and 1 <= prepared_attempt <= MAX_PREPARED_ATTEMPTS
            and submission_attempts == prepared_attempt
            and causal_extrinsic_evidence.get("status") == "exact_finalized_extrinsic"
        )
    if not exact_lineage:
        raise AxonPublishError("Axon terminal status has inconsistent prepared lineage")
    return {
        "schema_version": AXON_TERMINAL_EVIDENCE_SCHEMA_VERSION,
        "status": status,
        "identity": identity,
        "observation": observation,
        "submission_attempts": submission_attempts,
        "prepared_attempt": prepared_attempt,
        "causal_extrinsic_evidence": causal_extrinsic_evidence,
    }


def authorize_testnet_axon_publication(raw: str | None = None) -> None:
    """Require an exact operator acknowledgement immediately before an Axon chain write."""

    value = os.environ.get("AURELIUS_TESTNET_AXON_SERVE_ACK") if raw is None else raw
    if value != _TESTNET_AXON_SERVE_ACK:
        raise AxonPublishError("SN455 Axon publication acknowledgement is absent or invalid")


def _strict_json_object(path: Path):
    try:
        document = read_json_evidence(
            path,
            label=f"{path.name} publish evidence",
            max_bytes=_MAX_PUBLISH_EVIDENCE_BYTES,
            require_canonical=True,
            require_mode=0o600,
        )
    except EvidenceReadError as exc:
        raise AxonPublishError(f"{path.name} is unreadable publish evidence") from exc
    return document.value


def _publish_json_payload(value) -> bytes:
    try:
        payload = canonical_json_bytes(value)
    except EvidenceReadError as exc:
        raise AxonPublishError("Axon publish evidence is not canonical JSON data") from exc
    if not 1 <= len(payload) <= _MAX_PUBLISH_EVIDENCE_BYTES:
        raise AxonPublishError("Axon publish evidence exceeds its bounded JSON posture")
    return payload


def _write_all(descriptor, payload) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while persisting Axon publish evidence")
        offset += written


def _claim_json(path: Path, value) -> None:
    payload = _publish_json_payload(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AxonPublishError("Axon publish evidence is not one regular file")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    payload = _publish_json_payload(value)
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0),
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AxonPublishError("temporary Axon evidence is not one regular file")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def guarded_publish_once(
    state_dir,
    identity,
    observe,
    publish,
    *,
    authorize_publish=None,
    post_submit_observe_attempts=1,
    post_submit_observe_delay=0.0,
    sleeper=None,
    plan=None,
    response_envelope=None,
    validate_response=None,
    verify_response=None,
):
    """Publish at most once across process and Compose-project restarts.

    ``state_dir`` is host-stable in the external-testnet profile. An unresolved intent is never
    retried: a later process may continue only when a finalized chain read already equals the
    exact intended Axon tuple. Ordinary localnet mode does not select this helper.
    """
    if not isinstance(identity, dict) or not identity:
        raise AxonPublishError("guarded Axon publish identity must be a non-empty object")
    if (
        isinstance(post_submit_observe_attempts, bool)
        or not isinstance(post_submit_observe_attempts, int)
        or not 1 <= post_submit_observe_attempts <= 120
    ):
        raise AxonPublishError("post-submit Axon read-back attempts must be between 1 and 120")
    if (
        isinstance(post_submit_observe_delay, bool)
        or not isinstance(post_submit_observe_delay, (int, float))
        or not 0 <= post_submit_observe_delay <= 10
    ):
        raise AxonPublishError("post-submit Axon read-back delay must be between 0 and 10 seconds")
    sleep = time.sleep if sleeper is None else sleeper
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise AxonPublishError("Axon publish state must be a non-symlink directory")
    intent_path = state_dir / "intent.json"
    sdk_result_path = state_dir / "sdk-result.json"
    finalized_path = state_dir / "finalized.json"

    if finalized_path.exists():
        finalized = _strict_json_object(finalized_path)
        if finalized.get("identity") != identity:
            raise AxonPublishError("finalized Axon evidence belongs to a different tuple")
        observation = observe()
        if observation.get("matches") is not True:
            raise AxonPublishError("previously finalized Axon tuple has drifted on chain")
        return {"status": "finalized_replay", "observation": observation}

    if intent_path.exists():
        intent = _strict_json_object(intent_path)
        if intent.get("identity") != identity:
            raise AxonPublishError("unresolved Axon intent belongs to a different tuple")
        observation = observe()
        if observation.get("matches") is not True:
            raise AxonPublishError(
                "Axon publication intent is unresolved and the finalized chain does not show "
                "the intended tuple; refusing an ambiguous retry"
            )
        causal_evidence = {
            "status": "unavailable_endpoint_state_only",
            "reason": "no durable SDK response coordinates survived the ambiguous submission",
        }
        if sdk_result_path.exists():
            sdk_result = _strict_json_object(sdk_result_path)
            if verify_response is not None:
                causal_evidence = verify_response(sdk_result)
        finalized = {
            "schema_version": 1,
            "status": "recovered_from_finalized_chain",
            "identity": identity,
            "observation": observation,
            "submission_attempts": 1,
            "causal_extrinsic_evidence": causal_evidence,
        }
        _atomic_json(finalized_path, finalized)
        return finalized

    observation = observe()
    if observation.get("matches") is True:
        finalized = {
            "schema_version": 1,
            "status": "already_exact_no_submission",
            "identity": identity,
            "observation": observation,
            "submission_attempts": 0,
        }
        _atomic_json(finalized_path, finalized)
        return finalized

    # A missing acknowledgement is a configuration refusal, not an ambiguous chain attempt. Run
    # this after exact read-back (which can prove that no write is needed) but before the durable
    # intent that permanently forbids retry after any possible submission.
    if authorize_publish is not None:
        authorize_publish()

    plan_evidence = plan() if plan is not None else None
    if plan_evidence is not None and not isinstance(plan_evidence, dict):
        raise AxonPublishError("ServeAxon plan evidence must be an object")

    _claim_json(
        intent_path,
        {
            "schema_version": 1,
            "status": "submission_authorized",
            "identity": identity,
            "submission_attempts_authorized": 1,
            "sdk_plan": plan_evidence,
        },
    )
    # Exactly one call after the durable intent. Any exception is ambiguous and leaves the intent
    # in place; this helper will never call ``publish`` again for it.
    response = publish()
    envelope = (
        response_envelope(response)
        if response_envelope is not None
        else {
            "schema_version": 1,
            "kind": "bittensor_v11_sdk_response",
            "success": getattr(response, "success", None),
        }
    )
    if not isinstance(envelope, dict):
        raise AxonPublishError("ServeAxon SDK response evidence must be an object")
    _claim_json(sdk_result_path, envelope)
    if validate_response is not None:
        validate_response(response, envelope)
    if envelope.get("success") is not True:
        raise AxonPublishError("the single serve_axon submission did not succeed")
    causal_evidence = (
        verify_response(envelope) if verify_response is not None else {"status": "not_requested"}
    )
    observation = None
    for attempt in range(post_submit_observe_attempts):
        observation = observe()
        if observation.get("matches") is True:
            break
        if attempt + 1 < post_submit_observe_attempts:
            sleep(post_submit_observe_delay)
    if observation is None or observation.get("matches") is not True:
        raise AxonPublishError("serve_axon returned success but finalized read-back differs")
    finalized = {
        "schema_version": 1,
        "status": "finalized_exact",
        "identity": identity,
        "observation": observation,
        "submission_attempts": 1,
        "causal_extrinsic_evidence": causal_evidence,
    }
    _atomic_json(finalized_path, finalized)
    return finalized


def blacklist_decision(
    registered: Optional[frozenset], caller_hotkey: Optional[str]
) -> tuple[bool, str]:
    """Pure decision (unit-pinned): True = blacklist; never-synced is fail closed."""
    if not caller_hotkey:
        return True, "no authenticated Bittensor hotkey on the request"
    if registered is None:
        return True, "metagraph never synced — validator permit unavailable"
    if caller_hotkey not in registered:
        return True, "caller lacks validator permit on this netuid"
    return False, "ok"


def _declared_btauth_sender(headers, header_name: str) -> str:
    """Parse one bounded, case-insensitive sender header without treating it as authority."""

    try:
        items = iter(headers.items())
    except (AttributeError, TypeError, ValueError):
        raise QueryAuthorizationError(
            401, "Bittensor sender header collection is malformed"
        ) from None
    values = []
    try:
        for index, item in enumerate(items):
            if index >= _MAX_AUTH_HEADERS:
                raise QueryAuthorizationError(
                    401, "Bittensor sender header collection is oversized"
                )
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise QueryAuthorizationError(
                    401, "Bittensor sender header collection is malformed"
                )
            key, value = item
            if isinstance(key, str) and key.lower() == header_name.lower():
                values.append(value)
    except QueryAuthorizationError:
        raise
    except (TypeError, ValueError):
        raise QueryAuthorizationError(
            401, "Bittensor sender header collection is malformed"
        ) from None
    if len(values) != 1:
        raise QueryAuthorizationError(401, "Bittensor sender header must occur exactly once")
    sender = values[0]
    if (
        not isinstance(sender, str)
        or not sender.isascii()
        or not 3 <= len(sender.encode("ascii")) <= _MAX_AUTH_HOTKEY_BYTES
        or any(not 0x21 <= ord(character) <= 0x7E for character in sender)
    ):
        raise QueryAuthorizationError(401, "Bittensor sender header has an invalid shape")
    return sender


def build_envelope_json(
    minter: WorkIdMinter,
    task_type,
    schema_version,
    config_json,
    request_id="",
    validator_hotkey_ss58="",
    fanout_window=-1,
    validator_version="",
    protocol_version=PROTOCOL_VERSION,
    candidate_source=CANDIDATE_SOURCE_INLINE,
    *,
    candidate_resolver=None,
) -> str:
    """The forward body, refusals-as-empty (the validator maps an empty envelope to
    MinerUnreachable): size cap, miner-owned candidate resolution, then the SAME minter as http
    mode. ``config_json`` is decoded only for explicit inline compatibility; staging mode never
    treats validator bytes as the candidate."""
    try:
        if (
            not isinstance(config_json, str)
            or len(config_json.encode("utf-8")) > MAX_CONFIG_JSON_BYTES
        ):
            return ""
        request = {
            "request_id": request_id,
            "validator_hotkey_ss58": validator_hotkey_ss58,
            "fanout_window": fanout_window,
            "validator_version": validator_version,
            "protocol_version": protocol_version,
            "candidate_source": candidate_source,
            "task_type": task_type,
            "schema_version": int(schema_version),
        }
        if candidate_source == CANDIDATE_SOURCE_INLINE:
            request["config"] = parse_canonical_json(config_json)
        resolve_candidate = candidate_resolver or candidate_for_request
        envelope = minter.envelope(
            resolve_candidate(request),
            validator_hotkey_ss58=validator_hotkey_ss58,
            request_id=request_id,
            fanout_window=fanout_window,
        )
        return json.dumps(envelope)
    except (UnicodeError, ValueError, KeyError, TypeError):
        return ""


def make_btauth_authorizer(bt, wallet, registered: RegisteredSet, *, nonce_store=None):
    """Verify raw btauth/1 bytes and require a current validator permit.

    The entrypoint supplies Aurelius' crash-safe store.  The in-memory default exists only for
    isolated construction tests; production startup requires ``AURELIUS_BITTENSOR_NONCE_DB``.
    """

    if nonce_store is None:
        nonce_store = bt.http_auth.InMemoryNonceStore()
    self_hotkey = str(wallet.hotkey.ss58_address)

    def authorize(headers, body, *, method, path):
        permitted = registered.hotkeys
        if permitted is None:
            raise QueryAuthorizationError(503, "finalized validator permit set is unavailable")
        header_name = getattr(bt.http_auth, "HEADER_HOTKEY", "X-Bittensor-Hotkey")
        declared_sender = _declared_btauth_sender(headers, header_name)
        if declared_sender not in permitted:
            raise QueryAuthorizationError(403, "declared sender lacks validator permit")
        try:
            caller = bt.http_auth.verify(
                headers,
                body,
                method=method,
                path=path,
                self_hotkey_ss58=self_hotkey,
                nonce_store=nonce_store,
            )
        except bt.http_auth.AuthError as exc:
            raise QueryAuthorizationError(401, f"Bittensor authentication failed: {exc}") from None
        caller_hotkey = getattr(caller, "hotkey_ss58", None)
        if caller_hotkey != declared_sender:
            raise QueryAuthorizationError(
                401, "cryptographically verified caller differs from declared sender"
            )
        latest_permitted = registered.hotkeys
        denied, reason = blacklist_decision(latest_permitted, caller_hotkey)
        if denied:
            status = 503 if latest_permitted is None else 403
            raise QueryAuthorizationError(status, reason)
        return caller_hotkey

    return authorize


def durable_btauth_nonce_store(environ=None) -> DurableNonceStore:
    """Open the required restart-safe replay store from explicit operator configuration."""

    env = os.environ if environ is None else environ
    path = env.get("AURELIUS_BITTENSOR_NONCE_DB")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("AURELIUS_BITTENSOR_NONCE_DB is required in btauth miner mode")
    return DurableNonceStore(path)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AxonPublishError(f"{name} is required for guarded Axon publication")
    return value.strip()


def _prepare_publish_state_dir(
    raw: str,
    runtime452_action_claim: Runtime452ActionClaim | None = None,
) -> Path | Runtime452ActionJournalCustody:
    path = Path(raw)
    if runtime452_action_claim is not None:
        if path != Path(runtime452_action_claim.journal_path):
            raise AxonPublishError("runtime452 Axon journal path differs from its action claim")
        try:
            return Runtime452ActionJournalCustody(runtime452_action_claim)
        except Runtime452ActionJournalError as exc:
            raise AxonPublishError("runtime452 Axon journal installation is invalid") from exc
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise AxonPublishError("AURELIUS_AXON_PUBLISH_STATE_DIR cannot be created safely") from exc
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_only is None:
        raise AxonPublishError("secure Axon publish directory opening is unavailable")
    directory = None
    probe_descriptor = None
    probe_name = f".startup-probe-{os.getpid()}-{secrets.token_hex(8)}"
    probe_exists = False
    try:
        directory = os.open(
            path,
            os.O_RDONLY | directory_only | nofollow | getattr(os, "O_CLOEXEC", 0),
        )
        before = os.fstat(directory)
        if (
            not stat.S_ISDIR(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o700
        ):
            raise AxonPublishError(
                "AURELIUS_AXON_PUBLISH_STATE_DIR must be runtime-owned with mode 0700"
            )
        probe_descriptor = os.open(
            probe_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory,
        )
        probe_exists = True
        probe_info = os.fstat(probe_descriptor)
        if (
            not stat.S_ISREG(probe_info.st_mode)
            or probe_info.st_nlink != 1
            or probe_info.st_uid != os.geteuid()
            or stat.S_IMODE(probe_info.st_mode) != 0o600
        ):
            raise AxonPublishError("Axon publish directory probe has an unsafe inode shape")
        os.fsync(probe_descriptor)
        os.close(probe_descriptor)
        probe_descriptor = None
        os.unlink(probe_name, dir_fd=directory)
        probe_exists = False
        os.fsync(directory)
        after = os.fstat(directory)
        pathname = os.stat(path, follow_symlinks=False)
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_uid")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields) or any(
            getattr(after, field) != getattr(pathname, field) for field in stable_fields
        ):
            raise AxonPublishError("AURELIUS_AXON_PUBLISH_STATE_DIR changed during validation")
    except AxonPublishError:
        raise
    except OSError as exc:
        raise AxonPublishError("AURELIUS_AXON_PUBLISH_STATE_DIR is unavailable or unsafe") from exc
    finally:
        if probe_descriptor is not None:
            os.close(probe_descriptor)
        if probe_exists and directory is not None:
            try:
                os.unlink(probe_name, dir_fd=directory)
            except OSError:
                pass
        if directory is not None:
            os.close(directory)
    return path


def _max_fee_rao(raw: str | None = None) -> int:
    value_raw = (
        os.environ.get("AURELIUS_AXON_MAX_FEE_TAO", str(_DEFAULT_MAX_FEE_TAO))
        if raw is None
        else raw
    )
    try:
        value = Decimal(value_raw)
    except (InvalidOperation, TypeError):
        raise AxonPublishError("AURELIUS_AXON_MAX_FEE_TAO must be decimal TAO") from None
    rao = value * _RAO_PER_TAO
    if (
        not value.is_finite()
        or value < 0
        or value > _MAX_MAX_FEE_TAO
        or rao != rao.to_integral_value()
    ):
        raise AxonPublishError(f"AURELIUS_AXON_MAX_FEE_TAO must be 0 rao..{_MAX_MAX_FEE_TAO} TAO")
    return int(rao)


def _safe_response_field(response, name):
    try:
        value = getattr(response, name)
    except Exception:  # noqa: BLE001 - capture only inert public scalars
        return None
    return value if isinstance(value, (str, int, bool)) or value is None else None


def _sdk_response_envelope(response):
    try:
        fee = getattr(response, "fee")
        fee_rao = getattr(fee, "rao", None)
        fee_netuid = getattr(fee, "netuid", None)
    except Exception:  # noqa: BLE001 - evidence capture precedes interpretation
        fee_rao = None
        fee_netuid = None
    if type(fee_rao) is not int or fee_rao < 0:
        fee_rao = None
    return {
        "schema_version": 1,
        "kind": "bittensor_v11_sdk_response",
        "response_type": f"{type(response).__module__}.{type(response).__qualname__}",
        "success": _safe_response_field(response, "success"),
        "message": _safe_response_field(response, "message"),
        "block_hash": _safe_response_field(response, "block_hash"),
        "extrinsic_id": _safe_response_field(response, "extrinsic_id"),
        "fee_tao_rao": fee_rao,
        "fee_netuid": fee_netuid if type(fee_netuid) is int and fee_netuid >= 0 else None,
    }


def _balance_rao(value, *, label, expected_netuid):
    netuid = getattr(value, "netuid", None)
    if type(netuid) is not int or netuid != expected_netuid:
        raise AxonPublishError(f"{label} has the wrong Balance netuid")
    candidate = getattr(value, "rao", value)
    if type(candidate) is not int or candidate < 0:
        raise AxonPublishError(f"{label} did not resolve to non-negative integer rao")
    return candidate


def _value(value):
    value = getattr(value, "value", value)
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, bytes):
        return "0x" + value.hex()
    return value


_SERVE_AXON_CALL_ARG_TYPES = {
    "netuid": "NetUid",
    "version": "u32",
    "ip": "u128",
    "port": "u16",
    "ip_type": "u8",
    "protocol": "u8",
    "placeholder1": "u8",
    "placeholder2": "u8",
}
_SERVE_AXON_CALL_ARG_ORDER = tuple(_SERVE_AXON_CALL_ARG_TYPES)
_SERVE_AXON_CALL_INDEX = "0x0704"
# Each admitted runtime is independently decoded by an attributed metadata test. Runtime 452 is
# covered by ``test_bittensor_v11_runtime452_metadata.py`` and runtime 453 by
# ``test_bittensor_v11_runtime453_metadata.py`` (ADR-0197). Runtime 458 has its own
# attributed metadata and exact GH200 endpoint byte tests (ADR-0252). Runtime 466 has
# separate attributed metadata/byte proofs (ADR-0286); future eras still refuse.
# Runtime467/468 are separately attributed and wire-tested in ADR-0290/0314.
# Runtime473 is attributed and wire-tested in ADR-0339; runtime475 in ADR-0359.
_REVIEWED_SERVE_AXON_RUNTIME_SPECS = frozenset({432, 450, 452, 453, 458, 466, 467, 468, 473, 475})


def _require_exact_serve_axon_decoded_call(
    decoded,
    *,
    expected_call_args,
    label,
):
    """Normalize only the exact SDK 11/runtime-432 decoded ServeAxon shape."""

    if type(decoded) is not dict or set(decoded) != {
        "call_index",
        "call_function",
        "call_module",
        "call_args",
        "call_hash",
    }:
        raise AxonPublishError(f"{label} has an unaudited SDK decode shape")
    call_index = decoded.get("call_index")
    raw_call_hash = decoded.get("call_hash")
    call_hash = raw_call_hash.lower() if type(raw_call_hash) is str else ""
    arguments = decoded.get("call_args")
    if (
        type(call_index) is not str
        or call_index.lower() != _SERVE_AXON_CALL_INDEX
        or re.fullmatch(r"0x[0-9a-f]{64}", call_hash) is None
        or type(arguments) is not list
    ):
        raise AxonPublishError(f"{label} lacks an exact index, hash, or argument list")
    pairs = []
    for argument in arguments:
        if (
            type(argument) is not dict
            or set(argument) != {"name", "type", "value"}
            or type(argument.get("name")) is not str
            or argument.get("type") != _SERVE_AXON_CALL_ARG_TYPES.get(argument.get("name"))
            or type(argument.get("value")) is not int
            or any(name == argument["name"] for name, _ in pairs)
        ):
            raise AxonPublishError(f"{label} argument shape or runtime type drifted")
        pairs.append((argument["name"], argument["value"]))
    if (
        decoded.get("call_module") != "SubtensorModule"
        or decoded.get("call_function") != "serve_axon"
        or tuple(name for name, _ in pairs) != _SERVE_AXON_CALL_ARG_ORDER
        or dict(pairs) != expected_call_args
    ):
        raise AxonPublishError(f"{label} differs from the exact typed ServeAxon intent")
    return call_hash


def _validate_planned_serve_axon_call(
    subtensor,
    call,
    *,
    runtime_pin,
    expected_call_args,
):
    """Decode the opaque SDK 11 CallBytes before they become signing authority."""

    if runtime_pin not in _REVIEWED_SERVE_AXON_RUNTIME_SPECS:
        raise AxonPublishError("ServeAxon runtime has no reviewed exact call schema")
    call_data = getattr(call, "data", None)
    call_spec_version = getattr(call, "spec_version", None)
    if (
        type(call_data) is not bytes
        or not call_data
        or type(call_spec_version) is not int
        or call_spec_version != runtime_pin
    ):
        raise AxonPublishError("planned ServeAxon lacks pinned-runtime SDK 11 call bytes")
    try:
        decoded = subtensor.decode_scale("Call", call_data)
    except Exception as exc:  # noqa: BLE001 - opaque bytes may never become signing authority
        raise AxonPublishError(
            "planned ServeAxon bytes could not be independently decoded"
        ) from exc
    call_hash = _require_exact_serve_axon_decoded_call(
        decoded,
        expected_call_args=expected_call_args,
        label="planned ServeAxon bytes",
    )
    expected_hash = "0x" + hashlib.blake2b(call_data, digest_size=32).hexdigest()
    sdk_call_hash = getattr(call, "call_hash", None)
    if (
        call_hash != expected_hash
        or type(sdk_call_hash) is not bytes
        or sdk_call_hash != bytes.fromhex(expected_hash[2:])
    ):
        raise AxonPublishError("planned ServeAxon decoded hash differs from its exact bytes")
    return call_data, call_spec_version, call_hash


def _raw_extrinsics(block):
    if not isinstance(block, dict) or not isinstance(block.get("block"), dict):
        raise AxonPublishError("ServeAxon chain_getBlock response is malformed")
    values = block["block"].get("extrinsics")
    if not isinstance(values, list) or not all(
        isinstance(value, str) and re.fullmatch(r"0x[0-9a-fA-F]*", value) for value in values
    ):
        raise AxonPublishError("ServeAxon finalized block has malformed raw extrinsics")
    return values


def _raw_extrinsic_hash(raw):
    return "0x" + hashlib.blake2b(bytes.fromhex(raw[2:]), digest_size=32).hexdigest()


def _event_parts(record):
    record = _value(record)
    if not isinstance(record, dict):
        raise AxonPublishError("ServeAxon finalized event is malformed")
    index = record.get("extrinsic_idx")
    if index is None:
        phase = _value(record.get("phase"))
        if phase in ("Initialization", "Finalization"):
            return None
        if isinstance(phase, int):
            index = phase
        elif isinstance(phase, dict):
            index = phase.get("ApplyExtrinsic", phase.get("apply_extrinsic"))
    event = record.get("event", record)
    if type(index) is not int or not isinstance(event, dict):
        raise AxonPublishError("ServeAxon finalized event phase is malformed")
    module = event.get("module_id", event.get("module", event.get("pallet")))
    name = event.get("event_id", event.get("event", event.get("variant")))
    if not isinstance(module, str) or not isinstance(name, str):
        raise AxonPublishError("ServeAxon finalized event lacks identity")
    return index, module, name, event.get("attributes", event.get("params"))


def _validate_serve_axon_fee(
    attributes,
    *,
    expected_payer: str,
    maximum_fee_rao: int,
    sdk_fee_rao: int | None,
):
    attributes = _value(attributes)
    if not isinstance(attributes, dict) or set(attributes) != {"who", "actual_fee", "tip"}:
        raise AxonPublishError("ServeAxon TransactionFeePaid shape drifted")
    payer = str(attributes["who"])
    actual_fee_rao = attributes["actual_fee"]
    tip_rao = attributes["tip"]
    if type(actual_fee_rao) is not int or actual_fee_rao < 0:
        raise AxonPublishError("ServeAxon actual fee is not non-negative integer rao")
    if type(tip_rao) is not int or tip_rao < 0:
        raise AxonPublishError("ServeAxon tip is not non-negative integer rao")
    if (
        payer != expected_payer
        or tip_rao != 0
        or actual_fee_rao > maximum_fee_rao
        or (sdk_fee_rao is not None and sdk_fee_rao != actual_fee_rao)
    ):
        raise AxonPublishError("ServeAxon finalized fee evidence differs from authority")
    return payer, actual_fee_rao, tip_rao


def _require_sdk_endpoint(subtensor, endpoint: str) -> None:
    if getattr(subtensor, "endpoint", None) != endpoint:
        raise AxonPublishError("Bittensor SDK endpoint differs from explicit chain authority")


def _validate_finalized_snapshot(
    snapshot,
    *,
    expected_chain: str,
    expected_genesis: str,
    runtime_pin: int,
):
    if (
        not isinstance(expected_chain, str)
        or not expected_chain
        or re.fullmatch(r"0x[0-9a-fA-F]{64}", expected_genesis) is None
        or type(runtime_pin) is not int
        or runtime_pin <= 0
    ):
        raise AxonPublishError("explicit Axon chain authority is malformed")
    genesis_hash = getattr(snapshot, "genesis_hash", None)
    if (
        not isinstance(getattr(snapshot, "head_hash", None), str)
        or re.fullmatch(r"0x[0-9a-fA-F]{64}", snapshot.head_hash) is None
        or type(getattr(snapshot, "block_number", None)) is not int
        or snapshot.block_number < 0
        or getattr(snapshot, "chain", None) != expected_chain
        or not isinstance(genesis_hash, str)
        or genesis_hash.lower() != expected_genesis.lower()
        or type(getattr(snapshot, "runtime_spec_version", None)) is not int
        or snapshot.runtime_spec_version != runtime_pin
    ):
        raise AxonPublishError("chain identity/runtime differs from explicit Axon authority")
    return snapshot


def _require_sdk_block_info(
    subtensor,
    *,
    endpoint: str,
    block_number: int,
    block_hash: str,
    extrinsic_index: int | None = None,
    extrinsic_hash: str | None = None,
):
    """Bind a typed SDK decode to the independently canonical raw block."""

    if re.fullmatch(r"0x[0-9a-fA-F]{64}", block_hash) is None:
        raise AxonPublishError("canonical finalized block hash is malformed")
    _require_sdk_endpoint(subtensor, endpoint)
    block = subtensor.block_info(block_number)
    if (
        block is None
        or type(getattr(block, "number", None)) is not int
        or block.number != block_number
        or not isinstance(getattr(block, "hash", None), str)
        or block.hash.lower() != block_hash.lower()
        or not isinstance(getattr(block, "extrinsics", None), list)
    ):
        raise AxonPublishError("Bittensor SDK BlockInfo differs from canonical finalized block")
    if extrinsic_index is not None:
        if not 0 <= extrinsic_index < len(block.extrinsics):
            raise AxonPublishError("Bittensor SDK BlockInfo omits the canonical extrinsic")
        decoded = block.extrinsics[extrinsic_index]
        decoded_hash = None
        if isinstance(decoded, dict):
            decoded_hash = decoded.get("extrinsic_hash", decoded.get("hash"))
        if decoded_hash is not None and (
            not isinstance(decoded_hash, str)
            or re.fullmatch(r"0x[0-9a-fA-F]{64}", decoded_hash) is None
            or extrinsic_hash is None
            or decoded_hash.lower() != extrinsic_hash.lower()
        ):
            raise AxonPublishError(
                "Bittensor SDK decoded extrinsic hash differs from canonical raw extrinsic"
            )
    return block


def _refresh_finalized_permits(
    registered: RegisteredSet,
    subtensor,
    rpc: FinalizedRpc,
    netuid: int,
    *,
    endpoint: str,
    expected_chain: str,
    expected_genesis: str,
    runtime_pin: int,
):
    """Atomically install a permit set only after one fully pinned finalized SDK read."""

    snapshot = _validate_finalized_snapshot(
        rpc.snapshot(),
        expected_chain=expected_chain,
        expected_genesis=expected_genesis,
        runtime_pin=runtime_pin,
    )
    _require_sdk_block_info(
        subtensor,
        endpoint=endpoint,
        block_number=snapshot.block_number,
        block_hash=snapshot.head_hash,
    )
    metagraph = subtensor.subnets.metagraph(
        netuid,
        block=snapshot.block_number,
        commitments=False,
    )
    if metagraph is None or getattr(metagraph, "netuid", None) != netuid:
        raise AxonPublishError("typed finalized metagraph differs from requested netuid")
    hotkeys = _permitted_validators_from_metagraph(metagraph)
    registered.hotkeys = hotkeys
    return snapshot


def _validate_prior_axon_attempt_authority(
    *,
    authority,
    signature_document,
    submission_document,
    resolution_document,
    identity,
    prior_attempt: int,
    next_attempt: int,
):
    """Require the complete signed-and-claimed prior attempt before a reprepare."""

    signature = validate_signature_evidence(
        signature_document,
        authority,
        identity=identity,
        attempt=prior_attempt,
    )
    signature_sha256 = hashlib.sha256(signature).hexdigest()
    validate_submission_claim_evidence(
        submission_document,
        authority,
        identity=identity,
        attempt=prior_attempt,
        signature_sha256=signature_sha256,
    )
    validate_next_attempt_authority(
        resolution_document,
        authority,
        identity=identity,
        prior_attempt=prior_attempt,
        signature_sha256=signature_sha256,
        next_attempt=next_attempt,
    )


def _guarded_external_publish(
    bt,
    wallet,
    netuid,
    subtensor,
    state_dir,
    *,
    external_ip,
    external_port,
    runtime452_action_claim: Runtime452ActionClaim | None = None,
):
    """Durable v11 ``ServeAxon`` publication, bound to explicit deployment identity."""
    selected_runtime452 = os.environ.get("AURELIUS_ENV") in RUNTIME452_PROFILE_IDS
    if selected_runtime452 != (runtime452_action_claim is not None):
        raise AxonPublishError("runtime452 Axon action claim selection differs from profile")
    runtime452_journal = (
        state_dir if isinstance(state_dir, Runtime452ActionJournalCustody) else None
    )
    if selected_runtime452 != (runtime452_journal is not None) or (
        runtime452_journal is not None and runtime452_journal.claim is not runtime452_action_claim
    ):
        raise AxonPublishError("runtime452 Axon action lacks its exact installed journal custody")
    state_root = runtime452_journal.path if runtime452_journal is not None else Path(state_dir)

    def journal_boundary() -> None:
        if runtime452_journal is None:
            return
        try:
            runtime452_journal.revalidate()
        except Runtime452ActionJournalError as exc:
            raise AxonPublishError("runtime452 Axon journal custody changed") from exc

    def journal_name(path: Path) -> str:
        if path.parent != state_root or path.name in {"", ".", ".."}:
            raise AxonPublishError("Axon publish evidence escaped its held journal")
        return path.name

    def journal_exists(path: Path) -> bool:
        if runtime452_journal is None:
            return path.exists()
        try:
            return runtime452_journal.exists(journal_name(path))
        except Runtime452ActionJournalError as exc:
            raise AxonPublishError("runtime452 Axon evidence cannot be inspected") from exc

    def journal_read(path: Path):
        if runtime452_journal is None:
            return _strict_json_object(path)
        try:
            return runtime452_journal.read_json(journal_name(path))
        except Runtime452ActionJournalError as exc:
            raise AxonPublishError("runtime452 Axon evidence cannot be read") from exc

    def journal_claim(path: Path, value) -> str:
        if runtime452_journal is None:
            _claim_json(path, value)
            return "claimed_exact"
        try:
            return runtime452_journal.claim_bytes(journal_name(path), _publish_json_payload(value))
        except Runtime452ActionJournalError as exc:
            raise AxonPublishError("runtime452 Axon evidence cannot be claimed") from exc

    journal_boundary()
    _require_runtime452_axon_wallet(runtime452_action_claim, wallet)
    chain_id = _required_env("CHAIN_ID")
    expected_chain = _required_env("AURELIUS_AXON_EXPECTED_CHAIN_NAME")
    expected_genesis = _required_env("AURELIUS_AXON_EXPECTED_GENESIS_HASH")
    runtime_raw = _required_env("AURELIUS_RUNTIME_SPEC_PIN")
    if not runtime_raw.isascii() or not runtime_raw.isdecimal() or int(runtime_raw) <= 0:
        raise AxonPublishError("AURELIUS_RUNTIME_SPEC_PIN must be a positive decimal integer")
    runtime_pin = int(runtime_raw)
    transaction_raw = _required_env("AURELIUS_RUNTIME_TRANSACTION_VERSION_PIN")
    if (
        not transaction_raw.isascii()
        or not transaction_raw.isdecimal()
        or int(transaction_raw) <= 0
    ):
        raise AxonPublishError(
            "AURELIUS_RUNTIME_TRANSACTION_VERSION_PIN must be a positive decimal integer"
        )
    transaction_pin = int(transaction_raw)
    if type(netuid) is not int or netuid < 0:
        raise AxonPublishError("guarded Axon netuid must be a non-negative integer")
    max_fee_rao = _max_fee_rao()
    endpoint = os.environ["SUBSTRATE_URL"]
    if runtime452_action_claim is not None and (
        netuid != runtime452_action_claim.netuid
        or external_ip != runtime452_action_claim.endpoint_ip
        or external_port != runtime452_action_claim.endpoint_port
        or endpoint != runtime452_action_claim.chain_endpoint
        or expected_chain != runtime452_action_claim.chain_name
        or expected_genesis != runtime452_action_claim.genesis_hash
        or runtime_pin != runtime452_action_claim.runtime_spec_version
        or transaction_pin != runtime452_action_claim.transaction_version
    ):
        raise AxonPublishError("runtime452 Axon guarded coordinates differ from action claim")
    journal_boundary()
    _require_sdk_endpoint(subtensor, endpoint)

    intent = bt.ServeAxon(
        netuid=netuid,
        ip=external_ip,
        port=external_port,
        protocol=4,
        version=1,
    )
    rpc = FinalizedRpc(endpoint)
    try:
        journal_boundary()
        initial = rpc.snapshot()
        observe_local_genesis = (
            expected_genesis == "observe-once-testlab"
            and os.environ.get("AURELIUS_ENV") == "testlab"
            and chain_id == "localnet"
        )
        if (
            not observe_local_genesis
            and re.fullmatch(r"0x[0-9a-fA-F]{64}", expected_genesis) is None
        ):
            raise AxonPublishError(
                "AURELIUS_AXON_EXPECTED_GENESIS_HASH must be an exact hash outside local testlab"
            )
        expected_genesis_hash = (
            initial.genesis_hash.lower() if observe_local_genesis else expected_genesis.lower()
        )
        _validate_finalized_snapshot(
            initial,
            expected_chain=expected_chain,
            expected_genesis=expected_genesis_hash,
            runtime_pin=runtime_pin,
        )
        if getattr(initial, "transaction_version", None) != transaction_pin:
            raise AxonPublishError("chain transaction version differs from explicit Axon authority")
    except Exception:
        rpc.close()
        raise
    identity = {
        "chain_id": chain_id,
        "chain_name": expected_chain,
        "genesis_hash": expected_genesis_hash,
        "runtime_spec_version": runtime_pin,
        "runtime_transaction_version": transaction_pin,
        "netuid": netuid,
        "hotkey_ss58": str(wallet.hotkey.ss58_address),
        "coldkey_ss58": str(wallet.coldkeypub.ss58_address),
        "axon": {
            "version": 1,
            "ip": external_ip,
            "port": external_port,
            "ip_type": ipaddress.ip_address(external_ip).version,
            "protocol": 4,
            "placeholder1": 0,
            "placeholder2": 0,
        },
        "sdk_policy": {
            "max_fee_tao_rao": max_fee_rao,
            "max_spend_tao_rao": 0,
            "allowed_netuids": [netuid],
            "allow_raw_calls": False,
        },
    }
    if runtime452_action_claim is not None:
        identity["runtime452_action_claim"] = {
            "action": runtime452_action_claim.action,
            "actor": runtime452_action_claim.actor,
            "claim_index": runtime452_action_claim.claim_index,
            "claim_sha256": runtime452_action_claim.claim_sha256,
            "generation_marker_sha256": runtime452_action_claim.generation_marker_sha256,
            "burn_handoff_sha256": runtime452_action_claim.burn_handoff_sha256,
        }

    def observe(*, retained_observation=None):
        journal_boundary()
        snapshot = _validate_finalized_snapshot(
            rpc.snapshot(),
            expected_chain=expected_chain,
            expected_genesis=expected_genesis_hash,
            runtime_pin=runtime_pin,
        )
        if getattr(snapshot, "transaction_version", None) != transaction_pin:
            raise AxonPublishError("chain transaction version differs from explicit Axon authority")
        block_number = snapshot.block_number
        block_hash = snapshot.head_hash
        if retained_observation is not None:
            if (
                type(retained_observation) is not dict
                or set(retained_observation)
                != {"matches", "finalized_head", "finalized_block_number", "actual"}
                or retained_observation.get("matches") is not True
                or type(retained_observation.get("finalized_block_number")) is not int
                or not 0 <= retained_observation["finalized_block_number"] <= snapshot.block_number
                or not isinstance(retained_observation.get("finalized_head"), str)
                or re.fullmatch(r"0x[0-9a-f]{64}", retained_observation["finalized_head"]) is None
            ):
                raise AxonPublishError("retained Axon endpoint observation is malformed")
            block_number = retained_observation["finalized_block_number"]
            block_hash = retained_observation["finalized_head"]
            runtime = rpc.call("state_getRuntimeVersion", [block_hash])
            if not isinstance(runtime, dict) or (
                runtime.get("specVersion"),
                runtime.get("transactionVersion"),
            ) != (runtime_pin, transaction_pin):
                raise AxonPublishError("retained Axon endpoint observation has another runtime")
        canonical_hash = rpc.call("chain_getBlockHash", [block_number])
        if not isinstance(canonical_hash, str) or canonical_hash.lower() != block_hash:
            raise AxonPublishError("Axon endpoint observation is not canonical finalized authority")
        _require_sdk_block_info(
            subtensor,
            endpoint=endpoint,
            block_number=block_number,
            block_hash=block_hash,
        )
        metagraph = subtensor.subnets.metagraph(netuid, block=block_number, commitments=False)
        if metagraph is None or getattr(metagraph, "netuid", None) != netuid:
            raise AxonPublishError(
                "typed metagraph is absent or differs from the Axon publication netuid"
            )
        neuron = metagraph.by_hotkey(identity["hotkey_ss58"])
        actual = None
        if neuron is not None:
            published_endpoint = parse_published_endpoint(neuron.axon)
            raw_axon = None
            raw_axons = getattr(metagraph, "raw", {}).get("axons")
            if isinstance(raw_axons, list) and 0 <= neuron.uid < len(raw_axons):
                raw_axon = raw_axons[neuron.uid]
            actual = {
                "hotkey_ss58": neuron.hotkey,
                "coldkey_ss58": neuron.coldkey,
                "version": raw_axon.get("version") if isinstance(raw_axon, dict) else None,
                "ip": published_endpoint.ip if published_endpoint else None,
                "port": published_endpoint.port if published_endpoint else None,
                "ip_type": raw_axon.get("ip_type") if isinstance(raw_axon, dict) else None,
                "protocol": raw_axon.get("protocol") if isinstance(raw_axon, dict) else None,
                "placeholder1": (
                    raw_axon.get("placeholder1") if isinstance(raw_axon, dict) else None
                ),
                "placeholder2": (
                    raw_axon.get("placeholder2") if isinstance(raw_axon, dict) else None
                ),
            }
        return {
            "matches": actual
            == {
                "hotkey_ss58": identity["hotkey_ss58"],
                "coldkey_ss58": identity["coldkey_ss58"],
                **identity["axon"],
            },
            "finalized_head": block_hash,
            "finalized_block_number": block_number,
            "actual": actual,
        }

    prepared = {}
    expected_call_args = {
        "netuid": netuid,
        "version": 1,
        "ip": int(ipaddress.ip_address(external_ip)),
        "port": external_port,
        "ip_type": ipaddress.ip_address(external_ip).version,
        "protocol": 4,
        "placeholder1": 0,
        "placeholder2": 0,
    }

    def plan():
        journal_boundary()
        policy = bt.Policy(
            max_fee_tao=bt.Balance.from_rao(max_fee_rao),
            max_spend_tao=bt.Balance.from_rao(0),
            allowed_netuids=[netuid],
            allow_raw_calls=False,
        )
        if (
            _balance_rao(
                policy.max_fee_tao,
                label="ServeAxon policy maximum fee",
                expected_netuid=0,
            )
            != max_fee_rao
            or _balance_rao(
                policy.max_spend_tao,
                label="ServeAxon policy maximum spend",
                expected_netuid=0,
            )
            != 0
            or policy.allowed_netuids != [netuid]
            or policy.allow_raw_calls is not False
        ):
            raise AxonPublishError("constructed ServeAxon Policy differs from authority")
        sdk_plan = subtensor.plan(intent, wallet, policy=policy)
        fee_rao = _balance_rao(
            getattr(sdk_plan, "fee", None),
            label="planned ServeAxon fee",
            expected_netuid=0,
        )
        if (
            getattr(sdk_plan, "ok", None) is not True
            or getattr(sdk_plan, "violations", None) not in ([], ())
            or getattr(sdk_plan, "op", None) != "serve_axon"
            or getattr(sdk_plan, "signer", None) != "hotkey"
            or getattr(sdk_plan, "signer_address", None) != identity["hotkey_ss58"]
            or fee_rao > max_fee_rao
        ):
            raise AxonPublishError("Bittensor v11 rejected the exact capped ServeAxon plan")
        call_data, call_spec_version, call_hash = _validate_planned_serve_axon_call(
            subtensor,
            getattr(sdk_plan, "call", None),
            runtime_pin=runtime_pin,
            expected_call_args=expected_call_args,
        )
        prepared["plan"] = sdk_plan
        return {
            "op": "serve_axon",
            "signer": "hotkey",
            "signer_address": identity["hotkey_ss58"],
            "fee_tao_rao": fee_rao,
            "summary": str(getattr(sdk_plan, "summary", "")),
            "effects": [str(value) for value in getattr(sdk_plan, "effects", ())],
            "warnings": [str(value) for value in getattr(sdk_plan, "warnings", ())],
            "call_data_sha256": hashlib.sha256(call_data).hexdigest(),
            "call_spec_version": call_spec_version,
            "call_hash": call_hash,
            "call": {
                "module": "SubtensorModule",
                "function": "serve_axon",
                "args": expected_call_args,
            },
        }

    expected_response_type = (
        f"{bt.ExtrinsicResult.__module__}.{bt.ExtrinsicResult.__qualname__}"
        if hasattr(bt, "ExtrinsicResult")
        else None
    )

    def validate_response(response, envelope):
        if expected_response_type is None or type(response) is not bt.ExtrinsicResult:
            raise AxonPublishError("ServeAxon returned an unexpected SDK response type")
        if envelope.get("response_type") != expected_response_type:
            raise AxonPublishError("ServeAxon SDK response type evidence is inconsistent")
        if envelope.get("fee_netuid") != 0 or type(envelope.get("fee_tao_rao")) is not int:
            raise AxonPublishError("ServeAxon SDK response lacks an exact TAO transaction fee")
        if not 0 <= envelope["fee_tao_rao"] <= max_fee_rao:
            raise AxonPublishError("ServeAxon SDK response fee exceeds the authorized cap")

    def verify_response(envelope, *, authority, signature):
        journal_boundary()
        if (
            envelope.get("schema_version") != 1
            or envelope.get("kind") != "bittensor_v11_sdk_response"
        ):
            raise AxonPublishError("durable ServeAxon SDK response has an unknown shape")
        if (
            expected_response_type is None
            or envelope.get("response_type") != expected_response_type
            or envelope.get("fee_netuid") != 0
            or (
                envelope.get("fee_tao_rao") is not None
                and (
                    type(envelope.get("fee_tao_rao")) is not int
                    or not 0 <= envelope["fee_tao_rao"] <= max_fee_rao
                )
            )
        ):
            return {
                "status": "unavailable_endpoint_state_only",
                "reason": "durable SDK response lacks exact v11 TAO-fee evidence",
            }
        block_hash = envelope.get("block_hash")
        extrinsic_id = envelope.get("extrinsic_id")
        match = re.fullmatch(r"([0-9]+)-([0-9]+)", str(extrinsic_id or ""))
        if re.fullmatch(r"0x[0-9a-fA-F]{64}", str(block_hash or "")) is None or match is None:
            return {
                "status": "unavailable_endpoint_state_only",
                "reason": "SDK response exposed no independently auditable block/extrinsic coordinates",
            }
        block_number, extrinsic_index = (int(value) for value in match.groups())
        snapshot = _validate_finalized_snapshot(
            rpc.snapshot(),
            expected_chain=expected_chain,
            expected_genesis=expected_genesis_hash,
            runtime_pin=runtime_pin,
        )
        if getattr(snapshot, "transaction_version", None) != transaction_pin:
            raise AxonPublishError("ServeAxon response finality has another transaction version")
        if snapshot.block_number < block_number:
            raise AxonPublishError("ServeAxon response block is not finalized")
        canonical_hash = rpc.call("chain_getBlockHash", [block_number])
        runtime = rpc.call("state_getRuntimeVersion", [block_hash])
        if (
            not isinstance(canonical_hash, str)
            or canonical_hash.lower() != block_hash.lower()
            or not isinstance(runtime, dict)
            or runtime.get("specVersion") != runtime_pin
            or runtime.get("transactionVersion") != transaction_pin
        ):
            raise AxonPublishError("ServeAxon response coordinates are not canonical authority")
        raw = _raw_extrinsics(rpc.call("chain_getBlock", [block_hash]))
        if not 0 <= extrinsic_index < len(raw):
            raise AxonPublishError("ServeAxon response extrinsic index is outside its raw block")
        extrinsic_hash = _raw_extrinsic_hash(raw[extrinsic_index])
        decoded_block = _require_sdk_block_info(
            subtensor,
            endpoint=endpoint,
            block_number=block_number,
            block_hash=block_hash,
            extrinsic_index=extrinsic_index,
            extrinsic_hash=extrinsic_hash,
        )
        decoded = decoded_block.extrinsics
        extrinsic = decoded[extrinsic_index]
        if not authority.scan_start_block <= block_number <= authority.scan_end_block:
            raise AxonPublishError("ServeAxon response lies outside its prepared mortal era")
        candidate = decoded_scan_candidate(
            extrinsic,
            raw[extrinsic_index],
            block_hash=block_hash,
            block_number=block_number,
            extrinsic_index=extrinsic_index,
            authority=authority,
            signature=signature,
        )
        if candidate is None:
            raise _AxonCoordinateMismatch(
                "ServeAxon response coordinate does not contain the durable prepared signature"
            )
        call = extrinsic.get("call") if isinstance(extrinsic, dict) else None
        _require_exact_serve_axon_decoded_call(
            call,
            expected_call_args=expected_call_args,
            label="finalized ServeAxon call",
        )
        if (
            extrinsic.get("address") != identity["hotkey_ss58"]
            or extrinsic.get("signature") is None
        ):
            raise AxonPublishError("ServeAxon finalized call differs from exact typed intent")
        records = _value(subtensor.query(bt.storage.System.Events, [], block=block_number))
        if not isinstance(records, list):
            raise AxonPublishError("ServeAxon finalized events are unavailable")
        decoded_events = [_event_parts(value) for value in records]
        applied = [
            value for value in decoded_events if value is not None and value[0] == extrinsic_index
        ]
        if any(module == "System" and name == "ExtrinsicFailed" for _, module, name, _ in applied):
            raise AxonPublishError("ServeAxon finalized with ExtrinsicFailed")
        success = [value for value in applied if value[1:3] == ("System", "ExtrinsicSuccess")]
        fees = [
            value for value in applied if value[1:3] == ("TransactionPayment", "TransactionFeePaid")
        ]
        if len(success) != 1 or len(fees) != 1:
            raise AxonPublishError("ServeAxon lacks unique success and transaction-fee evidence")
        sdk_fee_rao = envelope.get("fee_tao_rao")
        payer, actual_fee_rao, tip_rao = _validate_serve_axon_fee(
            fees[0][3],
            expected_payer=identity["coldkey_ss58"],
            maximum_fee_rao=max_fee_rao,
            sdk_fee_rao=sdk_fee_rao,
        )
        return {
            "status": "exact_finalized_extrinsic",
            "block_hash": block_hash.lower(),
            "block_number": block_number,
            "extrinsic_index": extrinsic_index,
            "extrinsic_hash": extrinsic_hash,
            "transaction_fee_payer_ss58": payer,
            "actual_fee_tao_rao": actual_fee_rao,
            "transaction_tip_tao_rao": tip_rao,
            "maximum_fee_tao_rao": max_fee_rao,
            "sdk_fee_tao_rao": sdk_fee_rao,
            "prepared_sha256": authority.prepared_sha256,
            "payload_sha256": authority.payload_sha256,
            "signature_sha256": hashlib.sha256(signature).hexdigest(),
        }

    finalized_path = state_root / "finalized.json"
    evidence_identity = {
        "operation": "serve_axon",
        "deployment_chain_id": chain_id,
        "target_netuid": netuid,
        "signer_address": identity["hotkey_ss58"],
    }

    def attempt_path(kind: str, attempt: int) -> Path:
        return state_root / f"{kind}-{attempt}.json"

    def require_plan_document(document, *, call_data_sha256):
        if (
            not isinstance(document, dict)
            or set(document)
            != {
                "op",
                "signer",
                "signer_address",
                "fee_tao_rao",
                "summary",
                "effects",
                "warnings",
                "call_data_sha256",
                "call_spec_version",
                "call_hash",
                "call",
            }
            or document.get("op") != "serve_axon"
            or document.get("signer") != "hotkey"
            or document.get("signer_address") != identity["hotkey_ss58"]
            or type(document.get("fee_tao_rao")) is not int
            or not 0 <= document["fee_tao_rao"] <= max_fee_rao
            or not isinstance(document.get("summary"), str)
            or not isinstance(document.get("effects"), list)
            or not all(isinstance(value, str) for value in document["effects"])
            or not isinstance(document.get("warnings"), list)
            or not all(isinstance(value, str) for value in document["warnings"])
            or document.get("call_data_sha256") != call_data_sha256
            or document.get("call_spec_version") != runtime_pin
            or re.fullmatch(r"0x[0-9a-f]{64}", str(document.get("call_hash", ""))) is None
            or document.get("call")
            != {
                "module": "SubtensorModule",
                "function": "serve_axon",
                "args": expected_call_args,
            }
        ):
            raise AxonPublishError("durable ServeAxon plan differs from exact typed authority")

    def validate_anchor(authority):
        journal_boundary()
        snapshot = _validate_finalized_snapshot(
            rpc.snapshot(),
            expected_chain=expected_chain,
            expected_genesis=expected_genesis_hash,
            runtime_pin=runtime_pin,
        )
        current_hash = rpc.call("chain_getBlockHash", [authority.era_current])
        birth_hash = rpc.call("chain_getBlockHash", [authority.era_birth])
        if (
            snapshot.block_number < authority.era_current
            or getattr(snapshot, "transaction_version", None) != transaction_pin
            or not isinstance(current_hash, str)
            or re.fullmatch(r"0x[0-9a-fA-F]{64}", current_hash) is None
            or not isinstance(birth_hash, str)
        ):
            raise AxonPublishError("prepared ServeAxon era lacks canonical finalized anchors")
        validate_era_chain_anchor(
            authority,
            current_finalized_block_number=authority.era_current,
            birth_block_hash=birth_hash,
        )
        return current_hash.lower()

    def load_authority(attempt):
        document = journal_read(attempt_path("authority", attempt))
        if (
            set(document)
            != {
                "schema_version",
                "status",
                "identity",
                "attempt",
                "sdk_plan",
                "prepared",
                "era_current_block_hash",
                "submit_policy",
            }
            or document.get("schema_version") != 1
            or document.get("status") != "prepared_before_signature"
            or document.get("identity") != identity
            or document.get("attempt") != attempt
            or not isinstance(document.get("prepared"), dict)
            or document.get("submit_policy")
            != {
                "maximum_fee_tao_rao": max_fee_rao,
                "maximum_spend_tao_rao": 0,
                "allowed_netuids": [netuid],
                "allow_raw_calls": True,
            }
        ):
            raise AxonPublishError("durable ServeAxon authority has an unaudited shape")
        unsigned_document, authority = validate_prepared_evidence(
            document["prepared"],
            identity=evidence_identity,
            attempt=attempt,
            expected_address=identity["hotkey_ss58"],
            expected_genesis_hash=expected_genesis_hash,
            expected_spec_version=runtime_pin,
            expected_transaction_version=transaction_pin,
            expected_period=_EXTRINSIC_PERIOD_BLOCKS,
            expected_tip=0,
        )
        require_plan_document(document.get("sdk_plan"), call_data_sha256=authority.call_data_sha256)
        current_hash = validate_anchor(authority)
        if (
            not isinstance(document.get("era_current_block_hash"), str)
            or re.fullmatch(r"0x[0-9a-fA-F]{64}", document["era_current_block_hash"]) is None
            or document["era_current_block_hash"].lower() != current_hash.lower()
        ):
            raise AxonPublishError("durable ServeAxon era-current canonical hash drifted")
        return document, unsigned_document, authority

    def create_authority(attempt):
        journal_boundary()
        sdk_plan_document = plan()
        sdk_plan = prepared.pop("plan", None)
        if sdk_plan is None:
            raise AxonPublishError("typed ServeAxon plan did not retain its exact call")
        try:
            unsigned = subtensor.prepare_call(
                sdk_plan.call,
                address=identity["hotkey_ss58"],
                crypto_type=1,
                period=_EXTRINSIC_PERIOD_BLOCKS,
                tip=0,
            )
            authority = validate_unsigned_dict(
                unsigned.to_dict(),
                expected_address=identity["hotkey_ss58"],
                expected_genesis_hash=expected_genesis_hash,
                expected_spec_version=runtime_pin,
                expected_transaction_version=transaction_pin,
                expected_period=_EXTRINSIC_PERIOD_BLOCKS,
                expected_tip=0,
                expected_call_data_sha256=sdk_plan_document["call_data_sha256"],
            )
            era_current_block_hash = validate_anchor(authority)
            prepared_document = prepared_evidence(
                unsigned.to_dict(), authority, identity=evidence_identity, attempt=attempt
            )
        except AxonPublishError:
            raise
        except Exception as exc:  # noqa: BLE001 - prepare is pre-sign and safely retryable
            raise AxonPublishError("Bittensor could not prepare exact ServeAxon call") from exc
        journal_claim(
            attempt_path("authority", attempt),
            {
                "schema_version": 1,
                "status": "prepared_before_signature",
                "identity": identity,
                "attempt": attempt,
                "sdk_plan": sdk_plan_document,
                "prepared": prepared_document,
                "era_current_block_hash": era_current_block_hash,
                "submit_policy": {
                    "maximum_fee_tao_rao": max_fee_rao,
                    "maximum_spend_tao_rao": 0,
                    "allowed_netuids": [netuid],
                    "allow_raw_calls": True,
                },
            },
        )
        return load_authority(attempt)

    def load_or_create_signature(attempt, authority):
        path = attempt_path("signature", attempt)
        if journal_exists(path):
            document = journal_read(path)
        else:
            journal_boundary()
            try:
                signature = bt.WalletSigner(wallet, role="hotkey").sign(authority.payload)
                document = signature_evidence(
                    signature, authority, identity=evidence_identity, attempt=attempt
                )
            except Exception as exc:  # noqa: BLE001 - no claim exists; exact prepared may resume
                raise AxonPublishError(
                    "ServeAxon exact prepared payload could not be signed"
                ) from exc
            journal_claim(path, document)
        signature = validate_signature_evidence(
            document, authority, identity=evidence_identity, attempt=attempt
        )
        return document, signature

    def raw_submit_policy():
        policy = bt.Policy(
            max_fee_tao=bt.Balance.from_rao(max_fee_rao),
            max_spend_tao=bt.Balance.from_rao(0),
            allowed_netuids=[netuid],
            allow_raw_calls=True,
        )
        if (
            _balance_rao(
                policy.max_fee_tao,
                label="raw ServeAxon policy maximum fee",
                expected_netuid=0,
            )
            != max_fee_rao
            or _balance_rao(
                policy.max_spend_tao,
                label="raw ServeAxon policy maximum spend",
                expected_netuid=0,
            )
            != 0
            or policy.allowed_netuids != [netuid]
            or policy.allow_raw_calls is not True
        ):
            raise AxonPublishError("raw ServeAxon policy differs from exact planned bounds")
        return policy

    def exact_scan(attempt, unsigned_document, authority, signature):
        journal_boundary()
        encoded_call = unsigned_document.get("call_data")
        if (
            not isinstance(encoded_call, str)
            or re.fullmatch(r"0x(?:[0-9a-fA-F]{2})+", encoded_call) is None
            or hashlib.sha256(bytes.fromhex(encoded_call[2:])).hexdigest()
            != authority.call_data_sha256
        ):
            raise AxonPublishError("durable ServeAxon scan call bytes differ from authority")
        snapshot = _validate_finalized_snapshot(
            rpc.snapshot(),
            expected_chain=expected_chain,
            expected_genesis=expected_genesis_hash,
            runtime_pin=runtime_pin,
        )
        if getattr(snapshot, "transaction_version", None) != transaction_pin:
            raise AxonPublishError("ServeAxon recovery finality has another transaction version")
        if snapshot.block_number <= authority.scan_end_block:
            raise _AxonScanPending(
                "ServeAxon submission remains ambiguous inside its mortal era; scan-only recovery"
            )
        canonical_latest = rpc.call("chain_getBlockHash", [snapshot.block_number])
        if not isinstance(canonical_latest, str) or canonical_latest.lower() != snapshot.head_hash:
            raise AxonPublishError("ServeAxon recovery latest finalized head is not canonical")
        path = attempt_path("scan-resolution", attempt)
        retained = None
        scan_head = snapshot.head_hash
        scan_height = snapshot.block_number
        if journal_exists(path):
            retained = validate_scan_resolution_evidence(
                journal_read(path),
                authority,
                identity=evidence_identity,
                attempt=attempt,
                signature_sha256=hashlib.sha256(signature).hexdigest(),
            )
            retained_hash = rpc.call("chain_getBlockHash", [retained["finalized_block_number"]])
            if (
                retained["finalized_block_number"] > snapshot.block_number
                or not isinstance(retained_hash, str)
                or retained_hash.lower() != retained["finalized_head"]
            ):
                raise AxonPublishError("retained ServeAxon scan baseline is not canonical finality")
            scan_head = retained["finalized_head"]
            scan_height = retained["finalized_block_number"]
        candidates = []
        for block_number in range(authority.scan_start_block, authority.scan_end_block + 1):
            block_hash = rpc.call("chain_getBlockHash", [block_number])
            runtime = rpc.call("state_getRuntimeVersion", [block_hash])
            if (
                not isinstance(block_hash, str)
                or re.fullmatch(r"0x[0-9a-fA-F]{64}", block_hash) is None
                or not isinstance(runtime, dict)
                or runtime.get("specVersion") != runtime_pin
                or runtime.get("transactionVersion") != transaction_pin
            ):
                raise AxonPublishError("ServeAxon recovery encountered non-canonical runtime state")
            raw = _raw_extrinsics(rpc.call("chain_getBlock", [block_hash]))
            decoded_block = _require_sdk_block_info(
                subtensor,
                endpoint=endpoint,
                block_number=block_number,
                block_hash=block_hash,
            )
            if len(decoded_block.extrinsics) != len(raw):
                raise AxonPublishError("ServeAxon recovery raw/decoded extrinsic counts differ")
            for extrinsic_index, (decoded, raw_extrinsic) in enumerate(
                zip(decoded_block.extrinsics, raw, strict=True)
            ):
                if not isinstance(decoded, dict):
                    raise AxonPublishError(
                        "ServeAxon recovery cannot prove zero through an undecodable extrinsic"
                    )
                candidate = decoded_scan_candidate(
                    decoded,
                    raw_extrinsic,
                    block_hash=block_hash,
                    block_number=block_number,
                    extrinsic_index=extrinsic_index,
                    authority=authority,
                    signature=signature,
                )
                if candidate is None:
                    continue
                causal = verify_response(
                    {
                        "schema_version": 1,
                        "kind": "bittensor_v11_sdk_response",
                        "response_type": expected_response_type,
                        "success": True,
                        "message": "recovered from exact prepared-era scan",
                        "block_hash": block_hash,
                        "extrinsic_id": f"{block_number}-{extrinsic_index}",
                        "fee_tao_rao": None,
                        "fee_netuid": 0,
                    },
                    authority=authority,
                    signature=signature,
                )
                if causal.get("status") != "exact_finalized_extrinsic":
                    raise AxonPublishError("ServeAxon candidate lacks exact finalized proof")
                candidates.append(candidate)
        resolution = scan_resolution_evidence(
            authority,
            identity=evidence_identity,
            attempt=attempt,
            signature_sha256=hashlib.sha256(signature).hexdigest(),
            finalized_head=scan_head,
            finalized_block_number=scan_height,
            candidates=candidates,
        )
        final_baseline_hash = rpc.call("chain_getBlockHash", [scan_height])
        if not isinstance(final_baseline_hash, str) or final_baseline_hash.lower() != scan_head:
            raise AxonPublishError("ServeAxon scan baseline changed while proving its exact era")
        if retained is not None:
            if retained != resolution:
                raise AxonPublishError("ServeAxon prepared-era scan replay differs")
            return retained
        journal_claim(path, resolution)
        return resolution

    def replay_finalized():
        """Re-prove a retained terminal graph without restamping or reauthorizing a write."""

        retained = journal_read(finalized_path)
        if (
            set(retained)
            != {
                "schema_version",
                "status",
                "identity",
                "observation",
                "submission_attempts",
                "prepared_attempt",
                "causal_extrinsic_evidence",
            }
            or retained.get("identity") != identity
            or type(retained.get("causal_extrinsic_evidence")) is not dict
        ):
            raise AxonPublishError("retained finalized Axon terminal contract is malformed")
        rebuilt = finalized_axon_evidence_document(
            status=retained["status"],
            identity=identity,
            observation=retained["observation"],
            submission_attempts=retained["submission_attempts"],
            prepared_attempt=retained["prepared_attempt"],
            causal_extrinsic_evidence=retained["causal_extrinsic_evidence"],
        )
        if rebuilt != retained:
            raise AxonPublishError("retained finalized Axon schema or lineage differs")
        if (
            observe().get("matches") is not True
            or observe(retained_observation=retained["observation"]) != retained["observation"]
        ):
            raise AxonPublishError("retained finalized Axon endpoint authority drifted")
        attempt = retained["prepared_attempt"]
        if retained["status"] == "already_exact_no_submission":
            if any(
                journal_exists(attempt_path(kind, index))
                for kind in _RUNTIME452_AXON_ATTEMPT_KINDS
                for index in (1, 2)
            ):
                raise AxonPublishError("already-exact Axon terminal has submitted lineage")
            return retained
        if runtime452_action_claim is not None and attempt != 1:
            raise AxonPublishError("runtime452 Axon terminal exceeds one signed extrinsic")
        _document, unsigned, authority = load_authority(attempt)
        signature = validate_signature_evidence(
            journal_read(attempt_path("signature", attempt)),
            authority,
            identity=evidence_identity,
            attempt=attempt,
        )
        validate_submission_claim_evidence(
            journal_read(attempt_path("submission-claim", attempt)),
            authority,
            identity=evidence_identity,
            attempt=attempt,
            signature_sha256=hashlib.sha256(signature).hexdigest(),
        )
        if retained["status"] == "finalized_exact":
            causal = verify_response(
                journal_read(attempt_path("sdk-result", attempt)),
                authority=authority,
                signature=signature,
            )
        else:
            if not journal_exists(attempt_path("scan-resolution", attempt)):
                raise AxonPublishError("recovered Axon terminal lacks its retained exact-era scan")
            resolution = exact_scan(attempt, unsigned, authority, signature)
            if resolution["status"] != "finalized_exact_match":
                raise AxonPublishError("recovered Axon terminal lacks one exact finalized match")
            candidate = resolution["candidates"][0]
            causal = verify_response(
                {
                    "schema_version": 1,
                    "kind": "bittensor_v11_sdk_response",
                    "response_type": expected_response_type,
                    "success": True,
                    "message": "recovered from exact prepared-era scan",
                    "block_hash": candidate["block_hash"],
                    "extrinsic_id": f"{candidate['block_number']}-{candidate['extrinsic_index']}",
                    "fee_tao_rao": None,
                    "fee_netuid": 0,
                },
                authority=authority,
                signature=signature,
            )
        if causal != retained["causal_extrinsic_evidence"]:
            raise AxonPublishError("retained finalized Axon causal authority drifted")
        return retained

    def finalize(*, status, attempt, causal_evidence, submission_attempts):
        if journal_exists(finalized_path):
            return replay_finalized()
        observation = observe()
        if observation.get("matches") is not True:
            raise AxonPublishError("finalized ServeAxon call did not produce the intended endpoint")
        document = finalized_axon_evidence_document(
            status=status,
            identity=identity,
            observation=observation,
            submission_attempts=submission_attempts,
            prepared_attempt=attempt,
            causal_extrinsic_evidence=causal_evidence,
        )
        journal_claim(finalized_path, document)
        return document

    def run_prepared_publish():
        if journal_exists(state_root / "intent.json"):
            raise AxonPublishError(
                "legacy execute-era Axon state is unsupported; discard dev state before SDK11 run"
            )
        if journal_exists(finalized_path):
            return replay_finalized()

        maximum_prepared_attempts = _axon_prepared_attempt_limit(
            runtime452_journal if runtime452_journal is not None else state_root,
            runtime452_action_claim,
        )

        has_authority = any(
            journal_exists(attempt_path("authority", attempt))
            for attempt in range(1, maximum_prepared_attempts + 1)
        )
        if not has_authority:
            observation = observe()
            if observation.get("matches") is True:
                return finalize(
                    status="already_exact_no_submission",
                    attempt=0,
                    causal_evidence={"status": "not_required"},
                    submission_attempts=0,
                )
            if os.environ.get("AURELIUS_ENV") == "testnet-full":
                authorize_testnet_axon_publication()

        def validate_prior_attempt_one():
            _prior_document, _prior_unsigned, prior_authority = load_authority(1)
            _validate_prior_axon_attempt_authority(
                authority=prior_authority,
                signature_document=journal_read(attempt_path("signature", 1)),
                submission_document=journal_read(attempt_path("submission-claim", 1)),
                resolution_document=journal_read(attempt_path("scan-resolution", 1)),
                identity=evidence_identity,
                prior_attempt=1,
                next_attempt=2,
            )
            return prior_authority

        for attempt in range(1, maximum_prepared_attempts + 1):
            authority_path = attempt_path("authority", attempt)
            if not journal_exists(authority_path):
                if attempt == 2:
                    validate_prior_attempt_one()
                _authority_document, unsigned_document, authority = create_authority(attempt)
            else:
                _authority_document, unsigned_document, authority = load_authority(attempt)
            if attempt == 2:
                prior_authority = validate_prior_attempt_one()
                validate_reprepared_attempt(prior_authority, authority)
            signature_document, signature = load_or_create_signature(attempt, authority)
            claim_path = attempt_path("submission-claim", attempt)
            result_path = attempt_path("sdk-result", attempt)
            claim_status = "already_exact"
            if not journal_exists(claim_path):
                claim = submission_claim_evidence(
                    authority,
                    identity=evidence_identity,
                    attempt=attempt,
                    signature_sha256=signature_document["signature_sha256"],
                )
                claim_status = journal_claim(claim_path, claim)
            if claim_status not in {"claimed_exact", "already_exact"}:
                raise AxonPublishError("ServeAxon submission claim has no exact winner status")
            validate_submission_claim_evidence(
                journal_read(claim_path),
                authority,
                identity=evidence_identity,
                attempt=attempt,
                signature_sha256=hashlib.sha256(signature).hexdigest(),
            )
            if claim_status == "claimed_exact":
                # Reconstruct only the exact durable unsigned bytes; there is no post-claim plan.
                _authority_document, unsigned_document, authority = load_authority(attempt)
                signature_document = journal_read(attempt_path("signature", attempt))
                signature = validate_signature_evidence(
                    signature_document,
                    authority,
                    identity=evidence_identity,
                    attempt=attempt,
                )
                validate_submission_claim_evidence(
                    journal_read(claim_path),
                    authority,
                    identity=evidence_identity,
                    attempt=attempt,
                    signature_sha256=hashlib.sha256(signature).hexdigest(),
                )
                unsigned = reconstruct_unsigned(bt, unsigned_document)
                journal_boundary()
                try:
                    response = subtensor.submit_signature(
                        unsigned,
                        signature,
                        policy=raw_submit_policy(),
                        wait_for_inclusion=True,
                        wait_for_finalization=True,
                    )
                except Exception as exc:  # noqa: BLE001 - durable claim makes this scan-only
                    raise AxonPublishError("ServeAxon prepared submission is ambiguous") from exc
                envelope = _sdk_response_envelope(response)
                journal_claim(result_path, envelope)
                validate_response(response, envelope)

            if journal_exists(result_path):
                envelope = journal_read(result_path)
                try:
                    causal = verify_response(
                        envelope,
                        authority=authority,
                        signature=signature,
                    )
                except _AxonCoordinateMismatch:
                    raise
                except Exception:  # noqa: BLE001 - durable claim makes this scan-only
                    causal = {"status": "requires_exact_prepared_era_scan"}
                if causal.get("status") == "exact_finalized_extrinsic":
                    return finalize(
                        status="finalized_exact",
                        attempt=attempt,
                        causal_evidence=causal,
                        submission_attempts=attempt,
                    )

            resolution = exact_scan(attempt, unsigned_document, authority, signature)
            if resolution["status"] == "finalized_exact_match":
                candidate = resolution["candidates"][0]
                causal = verify_response(
                    {
                        "schema_version": 1,
                        "kind": "bittensor_v11_sdk_response",
                        "response_type": expected_response_type,
                        "success": True,
                        "message": "recovered from exact prepared-era scan",
                        "block_hash": candidate["block_hash"],
                        "extrinsic_id": (
                            f"{candidate['block_number']}-{candidate['extrinsic_index']}"
                        ),
                        "fee_tao_rao": None,
                        "fee_netuid": 0,
                    },
                    authority=authority,
                    signature=signature,
                )
                return finalize(
                    status="recovered_from_prepared_era",
                    attempt=attempt,
                    causal_evidence=causal,
                    submission_attempts=attempt,
                )
            if resolution["status"] == "multiple_exact_matches_terminal":
                raise AxonPublishError("multiple exact prepared ServeAxon submissions finalized")
            if attempt == maximum_prepared_attempts:
                if runtime452_action_claim is not None:
                    raise AxonPublishError(
                        "runtime452 Axon claim permits no second signed extrinsic after exact "
                        "attempt-one no-match"
                    )
                raise AxonPublishError(
                    "both prepared ServeAxon attempts expired with exact no-match"
                )
        raise AxonPublishError("ServeAxon prepared lifecycle exhausted unexpectedly")

    try:
        if runtime452_journal is None:
            return run_prepared_publish()
        try:
            with runtime452_journal.locked():
                return run_prepared_publish()
        except Runtime452ActionJournalError as exc:
            raise AxonPublishError("runtime452 Axon journal lock or custody changed") from exc
    finally:
        rpc.close()


def _await_guarded_external_publish(
    publish,
    *,
    retry_delay_s: float = 1.0,
    sleeper=None,
):
    """Keep a serving process alive while its durable claimed attempt is scan-only.

    Re-entering ``publish`` is safe because the exclusive submission claim already exists:
    ``_guarded_external_publish`` can only re-verify the SDK coordinates or scan the exact
    prepared era. Every other error remains terminal and is deliberately not retried.
    """

    if (
        isinstance(retry_delay_s, bool)
        or not isinstance(retry_delay_s, (int, float))
        or not 0 < retry_delay_s <= 30
    ):
        raise ValueError("Axon publication recovery delay must be in (0, 30] seconds")
    sleep = time.sleep if sleeper is None else sleeper
    announced = False
    while True:
        try:
            result = publish()
        except _AxonScanPending:
            if not announced:
                print(
                    "[miner:btauth] durable Axon submission is scan-only; "
                    "waiting for finalized recovery without resubmission",
                    flush=True,
                )
                announced = True
            sleep(retry_delay_s)
            continue
        if announced:
            print(
                "[miner:btauth] durable Axon publication recovered from finalized evidence",
                flush=True,
            )
        return result


def _permitted_validators_from_metagraph(metagraph) -> frozenset:
    """Return the exact v11 typed validator set or fail closed on shape drift."""

    neurons = getattr(metagraph, "neurons", None)
    validators = getattr(metagraph, "validators", None)
    if not isinstance(neurons, list) or not isinstance(validators, list):
        raise ValueError("v11 metagraph neurons/validators are unavailable")
    expected = set()
    for neuron in neurons:
        permit = getattr(neuron, "validator_permit", None)
        if type(permit) is not bool:
            raise ValueError("v11 metagraph validator_permit entries must be boolean")
        if permit:
            hotkey = str(getattr(neuron, "hotkey", "") or "")
            if not hotkey or hotkey in expected:
                raise ValueError("validator-permit hotkeys are empty or non-unique")
            expected.add(hotkey)
    actual = [str(getattr(neuron, "hotkey", "") or "") for neuron in validators]
    if not all(actual) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("v11 metagraph validators projection is inconsistent")
    return frozenset(expected)


def advertised_port(bind_port: int, raw: str | None = None) -> int:
    """Resolve the independently advertised Axon port.

    Localnet keeps the historical same-port default. A NAT/profile deployment supplies the env
    explicitly; malformed or out-of-range values fail startup instead of publishing an endpoint
    different from the operator's intended mapping.
    """
    value = os.environ.get("AURELIUS_AXON_EXTERNAL_PORT") if raw is None else raw
    if value is None:
        return bind_port
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError("AURELIUS_AXON_EXTERNAL_PORT must be decimal digits")
    port = int(value)
    if not 1 <= port <= 65_535:
        raise ValueError("AURELIUS_AXON_EXTERNAL_PORT must be between 1 and 65535")
    return port


def serve_axon_forever(*, hotkey_name: str, netuid: int, port: int = 8091) -> None:
    validate_miner_startup_posture()
    selected_runtime452 = os.environ.get("AURELIUS_ENV") in RUNTIME452_PROFILE_IDS
    configured_external_ip = os.environ.get("AURELIUS_AXON_EXTERNAL_IP")
    if selected_runtime452 and not configured_external_ip:
        raise AxonPublishError(
            "runtime452 Axon publication requires an explicit external IP before wallet access"
        )
    if selected_runtime452:
        endpoint_policy = AxonEndpointPolicy.build(
            environment=os.environ.get("AURELIUS_ENV", "prod"),
            profile=os.environ.get("AURELIUS_AXON_ENDPOINT_PROFILE", "public"),
            allowed_cidrs=os.environ.get("AURELIUS_AXON_ALLOWED_CIDRS"),
        )
        external_ip = endpoint_policy.require(configured_external_ip)
        external_port = advertised_port(port)
        runtime452_action_claim = _runtime452_axon_claim_preflight(
            os.environ,
            hotkey_name=hotkey_name,
            netuid=netuid,
            external_ip=external_ip,
            external_port=external_port,
        )
    else:
        # Also reject leaked profile evidence before any local state is created. Endpoint
        # resolution stays in its historical location after SDK/wallet loading on other profiles.
        runtime452_action_claim = _runtime452_axon_claim_preflight(
            os.environ,
            hotkey_name=hotkey_name,
            netuid=netuid,
            external_ip="",
            external_port=0,
        )
        external_ip = ""
        external_port = 0
    # The exact profile does not create even its durable journal directory until the immutable
    # public claim has been joined and validated. Wallet loading and RPC construction follow.
    publish_state_dir = _prepare_publish_state_dir(
        os.environ["AURELIUS_AXON_PUBLISH_STATE_DIR"],
        runtime452_action_claim,
    )
    try:
        bt = require_bittensor_v11()
    except ImportError as e:
        raise NotImplementedError(
            "miner btauth mode needs the Bittensor SDK — install the optional `bt` extra "
            "(the chain-bt image); AURELIUS_TRANSPORT=http needs none of this"
        ) from e

    serving_config = ServingConfig()
    # Load once and bind BOTH btauth and application work-id signatures to this exact hotkey.
    wallet = load_bittensor_wallet(bt)
    _require_runtime452_axon_wallet(runtime452_action_claim, wallet)
    minter = minter_from_bittensor_wallet(
        hotkey_name, wallet, clock=os.environ.get("AURELIUS_CLOCK", "frozen")
    )
    registered = RegisteredSet()
    if not selected_runtime452:
        external_ip = configured_external_ip or socket.gethostbyname(socket.gethostname())
        endpoint_policy = AxonEndpointPolicy.build(
            environment=os.environ.get("AURELIUS_ENV", "prod"),
            profile=os.environ.get("AURELIUS_AXON_ENDPOINT_PROFILE", "public"),
            allowed_cidrs=os.environ.get("AURELIUS_AXON_ALLOWED_CIDRS"),
        )
        external_ip = endpoint_policy.require(external_ip)
        external_port = advertised_port(port)
    nonce_store = durable_btauth_nonce_store()
    endpoint = os.environ["SUBSTRATE_URL"]
    subtensor = None
    permit_rpc = None
    servers = []
    try:
        serving_config.start()
        subtensor = open_subtensor(bt, endpoint)
        # No listener exists yet. Chain identity, durable publish recovery/submission, and one
        # finalized permit view must all succeed before an attacker can occupy a request slot.
        try:
            publication = _await_guarded_external_publish(
                lambda: _guarded_external_publish(
                    bt,
                    wallet,
                    netuid,
                    subtensor,
                    publish_state_dir,
                    external_ip=external_ip,
                    external_port=external_port,
                    runtime452_action_claim=runtime452_action_claim,
                )
            )
        finally:
            if isinstance(publish_state_dir, Runtime452ActionJournalCustody):
                publish_state_dir.close()
        authority = publication.get("identity")
        if not isinstance(authority, dict):
            raise AxonPublishError("Axon publication omitted its explicit chain identity")
        permit_rpc = FinalizedRpc(endpoint)
        permit_snapshot = _refresh_finalized_permits(
            registered,
            subtensor,
            permit_rpc,
            netuid,
            endpoint=endpoint,
            expected_chain=authority["chain_name"],
            expected_genesis=authority["genesis_hash"],
            runtime_pin=authority["runtime_spec_version"],
        )
        try:
            sync_s = Decimal(os.environ.get("AURELIUS_MINER_METAGRAPH_SYNC_S", "30"))
        except InvalidOperation:
            raise ValueError("AURELIUS_MINER_METAGRAPH_SYNC_S must be decimal seconds") from None
        if not sync_s.is_finite() or not Decimal("1") <= sync_s <= Decimal("3600"):
            raise ValueError("AURELIUS_MINER_METAGRAPH_SYNC_S must be between 1 and 3600 seconds")

        print(
            f"[miner:btauth] guarded publication status={publication['status']} "
            f"finalized_block={publication['observation']['finalized_block_number']}",
            flush=True,
        )
        print(
            f"[miner:btauth] finalized validator permits synced at "
            f"{permit_snapshot.block_number}; opening serving sockets",
            flush=True,
        )

        def status():
            return 200, {
                "status": "ok",
                "hotkey": minter.hotkey_ss58,
                "axon": {"port": port, "served": True},
            }

        stats = ServingStats()
        limiter = caller_rate_limiter()
        http_port = int(os.environ.get("AURELIUS_MINER_HTTP_PORT", "8092") or 0)
        if http_port:
            query_enabled = plain_query_allowed()
            bind_host = "0.0.0.0" if query_enabled else "127.0.0.1"
            sidecar = make_server(
                minter,
                http_port,
                status=status,
                query_enabled=query_enabled,
                bind_host=bind_host,
                stats=stats,
                rate_limiter=limiter,
                serving_config=serving_config,
            )
            servers.append(sidecar)
            threading.Thread(target=sidecar.serve_forever, daemon=True).start()
            mode = "health + /v1/query" if query_enabled else "health-only"
            print(f"[miner:btauth] plain sidecar on {bind_host}:{http_port} ({mode})", flush=True)

        authorizer = make_btauth_authorizer(bt, wallet, registered, nonce_store=nonce_store)
        authenticated = make_server(
            minter,
            port,
            status=status,
            query_enabled=True,
            bind_host="0.0.0.0",
            request_authorizer=authorizer,
            stats=stats,
            rate_limiter=limiter,
            serving_config=serving_config,
        )
        servers.append(authenticated)
        threading.Thread(target=authenticated.serve_forever, daemon=True).start()
        print(f"[miner:btauth] authenticated HTTP bound on 0.0.0.0:{port}", flush=True)
        print(
            f"[miner:btauth] {minter.hotkey_ss58} published {external_ip}:{external_port} "
            f"(bind :{port}) on netuid {netuid}",
            flush=True,
        )
        start_status_line(stats, "[miner:btauth]")

        while True:
            time.sleep(float(sync_s))
            try:
                _refresh_finalized_permits(
                    registered,
                    subtensor,
                    permit_rpc,
                    netuid,
                    endpoint=endpoint,
                    expected_chain=authority["chain_name"],
                    expected_genesis=authority["genesis_hash"],
                    runtime_pin=authority["runtime_spec_version"],
                )
            except Exception as e:  # noqa: BLE001 — retain only the previously proven set
                print(f"[miner:btauth] finalized permit refresh failed; keeping last-good: {e}")
    finally:
        serving_config.close()
        for server in servers:
            server.shutdown()
            server.server_close()
        if permit_rpc is not None:
            permit_rpc.close()
        nonce_store.close()
        close_subtensor = getattr(subtensor, "close", None)
        if callable(close_subtensor):
            close_subtensor()
