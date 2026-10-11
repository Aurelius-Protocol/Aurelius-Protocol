"""Bittensor v11 btauth miner, validator-permit admission, and guarded publication."""

import hashlib
import json
import inspect
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from io import BytesIO
from types import SimpleNamespace

import pytest

from subnet.miner.axon_serve import (
    AxonPublishError,
    RegisteredSet,
    _AxonScanPending,
    _await_guarded_external_publish,
    _guarded_external_publish,
    _event_parts,
    _max_fee_rao,
    _prepare_publish_state_dir,
    _refresh_finalized_permits,
    _require_sdk_block_info,
    _strict_json_object,
    _validate_prior_axon_attempt_authority,
    _validate_planned_serve_axon_call,
    _validate_serve_axon_fee,
    advertised_port,
    authorize_testnet_axon_publication,
    blacklist_decision,
    build_envelope_json,
    guarded_publish_once,
    make_btauth_authorizer,
    _permitted_validators_from_metagraph,
    serve_axon_forever,
)
from subnet.miner.candidates import candidate_for_request
from subnet.miner.cli import _serve_mode
from subnet.miner.serve import (
    BoundedThreadingHTTPServer,
    MAX_HTTP_QUERY_BYTES,
    QueryAuthorizationError,
    ServingStats,
    WorkIdMinter,
    _handler,
    make_server,
    plain_query_allowed,
    serve as serve_plain_http,
    validate_miner_startup_posture,
)
from subnet.miner.staging import StagingPool
from subnet.axon_protocol import (
    CANDIDATE_SOURCE_STAGING,
    MAX_CONFIG_JSON_BYTES,
    PROTOCOL_VERSION,
)
from subnet.shared.signing import keypair_from_seed, sign
from subnet.shared.ss58 import ss58_encode
from subnet.operator.prepared_extrinsic import (
    BITTENSOR_V11_SIGNED_EXTENSIONS,
    PreparedExtrinsicError,
    scan_resolution_evidence,
    signature_evidence,
    submission_claim_evidence,
    validate_unsigned_dict,
)

pytestmark = pytest.mark.miner

_TASK_CONFIG = json.dumps({"payload": "axon-item"})
_VALIDATOR = WorkIdMinter("validator-hotkey").hotkey_ss58
_REQUEST_ID = "ab" * 32
_WINDOW = 7
_EXTERNAL_MINER_PUBLIC_KEY, _EXTERNAL_MINER_SECRET_KEY = keypair_from_seed(b"a" * 32)
_EXTERNAL_MINER_SS58 = ss58_encode(_EXTERNAL_MINER_PUBLIC_KEY)
_EXTERNAL_GENESIS = "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"
_EXTERNAL_ERA_BIRTH_HASH = "0x" + "cd" * 32
_EXTERNAL_ERA_CURRENT_HASH = "0x" + "ce" * 32
_EXTERNAL_RESULT_BLOCK_HASH = "0x" + "44" * 32


def _build(minter, config_json=_TASK_CONFIG, **overrides):
    values = {
        "request_id": _REQUEST_ID,
        "validator_hotkey_ss58": _VALIDATOR,
        "fanout_window": _WINDOW,
        "validator_version": "1.2.3",
    }
    values.update(overrides)
    return build_envelope_json(minter, "synthetic", 1, config_json, **values)


def test_serve_mode_env_seam(monkeypatch):
    monkeypatch.delenv("AURELIUS_TRANSPORT", raising=False)
    assert _serve_mode() == "http"  # hermetic default
    monkeypatch.setenv("AURELIUS_TRANSPORT", "btauth")
    assert _serve_mode() == "btauth"


def test_module_import_is_bittensor_free():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import subnet.miner.axon_serve; assert 'bittensor' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_btauth_mode_without_bt_is_a_clean_seam(monkeypatch, tmp_path):
    import subnet.miner.axon_serve as subject

    from subnet.miner.axon_serve import serve_axon_forever

    monkeypatch.setenv("AURELIUS_ENV", "testlab")
    monkeypatch.setenv("AURELIUS_TRANSPORT", "btauth")
    monkeypatch.setenv("AURELIUS_BITTENSOR_NONCE_DB", "/unused/nonces.sqlite3")
    monkeypatch.setenv("AURELIUS_AXON_PUBLISH_STATE_DIR", str(tmp_path / "axon-publish"))
    monkeypatch.setenv("AURELIUS_AXON_EXPECTED_CHAIN_NAME", "Development")
    monkeypatch.setenv("AURELIUS_AXON_EXPECTED_GENESIS_HASH", "observe-once-testlab")
    monkeypatch.setenv("AURELIUS_RUNTIME_SPEC_PIN", "432")
    monkeypatch.setenv("CHAIN_ID", "localnet")
    monkeypatch.setenv("SUBSTRATE_URL", "ws://subtensor:9944")
    monkeypatch.setattr(
        subject,
        "require_bittensor_v11",
        lambda: (_ for _ in ()).throw(ImportError("not installed")),
    )
    with pytest.raises(NotImplementedError, match="Bittensor"):
        serve_axon_forever(hotkey_name="miner-hotkey", netuid=2)


def test_blacklist_posture_pinned():
    permitted = frozenset({"5Validator"})
    # normal operation: validator-permit callers pass; other registered actors are refused
    assert blacklist_decision(permitted, "5Validator") == (False, "ok")
    denied, reason = blacklist_decision(permitted, "5RegisteredMiner")
    assert denied and "validator permit" in reason
    # a request with no authenticated hotkey is ALWAYS refused
    assert blacklist_decision(permitted, None)[0] is True
    assert blacklist_decision(None, None)[0] is True
    # NEVER-synced fails closed: candidate bytes are not public before chain authorization exists.
    denied, reason = blacklist_decision(None, "5Stranger")
    assert denied is True
    assert "never synced" in reason


def test_registered_set_starts_never_synced():
    assert RegisteredSet().hotkeys is None


def test_metagraph_admission_extracts_only_validator_permit_hotkeys():
    neurons = [
        SimpleNamespace(hotkey="5ValidatorA", validator_permit=True),
        SimpleNamespace(hotkey="5Miner", validator_permit=False),
        SimpleNamespace(hotkey="5ValidatorB", validator_permit=True),
    ]
    metagraph = SimpleNamespace(neurons=neurons, validators=[neurons[0], neurons[2]])
    assert _permitted_validators_from_metagraph(metagraph) == frozenset(
        {"5ValidatorA", "5ValidatorB"}
    )

    metagraph.validators = None
    with pytest.raises(ValueError, match="neurons/validators"):
        _permitted_validators_from_metagraph(metagraph)

    metagraph.validators = [neurons[0], neurons[2]]
    neurons[0].validator_permit = 1
    with pytest.raises(ValueError, match="boolean"):
        _permitted_validators_from_metagraph(metagraph)


def test_guarded_axon_publish_skips_an_already_exact_finalized_tuple(tmp_path):
    calls = {"publish": 0}

    def publish():
        calls["publish"] += 1
        raise AssertionError("an exact finalized tuple must not be submitted again")

    result = guarded_publish_once(
        tmp_path,
        {"hotkey": "5Miner", "ip": "172.30.55.10", "port": 8090},
        lambda: {"matches": True, "finalized_block_number": 100},
        publish,
    )
    assert result["status"] == "already_exact_no_submission"
    assert result["submission_attempts"] == 0
    assert calls["publish"] == 0

    replay = guarded_publish_once(
        tmp_path,
        {"hotkey": "5Miner", "ip": "172.30.55.10", "port": 8090},
        lambda: {"matches": True, "finalized_block_number": 101},
        publish,
    )
    assert replay["status"] == "finalized_replay"
    assert calls["publish"] == 0


def test_guarded_axon_publish_never_retries_an_ambiguous_intent(tmp_path):
    observations = iter(
        (
            {"matches": False, "finalized_block_number": 100},
            {"matches": False, "finalized_block_number": 101},
            {"matches": False, "finalized_block_number": 102},
        )
    )
    calls = {"publish": 0}

    def publish():
        calls["publish"] += 1
        raise TimeoutError("response lost after possible submission")

    identity = {"hotkey": "5Miner", "ip": "172.30.55.10", "port": 8090}
    with pytest.raises(TimeoutError):
        guarded_publish_once(tmp_path, identity, lambda: next(observations), publish)
    assert calls["publish"] == 1
    assert (tmp_path / "intent.json").exists()
    assert not (tmp_path / "finalized.json").exists()

    with pytest.raises(AxonPublishError, match="refusing an ambiguous retry"):
        guarded_publish_once(tmp_path, identity, lambda: next(observations), publish)
    assert calls["publish"] == 1


def test_scan_pending_publish_waits_and_reenters_only_the_durable_guard():
    terminal = {"status": "finalized_exact"}
    outcomes = iter(
        (
            _AxonScanPending("finality has not caught up"),
            _AxonScanPending("prepared era remains live"),
            terminal,
        )
    )
    calls = 0
    delays = []

    def publish():
        nonlocal calls
        calls += 1
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert (
        _await_guarded_external_publish(
            publish,
            retry_delay_s=0.25,
            sleeper=delays.append,
        )
        is terminal
    )
    assert calls == 3
    assert delays == [0.25, 0.25]


def test_scan_pending_publish_does_not_retry_other_failures():
    calls = 0
    delays = []

    def publish():
        nonlocal calls
        calls += 1
        raise AxonPublishError("terminal authority drift")

    with pytest.raises(AxonPublishError, match="terminal authority drift"):
        _await_guarded_external_publish(publish, sleeper=delays.append)
    assert calls == 1
    assert delays == []


def test_guarded_axon_publish_recovers_only_from_exact_finalized_readback(tmp_path):
    identity = {"hotkey": "5Miner", "ip": "172.30.55.10", "port": 8090}
    observations = iter(
        (
            {"matches": False, "finalized_block_number": 100},
            {"matches": True, "finalized_block_number": 105},
        )
    )

    def ambiguous():
        raise TimeoutError("response lost")

    with pytest.raises(TimeoutError):
        guarded_publish_once(tmp_path, identity, lambda: next(observations), ambiguous)

    recovered = guarded_publish_once(
        tmp_path,
        identity,
        lambda: next(observations),
        lambda: pytest.fail("recovery must not submit"),
    )
    assert recovered["status"] == "recovered_from_finalized_chain"
    assert recovered["submission_attempts"] == 1


def test_guarded_axon_publish_records_one_successful_submission(tmp_path):
    class Response:
        success = True

    observations = iter(
        (
            {"matches": False, "finalized_block_number": 100},
            {"matches": True, "finalized_block_number": 103},
        )
    )
    calls = {"publish": 0}

    def publish():
        calls["publish"] += 1
        return Response()

    result = guarded_publish_once(
        tmp_path,
        {"hotkey": "5Miner", "ip": "172.30.55.10", "port": 8090},
        lambda: next(observations),
        publish,
    )
    assert result["status"] == "finalized_exact"
    assert result["submission_attempts"] == 1
    assert calls["publish"] == 1


def test_guarded_axon_publish_polls_finalized_readback_without_resubmitting(tmp_path):
    class Response:
        success = True

    observations = iter(
        (
            {"matches": False, "finalized_block_number": 100},
            {"matches": False, "finalized_block_number": 101},
            {"matches": False, "finalized_block_number": 102},
            {"matches": True, "finalized_block_number": 103},
        )
    )
    calls = {"publish": 0}
    delays = []

    def publish():
        calls["publish"] += 1
        return Response()

    result = guarded_publish_once(
        tmp_path,
        {"hotkey": "5Miner", "ip": "172.30.55.10", "port": 8090},
        lambda: next(observations),
        publish,
        post_submit_observe_attempts=3,
        post_submit_observe_delay=0.25,
        sleeper=delays.append,
    )
    assert result["status"] == "finalized_exact"
    assert result["observation"]["finalized_block_number"] == 103
    assert calls["publish"] == 1
    assert delays == [0.25, 0.25]


def test_guarded_axon_publish_poll_exhaustion_remains_non_retryable(tmp_path):
    class Response:
        success = True

    identity = {"hotkey": "5Miner", "ip": "172.30.55.10", "port": 8090}
    observations = iter(
        (
            {"matches": False, "finalized_block_number": 100},
            {"matches": False, "finalized_block_number": 101},
            {"matches": False, "finalized_block_number": 102},
        )
    )
    calls = {"publish": 0}

    def publish():
        calls["publish"] += 1
        return Response()

    with pytest.raises(AxonPublishError, match="finalized read-back differs"):
        guarded_publish_once(
            tmp_path,
            identity,
            lambda: next(observations),
            publish,
            post_submit_observe_attempts=2,
        )
    assert calls["publish"] == 1
    assert (tmp_path / "intent.json").exists()
    assert not (tmp_path / "finalized.json").exists()

    with pytest.raises(AxonPublishError, match="refusing an ambiguous retry"):
        guarded_publish_once(
            tmp_path,
            identity,
            lambda: {"matches": False, "finalized_block_number": 103},
            lambda: pytest.fail("an unresolved intent must never resubmit"),
        )
    recovered = guarded_publish_once(
        tmp_path,
        identity,
        lambda: {"matches": True, "finalized_block_number": 104},
        lambda: pytest.fail("exact recovery must never resubmit"),
    )
    assert recovered["status"] == "recovered_from_finalized_chain"
    assert calls["publish"] == 1


def test_testnet_axon_ack_is_exact_and_checked_before_intent(tmp_path):
    expected = "I_ACKNOWLEDGE_UP_TO_FOUR_SN455_AXON_ENDPOINT_UPDATES"
    authorize_testnet_axon_publication(expected)
    for invalid in (None, "", expected + " ", "I_ACKNOWLEDGE_SN455_AXON_ENDPOINT_UPDATES"):
        with pytest.raises(AxonPublishError, match="acknowledgement"):
            authorize_testnet_axon_publication(invalid)

    with pytest.raises(AxonPublishError, match="acknowledgement"):
        guarded_publish_once(
            tmp_path,
            {"hotkey": "5Miner", "ip": "172.30.55.10", "port": 8090},
            lambda: {"matches": False, "finalized_block_number": 100},
            lambda: pytest.fail("missing acknowledgement must not publish"),
            authorize_publish=lambda: authorize_testnet_axon_publication("wrong"),
        )
    assert not (tmp_path / "intent.json").exists()


def test_already_exact_axon_needs_no_mutation_acknowledgement(tmp_path):
    result = guarded_publish_once(
        tmp_path,
        {"hotkey": "5Miner", "ip": "172.30.55.10", "port": 8090},
        lambda: {"matches": True, "finalized_block_number": 100},
        lambda: pytest.fail("an exact tuple must not publish"),
        authorize_publish=lambda: pytest.fail("a no-write read-back must not request authority"),
    )
    assert result["status"] == "already_exact_no_submission"


def test_publish_evidence_rejects_same_size_in_place_mutation_while_reading(tmp_path, monkeypatch):
    from subnet.operator import evidence_reader

    path = tmp_path / "finalized.json"
    original = b'{"generation":"aaaaaaaa"}\n'
    replacement = b'{"generation":"bbbbbbbb"}\n'
    assert len(original) == len(replacement)
    path.write_bytes(original)
    path.chmod(0o600)
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    original_read = evidence_reader.os.read
    mutated = False

    def mutate_then_read(descriptor, count):
        nonlocal mutated
        if not mutated:
            mutated = True
            with path.open("r+b") as stream:
                stream.write(replacement)
                stream.flush()
                os.fsync(stream.fileno())
        return original_read(descriptor, count)

    monkeypatch.setattr(evidence_reader.os, "read", mutate_then_read)

    with pytest.raises(AxonPublishError, match="unreadable publish evidence"):
        _strict_json_object(path)
    assert path.read_bytes() == replacement


def test_publish_state_dir_is_runtime_owned_mode_0700_before_probe(tmp_path):
    path = tmp_path / "axon-publish"

    assert _prepare_publish_state_dir(str(path)) == path
    info = path.stat()
    assert info.st_uid == os.geteuid()
    assert info.st_mode & 0o777 == 0o700
    assert list(path.iterdir()) == []

    path.chmod(0o755)
    with pytest.raises(AxonPublishError, match="runtime-owned with mode 0700"):
        _prepare_publish_state_dir(str(path))
    assert list(path.iterdir()) == []


def test_publish_state_dir_rejects_symlink_and_path_swap_during_probe(tmp_path, monkeypatch):
    import subnet.miner.axon_serve as subject

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(AxonPublishError, match="unavailable or unsafe"):
        _prepare_publish_state_dir(str(alias))

    path = tmp_path / "raced"
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    original_open = subject.os.open
    swapped = False

    def swap_before_probe(file, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if isinstance(file, str) and file.startswith(".startup-probe-") and not swapped:
            swapped = True
            path.rename(tmp_path / "opened-directory")
            replacement.rename(path)
        return original_open(file, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(subject.os, "open", swap_before_probe)
    with pytest.raises(AxonPublishError, match="changed during validation"):
        _prepare_publish_state_dir(str(path))


def _install_external_publish_rpc(monkeypatch, *, finalized_block_number=123):
    import subnet.miner.axon_serve as subject

    instances = []
    pre_finality_head = "0x" + "aa" * 32

    class Rpc:
        def __init__(self, endpoint):
            assert endpoint == "wss://test.example:443"
            self.closed = False
            instances.append(self)

        def snapshot(self):
            block_number = (
                finalized_block_number()
                if callable(finalized_block_number)
                else finalized_block_number
            )
            head_hash = "0x" + "ab" * 32 if block_number == 123 else pre_finality_head
            return SimpleNamespace(
                head_hash=head_hash,
                block_number=block_number,
                genesis_hash=_EXTERNAL_GENESIS,
                chain="Bittensor",
                runtime_spec_version=432,
                transaction_version=1,
            )

        def call(self, method, params):
            values = {
                ("chain_getBlockHash", (123,)): "0x" + "ab" * 32,
                ("chain_getBlockHash", (110,)): pre_finality_head,
                ("chain_getBlockHash", (64,)): _EXTERNAL_ERA_BIRTH_HASH,
                ("chain_getBlockHash", (100,)): _EXTERNAL_ERA_CURRENT_HASH,
                ("chain_getBlockHash", (120,)): _EXTERNAL_RESULT_BLOCK_HASH,
                ("state_getRuntimeVersion", (_EXTERNAL_RESULT_BLOCK_HASH,)): {
                    "specVersion": 432,
                    "transactionVersion": 1,
                },
                ("chain_getBlock", (_EXTERNAL_RESULT_BLOCK_HASH,)): {
                    "block": {"extrinsics": _external_raw_extrinsics()}
                },
            }
            return values[(method, tuple(params))]

        def close(self):
            self.closed = True

    monkeypatch.setattr(subject, "FinalizedRpc", Rpc)
    monkeypatch.setenv(
        "CHAIN_ID",
        f"bittensor-testnet:{_EXTERNAL_GENESIS[2:]}",
    )
    monkeypatch.setenv("AURELIUS_RUNTIME_SPEC_PIN", "432")
    monkeypatch.setenv("AURELIUS_RUNTIME_TRANSACTION_VERSION_PIN", "1")
    monkeypatch.setenv("AURELIUS_AXON_EXPECTED_CHAIN_NAME", "Bittensor")
    monkeypatch.setenv(
        "AURELIUS_AXON_EXPECTED_GENESIS_HASH",
        _EXTERNAL_GENESIS,
    )
    monkeypatch.setenv("AURELIUS_ENV", "testnet-full")
    monkeypatch.setenv("SUBSTRATE_URL", "wss://test.example:443")
    return instances


def _external_publish_metagraph(*, endpoint):
    neuron = SimpleNamespace(
        uid=0,
        hotkey=_EXTERNAL_MINER_SS58,
        coldkey="5Coldkey",
        axon=endpoint,
    )
    raw_axon = {
        "version": 1,
        "ip_type": 4,
        "protocol": 4,
        "placeholder1": 0,
        "placeholder2": 0,
    }
    return SimpleNamespace(
        netuid=455,
        raw={"axons": [raw_axon]},
        by_hotkey=lambda hotkey: neuron if hotkey == neuron.hotkey else None,
    )


def _external_publish_plan_call():
    data = b"\x01\x02"
    return SimpleNamespace(
        data=data,
        spec_version=432,
        call_hash=hashlib.blake2b(data, digest_size=32).digest(),
    )


def _external_unsigned():
    call_data = b"\x01\x02"
    included = bytes.fromhex("4502") + bytes((7 << 2, 0, 0))
    signed = (
        (432).to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + bytes.fromhex(_EXTERNAL_GENESIS[2:])
        + bytes.fromhex(_EXTERNAL_ERA_CURRENT_HASH[2:])
        + b"\x00"
    )
    payload = call_data + included + signed
    return {
        "call_data": "0x" + call_data.hex(),
        "address": _EXTERNAL_MINER_SS58,
        "public_key": "0x" + _EXTERNAL_MINER_PUBLIC_KEY.hex(),
        "crypto_type": 1,
        "era": {"period": 64, "current": 100},
        "nonce": 7,
        "tip": 0,
        "tip_asset_id": None,
        "genesis_hash": _EXTERNAL_GENESIS,
        "era_block_hash": _EXTERNAL_ERA_CURRENT_HASH,
        "spec_version": 432,
        "transaction_version": 1,
        "metadata_hash": None,
        "payload": "0x" + payload.hex(),
        "payload_json": {
            "address": _EXTERNAL_MINER_SS58,
            "blockHash": _EXTERNAL_ERA_CURRENT_HASH,
            "genesisHash": _EXTERNAL_GENESIS,
            "method": "0x" + call_data.hex(),
            "nonce": "0x00000007",
            "specVersion": "0x000001b0",
            "tip": "0x" + "00" * 16,
            "transactionVersion": "0x00000001",
            "era": "0x4502",
            "version": 4,
            "signedExtensions": list(BITTENSOR_V11_SIGNED_EXTENSIONS),
            "mode": 0,
            "metadataHash": None,
        },
        "included_in_extrinsic": "0x" + included.hex(),
        "included_in_signed_data": "0x" + signed.hex(),
    }


class _ExternalUnsignedExtrinsic:
    def __init__(self, value):
        self.value = value

    @classmethod
    def from_dict(cls, value):
        return cls(value)

    def to_dict(self):
        return self.value


class _ExternalWalletSigner:
    last_signature = None

    def __init__(self, wallet, *, role):
        assert wallet.hotkey.ss58_address == _EXTERNAL_MINER_SS58
        assert role == "hotkey"

    def sign(self, payload):
        signature = sign(payload, (_EXTERNAL_MINER_PUBLIC_KEY, _EXTERNAL_MINER_SECRET_KEY))
        type(self).last_signature = signature
        return signature


def _external_raw(body: bytes) -> str:
    assert 0 < len(body) < 64
    return "0x" + bytes((len(body) << 2,)).hex() + body.hex()


def _external_raw_extrinsics():
    return [_external_raw(b"\x00"), _external_raw(b"\x01\x02")]


def _external_raw_hash(raw: str) -> str:
    return "0x" + hashlib.blake2b(bytes.fromhex(raw[2:]), digest_size=32).hexdigest()


def _external_prepared_signature() -> bytes:
    return sign(
        bytes.fromhex(_external_unsigned()["payload"][2:]),
        (_EXTERNAL_MINER_PUBLIC_KEY, _EXTERNAL_MINER_SECRET_KEY),
    )


def test_axon_attempt_two_requires_attempt_one_submission_claim_authority():
    identity = {
        "operation": "serve_axon",
        "deployment_chain_id": f"bittensor-testnet:{_EXTERNAL_GENESIS[2:]}",
        "target_netuid": 455,
        "signer_address": _EXTERNAL_MINER_SS58,
    }
    authority = validate_unsigned_dict(
        _external_unsigned(),
        expected_address=_EXTERNAL_MINER_SS58,
        expected_genesis_hash=_EXTERNAL_GENESIS,
        expected_spec_version=432,
        expected_transaction_version=1,
        expected_period=64,
        expected_tip=0,
        expected_call_data_sha256=hashlib.sha256(b"\x01\x02").hexdigest(),
    )
    signature = _external_prepared_signature()
    signature_document = signature_evidence(
        signature,
        authority,
        identity=identity,
        attempt=1,
    )
    submission_document = submission_claim_evidence(
        authority,
        identity=identity,
        attempt=1,
        signature_sha256=signature_document["signature_sha256"],
    )
    resolution_document = scan_resolution_evidence(
        authority,
        identity=identity,
        attempt=1,
        signature_sha256=signature_document["signature_sha256"],
        finalized_head="0x" + "ef" * 32,
        finalized_block_number=authority.scan_end_block + 1,
        candidates=[],
    )

    _validate_prior_axon_attempt_authority(
        authority=authority,
        signature_document=signature_document,
        submission_document=submission_document,
        resolution_document=resolution_document,
        identity=identity,
        prior_attempt=1,
        next_attempt=2,
    )

    with pytest.raises(PreparedExtrinsicError, match="submission claim"):
        _validate_prior_axon_attempt_authority(
            authority=authority,
            signature_document=signature_document,
            submission_document={},
            resolution_document=resolution_document,
            identity=identity,
            prior_attempt=1,
            next_attempt=2,
        )


def _external_decoded_extrinsics(*, signature: bytes | None = None):
    raw = _external_raw_extrinsics()
    signature = (
        _ExternalWalletSigner.last_signature or _external_prepared_signature()
        if signature is None
        else signature
    )
    return [
        {"extrinsic_hash": _external_raw_hash(raw[0])},
        {
            "extrinsic_hash": _external_raw_hash(raw[1]),
            "extrinsic_length": len(bytes.fromhex(raw[1][2:])) - 1,
            "address": _EXTERNAL_MINER_SS58,
            "signature": {"Sr25519": "0x" + signature.hex()},
            "era": [64, 36],
            "nonce": 7,
            "tip": 0,
            "mode": {"mode": "Disabled"},
            "call": {
                "call_hash": "0x" + hashlib.blake2b(b"\x01\x02", digest_size=32).hexdigest(),
                "call_index": "0x0704",
                "call_module": "SubtensorModule",
                "call_function": "serve_axon",
                "call_args": [
                    {"name": "netuid", "type": "NetUid", "value": 455},
                    {"name": "version", "type": "u32", "value": 1},
                    {"name": "ip", "type": "u128", "value": 2_887_661_322},
                    {"name": "port", "type": "u16", "value": 8090},
                    {"name": "ip_type", "type": "u8", "value": 4},
                    {"name": "protocol", "type": "u8", "value": 4},
                    {"name": "placeholder1", "type": "u8", "value": 0},
                    {"name": "placeholder2", "type": "u8", "value": 0},
                ],
            },
        },
    ]


def _external_expected_axon_call_args():
    return {
        "netuid": 455,
        "version": 1,
        "ip": 2_887_661_322,
        "port": 8090,
        "ip_type": 4,
        "protocol": 4,
        "placeholder1": 0,
        "placeholder2": 0,
    }


@pytest.mark.parametrize("runtime_pin", [432, 450, 452, 453, 458])
def test_planned_serve_axon_accepts_only_exact_sdk11_callbytes_decode(runtime_pin):
    decoded = _external_decoded_extrinsics()[1]["call"]

    class Subtensor:
        def decode_scale(self, type_string, data):
            assert (type_string, data) == ("Call", b"\x01\x02")
            return decoded

    plan_call = _external_publish_plan_call()
    plan_call.spec_version = runtime_pin
    data, spec_version, call_hash = _validate_planned_serve_axon_call(
        Subtensor(),
        plan_call,
        runtime_pin=runtime_pin,
        expected_call_args=_external_expected_axon_call_args(),
    )

    assert data == b"\x01\x02"
    assert spec_version == runtime_pin
    assert call_hash == "0x" + hashlib.blake2b(data, digest_size=32).hexdigest()


def test_planned_serve_axon_rejects_unreviewed_runtime_even_when_layout_matches():
    class Subtensor:
        def decode_scale(self, _type_string, _data):
            return _external_decoded_extrinsics()[1]["call"]

    plan_call = _external_publish_plan_call()
    plan_call.spec_version = 451
    with pytest.raises(AxonPublishError, match="no reviewed exact call schema"):
        _validate_planned_serve_axon_call(
            Subtensor(),
            plan_call,
            runtime_pin=451,
            expected_call_args=_external_expected_axon_call_args(),
        )


@pytest.mark.parametrize(
    "fault",
    [
        "spec-version",
        "call-index",
        "module",
        "argument-order",
        "argument-type",
        "argument-bool",
        "argument-value",
        "call-hash",
        "sdk-call-hash",
        "extra-field",
    ],
)
def test_planned_serve_axon_rejects_sdk11_callbytes_decode_drift(fault):
    decoded = json.loads(json.dumps(_external_decoded_extrinsics()[1]["call"]))
    spec_version = 432
    sdk_call_hash = hashlib.blake2b(b"\x01\x02", digest_size=32).digest()
    if fault == "spec-version":
        spec_version = 433
    elif fault == "call-index":
        decoded["call_index"] = "0x0705"
    elif fault == "module":
        decoded["call_module"] = "Balances"
    elif fault == "argument-order":
        decoded["call_args"][0], decoded["call_args"][1] = (
            decoded["call_args"][1],
            decoded["call_args"][0],
        )
    elif fault == "argument-type":
        decoded["call_args"][0]["type"] = "u16"
    elif fault == "argument-bool":
        decoded["call_args"][1]["value"] = True
    elif fault == "argument-value":
        decoded["call_args"][0]["value"] = 456
    elif fault == "call-hash":
        decoded["call_hash"] = "0x" + "00" * 32
    elif fault == "sdk-call-hash":
        sdk_call_hash = b"\x00" * 32
    else:
        decoded["unexpected"] = True

    class Subtensor:
        def decode_scale(self, _type_string, _data):
            return decoded

    with pytest.raises(AxonPublishError):
        _validate_planned_serve_axon_call(
            Subtensor(),
            SimpleNamespace(
                data=b"\x01\x02",
                spec_version=spec_version,
                call_hash=sdk_call_hash,
            ),
            runtime_pin=432,
            expected_call_args=_external_expected_axon_call_args(),
        )


def _external_fee_events(*, payer="5Coldkey"):
    return SimpleNamespace(
        value=[
            {
                "phase": "Initialization",
                "extrinsic_idx": None,
                "event": {
                    "module_id": "Balances",
                    "event_id": "Issued",
                    "attributes": {"amount": 1_000_000_000},
                },
            },
            {
                "extrinsic_idx": 1,
                "event": {
                    "module_id": "TransactionPayment",
                    "event_id": "TransactionFeePaid",
                    "attributes": {
                        "who": payer,
                        "actual_fee": 2_000_000,
                        "tip": 0,
                    },
                },
            },
            {
                "extrinsic_idx": 1,
                "event": {
                    "module_id": "System",
                    "event_id": "ExtrinsicSuccess",
                    "attributes": {},
                },
            },
        ]
    )


def test_external_publish_event_parser_ignores_only_known_non_extrinsic_phases():
    initialization = {
        "phase": "Initialization",
        "extrinsic_idx": None,
        "event": {
            "module_id": "Balances",
            "event_id": "Issued",
            "attributes": {"amount": 1},
        },
    }
    finalization = {
        "phase": "Finalization",
        "extrinsic_idx": None,
        "event": {
            "module_id": "System",
            "event_id": "CodeUpdated",
            "attributes": {},
        },
    }
    assert _event_parts(initialization) is None
    assert _event_parts(finalization) is None

    malformed = dict(initialization, phase="UnknownPhase")
    with pytest.raises(AxonPublishError, match="event phase is malformed"):
        _event_parts(malformed)


def test_zero_fee_publication_cap_refuses_any_finalized_spend():
    assert _max_fee_rao("0") == 0
    assert _validate_serve_axon_fee(
        {"who": "5Coldkey", "actual_fee": 0, "tip": 0},
        expected_payer="5Coldkey",
        maximum_fee_rao=0,
        sdk_fee_rao=0,
    ) == ("5Coldkey", 0, 0)
    for fee, tip in ((1, 0), (0, 1)):
        with pytest.raises(AxonPublishError, match="finalized fee evidence differs"):
            _validate_serve_axon_fee(
                {"who": "5Coldkey", "actual_fee": fee, "tip": tip},
                expected_payer="5Coldkey",
                maximum_fee_rao=0,
                sdk_fee_rao=fee,
            )
    for invalid in ("-0.000000001", "0.0000000001", "Infinity", "NaN"):
        with pytest.raises(AxonPublishError):
            _max_fee_rao(invalid)


def test_external_publish_fee_requires_bound_owner_coldkey():
    attributes = {"who": "5Coldkey", "actual_fee": 2_000_000, "tip": 0}
    assert _validate_serve_axon_fee(
        attributes,
        expected_payer="5Coldkey",
        maximum_fee_rao=10_000_000,
        sdk_fee_rao=2_000_000,
    ) == ("5Coldkey", 2_000_000, 0)

    with pytest.raises(AxonPublishError, match="finalized fee evidence differs"):
        _validate_serve_axon_fee(
            dict(attributes, who=_EXTERNAL_MINER_SS58),
            expected_payer="5Coldkey",
            maximum_fee_rao=10_000_000,
            sdk_fee_rao=2_000_000,
        )


def test_external_publish_integration_skips_exact_finalized_v11_tuple(monkeypatch, tmp_path):
    rpc_instances = _install_external_publish_rpc(monkeypatch)
    intents = []

    class ServeAxon:
        def __init__(self, **kwargs):
            intents.append(kwargs)

    class Subtensor:
        endpoint = "wss://test.example:443"
        subnets = SimpleNamespace(
            metagraph=lambda netuid, **kwargs: _external_publish_metagraph(
                endpoint="172.30.55.10:8090"
            )
        )

        def block_info(self, block):
            assert block == 123
            return SimpleNamespace(number=123, hash="0x" + "ab" * 32, extrinsics=[])

        def submit_signature(self, *_args, **_kwargs):
            pytest.fail("an exact finalized endpoint must not submit ServeAxon")

    class Balance:
        def __init__(self, rao, netuid=0):
            self.rao = rao
            self.netuid = netuid

        @classmethod
        def from_rao(cls, value, netuid=0):
            return cls(value, netuid)

    class Policy:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    result = _guarded_external_publish(
        SimpleNamespace(ServeAxon=ServeAxon, Balance=Balance, Policy=Policy),
        SimpleNamespace(
            hotkey=SimpleNamespace(ss58_address=_EXTERNAL_MINER_SS58),
            coldkeypub=SimpleNamespace(ss58_address="5Coldkey"),
        ),
        455,
        Subtensor(),
        tmp_path,
        external_ip="172.30.55.10",
        external_port=8090,
    )
    assert result["status"] == "already_exact_no_submission"
    assert result["observation"]["finalized_block_number"] == 123
    assert intents == [
        {
            "netuid": 455,
            "ip": "172.30.55.10",
            "port": 8090,
            "protocol": 4,
            "version": 1,
        }
    ]
    assert rpc_instances[0].closed is True


def test_external_publish_waits_for_independent_finality_without_resubmitting(
    monkeypatch, tmp_path
):
    finality = {"caught_up": False}
    rpc_instances = _install_external_publish_rpc(
        monkeypatch,
        finalized_block_number=lambda: 123 if finality["caught_up"] else 110,
    )
    monkeypatch.setenv(
        "AURELIUS_TESTNET_AXON_SERVE_ACK",
        "I_ACKNOWLEDGE_UP_TO_FOUR_SN455_AXON_ENDPOINT_UPDATES",
    )
    metagraphs = iter(
        (
            _external_publish_metagraph(endpoint=""),
            _external_publish_metagraph(endpoint="172.30.55.10:8090"),
        )
    )
    prepare_calls = []
    submit_calls = []
    plan_calls = []

    class ServeAxon:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Subtensor:
        endpoint = "wss://test.example:443"
        subnets = SimpleNamespace(metagraph=lambda *_args, **_kwargs: next(metagraphs))

        def block_info(self, block):
            if block in (110, 123):
                block_hash = "0x" + ("aa" if block == 110 else "ab") * 32
                return SimpleNamespace(number=block, hash=block_hash, extrinsics=[])
            assert block == 120
            return SimpleNamespace(
                number=120,
                hash=_EXTERNAL_RESULT_BLOCK_HASH,
                extrinsics=_external_decoded_extrinsics(),
            )

        def plan(self, *args, **kwargs):
            plan_calls.append((args, kwargs))
            return SimpleNamespace(
                ok=True,
                violations=[],
                op="serve_axon",
                signer="hotkey",
                signer_address=_EXTERNAL_MINER_SS58,
                fee=Balance(2_000_000),
                call=_external_publish_plan_call(),
                summary="bounded",
                effects=["publish endpoint"],
                warnings=[],
            )

        def decode_scale(self, type_string, data):
            assert type_string == "Call"
            assert data == b"\x01\x02"
            return _external_decoded_extrinsics()[1]["call"]

        def prepare_call(self, *args, **kwargs):
            prepare_calls.append((args, kwargs))
            return _ExternalUnsignedExtrinsic(_external_unsigned())

        def submit_signature(self, *args, **kwargs):
            submit_calls.append((args, kwargs))
            return ExtrinsicResult(
                success=True,
                message="finalized",
                block_hash=_EXTERNAL_RESULT_BLOCK_HASH,
                extrinsic_id="120-1",
                fee=Balance(2_000_000),
            )

        def query(self, _item, _params, *, block):
            assert block == 120
            return _external_fee_events()

    class Balance:
        def __init__(self, rao, netuid=0):
            self.rao = rao
            self.netuid = netuid

        @classmethod
        def from_rao(cls, value, netuid=0):
            return cls(value, netuid)

    class ExtrinsicResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Policy:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address=_EXTERNAL_MINER_SS58),
        coldkeypub=SimpleNamespace(ss58_address="5Coldkey"),
    )
    bt = SimpleNamespace(
        ServeAxon=ServeAxon,
        Balance=Balance,
        Policy=Policy,
        ExtrinsicResult=ExtrinsicResult,
        UnsignedExtrinsic=_ExternalUnsignedExtrinsic,
        WalletSigner=_ExternalWalletSigner,
        storage=SimpleNamespace(System=SimpleNamespace(Events="System.Events")),
    )
    subtensor = Subtensor()
    delays = []

    def advance_finality(delay):
        delays.append(delay)
        finality["caught_up"] = True

    result = _await_guarded_external_publish(
        lambda: _guarded_external_publish(
            bt,
            wallet,
            455,
            subtensor,
            tmp_path,
            external_ip="172.30.55.10",
            external_port=8090,
        ),
        retry_delay_s=0.25,
        sleeper=advance_finality,
    )
    assert result["status"] == "finalized_exact"
    assert delays == [0.25]
    [(prepare_args, prepare_kwargs)] = prepare_calls
    [(submit_args, submit_kwargs)] = submit_calls
    [(plan_args, plan_kwargs)] = plan_calls
    assert plan_args[1] is wallet
    assert plan_args[0].kwargs == {
        "netuid": 455,
        "ip": "172.30.55.10",
        "port": 8090,
        "protocol": 4,
        "version": 1,
    }
    assert plan_kwargs["policy"].allow_raw_calls is False
    assert len(prepare_args) == 1
    assert vars(prepare_args[0]) == vars(_external_publish_plan_call())
    assert prepare_kwargs == {
        "address": _EXTERNAL_MINER_SS58,
        "crypto_type": 1,
        "period": 64,
        "tip": 0,
    }
    assert type(submit_args[0]) is _ExternalUnsignedExtrinsic
    assert submit_args[0].to_dict() == _external_unsigned()
    assert len(submit_args[1]) == 64
    assert submit_kwargs["policy"].allow_raw_calls is True
    assert submit_kwargs["policy"].allowed_netuids == [455]
    assert submit_kwargs["policy"].max_spend_tao.rao == 0
    assert submit_kwargs["policy"].max_fee_tao.rao == 10_000_000
    assert {key: value for key, value in submit_kwargs.items() if key != "policy"} == {
        "wait_for_inclusion": True,
        "wait_for_finalization": True,
    }
    assert len(rpc_instances) == 2
    assert all(instance.closed for instance in rpc_instances)


def _install_external_recovery_harness(monkeypatch, root, *, include_match=True):
    """Hermetic chain transcript for exercising the production prepared lifecycle."""

    import subnet.miner.axon_serve as subject

    _install_external_publish_rpc(monkeypatch)
    monkeypatch.setenv(
        "AURELIUS_TESTNET_AXON_SERVE_ACK",
        "I_ACKNOWLEDGE_UP_TO_FOUR_SN455_AXON_ENDPOINT_UPDATES",
    )
    state = {
        "height": 100,
        "include_match": include_match,
        "submitted": False,
        "submit_lost": True,
        "plans": 0,
        "prepares": 0,
        "signs": 0,
        "submits": 0,
        "signature": None,
    }

    def block_hash(number):
        return {
            64: _EXTERNAL_ERA_BIRTH_HASH,
            100: _EXTERNAL_ERA_CURRENT_HASH,
            120: _EXTERNAL_RESULT_BLOCK_HASH,
        }.get(number, "0x" + number.to_bytes(32, "big").hex())

    def has_match(number):
        return number == 120 and state["submitted"] and state["include_match"]

    class Rpc:
        def __init__(self, endpoint):
            assert endpoint == "wss://test.example:443"

        def snapshot(self):
            return SimpleNamespace(
                head_hash=block_hash(state["height"]),
                block_number=state["height"],
                genesis_hash=_EXTERNAL_GENESIS,
                chain="Bittensor",
                runtime_spec_version=432,
                transaction_version=1,
            )

        def call(self, method, params):
            if method == "chain_getBlockHash":
                if state.get("bad_baseline") == params[0]:
                    return "0x" + "00" * 32
                return block_hash(params[0])
            if method == "state_getRuntimeVersion":
                return {"specVersion": 432, "transactionVersion": 1}
            if method == "chain_getBlock":
                populated = params == [_EXTERNAL_RESULT_BLOCK_HASH] and has_match(120)
                return {"block": {"extrinsics": _external_raw_extrinsics() if populated else []}}
            raise AssertionError((method, params))

        def close(self):
            pass

    class Balance:
        def __init__(self, rao, netuid=0):
            self.rao = rao
            self.netuid = netuid

        @classmethod
        def from_rao(cls, value, netuid=0):
            return cls(value, netuid)

    class Policy:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ServeAxon:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ExtrinsicResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class WalletSigner(_ExternalWalletSigner):
        def sign(self, payload):
            state["signs"] += 1
            signature = super().sign(payload)
            state["signature"] = signature
            return signature

    def metagraph(netuid, *, block, commitments):
        assert netuid == 455 and commitments is False
        published = state.get("preexisting_endpoint", False) or (
            state["submitted"] and state["include_match"] and block >= 120
        )
        return _external_publish_metagraph(endpoint="172.30.55.10:8090" if published else "")

    class Subtensor:
        endpoint = "wss://test.example:443"
        subnets = SimpleNamespace(metagraph=metagraph)

        def plan(self, _intent, _wallet, **_kwargs):
            state["plans"] += 1
            return SimpleNamespace(
                ok=True,
                violations=[],
                op="serve_axon",
                signer="hotkey",
                signer_address=_EXTERNAL_MINER_SS58,
                fee=Balance(2_000_000),
                call=_external_publish_plan_call(),
                summary="bounded",
                effects=["publish endpoint"],
                warnings=[],
            )

        def decode_scale(self, type_string, data):
            assert (type_string, data) == ("Call", b"\x01\x02")
            return _external_decoded_extrinsics()[1]["call"]

        def prepare_call(self, _call, **_kwargs):
            state["prepares"] += 1
            return _ExternalUnsignedExtrinsic(_external_unsigned())

        def submit_signature(self, _unsigned, signature, **_kwargs):
            state["submits"] += 1
            state["submitted"] = True
            state["signature"] = signature
            if state["submit_lost"]:
                raise TimeoutError("planted response loss after submission")
            state["height"] = 123
            return ExtrinsicResult(
                success=True,
                message="finalized",
                block_hash=_EXTERNAL_RESULT_BLOCK_HASH,
                extrinsic_id="120-1",
                fee=Balance(2_000_000),
            )

        def block_info(self, number):
            decoded = (
                _external_decoded_extrinsics(signature=state["signature"])
                if has_match(number)
                else []
            )
            return SimpleNamespace(number=number, hash=block_hash(number), extrinsics=decoded)

        def query(self, _item, _params, *, block):
            assert block == 120
            return _external_fee_events()

    monkeypatch.setattr(subject, "FinalizedRpc", Rpc)
    bt = SimpleNamespace(
        ServeAxon=ServeAxon,
        Balance=Balance,
        Policy=Policy,
        ExtrinsicResult=ExtrinsicResult,
        UnsignedExtrinsic=_ExternalUnsignedExtrinsic,
        WalletSigner=WalletSigner,
        storage=SimpleNamespace(System=SimpleNamespace(Events="System.Events")),
    )
    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address=_EXTERNAL_MINER_SS58),
        coldkeypub=SimpleNamespace(ss58_address="5Coldkey"),
    )
    subtensor = Subtensor()

    def publish(*, custody=None, claim=None):
        return subject._guarded_external_publish(
            bt,
            wallet,
            455,
            subtensor,
            root if custody is None else custody,
            external_ip="172.30.55.10",
            external_port=8090,
            runtime452_action_claim=claim,
        )

    return subject, publish, state


def _runtime452_journal_double(subject, monkeypatch, root):
    """Unit storage double only; Linux tests separately prove the root recorder protocol."""

    claim = SimpleNamespace(
        actor="miner-0",
        action=subject.RUNTIME452_SERVE_AXON_ACTION,
        claim_index=1,
        claim_sha256="a" * 64,
        generation_marker_sha256="b" * 64,
        burn_handoff_sha256="c" * 64,
        netuid=455,
        endpoint_ip="172.30.55.10",
        endpoint_port=8090,
        chain_endpoint="wss://test.example:443",
        chain_name="Bittensor",
        genesis_hash=_EXTERNAL_GENESIS,
        runtime_spec_version=432,
        transaction_version=1,
        signer_hotkey_ss58=_EXTERNAL_MINER_SS58,
        signer_coldkey_ss58="5Coldkey",
    )
    monkeypatch.setenv("AURELIUS_ENV", subject.RUNTIME452_PROFILE_ID)
    monkeypatch.setattr(
        subject,
        "RUNTIME452_ACTION_SIGNERS",
        {"miner-0": {"hotkey_ss58": _EXTERNAL_MINER_SS58, "coldkey_ss58": "5Coldkey"}},
    )
    custody = object.__new__(subject.Runtime452ActionJournalCustody)
    custody.path = root
    custody.claim = claim
    control = {"active": False, "fault": None}

    @contextmanager
    def locked():
        assert control["active"] is False
        control["active"] = True
        try:
            yield
        finally:
            control["active"] = False

    def exists(name):
        assert control["active"]
        return (root / name).exists()

    def read_json(name):
        assert control["active"]
        return subject._strict_json_object(root / name)

    def claim_bytes(name, payload):
        assert control["active"]
        if (root / name).exists():
            assert (root / name).read_bytes() == payload
            return "already_exact"
        subject._claim_json(root / name, json.loads(payload))
        fault = control["fault"]
        if (
            (fault == "after_scan" and name == "scan-resolution-1.json")
            or (fault == "after_final" and name == "finalized.json")
            or (fault == "lost_submission_ack" and name == "submission-claim-1.json")
        ):
            control["fault"] = None
            raise subject.Runtime452ActionJournalError("planted durable-record ACK loss")
        if fault == "already_submission_ack" and name == "submission-claim-1.json":
            control["fault"] = None
            return "already_exact"
        return "claimed_exact"

    custody.locked = locked
    custody.revalidate = lambda: None
    custody.exists = exists
    custody.read_json = read_json
    custody.claim_bytes = claim_bytes
    return custody, claim, control


@pytest.mark.parametrize("matched", [False, True])
def test_runtime452_axon_scan_crash_preserves_baseline_after_chain_advances(
    monkeypatch, tmp_path, matched
):
    subject, publish, state = _install_external_recovery_harness(
        monkeypatch, tmp_path, include_match=matched
    )
    custody, claim, control = _runtime452_journal_double(subject, monkeypatch, tmp_path)
    with pytest.raises(AxonPublishError, match="submission is ambiguous"):
        publish(custody=custody, claim=claim)
    assert control["active"] is False
    state["height"] = 164
    control["fault"] = "after_scan"
    with pytest.raises(AxonPublishError, match="cannot be claimed"):
        publish(custody=custody, claim=claim)
    path = tmp_path / "scan-resolution-1.json"
    retained = path.read_bytes()
    assert not (tmp_path / "finalized.json").exists()
    state["height"] = 165
    if matched:
        result = publish(custody=custody, claim=claim)
        assert result["status"] == "recovered_from_prepared_era"
    else:
        with pytest.raises(AxonPublishError, match="no second signed extrinsic"):
            publish(custody=custody, claim=claim)
    assert path.read_bytes() == retained
    assert json.loads(retained)["finalized_block_number"] == 164
    assert state["plans"] == state["prepares"] == state["signs"] == state["submits"] == 1
    assert not (tmp_path / "authority-2.json").exists()


@pytest.mark.parametrize("fault", ["lost_submission_ack", "already_submission_ack"])
def test_runtime452_axon_retained_claim_ack_is_scan_only(monkeypatch, tmp_path, fault):
    subject, publish, state = _install_external_recovery_harness(monkeypatch, tmp_path)
    custody, claim, control = _runtime452_journal_double(subject, monkeypatch, tmp_path)
    control["fault"] = fault
    with pytest.raises(AxonPublishError):
        publish(custody=custody, claim=claim)
    assert (tmp_path / "submission-claim-1.json").exists()
    assert state["submits"] == 0
    state["height"] = 164
    with pytest.raises(AxonPublishError, match="no second signed extrinsic"):
        publish(custody=custody, claim=claim)
    assert state["submits"] == 0
    assert state["plans"] == state["prepares"] == state["signs"] == 1


def test_runtime452_axon_lost_final_ack_reproves_without_restamping(monkeypatch, tmp_path):
    subject, publish, state = _install_external_recovery_harness(monkeypatch, tmp_path)
    custody, claim, control = _runtime452_journal_double(subject, monkeypatch, tmp_path)
    state["submit_lost"] = False
    control["fault"] = "after_final"
    with pytest.raises(AxonPublishError, match="cannot be claimed"):
        publish(custody=custody, claim=claim)
    path = tmp_path / "finalized.json"
    retained = path.read_bytes()
    state["height"] = 165
    assert publish(custody=custody, claim=claim) == json.loads(retained)
    assert path.read_bytes() == retained
    assert state["plans"] == state["prepares"] == state["signs"] == state["submits"] == 1


@pytest.mark.parametrize(
    "fault", ["schema", "causal", "signature", "submission", "observation", "baseline"]
)
def test_live_exact_endpoint_does_not_bless_forged_axon_terminal_lineage(
    monkeypatch, tmp_path, fault
):
    subject, publish, state = _install_external_recovery_harness(monkeypatch, tmp_path)
    custody, claim, _control = _runtime452_journal_double(subject, monkeypatch, tmp_path)
    state["submit_lost"] = False
    publish(custody=custody, claim=claim)
    path = tmp_path / "finalized.json"
    retained = json.loads(path.read_bytes())
    if fault == "schema":
        retained["schema_version"] = 1
    elif fault == "causal":
        retained["causal_extrinsic_evidence"]["signature_sha256"] = "0" * 64
    elif fault in {"signature", "submission"}:
        (
            tmp_path / ("signature-1.json" if fault == "signature" else "submission-claim-1.json")
        ).unlink()
    elif fault == "observation":
        retained["observation"]["actual"]["port"] = 8091
    else:
        state["bad_baseline"] = retained["observation"]["finalized_block_number"]
    path.write_bytes(subject._publish_json_payload(retained))
    state["height"] = 165
    with pytest.raises(AxonPublishError):
        publish(custody=custody, claim=claim)
    assert state["plans"] == state["prepares"] == state["signs"] == state["submits"] == 1


@pytest.mark.parametrize(
    ("decoded_signature", "fee_payer", "expected_error"),
    (
        (None, "5Coldkey", None),
        (
            b"\x99" * 64,
            "5Coldkey",
            "does not contain the durable prepared signature",
        ),
    ),
    ids=(
        "exact-durable-signature-and-owner-payer",
        "wrong-signature-same-call-rejected",
    ),
)
def test_external_publish_records_exact_finalized_causal_fee_evidence(
    monkeypatch, tmp_path, decoded_signature, fee_payer, expected_error
):
    import subnet.miner.axon_serve as subject

    block_hash = _EXTERNAL_RESULT_BLOCK_HASH
    monkeypatch.setenv(
        "CHAIN_ID",
        f"bittensor-testnet:{_EXTERNAL_GENESIS[2:]}",
    )
    monkeypatch.setenv("AURELIUS_AXON_EXPECTED_CHAIN_NAME", "Bittensor")
    monkeypatch.setenv(
        "AURELIUS_AXON_EXPECTED_GENESIS_HASH",
        _EXTERNAL_GENESIS,
    )
    monkeypatch.setenv("AURELIUS_RUNTIME_SPEC_PIN", "432")
    monkeypatch.setenv("AURELIUS_RUNTIME_TRANSACTION_VERSION_PIN", "1")
    monkeypatch.setenv("AURELIUS_ENV", "testnet-full")
    monkeypatch.setenv("SUBSTRATE_URL", "wss://test.example:443")
    monkeypatch.setenv(
        "AURELIUS_TESTNET_AXON_SERVE_ACK",
        "I_ACKNOWLEDGE_UP_TO_FOUR_SN455_AXON_ENDPOINT_UPDATES",
    )

    class Rpc:
        def __init__(self, endpoint):
            assert endpoint == "wss://test.example:443"

        def snapshot(self):
            return SimpleNamespace(
                head_hash="0x" + "55" * 32,
                block_number=123,
                genesis_hash=_EXTERNAL_GENESIS,
                chain="Bittensor",
                runtime_spec_version=432,
                transaction_version=1,
            )

        def call(self, method, params):
            values = {
                ("chain_getBlockHash", (123,)): "0x" + "55" * 32,
                ("chain_getBlockHash", (64,)): _EXTERNAL_ERA_BIRTH_HASH,
                ("chain_getBlockHash", (100,)): _EXTERNAL_ERA_CURRENT_HASH,
                ("chain_getBlockHash", (120,)): block_hash,
                ("state_getRuntimeVersion", (block_hash,)): {
                    "specVersion": 432,
                    "transactionVersion": 1,
                },
                ("chain_getBlock", (block_hash,)): {
                    "block": {"extrinsics": _external_raw_extrinsics()}
                },
            }
            return values[(method, tuple(params))]

        def close(self):
            pass

    monkeypatch.setattr(subject, "FinalizedRpc", Rpc)

    class Balance:
        def __init__(self, rao, netuid=0):
            self.rao = rao
            self.netuid = netuid

        @classmethod
        def from_rao(cls, value, netuid=0):
            return cls(value, netuid)

    class Policy:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ServeAxon:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ExtrinsicResult:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address=_EXTERNAL_MINER_SS58),
        coldkeypub=SimpleNamespace(ss58_address="5Coldkey"),
    )
    metagraphs = iter(
        (
            _external_publish_metagraph(endpoint=""),
            _external_publish_metagraph(endpoint="172.30.55.10:8090"),
        )
    )

    class Subtensor:
        endpoint = "wss://test.example:443"
        subnets = SimpleNamespace(metagraph=lambda *_args, **_kwargs: next(metagraphs))

        def plan(self, _intent, _wallet, **_kwargs):
            return SimpleNamespace(
                ok=True,
                violations=[],
                op="serve_axon",
                signer="hotkey",
                signer_address=_EXTERNAL_MINER_SS58,
                fee=Balance(2_000_000),
                call=_external_publish_plan_call(),
                summary="bounded",
                effects=[],
                warnings=[],
            )

        def decode_scale(self, type_string, data):
            assert type_string == "Call"
            assert data == b"\x01\x02"
            return _external_decoded_extrinsics()[1]["call"]

        def prepare_call(self, _call, **_kwargs):
            return _ExternalUnsignedExtrinsic(_external_unsigned())

        def submit_signature(self, _unsigned, _signature, **_kwargs):
            return ExtrinsicResult(
                success=True,
                message="finalized",
                block_hash=block_hash,
                extrinsic_id="120-1",
                fee=Balance(2_000_000),
            )

        def block_info(self, block_number):
            if block_number == 123:
                return SimpleNamespace(number=123, hash="0x" + "55" * 32, extrinsics=[])
            assert block_number == 120
            return SimpleNamespace(
                number=120,
                hash=block_hash,
                extrinsics=_external_decoded_extrinsics(signature=decoded_signature),
            )

        def query(self, _item, _params, *, block):
            assert block == 120
            return _external_fee_events(payer=fee_payer)

    bt = SimpleNamespace(
        ServeAxon=ServeAxon,
        Balance=Balance,
        Policy=Policy,
        ExtrinsicResult=ExtrinsicResult,
        UnsignedExtrinsic=_ExternalUnsignedExtrinsic,
        WalletSigner=_ExternalWalletSigner,
        storage=SimpleNamespace(System=SimpleNamespace(Events="System.Events")),
    )
    if expected_error is not None:
        with pytest.raises(AxonPublishError, match=expected_error):
            _guarded_external_publish(
                bt,
                wallet,
                455,
                Subtensor(),
                tmp_path,
                external_ip="172.30.55.10",
                external_port=8090,
            )
        assert not (tmp_path / "finalized.json").exists()
        return

    result = _guarded_external_publish(
        bt,
        wallet,
        455,
        Subtensor(),
        tmp_path,
        external_ip="172.30.55.10",
        external_port=8090,
    )

    causal = result["causal_extrinsic_evidence"]
    assert causal["status"] == "exact_finalized_extrinsic"
    assert causal["transaction_fee_payer_ss58"] == "5Coldkey"
    assert causal["actual_fee_tao_rao"] == 2_000_000
    assert causal["transaction_tip_tao_rao"] == 0
    assert causal["sdk_fee_tao_rao"] == 2_000_000


def test_advertised_axon_port_is_independent_and_strict(monkeypatch):
    monkeypatch.delenv("AURELIUS_AXON_EXTERNAL_PORT", raising=False)
    assert advertised_port(8091) == 8091

    monkeypatch.setenv("AURELIUS_AXON_EXTERNAL_PORT", "18094")
    assert advertised_port(8091) == 18094

    for malformed in ("", "0", "65536", " 18091", "+18091", "18_091", "١٨٠٩١"):
        monkeypatch.setenv("AURELIUS_AXON_EXTERNAL_PORT", malformed)
        with pytest.raises(ValueError, match="AURELIUS_AXON_EXTERNAL_PORT"):
            advertised_port(8091)


def test_forward_body_mints_the_same_envelope_as_http_mode(monkeypatch):
    # inline candidate carriage is confined to the local testlab posture; the parity claim of
    # this test is about the WIRE, so it runs under exactly that posture.
    monkeypatch.setenv("AURELIUS_ENV", "testlab")
    monkeypatch.setenv("CHAIN_ID", "localnet")
    monkeypatch.setenv("AURELIUS_AXON_EXPECTED_GENESIS_HASH", "observe-once-testlab")
    monkeypatch.setenv("SUBSTRATE_URL", "ws://subtensor:9944")
    monkeypatch.setenv("AURELIUS_TRANSPORT", "http")
    minter = WorkIdMinter("miner-hotkey")
    payload = _build(minter)
    envelope = json.loads(payload)
    # the §3.1 shape, minted by the SAME minter http mode uses
    assert envelope["task_type"] == "synthetic"
    assert envelope["work_id"]["hotkey_ss58"] == minter.hotkey_ss58
    assert envelope["work_id"]["validator_hotkey_ss58"] == _VALIDATOR
    assert envelope["work_id"]["request_id"] == _REQUEST_ID
    assert envelope["work_id"]["fanout_window"] == _WINDOW
    # fresh work_id per response, same content_hash (the serve contract)
    envelope2 = json.loads(_build(minter))
    assert envelope2["content_hash"] == envelope["content_hash"]
    assert envelope2["work_id"]["nonce"] != envelope["work_id"]["nonce"]
    # the coldkey never appears anywhere in the wire payload
    from subnet.miner.wallet import load_coldkey

    _, coldkey_ss58 = load_coldkey("miner-coldkey", unlocked=True)
    assert coldkey_ss58 not in payload


def test_forward_staging_mode_signs_miner_pool_candidate_not_validator_config(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AURELIUS_ENV", "testlab")
    monkeypatch.setenv("CHAIN_ID", "localnet")
    monkeypatch.setenv("AURELIUS_AXON_EXPECTED_GENESIS_HASH", "observe-once-testlab")
    monkeypatch.setenv("SUBSTRATE_URL", "ws://subtensor:9944")
    monkeypatch.setenv("AURELIUS_TRANSPORT", "http")
    pool = StagingPool(tmp_path / "staging.jsonl")
    pool.add(
        {
            "task_type": "synthetic",
            "schema_version": 1,
            "config": {"payload": "miner-owned"},
        }
    )
    schema = {
        "schema_version": 1,
        "required": ["payload"],
        "additional_fields": False,
        "fields": {"payload": {"type": "string"}},
    }

    def resolve(request):
        return candidate_for_request(request, pool=pool, schema_loader=lambda _task_type: schema)

    payload = build_envelope_json(
        WorkIdMinter("miner-hotkey"),
        "synthetic",
        1,
        json.dumps({"payload": "validator-supplied-must-be-ignored"}),
        request_id=_REQUEST_ID,
        validator_hotkey_ss58=_VALIDATOR,
        fanout_window=_WINDOW,
        validator_version="1.2.3",
        protocol_version=PROTOCOL_VERSION,
        candidate_source=CANDIDATE_SOURCE_STAGING,
        candidate_resolver=resolve,
    )
    envelope = json.loads(payload)
    assert json.loads(envelope["canonical_config"]) == {"payload": "miner-owned"}


@pytest.mark.parametrize(
    "config_json",
    [
        "x" * (MAX_CONFIG_JSON_BYTES + 1),  # over the size cap
        "not json",
        "\ud800",  # invalid UTF-8 must remain a refusal, never escape the Axon handler
        None,  # not a string at all
    ],
)
def test_forward_refusals_leave_the_envelope_empty(config_json):
    # NOTE: a config that PARSES stays opaque data (the http path's posture — canonicalization
    # accepts any JSON value); the axon wire must never be stricter than http, only size-capped
    minter = WorkIdMinter("miner-hotkey")
    assert _build(minter, config_json) == ""


def test_http_handler_binds_declared_audience_to_authenticated_btauth_caller():
    from io import BytesIO

    minter = WorkIdMinter("miner-hotkey")
    candidate_calls = []
    request = {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "item"},
        "request_id": _REQUEST_ID,
        "validator_hotkey_ss58": _VALIDATOR,
        "fanout_window": _WINDOW,
    }
    body = json.dumps(request).encode()

    def invoke(caller):
        cls = _handler(
            minter,
            candidate_resolver=lambda value: candidate_calls.append(value) or value,
            request_authorizer=lambda *_args, **_kwargs: caller,
        )
        handler = cls.__new__(cls)
        handler.path = "/v1/query"
        handler.headers = {"content-length": str(len(body))}
        handler.rfile = BytesIO(body)
        replies = []
        handler._json = lambda code, response: replies.append((code, response))
        handler.do_POST()
        return replies[-1]

    code, envelope = invoke(_VALIDATOR)
    assert code == 200
    assert envelope["work_id"]["validator_hotkey_ss58"] == _VALIDATOR
    assert len(candidate_calls) == 1
    code, response = invoke("5DifferentValidator")
    assert code == 403 and "differs" in response["error"]
    assert len(candidate_calls) == 1  # authentication mismatch precedes candidate extraction


def test_btauth_authorizer_verifies_before_permit_and_reuses_one_nonce_store():
    seen = []

    class AuthError(Exception):
        pass

    class Auth:
        class InMemoryNonceStore:
            pass

        @staticmethod
        def verify(headers, body, **kwargs):
            seen.append((headers, body, kwargs))
            return SimpleNamespace(hotkey_ss58="5Validator")

    Auth.AuthError = AuthError

    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner"))
    registered = RegisteredSet()
    nonce_store = object()
    authorize = make_btauth_authorizer(
        SimpleNamespace(http_auth=Auth), wallet, registered, nonce_store=nonce_store
    )
    with pytest.raises(QueryAuthorizationError, match="unavailable") as unavailable:
        authorize({"X-Bittensor-Hotkey": "5Validator"}, b"{}", method="POST", path="/v1/query")
    assert unavailable.value.status == 503
    assert seen == []  # no signature work and, critically, no nonce commit before finalized sync
    registered.hotkeys = frozenset({"5Validator"})
    with pytest.raises(QueryAuthorizationError, match="lacks validator permit") as denied:
        authorize({"X-Bittensor-Hotkey": "5Stranger"}, b"{}", method="POST", path="/v1/query")
    assert denied.value.status == 403
    assert seen == []
    assert (
        authorize(
            {"X-Bittensor-Hotkey": "5Validator"},
            b"{}",
            method="POST",
            path="/v1/query",
        )
        == "5Validator"
    )
    assert seen[0][2]["self_hotkey_ss58"] == "5Miner"
    assert seen[0][2]["nonce_store"] is nonce_store


def test_btauth_declared_sender_is_only_a_prefilter_and_verified_caller_is_rechecked():
    calls = []

    class AuthError(Exception):
        pass

    class Auth:
        class InMemoryNonceStore:
            pass

        @staticmethod
        def verify(headers, body, **kwargs):
            calls.append((headers, body, kwargs))
            return SimpleNamespace(hotkey_ss58="5DifferentValidator")

    Auth.AuthError = AuthError
    registered = RegisteredSet()
    registered.hotkeys = frozenset({"5Validator", "5DifferentValidator"})
    authorize = make_btauth_authorizer(
        SimpleNamespace(http_auth=Auth),
        SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner")),
        registered,
    )

    with pytest.raises(
        QueryAuthorizationError, match="caller differs from declared sender"
    ) as mismatch:
        authorize(
            {"X-Bittensor-Hotkey": "5Validator"},
            b"{}",
            method="POST",
            path="/v1/query",
        )
    assert mismatch.value.status == 401
    assert len(calls) == 1  # verify ran; the untrusted header was never accepted as authority


def test_btauth_caller_is_rechecked_against_latest_permit_set_after_nonce_commit():
    registered = RegisteredSet()
    registered.hotkeys = frozenset({"5Validator"})

    class Auth:
        class AuthError(Exception):
            pass

        class InMemoryNonceStore:
            pass

        @staticmethod
        def verify(*_args, **_kwargs):
            # Models a finalized refresh racing with cryptographic verification. The SDK has
            # accepted the nonce, but candidate extraction must observe the permit removal.
            registered.hotkeys = frozenset()
            return SimpleNamespace(hotkey_ss58="5Validator")

    authorize = make_btauth_authorizer(
        SimpleNamespace(http_auth=Auth),
        SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner")),
        registered,
    )
    with pytest.raises(QueryAuthorizationError, match="lacks validator permit") as removed:
        authorize(
            {"X-Bittensor-Hotkey": "5Validator"},
            b"{}",
            method="POST",
            path="/v1/query",
        )
    assert removed.value.status == 403


def test_btauth_sender_header_rejects_duplicates_and_invalid_shape_before_verify():
    class Headers:
        def items(self):
            return [
                ("X-Bittensor-Hotkey", "5Validator"),
                ("x-bittensor-hotkey", "5Other"),
            ]

    class Auth:
        class AuthError(Exception):
            pass

        class InMemoryNonceStore:
            pass

        @staticmethod
        def verify(*_args, **_kwargs):
            pytest.fail("malformed sender headers must not reach cryptographic verification")

    registered = RegisteredSet()
    registered.hotkeys = frozenset({"5Validator", "5Other"})
    authorize = make_btauth_authorizer(
        SimpleNamespace(http_auth=Auth),
        SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Miner")),
        registered,
    )
    with pytest.raises(QueryAuthorizationError, match="sender header") as malformed:
        authorize(Headers(), b"{}", method="POST", path="/v1/query")
    assert malformed.value.status == 401


def test_make_server_status_seam():
    """The readiness-gated /health: 503 until served, 200 after — no socket/wall clock."""

    minter = WorkIdMinter("miner-hotkey")
    state = {"served": False}

    def status():
        if state["served"]:
            return 200, {"status": "ok"}
        return 503, {"status": "publishing"}

    handler = _handler(minter, status=status).__new__(_handler(minter, status=status))
    handler.path = "/health"
    replies = []
    handler._json = lambda code, body: replies.append((code, body))
    handler.do_GET()
    assert replies[-1] == (503, {"status": "publishing"})
    state["served"] = True
    handler.do_GET()
    assert replies[-1] == (200, {"status": "ok"})


def test_health_reports_serving_counters_and_readiness_keeps_them():
    """/health carries aggregated serving telemetry; a non-200 readiness body does not."""

    minter = WorkIdMinter("miner-hotkey")
    stats = ServingStats()
    stats.note_served("5" * 48)
    stats.note_auth_rejected()
    stats.note_saturated()

    plain = _handler(minter, stats=stats).__new__(_handler(minter, stats=stats))
    plain.path = "/health"
    replies = []
    plain._json = lambda code, body: replies.append((code, body))
    plain.do_GET()
    code, body = replies[-1]
    assert code == 200
    assert body["stats"]["queries_served"] == 1
    assert body["stats"]["auth_rejected"] == 1
    assert body["stats"]["saturated"] == 1
    assert body["stats"]["last_caller_hotkey"] == "5" * 48
    assert body["stats"]["last_activity_at"] is not None

    gated = _handler(minter, status=lambda: (503, {"status": "publishing"}), stats=stats)
    handler = gated.__new__(gated)
    handler.path = "/health"
    handler._json = lambda c, b: replies.append((c, b))
    handler.do_GET()
    assert replies[-1] == (503, {"status": "publishing"})


def test_startup_posture_reports_every_missing_btauth_key_at_once():
    """One restart reports the whole misconfiguration, not one missing key per restart."""

    with pytest.raises(ValueError) as excinfo:
        validate_miner_startup_posture(
            {
                "AURELIUS_ENV": "testlab",
                "AURELIUS_TRANSPORT": "btauth",
                "AURELIUS_AXON_PUBLISH_STATE_DIR": "/tmp/publish",
                "AURELIUS_AXON_EXPECTED_CHAIN_NAME": "name",
            }
        )
    message = str(excinfo.value)
    for key in (
        "AURELIUS_BITTENSOR_NONCE_DB",
        "AURELIUS_AXON_EXPECTED_GENESIS_HASH",
        "AURELIUS_RUNTIME_SPEC_PIN",
        "CHAIN_ID",
        "SUBSTRATE_URL",
    ):
        assert key in message


def test_caller_rate_limiter_is_per_caller_and_window_bounded():
    from subnet.miner.serve import CallerRateLimiter

    now = {"t": 1000.0}
    limiter = CallerRateLimiter(2, window_s=60.0, clock=lambda: now["t"])
    assert limiter.allow("caller-a") is True
    assert limiter.allow("caller-a") is True
    assert limiter.allow("caller-a") is False
    assert limiter.allow("caller-b") is True  # independent quota per caller
    now["t"] += 61.0
    assert limiter.allow("caller-a") is True  # the window slides

    unlimited = CallerRateLimiter(0, window_s=60.0)
    assert all(unlimited.allow("same") for _ in range(100))


def test_query_rate_limit_answers_429_and_counts():
    from io import BytesIO

    from subnet.miner.serve import CallerRateLimiter

    minter = WorkIdMinter("miner-hotkey")
    stats = ServingStats()
    limiter = CallerRateLimiter(1, window_s=60.0)
    request = {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "item"},
        "request_id": _REQUEST_ID,
        "validator_hotkey_ss58": _VALIDATOR,
        "fanout_window": _WINDOW,
    }
    body = json.dumps(request).encode()
    cls = _handler(
        minter,
        candidate_resolver=lambda value: value,
        request_authorizer=lambda *_args, **_kwargs: _VALIDATOR,
        stats=stats,
        rate_limiter=limiter,
    )
    replies = []

    def invoke():
        handler = cls.__new__(cls)
        handler.path = "/v1/query"
        handler.headers = {"content-length": str(len(body))}
        handler.rfile = BytesIO(body)
        handler._json = lambda code, response: replies.append((code, response))
        handler.do_POST()

    invoke()
    invoke()
    assert replies[0][0] == 200
    assert replies[1] == (429, {"error": "caller query rate exceeded"})
    assert stats.snapshot()["rate_limited"] == 1


@pytest.mark.parametrize("authenticated", [False, True], ids=["http", "btauth"])
@pytest.mark.parametrize("shared_limiter", [False, True], ids=["default", "shared"])
def test_make_server_constructs_and_enforces_caller_quota(
    monkeypatch, authenticated, shared_limiter
):
    """Exercise the real factory without binding a port or starting a serving thread."""
    from subnet.miner.serve import CallerRateLimiter

    monkeypatch.setattr(BoundedThreadingHTTPServer, "server_bind", lambda self: None)
    monkeypatch.setattr(BoundedThreadingHTTPServer, "server_activate", lambda self: None)
    limiter = CallerRateLimiter(1, window_s=60.0, clock=lambda: 1000.0)
    factory_calls = []

    def default_limiter():
        factory_calls.append(True)
        return limiter

    monkeypatch.setattr("subnet.miner.serve.caller_rate_limiter", default_limiter)
    kwargs = {"rate_limiter": limiter} if shared_limiter else {}
    stats = ServingStats()
    request = {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "item"},
        "request_id": _REQUEST_ID,
        "validator_hotkey_ss58": _VALIDATOR,
        "fanout_window": _WINDOW,
    }
    body = json.dumps(request).encode()
    with make_server(
        WorkIdMinter("miner-hotkey"),
        0,
        candidate_resolver=lambda value: value,
        request_authorizer=(lambda *_args, **_kwargs: _VALIDATOR) if authenticated else None,
        stats=stats,
        **kwargs,
    ) as server:
        assert factory_calls == ([] if shared_limiter else [True])
        replies = []
        for _ in range(2):
            cls = server.RequestHandlerClass
            handler = cls.__new__(cls)
            handler.client_address = ("127.0.0.1", 12345)
            handler.path = "/v1/query"
            handler.headers = {"content-length": str(len(body))}
            handler.rfile = BytesIO(body)
            handler._json = lambda code, response: replies.append((code, response))
            handler.do_POST()
        assert replies[0][0] == 200
        assert replies[1] == (429, {"error": "caller query rate exceeded"})
        assert stats.snapshot()["queries_served"] == 1
        assert stats.snapshot()["rate_limited"] == 1


def test_http_sidecar_can_be_health_only_and_caps_query_bodies():
    minter = WorkIdMinter("miner-hotkey")

    def invoke(handler_cls, *, length, body=b""):
        handler = handler_cls.__new__(handler_cls)
        handler.path = "/v1/query"
        handler.headers = {"content-length": str(length)}
        handler.rfile = BytesIO(body)
        replies = []
        handler._json = lambda code, response: replies.append((code, response))
        handler.do_POST()
        return replies[-1]

    assert invoke(_handler(minter, query_enabled=False), length=2, body=b"{}") == (
        404,
        {"error": "not found"},
    )
    assert invoke(_handler(minter), length=MAX_HTTP_QUERY_BYTES + 1)[0] == 413


def test_partial_query_body_timeout_is_bounded_and_never_reaches_candidate_selection():
    class TimedOutBody:
        def read(self, _length):
            raise TimeoutError("hermetic read deadline")

    candidates = []
    handler_cls = _handler(
        WorkIdMinter("miner-hotkey"),
        candidate_resolver=lambda value: candidates.append(value) or value,
    )
    handler = handler_cls.__new__(handler_cls)
    handler.path = "/v1/query"
    handler.headers = {"content-length": "100"}
    handler.rfile = TimedOutBody()
    replies = []
    handler._json = lambda code, response: replies.append((code, response))

    handler.do_POST()

    assert replies == [(408, {"error": "request body deadline exceeded"})]
    assert candidates == []


def test_bounded_http_server_rejects_saturation_without_starting_another_thread():
    class Request:
        def __init__(self):
            self.sent = b""
            self.closed = False

        def sendall(self, value):
            self.sent += value

        def shutdown(self, _how):
            pass

        def close(self):
            self.closed = True

    server = object.__new__(BoundedThreadingHTTPServer)
    server._request_slots = threading.BoundedSemaphore(1)
    assert server._request_slots.acquire(blocking=False)
    request = Request()

    server.process_request(request, ("127.0.0.1", 12345))

    assert request.sent.startswith(b"HTTP/1.1 503 Service Unavailable\r\n")
    assert b'"error":"server saturated"' in request.sent
    assert request.closed is True


def test_plain_query_requires_the_exact_local_chain_posture(monkeypatch):
    local = {
        "AURELIUS_ENV": "testlab",
        "AURELIUS_TRANSPORT": "btauth",
        "CHAIN_ID": "localnet",
        "AURELIUS_AXON_EXPECTED_GENESIS_HASH": "observe-once-testlab",
        "SUBSTRATE_URL": "ws://subtensor:9944",
    }
    assert plain_query_allowed(local) is True
    for key, external in (
        ("AURELIUS_ENV", "testnet-full"),
        ("CHAIN_ID", "bittensor-testnet:external"),
        ("AURELIUS_AXON_EXPECTED_GENESIS_HASH", "0x" + "ab" * 32),
        ("AURELIUS_TRANSPORT", "other"),
    ):
        assert plain_query_allowed(dict(local, **{key: external})) is False

    with pytest.raises(ValueError, match="CHAIN_ID=localnet"):
        validate_miner_startup_posture(
            {
                "AURELIUS_ENV": "testlab",
                "AURELIUS_TRANSPORT": "http",
                "CHAIN_ID": "bittensor-testnet:external",
                "AURELIUS_AXON_EXPECTED_GENESIS_HASH": "0x" + "ab" * 32,
                "SUBSTRATE_URL": "wss://test.finney.opentensor.ai:443",
            }
        )

    monkeypatch.setenv("AURELIUS_ENV", "testlab")
    monkeypatch.setenv("AURELIUS_TRANSPORT", "http")
    monkeypatch.setenv("CHAIN_ID", "bittensor-testnet:external")
    monkeypatch.setenv("AURELIUS_AXON_EXPECTED_GENESIS_HASH", "0x" + "ab" * 32)
    monkeypatch.setenv("SUBSTRATE_URL", "wss://test.finney.opentensor.ai:443")
    with pytest.raises(ValueError, match="CHAIN_ID=localnet"):
        serve_plain_http(hotkey_name="must-not-open", port=0)


@pytest.mark.parametrize(
    "endpoint",
    [
        "wss://test.finney.opentensor.ai:443",
        "ws://test.finney.opentensor.ai:9944",
        "ws://subtensor:9945",
        "ws://subtensor.example:9944",
        "ws://subtensor:9944/rpc",
        "ws://operator@subtensor:9944",
        "ws://subtensor:9944?network=testnet",
        " ws://subtensor:9944",
    ],
)
def test_observe_once_rejects_nonlocal_subtensor_authorities(endpoint):
    posture = {
        "AURELIUS_ENV": "testlab",
        "AURELIUS_TRANSPORT": "btauth",
        "AURELIUS_IDENTITY_SOURCE": "bittensor",
        "AURELIUS_CLOCK": "real",
        "AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET": "true",
        "AURELIUS_BITTENSOR_NONCE_DB": "/tmp/nonces.sqlite3",
        "AURELIUS_AXON_PUBLISH_STATE_DIR": "/tmp/axon-publish",
        "AURELIUS_AXON_EXPECTED_CHAIN_NAME": "Development",
        "AURELIUS_AXON_EXPECTED_GENESIS_HASH": "observe-once-testlab",
        "AURELIUS_RUNTIME_SPEC_PIN": "432",
        "CHAIN_ID": "localnet",
        "SUBSTRATE_URL": endpoint,
    }

    assert plain_query_allowed(posture) is False
    with pytest.raises(ValueError, match="SUBSTRATE_URL"):
        validate_miner_startup_posture(posture)


def test_observe_once_external_endpoint_fails_before_sdk_or_state_mutation(monkeypatch, tmp_path):
    import subnet.miner.axon_serve as subject

    publish_dir = tmp_path / "must-not-be-created"
    values = {
        "AURELIUS_ENV": "testlab",
        "AURELIUS_TRANSPORT": "btauth",
        "AURELIUS_IDENTITY_SOURCE": "bittensor",
        "AURELIUS_CLOCK": "real",
        "AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET": "true",
        "AURELIUS_BITTENSOR_NONCE_DB": str(tmp_path / "must-not-open.sqlite3"),
        "AURELIUS_AXON_PUBLISH_STATE_DIR": str(publish_dir),
        "AURELIUS_AXON_EXPECTED_CHAIN_NAME": "Development",
        "AURELIUS_AXON_EXPECTED_GENESIS_HASH": "observe-once-testlab",
        "AURELIUS_RUNTIME_SPEC_PIN": "432",
        "CHAIN_ID": "localnet",
        "SUBSTRATE_URL": "wss://test.finney.opentensor.ai:443",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        subject,
        "require_bittensor_v11",
        lambda: pytest.fail("SDK import must follow local-endpoint posture validation"),
    )

    with pytest.raises(ValueError, match="SUBSTRATE_URL"):
        subject.serve_axon_forever(hotkey_name="miner-0", netuid=455)
    assert not publish_dir.exists()


def test_finalized_permit_refresh_pins_sdk_reads_and_keeps_last_good_on_drift():
    head = "0x" + "ab" * 32

    class Rpc:
        drift = False

        def snapshot(self):
            return SimpleNamespace(
                head_hash=head,
                block_number=123,
                genesis_hash="0x" + ("cd" if self.drift else "ef") * 32,
                chain="Bittensor",
                runtime_spec_version=432,
            )

    neurons = [
        SimpleNamespace(hotkey="5Validator", validator_permit=True),
        SimpleNamespace(hotkey="5Miner", validator_permit=False),
    ]
    calls = []

    class Subtensor:
        endpoint = "wss://expected.example:443"
        subnets = SimpleNamespace(
            metagraph=lambda netuid, **kwargs: (
                calls.append((netuid, kwargs))
                or SimpleNamespace(netuid=netuid, neurons=neurons, validators=[neurons[0]])
            )
        )

        def block_info(self, block):
            return SimpleNamespace(number=block, hash=head, extrinsics=[])

    registered = RegisteredSet()
    rpc = Rpc()
    snapshot = _refresh_finalized_permits(
        registered,
        Subtensor(),
        rpc,
        455,
        endpoint="wss://expected.example:443",
        expected_chain="Bittensor",
        expected_genesis="0x" + "ef" * 32,
        runtime_pin=432,
    )
    assert snapshot.block_number == 123
    assert registered.hotkeys == frozenset({"5Validator"})
    assert calls == [(455, {"block": 123, "commitments": False})]

    rpc.drift = True
    with pytest.raises(AxonPublishError, match="chain identity/runtime"):
        _refresh_finalized_permits(
            registered,
            Subtensor(),
            rpc,
            455,
            endpoint="wss://expected.example:443",
            expected_chain="Bittensor",
            expected_genesis="0x" + "ef" * 32,
            runtime_pin=432,
        )
    assert registered.hotkeys == frozenset({"5Validator"})
    assert len(calls) == 1  # drift is rejected before an SDK metagraph read


def test_no_miner_listener_is_constructed_before_publication_and_first_permit_sync():
    source = inspect.getsource(serve_axon_forever)
    publication = source.index("publication = _await_guarded_external_publish(")
    guarded_publication = source.index("_guarded_external_publish(")
    first_permit_sync = source.index("permit_snapshot = _refresh_finalized_permits(")
    first_listener = source.index("make_server(")
    assert publication < guarded_publication < first_permit_sync < first_listener


@pytest.mark.parametrize("fault", ["endpoint", "number", "hash", "extrinsic_hash"])
def test_sdk_block_decode_must_match_independent_canonical_block(fault):
    block_hash = "0x" + "44" * 32
    extrinsic_hash = "0x" + "55" * 32
    endpoint = "wss://expected.example:443"
    decoded = {
        "address": "5Miner",
        "extrinsic_hash": extrinsic_hash,
        "call": {"call_module": "SubtensorModule", "call_function": "serve_axon"},
    }

    class Subtensor:
        def __init__(self):
            self.endpoint = "wss://different.example:443" if fault == "endpoint" else endpoint

        def block_info(self, block):
            return SimpleNamespace(
                number=block + 1 if fault == "number" else block,
                hash=("0x" + "66" * 32) if fault == "hash" else block_hash,
                extrinsics=[
                    {
                        **decoded,
                        "extrinsic_hash": (
                            "0x" + "77" * 32 if fault == "extrinsic_hash" else extrinsic_hash
                        ),
                    }
                ],
            )

    with pytest.raises(AxonPublishError, match="SDK"):
        _require_sdk_block_info(
            Subtensor(),
            endpoint=endpoint,
            block_number=120,
            block_hash=block_hash,
            extrinsic_index=0,
            extrinsic_hash=extrinsic_hash,
        )


def test_miner_production_posture_rejects_development_seams():
    secure = {
        "CONFIG_SIGNING_PUBKEY": "ab" * 32,
        "CENTRAL_API_URL": "http://central-api:4000",
        "AURELIUS_ENV": "prod",
        "AURELIUS_TRANSPORT": "btauth",
        "AURELIUS_IDENTITY_SOURCE": "bittensor",
        "AURELIUS_CLOCK": "real",
        "AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET": "true",
        "AURELIUS_AXON_ENDPOINT_PROFILE": "public",
        "AURELIUS_BITTENSOR_NONCE_DB": "/var/lib/aurelius/btauth-nonces.sqlite3",
        "AURELIUS_AXON_PUBLISH_STATE_DIR": "/var/lib/aurelius/axon-publish",
        "AURELIUS_AXON_EXPECTED_CHAIN_NAME": "Bittensor",
        "AURELIUS_AXON_EXPECTED_GENESIS_HASH": "0x" + "ab" * 32,
        "AURELIUS_RUNTIME_SPEC_PIN": "432",
        "CHAIN_ID": "bittensor-mainnet:reviewed-genesis",
        "SUBSTRATE_URL": "wss://mainnet.example:443",
    }
    validate_miner_startup_posture(secure)

    for endpoint in (
        "wss://operator@mainnet.example:443",
        "wss://mainnet.example:443/rpc",
        "wss://mainnet.example:443?unsafe=1",
        "wss://mainnet.example:443#fragment",
        " wss://mainnet.example:443",
    ):
        with pytest.raises(ValueError, match="SUBSTRATE_URL"):
            validate_miner_startup_posture(dict(secure, SUBSTRATE_URL=endpoint))

    for key, bad in (
        ("CONFIG_SIGNING_PUBKEY", ""),
        ("CENTRAL_API_URL", ""),
        ("AURELIUS_TRANSPORT", "http"),
        ("AURELIUS_IDENTITY_SOURCE", "deterministic"),
        ("AURELIUS_CLOCK", "frozen"),
        ("AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET", "false"),
        ("AURELIUS_BITTENSOR_NONCE_DB", ""),
        ("AURELIUS_AXON_PUBLISH_STATE_DIR", ""),
        ("AURELIUS_AXON_EXPECTED_CHAIN_NAME", ""),
        ("AURELIUS_AXON_EXPECTED_GENESIS_HASH", ""),
        ("AURELIUS_RUNTIME_SPEC_PIN", ""),
        ("CHAIN_ID", ""),
        ("SUBSTRATE_URL", ""),
        ("AURELIUS_AXON_ENDPOINT_PROFILE", "testlab-private"),
    ):
        broken = dict(secure, **{key: bad})
        with pytest.raises(ValueError, match=key):
            validate_miner_startup_posture(broken)

    # Explicit testlab is the only compatibility profile that may retain the HTTP responder.
    validate_miner_startup_posture(
        {
            "AURELIUS_ENV": "testlab",
            "AURELIUS_TRANSPORT": "http",
            "CHAIN_ID": "localnet",
            "AURELIUS_AXON_EXPECTED_GENESIS_HASH": "observe-once-testlab",
            "SUBSTRATE_URL": "ws://subtensor:9944",
        }
    )
