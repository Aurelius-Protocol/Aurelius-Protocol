"""Funding is a one-shot coldkey mutation; retries observe the same durable intent."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from subnet.miner.funding import FundingJournal, FundingRefused, amount_rao, transfer_call
from subnet.miner.funding_client import TREASURY
from subnet.shared.ss58 import ss58_encode
from subnet.shared.signing import keypair_from_seed, sign
from subnet.operator.prepared_extrinsic import BITTENSOR_V11_SIGNED_EXTENSIONS
import subnet.miner.funding as funding

pytestmark = pytest.mark.miner


@pytest.mark.parametrize(
    "bad",
    [
        "0",
        "-1",
        "NaN",
        "Infinity",
        "1e2",
        "0.0000000001",
        " 1",
        "1.0 ",
        True,
        "18446744073.709551615",
    ],
)
def test_amount_never_rounds_or_accepts_float_syntax(bad):
    with pytest.raises(FundingRefused):
        amount_rao(bad)


def test_transfer_is_one_exact_native_mainnet_call():
    hotkey = ss58_encode(bytes([1]) * 32)
    call = transfer_call(hotkey, amount_rao("10.125"))
    assert call[:2] == bytes.fromhex("0756")
    assert call[34:66] == bytes([1]) * 32
    assert call[66:70] == bytes.fromhex("25002500")
    assert int.from_bytes(call[70:], "little") == 10_125_000_000
    assert len(call) == 78
    assert TREASURY
    assert amount_rao("0.000000001") == 1
    assert amount_rao(str(Decimal("10.125"))) == 10_125_000_000


@pytest.mark.parametrize("version", [475, 477, 479])
@pytest.mark.parametrize(
    "fault", [None, "parent", "number", "runtime", "code", "metadata", "upgrade", "canonical"]
)
def test_finalized_funding_binds_parent_and_post_execution(version, fault):
    parent = "0x" + "11" * 32
    block = "0x" + "22" * 32
    metadata = b"reviewed metadata"
    monkeypatch_profile = (
        funding.hashlib.sha256(metadata).hexdigest(),
        funding.RUNTIME_PROFILES[version][1],
    )
    from unittest.mock import patch

    def call(method, params):
        digest = params[-1]
        if method == "chain_getBlockHash":
            return (
                "0x" + "33" * 32 if fault == "canonical" else (parent if params == [99] else block)
            )
        if method == "chain_getHeader":
            return {
                "parentHash": block if fault == "parent" else parent,
                "number": "0x65" if fault == "number" else "0x64",
            }
        if method == "state_getRuntimeVersion":
            return {
                "specVersion": 480
                if fault == "runtime" or (fault == "upgrade" and digest == block)
                else version,
                "transactionVersion": 1,
            }
        if method == "state_getStorageHash":
            return parent if fault == "code" else monkeypatch_profile[1]
        if method == "state_getMetadata":
            return "0x00" if fault == "metadata" else "0x" + metadata.hex()
        raise AssertionError(method)

    rpc = SimpleNamespace(call=call)
    with patch.dict(funding.RUNTIME_PROFILES, {version: monkeypatch_profile}):
        if fault is None:
            funding.matched_execution(rpc, {"block_number": 100, "block_hash": block}, version)
        else:
            with pytest.raises(FundingRefused):
                funding.matched_execution(rpc, {"block_number": 100, "block_hash": block}, version)


def test_native_move_all_sentinel_is_never_authorized():
    with pytest.raises(FundingRefused, match="move-all"):
        transfer_call(ss58_encode(bytes([1]) * 32), 2**64 - 1)


def test_submission_claim_is_atomic_and_durable(tmp_path):
    root = tmp_path / "funding"
    journal = FundingJournal(root)
    document = {"status": "submission_ambiguous_until_observed", "unsigned": {"nonce": 4}}

    def claim(_):
        with journal.lock():
            return journal.claim("submission", document)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(claim, range(4))) == 1
    assert FundingJournal(root).read("submission") == document
    assert (root / "submission.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((root / "submission.json").read_text()) == document
    with pytest.raises(FundingRefused):
        journal.claim("submission", {"unsigned": {"nonce": 5}})


def test_journal_refuses_symlinks_and_replacement_claims(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FundingRefused):
        FundingJournal(link)
    journal = FundingJournal(target)
    (target / "submission.json").symlink_to(tmp_path / "missing")
    with pytest.raises(FundingRefused):
        journal.claim("submission", {"nonce": 1})


def finalized_events():
    coldkey = ss58_encode(bytes([3]) * 32)
    hotkey = ss58_encode(bytes([2]) * 32)
    prepared = {
        "coldkey": coldkey,
        "hotkey": hotkey,
        "alpha_rao": 10_000_000_000,
        "fee_cap_tao_rao": 10_000_000,
    }

    def event(name, attributes, module="SubtensorModule"):
        return {
            "phase": "ApplyExtrinsic",
            "extrinsic_idx": 7,
            "event": {"module_id": module, "event_id": name, "attributes": attributes},
        }

    events = [
        event("StakeRemoved", [coldkey, hotkey, 5_000_000, 10_000_000_000, 37, 0]),
        event("StakeAdded", [TREASURY, hotkey, 5_000_000, 10_000_000_000, 37, 0]),
        event("StakeTransferred", [coldkey, TREASURY, hotkey, 37, 37, 5_000_000]),
        event(
            "TransactionFeePaid", {"who": coldkey, "actual_fee": 3, "tip": 0}, "TransactionPayment"
        ),
        event("ExtrinsicSuccess", {}, "System"),
    ]
    return events, prepared


def test_finality_proof_requires_actual_native_alpha_and_tao_fee():
    events, prepared = finalized_events()
    assert funding.transfer_event_index(events, prepared, 7) == 2


@pytest.mark.parametrize(
    "fault",
    ["alpha", "destination", "netuid", "phase", "fee", "tip", "order", "duplicate", "failed"],
)
def test_finalized_success_alone_never_proves_the_purchase(fault):
    events, prepared = finalized_events()
    if fault == "alpha":
        events[1]["event"]["attributes"][3] -= 1
    elif fault == "destination":
        events[2]["event"]["attributes"][1] = prepared["coldkey"]
    elif fault == "netuid":
        events[2]["event"]["attributes"][4] = 455
    elif fault == "phase":
        events[0]["extrinsic_idx"] = True
    elif fault in ("fee", "tip"):
        events[3]["event"]["attributes"]["actual_fee" if fault == "fee" else "tip"] = 10_000_001
    elif fault == "order":
        events[0], events[1] = events[1], events[0]
    elif fault == "duplicate":
        events.append(events[2])
    else:
        events[-1]["event"]["event_id"] = "ExtrinsicFailed"
    with pytest.raises(FundingRefused):
        funding.transfer_event_index(events, prepared, 7)


def flow(tmp_path, monkeypatch, version=475):
    pair = keypair_from_seed(b"f" * 32)
    coldkey = ss58_encode(pair[0])
    hotkey = ss58_encode(bytes([2]) * 32)
    calls = []
    call = transfer_call(hotkey, 10_000_000_000)
    era_hash = "0x" + "cd" * 32
    included = bytes.fromhex("45021c0000")
    signed = (
        version.to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + bytes.fromhex(funding.GENESIS_HASH[2:])
        + bytes.fromhex(era_hash[2:])
        + b"\x00"
    )
    unsigned = {
        "call_data": "0x" + call.hex(),
        "address": coldkey,
        "public_key": "0x" + pair[0].hex(),
        "crypto_type": 1,
        "era": {"period": 64, "current": 100},
        "nonce": 7,
        "tip": 0,
        "tip_asset_id": None,
        "genesis_hash": funding.GENESIS_HASH,
        "era_block_hash": era_hash,
        "spec_version": version,
        "transaction_version": 1,
        "metadata_hash": None,
        "payload": "0x" + (call + included + signed).hex(),
        "included_in_extrinsic": "0x" + included.hex(),
        "included_in_signed_data": "0x" + signed.hex(),
        "payload_json": {
            "address": coldkey,
            "blockHash": era_hash,
            "genesisHash": funding.GENESIS_HASH,
            "method": "0x" + call.hex(),
            "nonce": "0x00000007",
            "specVersion": "0x" + version.to_bytes(4, "big").hex(),
            "tip": "0x" + "00" * 16,
            "transactionVersion": "0x00000001",
            "era": "0x4502",
            "version": 4,
            "signedExtensions": list(BITTENSOR_V11_SIGNED_EXTENSIONS),
            "mode": 0,
            "metadataHash": None,
        },
    }

    class Wallet:
        def __init__(self):
            self.coldkeypub = SimpleNamespace(ss58_address=coldkey)

        @property
        def coldkey(self):
            calls.append("unlock")
            return SimpleNamespace(ss58_address=coldkey, sign=lambda message: sign(message, pair))

    class Client:
        collateral = None
        available = 20_000_000_000
        price = 500_000

        def at(self, block):
            assert block == 100
            return self

        def query(self, item, params):
            return {
                "Owner": coldkey,
                "Uids": 42,
                "TransferToggle": True,
                "SubtokenEnabled": True,
                "MinerCollateral": self.collateral,
            }[item[1]]

        def runtime(self, method, params):
            if method == "price":
                assert params == [37]
                return self.price
            return {coldkey: {37: {"available": self.available}}}

        staking = SimpleNamespace(get=lambda *args: SimpleNamespace(rao=20_000_000_000))
        balances = SimpleNamespace(get=lambda *args: SimpleNamespace(rao=100_000_000))

        def plan(self, *args, **kwargs):
            calls.append("plan")
            return SimpleNamespace(
                ok=True,
                op="transfer_stake",
                signer="coldkey",
                signer_address=coldkey,
                fee=SimpleNamespace(netuid=0, rao=5),
                call=SimpleNamespace(spec_version=version, data=call),
            )

        def prepare_call(self, *args, **kwargs):
            calls.append("prepare")
            return SimpleNamespace(to_dict=lambda: unsigned)

        def submit_signature(self, *args, **kwargs):
            calls.append("broadcast")
            assert FundingJournal(tmp_path / "intent").read("submission") is not None
            raise OSError("connection dropped after broadcast")

    bt = SimpleNamespace(
        Balance=SimpleNamespace(
            from_rao=lambda rao, netuid=0: SimpleNamespace(rao=rao, netuid=netuid)
        ),
        TransferStake=lambda **kwargs: SimpleNamespace(**kwargs),
        Policy=lambda **kwargs: kwargs,
        runtime_api=SimpleNamespace(
            StakeInfoRuntimeApi=SimpleNamespace(get_stake_availability_for_coldkeys="availability"),
            SwapRuntimeApi=SimpleNamespace(current_alpha_price="price"),
        ),
    )
    monkeypatch.setattr(
        funding,
        "chain_scope",
        lambda rpc: SimpleNamespace(block_number=100, runtime_spec_version=version),
    )

    class Rpc:
        def call(self, method, params):
            assert method == "chain_getBlockHash"
            return era_hash

    bt.UnsignedExtrinsic = SimpleNamespace(
        from_dict=lambda document: SimpleNamespace(to_dict=lambda: document)
    )
    args = dict(
        bt=bt,
        rpc=Rpc(),
        response={
            "identity": {"coldkey": coldkey, "hotkey": hotkey, "netuid": 37},
            "terms": {
                "status": "available",
                "price_policy_digest": "d" * 64,
                "runtime_spec_version": version,
            },
            "finalized_block": 100,
        },
        alpha_rao=10_000_000_000,
        max_fee_rao=10_000_000,
        journal=FundingJournal(tmp_path / "intent"),
    )
    return Client(), Wallet(), calls, args


@pytest.mark.parametrize("price", [0, True, "500000", 9999])
def test_native_transfer_minimum_refuses_before_wallet_unlock(tmp_path, monkeypatch, price):
    client, wallet, calls, args = flow(tmp_path, monkeypatch)
    client.price = price
    with pytest.raises(FundingRefused, match="minimum transfer"):
        funding.fund(client, wallet, execute=True, **args)
    assert calls == []
    assert args["journal"].read("prepared") is None


@pytest.mark.parametrize("version", [475, 477, 479])
def test_preview_never_unlocks_signs_or_prepares(tmp_path, monkeypatch, version):
    client, wallet, calls, args = flow(tmp_path, monkeypatch, version)
    result = funding.fund(client, wallet, execute=False, **args)
    assert result["status"] == "preview"
    assert result["alpha_rao"] == 10_000_000_000
    assert calls == ["plan"]
    assert list((tmp_path / "intent").iterdir()) == []


@pytest.mark.parametrize("version", [475, 477, 479])
def test_dropped_submission_reuses_intent_for_observation_without_resigning(
    tmp_path, monkeypatch, version
):
    client, wallet, calls, args = flow(tmp_path, monkeypatch, version)
    with pytest.raises(FundingRefused, match="ambiguous"):
        funding.fund(client, wallet, execute=True, **args)
    assert calls.count("broadcast") == 1
    assert args["journal"].read("signature") is not None
    before = calls[:]
    monkeypatch.setattr(funding, "observe", lambda *a: {"status": "submission_unresolved"})
    assert funding.fund(client, wallet, execute=True, **args) == {"status": "submission_unresolved"}
    assert calls == before


def test_observation_refuses_a_changed_durable_signature_before_reading_chain(
    tmp_path, monkeypatch
):
    client, wallet, _, args = flow(tmp_path, monkeypatch)
    with pytest.raises(FundingRefused, match="ambiguous"):
        funding.fund(client, wallet, execute=True, **args)
    (tmp_path / "intent" / "signature.json").write_text(json.dumps({"signature_hex": "00" * 64}))
    with pytest.raises(FundingRefused, match="retained funding signature differs"):
        funding.observe(client, args["rpc"], args["journal"], args["response"])


def test_signed_unsubmitted_intent_resumes_without_repreparing_or_resigning(tmp_path, monkeypatch):
    client, wallet, calls, args = flow(tmp_path, monkeypatch)
    original = args["journal"].claim

    def interrupted_claim(name, value):
        if name == "submission":
            raise OSError("process stopped before submission claim")
        return original(name, value)

    monkeypatch.setattr(args["journal"], "claim", interrupted_claim)
    with pytest.raises(OSError):
        funding.fund(client, wallet, execute=True, **args)
    assert args["journal"].read("signature") is not None
    assert args["journal"].read("submission") is None
    before = calls[:]
    monkeypatch.setattr(args["journal"], "claim", original)
    with pytest.raises(FundingRefused, match="ambiguous"):
        funding.fund(client, wallet, execute=True, **args)
    assert calls.count("prepare") == 1
    assert calls.count("unlock") == before.count("unlock")
    assert calls.count("broadcast") == 1


def test_a_changed_unsubmitted_intent_never_unlocks_or_broadcasts(tmp_path, monkeypatch):
    client, wallet, calls, args = flow(tmp_path, monkeypatch)
    original = args["journal"].claim

    def interrupted_claim(name, value):
        if name == "signature":
            raise OSError("stopped after unsigned preparation")
        return original(name, value)

    monkeypatch.setattr(args["journal"], "claim", interrupted_claim)
    with pytest.raises(OSError):
        funding.fund(client, wallet, execute=True, **args)
    before = calls[:]
    args["alpha_rao"] += 1
    with pytest.raises(FundingRefused):
        funding.fund(client, wallet, execute=True, **args)
    assert calls.count("unlock") == before.count("unlock")
    assert "broadcast" not in calls


@pytest.mark.parametrize("fault", ["collateral", "available", "terms"])
def test_blocked_credit_or_encumbered_stake_never_signs(tmp_path, monkeypatch, fault):
    client, wallet, calls, args = flow(tmp_path, monkeypatch)
    if fault == "collateral":
        client.collateral = {"locked": 15_000_000_000}
    elif fault == "available":
        client.available = 1
    else:
        args["response"]["terms"] = {
            "status": "unavailable",
            "reason": "mainnet_retention_unavailable",
        }
    with pytest.raises(FundingRefused):
        funding.fund(client, wallet, execute=True, **args)
    assert "unlock" not in calls
    assert "prepare" not in calls
    assert "broadcast" not in calls


@pytest.mark.parametrize("version", [475, 477, 479])
def test_reviewed_funding_runtime_preserves_one_shot_ambiguous_submission(
    tmp_path, monkeypatch, version
):
    client, wallet, calls, args = flow(tmp_path, monkeypatch, version)
    with pytest.raises(FundingRefused, match="ambiguous"):
        funding.fund(client, wallet, execute=True, **args)
    assert calls.count("broadcast") == 1
    assert args["journal"].read("prepared")["unsigned"]["spec_version"] == version


@pytest.mark.parametrize("version,successor", [(475, 477), (477, 479), (479, 480)])
def test_upgrade_during_preparation_refuses_before_unlock_or_sign(
    tmp_path, monkeypatch, version, successor
):
    client, wallet, calls, args = flow(tmp_path, monkeypatch, version)
    snapshots = iter(
        [
            SimpleNamespace(block_number=100, runtime_spec_version=version),
            SimpleNamespace(block_number=100, runtime_spec_version=successor),
        ]
    )
    monkeypatch.setattr(funding, "chain_scope", lambda rpc: next(snapshots))
    with pytest.raises(FundingRefused, match="prepared transfer"):
        funding.fund(client, wallet, execute=True, **args)
    assert "unlock" not in calls and "broadcast" not in calls


@pytest.mark.parametrize("version", [475, 477, 479])
@pytest.mark.parametrize(
    "fault", [None, "metadata", "code", "unknown", "float", "transaction", "genesis"]
)
def test_funding_scope_requires_exact_reviewed_identity(monkeypatch, version, fault):
    metadata = b"reviewed unit metadata"
    monkeypatch.setitem(
        funding.RUNTIME_PROFILES,
        version,
        (funding.hashlib.sha256(metadata).hexdigest(), "0x" + "ab" * 32),
    )
    snapshot = SimpleNamespace(
        genesis_hash=funding.GENESIS_HASH,
        runtime_spec_version=version,
        transaction_version=1,
        chain="Bittensor",
        head_hash="0x" + "cd" * 32,
    )
    if fault == "unknown":
        snapshot.runtime_spec_version = 478
    elif fault == "float":
        snapshot.runtime_spec_version = float(version)
    elif fault == "transaction":
        snapshot.transaction_version = 2
    elif fault == "genesis":
        snapshot.genesis_hash = "0x" + "00" * 32
    rpc = SimpleNamespace(
        snapshot=lambda: snapshot,
        call=lambda method, params: (
            ("0x" + "00" * 32 if fault == "code" else "0x" + "ab" * 32)
            if method == "state_getStorageHash"
            else "0x" + (metadata + (b"x" if fault == "metadata" else b"")).hex()
        ),
    )
    if fault is None:
        assert funding.chain_scope(rpc) is snapshot
    else:
        with pytest.raises(FundingRefused):
            funding.chain_scope(rpc)


@pytest.mark.parametrize("version", [475, 477, 479])
@pytest.mark.parametrize("free_rao", [0, 4, 5, 1_399_264])
def test_fee_cap_does_not_require_free_tao_reserve(tmp_path, monkeypatch, version, free_rao):
    client, wallet, calls, args = flow(tmp_path, monkeypatch, version)
    client.balances = SimpleNamespace(get=lambda _: SimpleNamespace(rao=free_rao))
    result = funding.fund(client, wallet, execute=False, **args)
    assert result["free_tao_rao"] == free_rao
    assert result["planned_fee_tao_rao"] == 5
    assert result["fee_payment"] == "chain_selected_tao_or_alpha"
    assert calls == ["plan"]


@pytest.mark.parametrize("free_rao", [-1, True, "0"])
def test_invalid_free_tao_balance_never_unlocks(tmp_path, monkeypatch, free_rao):
    client, wallet, calls, args = flow(tmp_path, monkeypatch)
    client.balances = SimpleNamespace(get=lambda _: SimpleNamespace(rao=free_rao))
    with pytest.raises(FundingRefused, match="invalid free TAO"):
        funding.fund(client, wallet, execute=True, **args)
    assert calls == []


def test_alpha_fee_fallback_keeps_exact_fee_cap_before_unlock(tmp_path, monkeypatch):
    client, wallet, calls, args = flow(tmp_path, monkeypatch, 477)
    client.balances = SimpleNamespace(get=lambda _: SimpleNamespace(rao=0))
    args["max_fee_rao"] = 4
    with pytest.raises(FundingRefused, match="planned transfer/fee"):
        funding.fund(client, wallet, execute=True, **args)
    assert calls == ["plan"]
    assert args["journal"].read("prepared") is None


@pytest.mark.parametrize("version", [478, 480])
def test_unreviewed_funding_refuses_before_preparation_unlock_or_broadcast(
    tmp_path, monkeypatch, version
):
    real_scope = funding.chain_scope
    client, wallet, calls, args = flow(tmp_path, monkeypatch, 477)
    monkeypatch.setattr(funding, "chain_scope", real_scope)
    args["rpc"] = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            genesis_hash=funding.GENESIS_HASH,
            runtime_spec_version=version,
            transaction_version=1,
            chain="Bittensor",
        )
    )
    with pytest.raises(FundingRefused, match="unreviewed funding chain/runtime"):
        funding.fund(client, wallet, execute=True, **args)
    assert calls == []
    assert args["journal"].read("submission") is None


def test_server_runtime_must_match_finalized_chain_before_planning(tmp_path, monkeypatch):
    client, wallet, calls, args = flow(tmp_path, monkeypatch, 479)
    args["response"]["terms"]["runtime_spec_version"] = 477
    with pytest.raises(FundingRefused, match="terms runtime differs"):
        funding.fund(client, wallet, execute=True, **args)
    assert calls == []
    assert args["journal"].read("prepared") is None


def test_upgrade_after_signing_never_claims_or_broadcasts_submission(tmp_path, monkeypatch):
    client, wallet, calls, args = flow(tmp_path, monkeypatch, 477)
    snapshots = iter(
        SimpleNamespace(block_number=100, runtime_spec_version=version)
        for version in (477, 477, 479)
    )
    monkeypatch.setattr(funding, "chain_scope", lambda rpc: next(snapshots))
    with pytest.raises(FundingRefused, match="runtime changed"):
        funding.fund(client, wallet, execute=True, **args)
    assert calls == ["plan", "prepare", "unlock"]
    assert args["journal"].read("signature") is not None
    assert args["journal"].read("submission") is None


@pytest.mark.parametrize("fault", [None, "code", "metadata"])
def test_runtime479_flow_checks_pinned_bytes_before_unlock(tmp_path, monkeypatch, fault):
    real_scope = funding.chain_scope
    client, wallet, calls, args = flow(tmp_path, monkeypatch, 479)
    monkeypatch.setattr(funding, "chain_scope", real_scope)
    from pathlib import Path

    raw = (Path(__file__).parent / "fixtures/runtime479/metadata.scale").read_bytes()
    snapshot = SimpleNamespace(
        genesis_hash=funding.GENESIS_HASH,
        runtime_spec_version=479,
        transaction_version=1,
        chain="Bittensor",
        head_hash="0x" + "cd" * 32,
        block_number=100,
    )

    def rpc_call(method, params):
        if method == "chain_getBlockHash":
            return snapshot.head_hash
        assert params[-1] == snapshot.head_hash
        if method == "state_getStorageHash":
            return "0x" + "00" * 32 if fault == "code" else funding.RUNTIME_PROFILES[479][1]
        assert method == "state_getMetadata"
        return "0x" + (raw + (b"x" if fault == "metadata" else b"")).hex()

    args["rpc"] = SimpleNamespace(snapshot=lambda: snapshot, call=rpc_call)
    with pytest.raises(FundingRefused, match="ambiguous" if fault is None else "differs"):
        funding.fund(client, wallet, execute=True, **args)
    if fault is None:
        assert calls == ["plan", "prepare", "unlock", "broadcast"]
        assert args["journal"].read("submission") is not None
    else:
        assert calls == []
        assert args["journal"].read("prepared") is None
