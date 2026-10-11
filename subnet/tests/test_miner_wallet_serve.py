"""Miner wallet gating + serve (plan §1.8, invariant 1): the coldkey is local-only and never in a
payload; serve mints a FRESH work_id per response (same content_hash, different verifying work_id)."""

import json

import pytest

from subnet.miner import serve
from subnet.miner.serve import WorkIdMinter, minter_from_bittensor_wallet, minter_from_env
from subnet.miner.wallet import ColdkeyLocked, load_coldkey, load_hotkey
from subnet.shared.signing import keypair_from_seed, sign, verify_work_id_v2
from subnet.shared.ss58 import ss58_encode

pytestmark = pytest.mark.miner

_TASK = {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "same"}}
_VALIDATOR = WorkIdMinter("validator-hotkey").hotkey_ss58
_REQUEST_ID = "ab" * 32
_WINDOW = 7


def _envelope(minter):
    return minter.envelope(
        _TASK,
        validator_hotkey_ss58=_VALIDATOR,
        request_id=_REQUEST_ID,
        fanout_window=_WINDOW,
    )


def test_hotkey_loads_freely():
    pair, ss58 = load_hotkey("miner-hotkey")
    assert ss58.startswith("5") and len(pair) == 2


def test_coldkey_locked_without_explicit_unlock():
    with pytest.raises(ColdkeyLocked):
        load_coldkey("miner-coldkey", unlocked=False)


def test_coldkey_unlocked_returns_pair():
    _pair, ss58 = load_coldkey("miner-coldkey", unlocked=True)
    assert ss58.startswith("5")


def test_fresh_work_id_per_response_same_content_hash():
    minter = WorkIdMinter("miner-hotkey")
    e1 = _envelope(minter)
    e2 = _envelope(minter)
    assert e1["content_hash"] == e2["content_hash"]  # same content
    assert e1["work_id"]["nonce"] != e2["work_id"]["nonce"]  # fresh nonce per response
    assert e1["work_id"]["signature_hex"] != e2["work_id"]["signature_hex"]  # distinct work_id


def test_nonce_does_not_repeat_when_the_process_local_minter_is_reconstructed():
    first = _envelope(WorkIdMinter("miner-hotkey"))
    after_restart = _envelope(WorkIdMinter("miner-hotkey"))
    assert first["work_id"]["nonce"] != after_restart["work_id"]["nonce"]


def test_coldkey_never_in_the_envelope():
    minter = WorkIdMinter("miner-hotkey")
    _pair, coldkey_ss58 = load_coldkey("miner-coldkey", unlocked=True)
    envelope = _envelope(minter)
    assert coldkey_ss58 not in json.dumps(envelope)  # the coldkey never leaves the miner
    assert envelope["work_id"]["hotkey_ss58"] == minter.hotkey_ss58  # only the hotkey signs


def test_injected_wallet_signer_authors_work_id_as_its_exact_ss58():
    pair = keypair_from_seed(b"\x17" * 32)
    ss58 = ss58_encode(pair[0])

    class Hotkey:
        ss58_address = ss58

        @staticmethod
        def sign(message):
            return sign(message, pair)

    class Wallet:
        hotkey = Hotkey()

    minter = minter_from_bittensor_wallet("ignored-dev-name", Wallet())
    envelope = _envelope(minter)
    work_id = envelope["work_id"]

    assert minter.hotkey_ss58 == ss58
    assert work_id["hotkey_ss58"] == ss58
    assert work_id["version"] == 2
    assert work_id["validator_hotkey_ss58"] == _VALIDATOR
    assert work_id["request_id"] == _REQUEST_ID
    assert work_id["fanout_window"] == _WINDOW
    assert verify_work_id_v2(
        pair[0],
        bytes.fromhex(work_id["signature_hex"]),
        content_hash=envelope["content_hash"],
        hotkey_ss58=ss58,
        validator_hotkey_ss58=_VALIDATOR,
        request_id=_REQUEST_ID,
        fanout_window=_WINDOW,
        timestamp=work_id["timestamp"],
        nonce=work_id["nonce"],
    )


def test_minter_identity_source_is_explicit_and_bittensor_wallet_backed(monkeypatch):
    pair = keypair_from_seed(b"\x18" * 32)

    class Hotkey:
        ss58_address = ss58_encode(pair[0])
        sign = staticmethod(lambda message: sign(message, pair))

    class Wallet:
        hotkey = Hotkey()

    wallet = Wallet()
    monkeypatch.setenv("AURELIUS_IDENTITY_SOURCE", "bittensor")
    monkeypatch.setattr(serve, "load_bittensor_wallet", lambda: wallet)

    minter = minter_from_env("ignored-dev-name")
    assert minter.hotkey_ss58 == wallet.hotkey.ss58_address

    monkeypatch.setenv("AURELIUS_IDENTITY_SOURCE", "misspelled")
    with pytest.raises(ValueError, match="unknown AURELIUS_IDENTITY_SOURCE"):
        minter_from_env("ignored-dev-name")
