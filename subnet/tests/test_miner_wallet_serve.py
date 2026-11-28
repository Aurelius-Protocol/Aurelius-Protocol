"""The miner's identity and its work-ids (plan §1.8, invariant 1): a signer comes only from an
address and a signing callable, the coldkey is never in a payload, and serve mints a FRESH work_id
per response (same content_hash, different verifying work_id)."""

import json

import pytest

from subnet.miner.serve import WorkIdMinter, minter_from_bittensor_wallet, minter_from_env
from subnet.shared.signing import sign, verify_work_id_v2
from .miner_fixtures import fixed_keypair, fixed_minter, mount_wallet

pytestmark = pytest.mark.miner

_TASK = {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "same"}}
_VALIDATOR = fixed_minter("validator-hotkey").hotkey_ss58
_REQUEST_ID = "ab" * 32
_WINDOW = 7


def _envelope(minter):
    return minter.envelope(
        _TASK,
        validator_hotkey_ss58=_VALIDATOR,
        request_id=_REQUEST_ID,
        fanout_window=_WINDOW,
    )


def test_a_signer_needs_an_address_and_a_signing_callable():
    pair, address = fixed_keypair("miner-hotkey")

    def signer(message):
        return sign(message, pair)

    with pytest.raises(TypeError):
        WorkIdMinter()
    with pytest.raises(ValueError, match="hotkey_ss58"):
        WorkIdMinter(hotkey_ss58="", signer=signer)
    with pytest.raises(ValueError, match="callable"):
        WorkIdMinter(hotkey_ss58=address, signer="not callable")
    assert WorkIdMinter(hotkey_ss58=address, signer=signer).hotkey_ss58 == address


def test_fresh_work_id_per_response_same_content_hash():
    minter = fixed_minter("miner-hotkey")
    e1 = _envelope(minter)
    e2 = _envelope(minter)
    assert e1["content_hash"] == e2["content_hash"]  # same content
    assert e1["work_id"]["nonce"] != e2["work_id"]["nonce"]  # fresh nonce per response
    assert e1["work_id"]["signature_hex"] != e2["work_id"]["signature_hex"]  # distinct work_id


def test_nonce_does_not_repeat_when_the_process_local_minter_is_reconstructed():
    first = _envelope(fixed_minter("miner-hotkey"))
    after_restart = _envelope(fixed_minter("miner-hotkey"))
    assert first["work_id"]["nonce"] != after_restart["work_id"]["nonce"]


def test_coldkey_never_in_the_envelope():
    minter = fixed_minter("miner-hotkey")
    _pair, coldkey_ss58 = fixed_keypair("miner-coldkey")
    envelope = _envelope(minter)
    assert coldkey_ss58 not in json.dumps(envelope)  # the coldkey never leaves the miner
    assert envelope["work_id"]["hotkey_ss58"] == minter.hotkey_ss58  # only the hotkey signs


def test_injected_wallet_signer_authors_work_id_as_its_exact_ss58(monkeypatch):
    pair, ss58 = fixed_keypair("wallet-hotkey")
    wallet = mount_wallet(monkeypatch, "wallet-hotkey")
    minter = minter_from_bittensor_wallet(wallet)
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


def test_the_wallet_hotkey_is_the_only_identity_source(monkeypatch):
    wallet = mount_wallet(monkeypatch, "wallet-hotkey")
    assert minter_from_env().hotkey_ss58 == wallet.hotkey.ss58_address


def test_a_wallet_hotkey_without_a_signer_is_refused():
    class Hotkey:
        ss58_address = "5HotkeyWithoutSign"

    class Wallet:
        hotkey = Hotkey()

    with pytest.raises(ValueError, match="sign"):
        minter_from_bittensor_wallet(Wallet())
