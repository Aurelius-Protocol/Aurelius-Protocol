"""Gamma funding receipt wire identity (gamma plan §9): golden parity + event-identity reuse.

Pins the receipt canonical bytes against the shared golden vectors and the reuse of the
DEPLOYED physical-event identity (`chain_id‖block_hash‖extrinsic_index‖event_index`, U+2016
joins) as `deposit_key` — one physical on-chain event keeps exactly one credit identity across
funding providers. Evidence representation only: nothing here recomputes chain facts.
"""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from subnet.shared.gamma_funding_receipt import (
    DEPOSIT_KEY_SEPARATOR,
    GammaFundingReceipt,
    GammaFundingReceiptRefused,
    canonical_bytes,
    deposit_key,
    from_map,
    signature_payload,
    verify_signature,
)

pytestmark = pytest.mark.shared

_VECTORS_PATH = Path(__file__).parent / "fixtures" / "gamma_funding_vectors.json"
_DOC = json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))
_CASES = [case for case in _DOC["cases"] if case["kind"] == "receipt"]
_VALID = [case for case in _CASES if "error" not in case["expect"]]
_REJECTED = [case for case in _CASES if "error" in case["expect"]]


def _by_name(name):
    return next(case for case in _CASES if case["name"] == name)


@pytest.mark.parametrize("case", _VALID, ids=[c["name"] for c in _VALID])
def test_golden_receipts_match_canonical_bytes(case):
    receipt = from_map(case["input"])
    assert receipt.canonical_bytes().hex() == case["expect"]["canonical_bytes_hex"]


@pytest.mark.parametrize("case", _REJECTED, ids=[c["name"] for c in _REJECTED])
def test_golden_receipt_rejections_pin_the_refusal_kind(case):
    with pytest.raises(GammaFundingReceiptRefused) as exc:
        from_map(case["input"])
    assert exc.value.kind == case["expect"]["error"]


def test_deposit_key_reuses_the_deployed_u2016_event_identity():
    chain = "bittensor-mainnet:" + "2f" * 32
    block = "0x" + "ef" * 32
    assert DEPOSIT_KEY_SEPARATOR == "‖"
    assert deposit_key(chain, block, 2, 5) == f"{chain}‖{block}‖2‖5"
    golden = _by_name("receipt_first_gamma_credit")["input"]
    assert from_map(golden).deposit_key == deposit_key(
        golden["chain_id"], golden["finalized_block_hash"], 2, 5
    )


def test_deposit_key_identity_is_rederived_not_trusted():
    base = deepcopy(_by_name("receipt_first_gamma_credit")["input"])
    for mutate in (
        {"extrinsic_index": 3},
        {"event_index": 6},
        {"finalized_block_hash": "0x" + "ce" * 32},
        {"chain_id": "bittensor-mainnet:" + "30" * 32},
    ):
        tampered = dict(base, **mutate)
        with pytest.raises(GammaFundingReceiptRefused, match="deposit_key_mismatch"):
            from_map(tampered)
    non_string = dict(base, deposit_key=7)
    with pytest.raises(GammaFundingReceiptRefused, match="invalid_funding_receipt"):
        from_map(non_string)


@pytest.mark.parametrize(
    ("mutation", "kind"),
    [
        ({"schema": "aurelius/gamma_funding_receipt/v2"}, "invalid_funding_receipt"),
        ({"funding_method": "alpha"}, "invalid_funding_receipt"),
        ({"funding_method": "GAMMA"}, "invalid_funding_receipt"),
        ({"extra": True}, "invalid_funding_receipt"),
        ({"ledger_transaction_id": 0}, "invalid_funding_receipt"),
        ({"extrinsic_index": -1}, "invalid_funding_receipt"),
        ({"event_index": -1}, "invalid_funding_receipt"),
        # field-invalid event components refuse BEFORE the deposit_key re-derivation
        ({"finalized_block_hash": "0x" + "zz" * 32}, "invalid_funding_receipt"),
        ({"chain_id": "bittensor-testnet:" + "2f" * 32}, "invalid_funding_receipt"),
        ({"reference": "0x" + "zz" * 32}, "invalid_funding_receipt"),
        ({"tx_hash": "zz" * 32}, "invalid_funding_receipt"),
        ({"netuid": 0}, "invalid_funding_receipt"),
        ({"netuid": True}, "invalid_funding_receipt"),
        ({"finalized_block_number": True}, "invalid_funding_receipt"),
        ({"intent_id": "bad id!"}, "invalid_funding_receipt"),
        ({"pack_id": "Bad-Pack"}, "invalid_funding_receipt"),
        ({"gamma_atoms": "1.0"}, "invalid_funding_receipt"),
        ({"work_tokens": "0"}, "invalid_funding_receipt"),
        ({"verification_profile": ""}, "invalid_funding_receipt"),
        ({"signing_key_id": "x" * 65}, "invalid_funding_receipt"),
    ],
)
def test_strict_from_map_rejections(mutation, kind):
    base = deepcopy(_by_name("receipt_first_gamma_credit")["input"])
    with pytest.raises(GammaFundingReceiptRefused, match=kind):
        from_map(dict(base, **mutation))


def test_signature_payload_is_domain_tag_sep_canonical_body():
    body = _by_name("receipt_first_gamma_credit")["input"]
    payload = signature_payload(body)
    assert payload == b"aurelius/gamma_funding_receipt/v1\n" + b"\x00" + canonical_bytes(body)


def test_ed25519_signature_round_trip_over_the_exact_payload():
    body = _by_name("receipt_first_gamma_credit")["input"]
    payload = signature_payload(body)
    signing = Ed25519PrivateKey.from_private_bytes(bytes([91]) * 32)
    signature = signing.sign(payload)
    pubkey_hex = signing.public_key().public_bytes_raw().hex()
    assert verify_signature(payload, signature, pubkey_hex)
    assert not verify_signature(payload + b"x", signature, pubkey_hex)
    tampered = bytearray(signature)
    tampered[31] ^= 1
    assert not verify_signature(payload, bytes(tampered), pubkey_hex)


def test_from_map_returns_an_isolated_copy():
    doc = deepcopy(_by_name("receipt_first_gamma_credit")["input"])
    receipt = from_map(doc)
    assert isinstance(receipt, GammaFundingReceipt)
    doc["gamma_atoms"] = "999999999999"
    assert receipt.body["gamma_atoms"] == "1000000000"
