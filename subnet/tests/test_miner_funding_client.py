"""Funding cannot trust an unsigned tariff, changed destination or forged credit receipt."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from subnet.miner.funding_client import CHAIN_ID, TREASURY, read_funding, verified_deposit
from subnet.miner.read_client import ReadUnavailable
from subnet.miner.serve import WorkIdMinter
from subnet.shared.canonicalization import canonical_bytes
from subnet.shared.config_snapshot import (
    canonical_hash,
    SNAPSHOT_SIGNATURE_DOMAIN,
    null_keys_present,
)
from subnet.shared.price_policy import digest, shipped_catalog
from subnet.shared.sn37_economics import GENESIS_HASH, TARIFF_VERSION
from subnet.shared.ss58 import ss58_encode

pytestmark = pytest.mark.miner


def prepared_response():
    minter = WorkIdMinter("funding-client-test")
    body = json.loads((Path(__file__).parent / "fixtures/sn37_config_v24.json").read_text())["body"]
    policy = {
        "schema_version": 2,
        "version": 1,
        "effective_from_block": 100,
        "cost_evidence_sha256": "a" * 64,
        "lanes": {"interp_sae": {"gpu_sae_verdict": {"cost_model_version": 2, "work_tokens": 10}}},
        "conversion": {"alpha_rao_per_work_token": 1_000_000_000},
        "capital": {"mode": "none"},
    }
    body["entries"]["price_policy"] = {"current": policy, "next": None}
    body["null_keys_present"] = null_keys_present(
        {k: v for k, v in body.items() if k != "null_keys_present"}
    )
    signing = Ed25519PrivateKey.from_private_bytes(bytes([47]) * 32)
    config_hash = canonical_hash(body)
    snapshot = {
        "version": 1,
        "body": body,
        "canonical_hash": config_hash,
        "signature_hex": signing.sign(
            SNAPSHOT_SIGNATURE_DOMAIN + f"{config_hash}|1".encode()
        ).hex(),
    }
    response = {
        "schema_version": 1,
        "identity": {
            "hotkey": minter.hotkey_ss58,
            "coldkey": ss58_encode(bytes([2]) * 32),
            "netuid": 37,
        },
        "config_hash": config_hash,
        "finalized_block": 200,
        "observed_at": 123,
        "account": {
            "balance": {
                "available_real": 0,
                "available_synthetic": 0,
                "deposited_lifetime": 0,
                "consumed_lifetime": 0,
                "synthetic_granted_lifetime": 0,
                "synthetic_consumed_lifetime": 0,
            },
            "receipts": {
                "real": [],
                "synthetic": [],
                "more_real": False,
                "more_synthetic": False,
                "limit_per_ledger": 25,
            },
        },
        "terms": {
            "status": "available",
            "chain_id": CHAIN_ID,
            "genesis_hash": GENESIS_HASH,
            "netuid": 37,
            "destination_coldkey": TREASURY,
            "tariff_version": TARIFF_VERSION,
            "price_policy_digest": digest(policy, shipped_catalog()),
            "base_alpha_rao_per_work_token": 1_000_000_000,
            "evaluation_work_tokens": 10,
            "retention": "quadratic_fresh_lots_v3",
            "fractional_carry": "beneficiary",
            "refundable": False,
            "runtime_spec_version": 475,
            "transaction_version": 1,
        },
    }
    args = dict(
        hotkey=minter.hotkey_ss58,
        signer=minter.sign_economics,
        config_pubkey=signing.public_key().public_bytes_raw().hex(),
        now=123,
        fetch=lambda url, body=None: deepcopy(response if body else snapshot),
    )
    return snapshot, response, args


@pytest.mark.parametrize("version", [475, 477, 479])
def test_signed_effective_policy_binds_real_purchase_terms(version):
    _, response, args = prepared_response()
    response["terms"]["runtime_spec_version"] = version
    assert read_funding("https://example.test", **args) == response


def test_forged_config_refuses_funding_before_requesting_terms():
    snapshot, _, args = prepared_response()
    snapshot["signature_hex"] = "00" * 64
    requests = []
    args["fetch"] = lambda url, body=None: requests.append(url) or snapshot
    with pytest.raises(ReadUnavailable, match="invalid_funding_config"):
        read_funding("https://example.test", **args)
    assert requests == ["https://example.test/api/config"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("destination_coldkey", ss58_encode(bytes([7]) * 32)),
        ("evaluation_work_tokens", 1),
        ("base_alpha_rao_per_work_token", 1),
        ("retention", "flat"),
        ("genesis_hash", "0x" + "00" * 32),
        ("price_policy_digest", "b" * 64),
        ("runtime_spec_version", 476),
        ("runtime_spec_version", 478),
        ("runtime_spec_version", 480),
        ("runtime_spec_version", 479.0),
        ("runtime_spec_version", "479"),
        ("runtime_spec_version", True),
        ("refundable", 0),
        ("transaction_version", True),
    ],
)
def test_unsigned_terms_cannot_change_execution_authority(field, value):
    _, response, args = prepared_response()
    response["terms"][field] = value
    with pytest.raises(ReadUnavailable):
        read_funding("https://example.test", **args)


@pytest.mark.parametrize(
    "bad_account",
    [
        None,
        {"balance": {"available_real": True}, "receipts": {"real": []}},
        {"balance": {"available_real": 0}, "receipts": {"real": [None]}},
    ],
)
def test_malformed_account_cannot_supply_funding_confirmation(bad_account):
    _, response, args = prepared_response()
    response["account"] = bad_account
    with pytest.raises(ReadUnavailable):
        read_funding("https://example.test", **args)


def test_honest_unavailable_is_readable_but_never_executable():
    _, response, args = prepared_response()
    response["terms"] = {"status": "unavailable", "reason": "mainnet_retention_unavailable"}
    assert read_funding("https://example.test", **args)["terms"]["status"] == "unavailable"
    response["observed_at"] = 0
    with pytest.raises(ReadUnavailable):
        read_funding("https://example.test", **args)


def test_signed_own_receipt_confirms_exact_deposit_and_forgery_never_confirms():
    _, response, _ = prepared_response()
    key = Ed25519PrivateKey.from_private_bytes(bytes([31]) * 32)
    public = key.public_key().public_bytes_raw().hex()
    receipt = {
        "receipt_version": 4,
        "kind": "deposit",
        "deposit_key": "exact-key",
        **response["identity"],
        "genesis_hash": GENESIS_HASH,
        "tariff_version": TARIFF_VERSION,
        "denomination": "alpha",
        "alpha_amount_rao": 10_000_000_000,
        "tokens_credited": 10,
        "rate_version": TARIFF_VERSION,
        "base_work_tokens_per_alpha": "1",
        "price_policy_digest": "a" * 64,
        "commitment_policy_digest": "b" * 64,
        "commitment_state_digest": "c" * 64,
        "retention_multiple": "1",
        "carry_before": "0",
        "carry_after": "0",
    }
    row = {"receipt": receipt, "signature_hex": key.sign(canonical_bytes(receipt)).hex()}
    response["account"]["receipts"]["real"] = [row]
    assert verified_deposit(response, "exact-key", receipt_pubkey=public) == receipt
    assert verified_deposit(response, "unrelated-key", receipt_pubkey=public) is None
    row["receipt"]["tokens_credited"] += 1
    with pytest.raises(ReadUnavailable, match="invalid_deposit_receipt"):
        verified_deposit(response, "exact-key", receipt_pubkey=public)


@pytest.mark.parametrize("version", [475, 477, 479])
def test_explicit_base_terms_keep_signed_price_and_fixed_destination(version):
    _, response, args = prepared_response()
    response["terms"]["runtime_spec_version"] = version
    response["terms"]["retention"] = "base_rate_v1"
    assert read_funding("https://example.test", **args) == response
    response["terms"]["evaluation_work_tokens"] = 11
    with pytest.raises(ReadUnavailable):
        read_funding("https://example.test", **args)


def test_signed_base_receipt_names_exact_policy_without_claiming_history():
    from subnet.miner.funding_client import BASE_PURCHASE_POLICY_DIGEST

    _, response, _ = prepared_response()
    key = Ed25519PrivateKey.from_private_bytes(bytes([48]) * 32)
    public = key.public_key().public_bytes_raw().hex()
    receipt = {
        "receipt_version": 4,
        "kind": "deposit",
        "deposit_key": "base-key",
        **response["identity"],
        "genesis_hash": GENESIS_HASH,
        "tariff_version": TARIFF_VERSION,
        "rate_version": TARIFF_VERSION,
        "denomination": "alpha",
        "alpha_amount_rao": 10_500_000_000,
        "tokens_credited": 10,
        "base_work_tokens_per_alpha": "1",
        "price_policy_digest": "a" * 64,
        "purchase_policy": "sn37-base-1wt-v1",
        "purchase_policy_digest": BASE_PURCHASE_POLICY_DIGEST,
        "retention_multiple": "1",
        "carry_before": "0",
        "carry_after": "1/2",
    }
    row = {"receipt": receipt, "signature_hex": key.sign(canonical_bytes(receipt)).hex()}
    response["account"]["receipts"]["real"] = [row]
    assert verified_deposit(response, "base-key", receipt_pubkey=public) == receipt
    for field, value in [("purchase_policy_digest", "b" * 64), ("retention_multiple", "20")]:
        changed = {**receipt, field: value}
        row["receipt"] = changed
        row["signature_hex"] = key.sign(canonical_bytes(changed)).hex()
        with pytest.raises(ReadUnavailable):
            verified_deposit(response, "base-key", receipt_pubkey=public)
