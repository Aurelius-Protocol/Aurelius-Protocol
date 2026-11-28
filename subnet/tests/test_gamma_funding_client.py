"""Gamma funding client: two-proof quote auth, quote/receipt verification, terms v2 fallback.

STRICT fake transports (mirrors ``test_miner_funding_client``): every server answer is built
byte-exactly in-test, every trust root is a test key, and no test opens a socket or a wallet.
"""

from copy import deepcopy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from subnet.miner.funding_client import CHAIN_ID
from subnet.miner.gamma_funding import (
    GAMMA_AUTH_DOMAIN,
    GAMMA_BINDING_DOMAIN,
    TERMS_V2_DOMAIN,
    GammaRefused,
    build_quote_auth,
    find_receipt_envelope,
    read_gamma_terms,
    request_gamma_quote,
    verify_gamma_receipt,
    verify_policy_envelope,
    verify_quote,
)
from subnet.miner.read_client import ReadUnavailable
from subnet.miner.serve import WorkIdMinter
from subnet.shared import gamma_intent, gamma_policy
from subnet.shared.canonicalization import canonical_bytes
from subnet.shared.config_snapshot import (
    SNAPSHOT_SIGNATURE_DOMAIN,
    canonical_hash,
    null_keys_present,
)
from subnet.shared.price_policy import digest as price_digest
from subnet.shared.price_policy import shipped_catalog
from subnet.shared.signing import keypair_from_seed, sign, verify
from subnet.shared.sn37_economics import GENESIS_HASH, TARIFF_VERSION
from subnet.shared.ss58 import ss58_encode

pytestmark = pytest.mark.miner

TREASURY = "5Gx14QffqwC8wNHv4wUvfCfE2zAUYDNvF9Z7LjNY81WQx7iL"
FAKE_CHAIN_ID = "bittensor-mainnet:" + "ab" * 32
CONFIG_KEY = Ed25519PrivateKey.from_private_bytes(bytes([47]) * 32)
QUOTE_KEY = Ed25519PrivateKey.from_private_bytes(bytes([31]) * 32)


def _public_hex(key) -> str:
    return key.public_key().public_bytes_raw().hex()


def _account(seed_byte: int) -> str:
    return ss58_encode(bytes([seed_byte]) * 32)


# --- legacy terms response (byte-exact for the deployed strict parser) ------------------------


def legacy_terms_response(hotkey: str, *, available=True):
    body = {"config_version": 1, "entries": {}}
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
    config_hash = canonical_hash(body)
    snapshot = {
        "version": 1,
        "body": body,
        "canonical_hash": config_hash,
        "signature_hex": CONFIG_KEY.sign(
            SNAPSHOT_SIGNATURE_DOMAIN + f"{config_hash}|1".encode()
        ).hex(),
    }
    terms = {"status": "unavailable", "reason": "mainnet_funding_not_selected"}
    if available:
        terms = {
            "status": "available",
            "chain_id": CHAIN_ID,
            "genesis_hash": GENESIS_HASH,
            "netuid": 37,
            "destination_coldkey": TREASURY,
            "tariff_version": TARIFF_VERSION,
            "price_policy_digest": price_digest(policy, shipped_catalog()),
            "base_alpha_rao_per_work_token": 1_000_000_000,
            "evaluation_work_tokens": 10,
            "retention": "quadratic_fresh_lots_v3",
            "fractional_carry": "beneficiary",
            "refundable": False,
            "runtime_spec_version": 475,
            "transaction_version": 1,
        }
    response = {
        "schema_version": 1,
        "identity": {"coldkey": _account(2), "hotkey": hotkey, "netuid": 37},
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
        "terms": terms,
    }
    return snapshot, response


# --- signed gamma policy + quote fixtures ------------------------------------------------------


def purchase_policy(*, netuid=37, chain_id=CHAIN_ID, genesis=GENESIS_HASH, version="gpp-2026-11"):
    body = {
        "schema": gamma_policy.SCHEMA,
        "service_id": "gamma-aurelius-mainnet",
        "chain_id": chain_id,
        "genesis_hash": genesis,
        "netuid": netuid,
        "policy_version": version,
        "packs": {
            "starter": {"gamma_atoms": "1000000000", "work_tokens": "1"},
            "ten": {"gamma_atoms": "10000000000", "work_tokens": "10"},
        },
        "activation_block": 1,
        "quote_lifetime_blocks": 300,
        "max_packs_per_intent": 1,
        "per_account_packs_per_day": 24,
        "global_packs_per_day": 10_000,
        "payer_mode": gamma_policy.PAYER_MODE,
        "runtime_profile": "gamma-mainnet-490",
        "governing_config_hash": "0x" + "cd" * 32,
        "signing_key_id": "operator-receipt-v1",
    }
    policy = gamma_policy.from_map(body)
    preimage = (
        b"aurelius/gamma_purchase_policy/v1\n"
        + policy.digest()[2:].encode()
        + b"|"
        + version.encode()
    )
    envelope = {
        "body": body,
        "signature_hex": CONFIG_KEY.sign(preimage).hex(),
        "policy_digest": policy.digest(),
    }
    return policy, envelope


def quote_intent(policy: gamma_policy.GammaPurchasePolicy, *, pack_id="starter", **overrides):
    pack = policy.pack(pack_id)
    assigns = {
        "intent_id": "gfi-" + "1" * 16,
        "beneficiary_coldkey": _account(2),
        "beneficiary_hotkey": _account(5),
        "funding_coldkey": _account(2),
        "gamma_api_hotkey": _account(9),
        "authorized_spender": _account(2),
        "gamma_atoms": str(pack["gamma_atoms"]),
        "work_tokens": str(pack["work_tokens"]),
        "config_hash": "0x" + "ab" * 32,
        "issued_at_finalized_block": 1000,
        "valid_from_block": 1000,
        "valid_until_block": 1000 + policy.quote_lifetime_blocks,
        "server_nonce": "0x" + "22" * 32,
        "client_request_id": "req-1",
    }
    assigns.update(overrides)
    return gamma_intent.build(policy, pack_id, **assigns)


def quote_response(intent: gamma_intent.GammaFundingIntent, *, key=QUOTE_KEY, key_id=None):
    return {
        "schema_version": 1,
        "intent_id": intent.intent_id,
        "reference": intent.reference_hex,
        "valid_from_block": intent.body["valid_from_block"],
        "valid_until_block": intent.body["valid_until_block"],
        "envelope": {
            "body": intent.body,
            "signature_hex": key.sign(intent.signature_payload()).hex(),
            "key_id": key_id or "operator-receipt-v1",
        },
    }


def verify_kwargs(policy, intent):
    return dict(
        policy=policy,
        beneficiary_coldkey=intent.body["beneficiary_coldkey"],
        beneficiary_hotkey=intent.body["beneficiary_hotkey"],
        funding_coldkey=intent.body["funding_coldkey"],
        gamma_api_hotkey=intent.body["gamma_api_hotkey"],
        config_hash=intent.body["config_hash"][2:],
        pack_id=intent.body["pack_id"],
        quote_pubkey=_public_hex(QUOTE_KEY),
    )


# --- signed policy discovery -------------------------------------------------------------------


def test_policy_discovery_verifies_against_the_shipped_config_root():
    policy, envelope = purchase_policy()
    verified = verify_policy_envelope(envelope, config_pubkey=_public_hex(CONFIG_KEY))
    assert verified.digest() == policy.digest()


@pytest.mark.parametrize("fault", ["signature", "digest", "chain", "netuid", "genesis", "shape"])
def test_policy_discovery_refuses_forged_or_foreign_envelopes(fault):
    policy, envelope = purchase_policy()
    if fault == "signature":
        envelope["signature_hex"] = (
            Ed25519PrivateKey.from_private_bytes(bytes([48]) * 32)
            .sign(
                b"aurelius/gamma_purchase_policy/v1\n"
                + policy.digest()[2:].encode()
                + b"|gpp-2026-11"
            )
            .hex()
        )
    elif fault == "digest":
        envelope = {**envelope, "policy_digest": "0x" + "00" * 32}
    elif fault == "chain":
        policy, envelope = purchase_policy(chain_id=FAKE_CHAIN_ID)
    elif fault == "netuid":
        policy, envelope = purchase_policy(netuid=38)
    elif fault == "genesis":
        policy, envelope = purchase_policy(genesis="0x" + "00" * 32)
    else:
        envelope = {**envelope, "policy_digest": "0x" + "00" * 32, "extra": 1}
    with pytest.raises((GammaRefused, gamma_policy.GammaPolicyRefused)):
        verify_policy_envelope(envelope, config_pubkey=_public_hex(CONFIG_KEY))


# --- terms_version 2 read with old-server fallback ---------------------------------------------


def gamma_methods_block(envelope):
    return {
        "terms_version": 2,
        "legacy": {"available": True},
        "gamma": {"available": True, "reason": "available", "policy": envelope},
    }


def test_terms_v2_read_validates_legacy_block_and_policy_envelope():
    hotkey = _account(5)
    snapshot, response = legacy_terms_response(hotkey)
    policy, envelope = purchase_policy()
    v2 = {**deepcopy(response), "funding_methods": gamma_methods_block(envelope)}
    requests = []

    def fetch(url, body=None):
        requests.append((url, body))
        return deepcopy(snapshot if body is None else v2)

    pair = keypair_from_seed(b"t" * 32)
    result = read_gamma_terms(
        "https://example.test",
        hotkey=hotkey,
        funding_terms_signer=lambda message: sign(message, pair),
        economics_signer=lambda message: sign(message, pair),
        config_pubkey=_public_hex(CONFIG_KEY),
        fetch=fetch,
        now=123,
    )
    assert result["terms_version"] == 2
    assert result["legacy"] == response
    assert result["funding_methods"]["gamma"]["available"] is True
    assert result["funding_methods"]["gamma"]["policy"]["packs"]["starter"] == {
        "gamma_atoms": 1_000_000_000,
        "work_tokens": 1,
    }
    sent = requests[1][1]
    assert sent["terms_version"] == 2 and set(sent) == {
        "hotkey",
        "netuid",
        "config_hash",
        "timestamp",
        "nonce",
        "terms_version",
        "signature_hex",
    }
    signed = (
        TERMS_V2_DOMAIN
        + b"\x00"
        + canonical_bytes({k: v for k, v in sent.items() if k != "signature_hex"})
    )
    assert verify(bytes.fromhex(sent["signature_hex"]), signed, pair[0])


def test_old_server_falls_back_to_the_plain_v1_read_without_gamma_fields():
    hotkey = _account(5)
    snapshot, response = legacy_terms_response(hotkey)
    calls = []

    def fetch(url, body=None):
        calls.append(body)
        if body is None:
            return deepcopy(snapshot)
        if body.get("terms_version") == 2:
            raise ReadUnavailable("http_error", 401)
        return deepcopy(response)

    pair = keypair_from_seed(b"t" * 32)
    result = read_gamma_terms(
        "https://example.test",
        hotkey=hotkey,
        funding_terms_signer=lambda message: sign(message, pair),
        economics_signer=lambda message: sign(message, pair),
        config_pubkey=_public_hex(CONFIG_KEY),
        fetch=fetch,
        now=123,
    )
    assert result["terms_version"] == 1
    assert result["funding_methods"] is None
    assert result["legacy"] == response
    assert calls[1].get("terms_version") == 2 and "terms_version" not in calls[3]


@pytest.mark.parametrize(
    "bad",
    [
        {"terms_version": 3},
        {"legacy": {"available": "yes"}},
        {"gamma": {"available": 1, "reason": "x"}},
    ],
)
def test_malformed_funding_methods_block_refuses(bad):
    hotkey = _account(5)
    snapshot, response = legacy_terms_response(hotkey)
    _, envelope = purchase_policy()
    block = gamma_methods_block(envelope)
    block.update(bad)
    v2 = {**deepcopy(response), "funding_methods": block}

    def fetch(url, body=None):
        return deepcopy(snapshot if body is None else v2)

    pair = keypair_from_seed(b"t" * 32)
    with pytest.raises(GammaRefused):
        read_gamma_terms(
            "https://example.test",
            hotkey=hotkey,
            funding_terms_signer=lambda message: sign(message, pair),
            economics_signer=lambda message: sign(message, pair),
            config_pubkey=_public_hex(CONFIG_KEY),
            fetch=fetch,
            now=123,
        )


# --- two-proof quote authorization -------------------------------------------------------------


def hotkey_and_coldkey_signers():
    hotkey_pair = keypair_from_seed(b"h" * 32)
    coldkey_pair = keypair_from_seed(b"c" * 32)
    hotkey_calls = []
    coldkey_calls = []

    def hotkey_signer(message):
        hotkey_calls.append(message)
        return sign(message, hotkey_pair)

    def coldkey_signer(message):
        coldkey_calls.append(message)
        return sign(message, coldkey_pair)

    return hotkey_signer, coldkey_signer, hotkey_pair, coldkey_pair, hotkey_calls, coldkey_calls


def test_quote_auth_builds_the_exact_g4_two_proof_bodies():
    hotkey_signer, coldkey_signer, hotkey_pair, coldkey_pair, hotkey_calls, coldkey_calls = (
        hotkey_and_coldkey_signers()
    )
    auth = build_quote_auth(
        beneficiary_coldkey=_account(2),
        beneficiary_hotkey=ss58_encode(hotkey_pair[0]),
        funding_coldkey=ss58_encode(coldkey_pair[0]),
        gamma_api_hotkey=_account(9),
        pack_id="starter",
        config_hash="ab" * 32,
        client_request_id="req-1",
        gamma_auth_signer=hotkey_signer,
        coldkey_signer=coldkey_signer,
        now=1000,
    )
    body = {
        k: v for k, v in auth.items() if k not in {"auth_signature_hex", "binding_signature_hex"}
    }
    assert hotkey_calls == [GAMMA_AUTH_DOMAIN + b"\x00" + canonical_bytes(body)]
    assert coldkey_calls == [GAMMA_BINDING_DOMAIN + canonical_bytes(auth["binding"])]
    assert verify(
        bytes.fromhex(auth["auth_signature_hex"]),
        hotkey_calls[0],
        hotkey_pair[0],
    )
    assert verify(
        bytes.fromhex(auth["binding_signature_hex"]),
        coldkey_calls[0],
        coldkey_pair[0],
    )
    assert auth["method"] == "quote"
    assert auth["route"] == "/api/miner/funding/gamma/intents"
    assert auth["binding"]["authorized_spender"] == auth["binding"]["funding_coldkey"]
    assert auth["expires_at"] - auth["issued_at"] == 60


def test_request_maps_typed_server_refusals_and_validates_the_response_shape():
    _, envelope = purchase_policy()
    policy = gamma_policy.from_map(envelope["body"])
    intent = quote_intent(policy)
    good = quote_response(intent)

    def fetch(url, body=None):
        assert url.endswith("/api/miner/funding/gamma/intents")
        assert set(body) == {"auth", "request"}
        assert set(body["request"]) == {
            "beneficiary_coldkey",
            "beneficiary_hotkey",
            "pack_id",
            "client_request_id",
        }
        return deepcopy(good)

    auth = {
        "beneficiary_coldkey": _account(2),
        "beneficiary_hotkey": _account(5),
        "pack_id": "starter",
        "client_request_id": "req-1",
    }
    assert request_gamma_quote("https://x.test", auth=auth, fetch=fetch) == good

    def refusing(url, body=None):
        raise ReadUnavailable("http_error", 429, {"error": "daily_cap_insufficient"})

    with pytest.raises(GammaRefused, match="daily_cap_insufficient"):
        request_gamma_quote("https://x.test", auth=auth, fetch=refusing)

    bad = {**deepcopy(good), "unexpected": 1}
    with pytest.raises(GammaRefused, match="invalid quote response"):
        request_gamma_quote("https://x.test", auth=auth, fetch=lambda u, body=None: bad)


# --- quote verification: happy path + every refusal --------------------------------------------


def test_a_policy_bound_quote_verifies_fully():
    policy, _ = purchase_policy()
    intent = quote_intent(policy)
    verified = verify_quote(quote_response(intent), **verify_kwargs(policy, intent))
    assert verified.intent_id == intent.intent_id
    assert verified.reference_hex == intent.reference_hex


@pytest.mark.parametrize(
    "fault",
    [
        "signature",
        "key_id",
        "policy",
        "amounts",
        "chain",
        "genesis",
        "netuid",
        "beneficiary",
        "api_hotkey",
        "config_hash",
        "reference",
        "pack",
    ],
)
def test_every_quote_mismatch_refuses_before_any_signing(fault):
    other_policy, _ = purchase_policy(version="gpp-other")
    policy, _ = purchase_policy()
    intent = quote_intent(policy)
    kwargs = verify_kwargs(policy, intent)
    response = quote_response(intent)
    if fault == "signature":
        response = quote_response(
            intent, key=Ed25519PrivateKey.from_private_bytes(bytes([50]) * 32)
        )
    elif fault == "key_id":
        response = quote_response(intent, key_id="unknown-key")
    elif fault == "policy":
        response = quote_response(intent)
        kwargs["policy"] = other_policy
    elif fault == "amounts":
        body = {**intent.body, "gamma_atoms": "2000000000"}
        forged = gamma_intent.from_map(body)
        response = quote_response(forged)
        kwargs["pack_id"] = "starter"
    elif fault == "chain":
        foreign, _ = purchase_policy(chain_id=FAKE_CHAIN_ID)
        response = quote_response(quote_intent(foreign))
    elif fault == "genesis":
        foreign, _ = purchase_policy(genesis="0x" + "01" * 32)
        response = quote_response(quote_intent(foreign))
    elif fault == "netuid":
        foreign, _ = purchase_policy(netuid=38)
        response = quote_response(quote_intent(foreign))
    elif fault == "beneficiary":
        body = {**intent.body, "beneficiary_coldkey": _account(7)}
        body["funding_coldkey"] = _account(7)
        body["authorized_spender"] = _account(7)
        response = quote_response(gamma_intent.from_map(body))
    elif fault == "api_hotkey":
        kwargs["gamma_api_hotkey"] = _account(8)
    elif fault == "config_hash":
        kwargs["config_hash"] = "cd" * 32
    elif fault == "reference":
        response = quote_response(intent)
        response["reference"] = "0x" + "00" * 32
    else:
        response = quote_response(intent)
        kwargs["pack_id"] = "ten"
    with pytest.raises((GammaRefused, gamma_intent.GammaIntentRefused)):
        verify_quote(response, **kwargs)


def test_a_gamma_api_hotkey_equal_to_the_beneficiary_identity_can_never_be_issued():
    # A server can never issue an intent whose API hotkey equals the beneficiary identity
    # (G1 from_map refuses it); a client-supplied one merely differs from the signed quote.
    policy, _ = purchase_policy()
    intent = quote_intent(policy)
    kwargs = verify_kwargs(policy, intent)
    kwargs["gamma_api_hotkey"] = intent.body["beneficiary_hotkey"]
    with pytest.raises(GammaRefused, match="beneficiary/binding"):
        verify_quote(quote_response(intent), **kwargs)
    body = {**intent.body, "gamma_api_hotkey": intent.body["beneficiary_hotkey"]}
    with pytest.raises(gamma_intent.GammaIntentRefused, match="gamma_api_hotkey_not_distinct"):
        gamma_intent.from_map(body)


# --- signed receipt: the only credit proof -----------------------------------------------------


def receipt_envelope(intent: gamma_intent.GammaFundingIntent, *, key=QUOTE_KEY, **overrides):
    from subnet.shared import gamma_funding_receipt

    body = {
        "schema": gamma_funding_receipt.SCHEMA,
        "funding_method": "gamma",
        "intent_id": intent.intent_id,
        "reference": intent.reference_hex,
        "pack_id": intent.body["pack_id"],
        "purchase_policy_version": intent.body["purchase_policy_version"],
        "purchase_policy_digest": intent.body["purchase_policy_digest"],
        "config_hash": intent.body["config_hash"],
        "beneficiary_coldkey": intent.body["beneficiary_coldkey"],
        "beneficiary_hotkey": intent.body["beneficiary_hotkey"],
        "chain_id": intent.body["chain_id"],
        "genesis_hash": intent.body["genesis_hash"],
        "netuid": intent.body["netuid"],
        "finalized_block_hash": "0x" + "ee" * 32,
        "finalized_block_number": 1200,
        "extrinsic_index": 3,
        "event_index": 11,
        "tx_hash": "0x" + "ff" * 32,
        "funding_coldkey": intent.body["funding_coldkey"],
        "gamma_api_hotkey": intent.body["gamma_api_hotkey"],
        "authorized_spender": intent.body["authorized_spender"],
        "gamma_atoms": intent.body["gamma_atoms"],
        "work_tokens": intent.body["work_tokens"],
        "deposit_key": gamma_funding_receipt.deposit_key(
            intent.body["chain_id"], "0x" + "ee" * 32, 3, 11
        ),
        "ledger_transaction_id": 9,
        "verification_profile": "gamma-mainnet-490",
        "signing_key_id": "operator-receipt-v1",
    }
    body.update(overrides)
    return {
        "body": body,
        "signature_hex": key.sign(gamma_funding_receipt.signature_payload(body)).hex(),
        "key_id": body["signing_key_id"],
    }


def test_a_fully_lineaged_signed_receipt_verifies():
    policy, _ = purchase_policy()
    intent = quote_intent(policy)
    envelope = receipt_envelope(intent)
    receipt = verify_gamma_receipt(
        envelope,
        intent={
            "intent_id": intent.intent_id,
            "reference_hex": intent.reference_hex,
            "pack_id": intent.body["pack_id"],
            "purchase_policy_version": intent.body["purchase_policy_version"],
            "purchase_policy_digest": intent.body["purchase_policy_digest"],
            "config_hash": intent.body["config_hash"],
            "beneficiary_coldkey": intent.body["beneficiary_coldkey"],
            "beneficiary_hotkey": intent.body["beneficiary_hotkey"],
            "funding_coldkey": intent.body["funding_coldkey"],
            "gamma_api_hotkey": intent.body["gamma_api_hotkey"],
            "authorized_spender": intent.body["authorized_spender"],
            "chain_id": intent.body["chain_id"],
            "genesis_hash": intent.body["genesis_hash"],
            "netuid": intent.body["netuid"],
            "gamma_atoms": intent.body["gamma_atoms"],
            "work_tokens": intent.body["work_tokens"],
        },
        quote_pubkey=_public_hex(QUOTE_KEY),
    )
    assert receipt["work_tokens"] == "1"
    assert receipt["deposit_key"].startswith(CHAIN_ID)


@pytest.mark.parametrize(
    "fault", ["signature", "amount", "reference", "beneficiary", "key_id", "work_tokens"]
)
def test_receipt_spoofs_and_lineage_breaks_refuse(fault):
    policy, _ = purchase_policy()
    intent = quote_intent(policy)
    overrides = {}
    if fault == "signature":
        envelope = receipt_envelope(
            intent, key=Ed25519PrivateKey.from_private_bytes(bytes([51]) * 32)
        )
    else:
        if fault == "amount":
            overrides["gamma_atoms"] = "9000000000"
        elif fault == "reference":
            overrides["reference"] = "0x" + "00" * 32
        elif fault == "beneficiary":
            overrides["beneficiary_coldkey"] = _account(7)
        elif fault == "work_tokens":
            overrides["work_tokens"] = "5"
        envelope = receipt_envelope(intent, **overrides)
    if fault == "key_id":
        envelope = {**envelope, "key_id": "other"}
    with pytest.raises((GammaRefused, ValueError)):
        verify_gamma_receipt(
            envelope,
            intent={
                "intent_id": intent.intent_id,
                "reference_hex": intent.reference_hex,
                "pack_id": intent.body["pack_id"],
                "purchase_policy_version": intent.body["purchase_policy_version"],
                "purchase_policy_digest": intent.body["purchase_policy_digest"],
                "config_hash": intent.body["config_hash"],
                "beneficiary_coldkey": intent.body["beneficiary_coldkey"],
                "beneficiary_hotkey": intent.body["beneficiary_hotkey"],
                "funding_coldkey": intent.body["funding_coldkey"],
                "gamma_api_hotkey": intent.body["gamma_api_hotkey"],
                "authorized_spender": intent.body["authorized_spender"],
                "chain_id": intent.body["chain_id"],
                "genesis_hash": intent.body["genesis_hash"],
                "netuid": intent.body["netuid"],
                "gamma_atoms": intent.body["gamma_atoms"],
                "work_tokens": intent.body["work_tokens"],
            },
            quote_pubkey=_public_hex(QUOTE_KEY),
        )


def test_a_balance_change_or_alpha_receipt_is_never_a_gamma_confirmation():
    policy, _ = purchase_policy()
    intent = quote_intent(policy)
    _, response = legacy_terms_response(_account(5))
    # A balance increase plus an alpha-shaped receipt row is NOT a gamma confirmation.
    response["account"]["balance"]["available_real"] += 10
    gamma_row = receipt_envelope(intent)["body"]
    response["account"]["receipts"]["real"] = [
        {
            "id": 1,
            "kind": "deposit",
            "amount": 10,
            "receipt": {
                "receipt_version": 4,
                "kind": "deposit",
                "deposit_key": "not-gamma",
                "intent_id": gamma_row["intent_id"],
                "reference": gamma_row["reference"],
            },
            "signature_hex": "00" * 64,
            "key_id": "operator-receipt-v1",
            "inserted_at": "2026-11-14T00:00:00Z",
        }
    ]
    assert find_receipt_envelope(response) is None
    # A gamma-schema row IS picked up as a candidate (the caller then signature-verifies it).
    signed = receipt_envelope(intent)
    response["account"]["receipts"]["real"][0]["receipt"] = signed["body"]
    response["account"]["receipts"]["real"][0]["signature_hex"] = signed["signature_hex"]
    assert find_receipt_envelope(response) == {
        "body": signed["body"],
        "signature_hex": signed["signature_hex"],
        "key_id": "operator-receipt-v1",
    }


def test_a_foreign_gamma_receipt_under_the_same_account_is_skipped_not_refused():
    # Two genuine purchases share one funding account, so the account read lists BOTH gamma
    # receipts. The journal's own receipt must still be found (and fully verify): the foreign
    # row is skipped during the scan — never mistaken for a forged lineage ("receipt lineage
    # differs: intent_id") of THIS journal.
    policy, _ = purchase_policy()
    mine = quote_intent(policy)
    other = quote_intent(policy, intent_id="gfi-" + "2" * 16, client_request_id="req-2")
    assert other.intent_id != mine.intent_id
    _, response = legacy_terms_response(_account(5))
    foreign = receipt_envelope(other)
    matching = receipt_envelope(mine)

    def row(row_id, envelope, inserted_at):
        return {
            "id": row_id,
            "kind": "deposit",
            "amount": 1,
            "receipt": envelope["body"],
            "signature_hex": envelope["signature_hex"],
            "key_id": envelope["key_id"],
            "inserted_at": inserted_at,
        }

    # The foreign receipt comes FIRST — the unfiltered scan used to stop there.
    response["account"]["receipts"]["real"] = [
        row(1, foreign, "2026-11-14T00:00:00Z"),
        row(2, matching, "2026-11-14T00:01:00Z"),
    ]
    intent = {
        "intent_id": mine.intent_id,
        "reference_hex": mine.reference_hex,
        "pack_id": mine.body["pack_id"],
        "purchase_policy_version": mine.body["purchase_policy_version"],
        "purchase_policy_digest": mine.body["purchase_policy_digest"],
        "config_hash": mine.body["config_hash"],
        "beneficiary_coldkey": mine.body["beneficiary_coldkey"],
        "beneficiary_hotkey": mine.body["beneficiary_hotkey"],
        "funding_coldkey": mine.body["funding_coldkey"],
        "gamma_api_hotkey": mine.body["gamma_api_hotkey"],
        "authorized_spender": mine.body["authorized_spender"],
        "chain_id": mine.body["chain_id"],
        "genesis_hash": mine.body["genesis_hash"],
        "netuid": mine.body["netuid"],
        "gamma_atoms": mine.body["gamma_atoms"],
        "work_tokens": mine.body["work_tokens"],
    }
    envelope = find_receipt_envelope(response, intent=intent)
    assert envelope is not None
    assert envelope["body"]["intent_id"] == mine.intent_id
    # The matching envelope fully verifies — no hard refusal from the foreign row.
    receipt = verify_gamma_receipt(envelope, intent=intent, quote_pubkey=_public_hex(QUOTE_KEY))
    assert receipt["intent_id"] == mine.intent_id
    # Without a prepared-intent selector the first gamma row still wins (candidate semantics).
    assert find_receipt_envelope(response)["body"]["intent_id"] == other.intent_id
    # A journal whose receipt the server has not shipped yet gets None, not the foreign row.
    assert find_receipt_envelope(response, intent={**intent, "intent_id": "gfi-absent"}) is None


# --- WorkIdMinter domain-restricted signers ----------------------------------------------------


def test_minter_signers_are_domain_restricted_and_verifiable():
    pair = keypair_from_seed(b"m" * 32)
    minter = WorkIdMinter(
        hotkey_ss58=ss58_encode(pair[0]), signer=lambda message: sign(message, pair)
    )
    terms = minter.sign_miner_funding_terms(TERMS_V2_DOMAIN + b"\x00" + b"body")
    auth = minter.sign_gamma_funding_auth(GAMMA_AUTH_DOMAIN + b"\x00" + b"body")
    assert verify(terms, TERMS_V2_DOMAIN + b"\x00" + b"body", pair[0])
    assert verify(auth, GAMMA_AUTH_DOMAIN + b"\x00" + b"body", pair[0])
    with pytest.raises(ValueError):
        minter.sign_miner_funding_terms(b"other-domain\x00body")
    with pytest.raises(ValueError):
        minter.sign_gamma_funding_auth(GAMMA_BINDING_DOMAIN + b"body")
