"""Gamma funding intent wire identity (gamma plan §8.2): golden parity + typed build seams.

Pins the EXACT 32-byte reference formula, the V1 own-account/distinct-hotkey invariants, the
policy-bound `build` cross-checks and the Ed25519 signature payload round-trip. The Elixir
twin asserts the same vector file.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from subnet.shared.canonicalization import CanonicalizationError, parse_canonical_json
from subnet.shared.gamma_intent import (
    REFERENCE_DOMAIN,
    SIGNATURE_DOMAIN,
    GammaIntentRefused,
    build,
    canonical_bytes,
    from_map,
    reference,
    signature_payload,
    ss58_decode_account,
    ss58_encode_account,
    verify_signature,
)
from subnet.shared.gamma_policy import GammaPolicyRefused, from_map as policy_from_map

pytestmark = pytest.mark.shared

_VECTORS_PATH = Path(__file__).parent / "fixtures" / "gamma_funding_vectors.json"
_DOC = json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))
_CASES = [case for case in _DOC["cases"] if case["kind"] == "intent"]
_VALID = [case for case in _CASES if "error" not in case["expect"]]
_REJECTED = [case for case in _CASES if "error" in case["expect"]]


def _by_name(name):
    return next(case for case in _CASES if case["name"] == name)


def base_policy_map():
    return deepcopy(_by_name("build_unknown_pack")["input"]["policy"])


def base_assigns():
    return deepcopy(_by_name("build_nil_policy")["input"]["assigns"])


@pytest.mark.parametrize("case", _VALID, ids=[c["name"] for c in _VALID])
def test_golden_intents_match_canonical_bytes_and_reference(case):
    intent = from_map(case["input"])
    assert intent.canonical_bytes().hex() == case["expect"]["canonical_bytes_hex"]
    assert intent.reference() == case["expect"]["reference_hex"]


@pytest.mark.parametrize("case", _REJECTED, ids=[c["name"] for c in _REJECTED])
def test_golden_intent_rejections_pin_the_refusal_kind(case):
    op = case["op"]
    if op == "parse_from_map":
        with pytest.raises(CanonicalizationError) as exc:
            from_map(parse_canonical_json(case["input"]))
        observed = exc.value.kind
    elif op == "build":
        raw_policy = case["input"]["policy"]
        policy = None if raw_policy is None else policy_from_map(raw_policy)
        with pytest.raises((GammaIntentRefused, GammaPolicyRefused)) as exc:
            build(policy, case["input"]["pack_id"], **case["input"]["assigns"])
        observed = exc.value.kind
    else:
        with pytest.raises(GammaIntentRefused) as exc:
            from_map(case["input"])
        observed = exc.value.kind
    assert observed == case["expect"]["error"]


def test_reference_formula_matches_an_inline_sha256():
    body = _by_name("intent_happy")["input"]
    inline = (
        "0x"
        + hashlib.sha256(
            b"aurelius:wt-gamma-purchase:v1" + b"\x00" + canonical_bytes(body)
        ).hexdigest()
    )
    assert reference(body) == inline == _by_name("intent_happy")["expect"]["reference_hex"]
    assert REFERENCE_DOMAIN == b"aurelius:wt-gamma-purchase:v1"  # no trailing newline (§8.2)


def test_signature_payload_binds_domain_body_and_raw_reference():
    body = _by_name("intent_happy")["input"]
    payload = signature_payload(body)
    encoded = canonical_bytes(body)
    assert payload.startswith(SIGNATURE_DOMAIN + b"\x00")
    assert payload[len(SIGNATURE_DOMAIN) + 1 : -32] == encoded
    assert payload[-32:] == bytes.fromhex(reference(body)[2:])


def test_build_derives_policy_identity_and_honours_verbatim_repeats():
    policy = policy_from_map(base_policy_map())
    assigns = base_assigns()
    intent = build(policy, "starter", **assigns)
    assert intent.body["service_id"] == policy.service_id
    assert intent.body["chain_id"] == policy.chain_id
    assert intent.body["purchase_policy_version"] == policy.policy_version
    assert intent.body["purchase_policy_digest"] == policy.digest()
    verbatim = dict(assigns, service_id=policy.service_id)
    assert build(policy, "starter", **verbatim).body == intent.body


def test_build_cross_checks_refuse_typed():
    policy = policy_from_map(base_policy_map())

    def expect_kind(kind, **overrides):
        with pytest.raises((GammaIntentRefused, GammaPolicyRefused)) as exc:
            build(policy, overrides.pop("pack_id", "starter"), **{**base_assigns(), **overrides})
        assert exc.value.kind == kind

    expect_kind("purchase_policy_mismatch", service_id="other-service")
    expect_kind("purchase_policy_mismatch", purchase_policy_digest="0x" + "ee" * 32)
    expect_kind("quote_lifetime_mismatch", valid_until_block=9_400_401)
    expect_kind("invalid_funding_intent", issued_at_finalized_block="soon")
    expect_kind("invalid_funding_intent", surprise=1)
    missing = base_assigns()
    del missing["server_nonce"]
    with pytest.raises(GammaIntentRefused, match="invalid_funding_intent"):
        build(policy, "starter", **missing)


def test_v1_invariants_and_account_domain():
    happy = deepcopy(_by_name("intent_happy")["input"])

    api_is_coldkey = dict(happy, gamma_api_hotkey=happy["beneficiary_coldkey"])
    with pytest.raises(GammaIntentRefused, match="gamma_api_hotkey_not_distinct"):
        from_map(api_is_coldkey)

    api_is_funding = dict(happy, gamma_api_hotkey=happy["funding_coldkey"])
    with pytest.raises(GammaIntentRefused, match="gamma_api_hotkey_not_distinct"):
        from_map(api_is_funding)

    bad_charset = dict(happy, beneficiary_hotkey="0OIlLucK")
    with pytest.raises(GammaIntentRefused, match="invalid_ss58_account"):
        from_map(bad_charset)

    wrong_prefix = dict(happy, authorized_spender=happy["authorized_spender"] + "2")
    with pytest.raises(GammaIntentRefused, match="invalid_ss58_account"):
        from_map(wrong_prefix)


def test_block_bounds_and_ordering():
    happy = _by_name("intent_happy")["input"]
    with pytest.raises(GammaIntentRefused, match="invalid_funding_intent"):
        from_map(dict(happy, issued_at_finalized_block=-1))
    with pytest.raises(GammaIntentRefused, match="invalid_funding_intent"):
        from_map(dict(happy, valid_until_block=4_294_967_296))
    with pytest.raises(GammaIntentRefused, match="valid_from_before_issued"):
        from_map(dict(happy, issued_at_finalized_block=happy["valid_from_block"] + 1))


def test_unknown_field_and_wrong_schema_reject():
    happy = _by_name("intent_happy")["input"]
    with pytest.raises(GammaIntentRefused, match="invalid_funding_intent"):
        from_map(dict(happy, schema="aurelius/gamma_funding_intent/v2"))
    with pytest.raises(GammaIntentRefused, match="invalid_funding_intent"):
        from_map(dict(happy, extra=True))


def test_dep_free_ss58_codec_round_trips_and_knows_alice():
    # public substrate dev vector //Alice (test vector only — never a case beneficiary)
    alice = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    assert ss58_decode_account(alice).hex() == (
        "d43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d"
    )
    key = bytes(range(32))
    encoded = ss58_encode_account(key)
    assert len(encoded) == 48 and ss58_decode_account(encoded) == key
    with pytest.raises(ValueError, match="32-byte"):
        ss58_encode_account(b"short")


def test_ed25519_signature_round_trip_over_the_exact_payload():
    body = _by_name("intent_happy")["input"]
    payload = signature_payload(body)
    signing = Ed25519PrivateKey.from_private_bytes(bytes([47]) * 32)
    signature = signing.sign(payload)
    pubkey_hex = signing.public_key().public_bytes_raw().hex()
    assert verify_signature(payload, signature, pubkey_hex)
    assert verify_signature(payload, signature.hex(), pubkey_hex)
    assert not verify_signature(payload[:-1] + b"X", signature, pubkey_hex)
    assert not verify_signature(payload, signature, "0z" + pubkey_hex[1:])  # not hex
    assert not verify_signature(payload, signature, pubkey_hex[:-2])  # wrong key length
    assert not verify_signature(payload, b"tooshort", pubkey_hex)
    assert not verify_signature("not-bytes", signature, pubkey_hex)
    tampered = bytearray(signature)
    tampered[0] ^= 1
    assert not verify_signature(payload, bytes(tampered), pubkey_hex)
