"""Gamma purchase policy wire identity (gamma plan §8.1): golden parity + fail-closed shape.

Owns the Python half of the shared golden vectors (the Elixir twin asserts the same file), the
typed-unavailable selection seam (`purchase_policy_unselected`) and the canonical-decimal wire
discipline. Pure representation only — no economic values are invented here.
"""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

from subnet.shared.canonicalization import CanonicalizationError, parse_canonical_json
from subnet.shared.gamma_policy import (
    GAMMA_ATOMS_MAX,
    SCHEMA,
    GammaPolicyRefused,
    WORK_TOKENS_MAX,
    canonical_bytes,
    digest,
    from_map,
    validate_selection,
)

pytestmark = pytest.mark.shared

_VECTORS_PATH = Path(__file__).parent / "fixtures" / "gamma_funding_vectors.json"
_DOC = json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))
_CASES = [case for case in _DOC["cases"] if case["kind"] == "policy"]
_VALID = [case for case in _CASES if "error" not in case["expect"]]
_REJECTED = [case for case in _CASES if "error" in case["expect"]]


def base_policy():
    return {
        "schema": SCHEMA,
        "service_id": "gamma-test-service",
        "chain_id": "bittensor-mainnet:" + "2f" * 32,
        "genesis_hash": "0x" + "2f" * 32,
        "netuid": 21,
        "policy_version": "gamma-test.v1",
        "packs": {"starter": {"gamma_atoms": "1000000000", "work_tokens": "1"}},
        "activation_block": 1000,
        "quote_lifetime_blocks": 300,
        "max_packs_per_intent": 1,
        "per_account_packs_per_day": 24,
        "global_packs_per_day": 10000,
        "payer_mode": "self_spend_direct_v1",
        "runtime_profile": "gamma-test-profile",
        "governing_config_hash": "0x" + "ab" * 32,
        "signing_key_id": "gamma-test-signing-key",
    }


@pytest.mark.parametrize("case", _VALID, ids=[c["name"] for c in _VALID])
def test_golden_policies_match_canonical_bytes_and_digest(case):
    policy = from_map(case["input"])
    assert policy.canonical_bytes().hex() == case["expect"]["canonical_bytes_hex"]
    assert policy.digest() == case["expect"]["digest_hex"]


@pytest.mark.parametrize("case", _REJECTED, ids=[c["name"] for c in _REJECTED])
def test_golden_policy_rejections_pin_the_refusal_kind(case):
    with pytest.raises(GammaPolicyRefused) as exc:
        from_map(case["input"])
    assert exc.value.kind == case["expect"]["error"]


def test_document_tag_and_case_partition_look_right():
    assert _DOC["schema_version"] == "aurelius/gamma_funding_vectors/v1"
    assert len(_VALID) == 3 and len(_REJECTED) == 12
    assert all(case["languages"] == ["python", "elixir"] for case in _CASES)


def test_unknown_and_missing_fields_reject():
    with_extra = base_policy()
    with_extra["operator_note"] = "please"
    with pytest.raises(GammaPolicyRefused, match="invalid_purchase_policy"):
        from_map(with_extra)
    missing = base_policy()
    del missing["quote_lifetime_blocks"]
    with pytest.raises(GammaPolicyRefused, match="invalid_purchase_policy"):
        from_map(missing)


def test_duplicate_json_keys_reject_at_the_parse_boundary():
    text = json.dumps(base_policy())[:-1] + ',"netuid":21}'
    with pytest.raises(CanonicalizationError) as exc:
        from_map(parse_canonical_json(text))
    assert exc.value.kind == "duplicate_key"


@pytest.mark.parametrize("amount", ["0", "-5", "1.0", "1e3", "+1", "01", " 1", "1 ", "", "0x10"])
def test_non_canonical_amounts_reject(amount):
    doc = base_policy()
    doc["packs"]["starter"]["gamma_atoms"] = amount
    with pytest.raises(GammaPolicyRefused, match="invalid_purchase_policy"):
        from_map(doc)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("netuid", 0),
        ("netuid", 65536),
        ("netuid", True),
        ("max_packs_per_intent", 2),
        ("max_packs_per_intent", True),
        ("payer_mode", "self_spend_direct_v2"),
        ("quote_lifetime_blocks", 0),
        ("activation_block", -1),
        ("service_id", "not printable \u00e9"),
        ("service_id", ""),
        ("policy_version", "bad version!"),
        ("governing_config_hash", "0x" + "AB" * 32),
        ("runtime_profile", ""),
    ],
)
def test_out_of_domain_fields_reject(field, value):
    doc = base_policy()
    doc[field] = value
    with pytest.raises(GammaPolicyRefused, match="invalid_purchase_policy"):
        from_map(doc)


def test_amount_bounds_are_u64_atoms_and_i64_work_tokens():
    atoms_max = base_policy()
    atoms_max["packs"]["starter"]["gamma_atoms"] = str(GAMMA_ATOMS_MAX)
    assert from_map(atoms_max).pack("starter")["gamma_atoms"] == GAMMA_ATOMS_MAX
    atoms_over = base_policy()
    atoms_over["packs"]["starter"]["gamma_atoms"] = str(GAMMA_ATOMS_MAX + 1)
    with pytest.raises(GammaPolicyRefused, match="invalid_purchase_policy"):
        from_map(atoms_over)
    tokens_max = base_policy()
    tokens_max["packs"]["starter"]["work_tokens"] = str(WORK_TOKENS_MAX)
    assert from_map(tokens_max).pack("starter")["work_tokens"] == WORK_TOKENS_MAX
    tokens_over = base_policy()
    tokens_over["packs"]["starter"]["work_tokens"] = str(WORK_TOKENS_MAX + 1)
    with pytest.raises(GammaPolicyRefused, match="invalid_purchase_policy"):
        from_map(tokens_over)


def test_retirement_block_orders_after_activation_and_null_rejects():
    equal = base_policy()
    equal["retirement_block"] = equal["activation_block"]
    assert from_map(equal).retirement_block == 1000
    early = base_policy()
    early["retirement_block"] = 999
    with pytest.raises(GammaPolicyRefused, match="retirement_before_activation"):
        from_map(early)
    explicit_null = base_policy()
    explicit_null["retirement_block"] = None
    with pytest.raises(GammaPolicyRefused, match="invalid_purchase_policy"):
        from_map(explicit_null)


def test_active_at_is_inclusive_at_both_boundaries():
    doc = base_policy()
    doc["retirement_block"] = 1300
    policy = from_map(doc)
    assert not policy.active_at(999)
    assert policy.active_at(1000)
    assert policy.active_at(1300)
    assert not policy.active_at(1301)
    assert from_map(base_policy()).active_at(4294967295)
    with pytest.raises(GammaPolicyRefused, match="invalid_block"):
        from_map(base_policy()).active_at(-1)


def test_pack_and_selection_seams_are_typed_not_silent():
    policy = from_map(base_policy())
    assert policy.pack("starter") == {"gamma_atoms": 1000000000, "work_tokens": 1}
    assert validate_selection(policy, "starter") == policy.pack("starter")
    with pytest.raises(GammaPolicyRefused, match="pack_unknown"):
        policy.pack("nosuch")
    with pytest.raises(GammaPolicyRefused, match="pack_id_invalid"):
        policy.pack("Bad_Id")
    with pytest.raises(GammaPolicyRefused, match="pack_id_invalid"):
        policy.pack(7)
    with pytest.raises(GammaPolicyRefused, match="purchase_policy_unselected"):
        validate_selection(None, "starter")


def test_every_bound_field_changes_the_policy_digest():
    original = base_policy()
    changes = [
        ("service_id", "gamma-test-service-2"),
        ("chain_id", "bittensor-mainnet:" + "30" * 32),
        ("genesis_hash", "0x" + "30" * 32),
        ("netuid", 22),
        ("policy_version", "gamma-test.v2"),
        ("packs", {"starter": {"gamma_atoms": "2000000000", "work_tokens": "1"}}),
        ("activation_block", 1001),
        ("quote_lifetime_blocks", 301),
        ("per_account_packs_per_day", 25),
        ("global_packs_per_day", 10001),
        ("runtime_profile", "gamma-test-profile-2"),
        ("governing_config_hash", "0x" + "ba" * 32),
        ("signing_key_id", "gamma-test-signing-key-2"),
    ]
    for field, value in changes:
        changed = deepcopy(original)
        changed[field] = value
        assert digest(changed) != digest(original), field
    with_retirement = deepcopy(original)
    with_retirement["retirement_block"] = 2000
    assert digest(with_retirement) != digest(original)


def test_from_map_returns_an_isolated_copy():
    doc = base_policy()
    policy = from_map(doc)
    doc["packs"]["starter"]["gamma_atoms"] = "999999999999"
    assert policy.pack("starter")["gamma_atoms"] == 1000000000
    assert canonical_bytes(policy.body) == canonical_bytes(from_map(base_policy()).body)


@seed(20261113)
@settings(max_examples=100, deadline=None)
@given(st.text(max_size=12))
def test_pack_amount_acceptance_is_exactly_the_canonical_decimals(text):
    """Whatever bytes arrive on the wire, acceptance is exactly `^[1-9][0-9]*$` within the
    u64 bound — zero/sign/fraction/exponent/leading-zero spellings can never carry value."""
    doc = base_policy()
    doc["packs"]["starter"]["gamma_atoms"] = text
    if re.fullmatch(r"[1-9][0-9]*", text):
        assume(int(text) <= GAMMA_ATOMS_MAX)
        assert from_map(doc).pack("starter")["gamma_atoms"] == int(text)
    else:
        with pytest.raises(GammaPolicyRefused, match="invalid_purchase_policy"):
            from_map(doc)
