"""Price schema 2 represents adopted unconditioned payouts without a fake reference."""

import hashlib
from copy import deepcopy

import pytest
from subnet.shared.canonicalization import canonical_bytes
from subnet.shared.price_policy import PricePolicyRefused, digest, resolve, validate

pytestmark = pytest.mark.shared
CATALOG = [["interp_sae", "gpu_sae_verdict", 2]]


def policy():
    return {
        "schema_version": 2,
        "version": 1,
        "effective_from_block": 100,
        "cost_evidence_sha256": "a" * 64,
        "lanes": {"interp_sae": {"gpu_sae_verdict": {"cost_model_version": 2, "work_tokens": 10}}},
        "conversion": {"alpha_rao_per_work_token": 1_000_000_000},
        "capital": {"mode": "none"},
    }


def test_none_has_no_reference_or_capital_authority_and_a_distinct_digest_domain():
    value = policy()
    expected = hashlib.sha256(b"aurelius/price-policy/v2\n" + canonical_bytes(value)).hexdigest()
    assert digest(value, CATALOG) == expected
    document = {"current": value, "next": None}
    assert validate(document, CATALOG) == document
    assert resolve(document, 100, CATALOG)["policy_digest"] == expected
    with pytest.raises(PricePolicyRefused, match="price_policy_not_effective"):
        resolve(document, 99, CATALOG)


@pytest.mark.parametrize(
    "capital",
    [
        None,
        {},
        {"mode": "NONE"},
        {"mode": "none", "payout_reference": {}},
        {"mode": "none", "exit_penalty": "2"},
        {"mode": "conditioned"},
        {
            "mode": "conditioned",
            "exit_penalty": "2",
            "window_blocks": 100,
            "payout_reference": {"interp_sae": "0"},
        },
    ],
)
def test_mode_shape_never_silently_disables_conditioning(capital):
    value = policy()
    value["capital"] = capital
    with pytest.raises(PricePolicyRefused):
        digest(value, CATALOG)


@pytest.mark.parametrize("schema", [None, True, 1, 2.0, "2", 3])
def test_schema_is_exact_and_cannot_reinterpret_legacy_prices(schema):
    value = policy()
    value["schema_version"] = schema
    with pytest.raises(PricePolicyRefused):
        digest(value, CATALOG)


def test_explicit_conditioned_mode_keeps_all_legacy_positive_bounds():
    value = policy()
    value["capital"] = {
        "mode": "conditioned",
        "exit_penalty": "2",
        "window_blocks": 100,
        "payout_reference": {"interp_sae": "100"},
    }
    digest(value, CATALOG)
    changed = deepcopy(value)
    changed["capital"]["payout_reference"]["interp_sae"] = "0"
    with pytest.raises(PricePolicyRefused):
        digest(changed, CATALOG)
