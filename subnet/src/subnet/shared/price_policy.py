# Miner dependency subset extracted from Aurelius v5 (02375403).
"""ADR-0269 exact prospective price identities; no IO, signing or activation.

Supports §12 27/36: caller-owned tuple allowlist, explicit effective boundaries
and domain-separated App-F identities. This validates representation, not cost
measurement, capital calibration, finalized provenance or permission to charge.
"""

from __future__ import annotations
from copy import deepcopy
from fractions import Fraction
import hashlib
import re
from subnet.shared.canonicalization import canonical_bytes

DOMAIN = b"aurelius/price-policy/v1\n"


MAX_INT = 2**63 - 1


_ID = re.compile(r"[a-z0-9_]{1,64}\Z")


_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,36})?\Z")


_VERSION_FIELDS = {
    "version",
    "effective_from_block",
    "cost_evidence_sha256",
    "lanes",
    "conversion",
    "capital",
}


class PricePolicyRefused(ValueError):
    def __init__(self, kind="invalid_price_policy"):
        self.kind = kind
        super().__init__(kind)


def shipped_catalog():
    """Reviewed composition data; never inferred from a remotely submitted policy."""
    from subnet.shared.cost_model import cost_model_version, derive_cost_class

    return [
        [task, derive_cost_class(task, {}), cost_model_version(task)]
        for task in ("interp_sae", "synthetic", "synthetic_deferred", "tool_observation")
    ]


def validate_entries(entries):
    _require(type(entries) is dict)
    for key in entries:
        _require(type(key) is str)
        _require(key == "price_policy" or "price_policy" not in key.split("."))
    if "price_policy" in entries:
        validate(entries["price_policy"], shipped_catalog())
    return deepcopy(entries)


def bounds() -> dict:
    return dict(
        max_lanes=16,
        max_classes_per_lane=32,
        max_catalog_tuples=512,
        max_identifier_bytes=64,
        max_integer=MAX_INT,
        max_decimal_bytes=128,
        max_decimal_fraction_places=36,
        max_policy_bytes=131072,
        min_exit_penalty="1",
        max_exit_penalty="5",
        catalog_required=True,
    )


def _require(condition, kind="invalid_price_policy"):
    if not condition:
        raise PricePolicyRefused(kind)


def _shape(value, fields):
    _require(type(value) is dict and set(value) == fields)


def _integer(value, minimum=1):
    _require(type(value) is int and minimum <= value <= MAX_INT)


def _identifier(value):
    _require(isinstance(value, str) and _ID.fullmatch(value) is not None)


def _decimal(value):
    _require(isinstance(value, str) and len(value) <= 128 and _DECIMAL.fullmatch(value) is not None)
    return Fraction(value)


def _catalog(entries):
    _require(type(entries) in (list, tuple) and 1 <= len(entries) <= 512)
    catalog = set()
    for entry in entries:
        _require(type(entry) in (list, tuple) and len(entry) == 3)
        task, cost_class, version = entry
        _identifier(task)
        _identifier(cost_class)
        _integer(version)
        item = (task, cost_class, version)
        _require(item not in catalog)
        catalog.add(item)
    return catalog


def _version(value, catalog):
    _shape(value, _VERSION_FIELDS)
    _integer(value["version"])
    _integer(value["effective_from_block"], 0)
    evidence = value["cost_evidence_sha256"]
    _require(isinstance(evidence, str) and re.fullmatch(r"[0-9a-f]{64}", evidence) is not None)
    lanes = value["lanes"]
    _require(type(lanes) is dict and 1 <= len(lanes) <= 16)
    for task, classes in lanes.items():
        _identifier(task)
        _require(type(classes) is dict and 1 <= len(classes) <= 32)
        for cost_class, price in classes.items():
            _identifier(cost_class)
            _shape(price, {"cost_model_version", "work_tokens"})
            _integer(price["cost_model_version"])
            _integer(price["work_tokens"])
            _require((task, cost_class, price["cost_model_version"]) in catalog)
    _shape(value["conversion"], {"alpha_rao_per_work_token"})
    _integer(value["conversion"]["alpha_rao_per_work_token"])
    capital = value["capital"]
    _shape(capital, {"exit_penalty", "window_blocks", "payout_reference"})
    _require(1 <= _decimal(capital["exit_penalty"]) <= 5)
    _integer(capital["window_blocks"])
    _shape(capital["payout_reference"], set(lanes))
    for reference in capital["payout_reference"].values():
        _require(_decimal(reference) > 0)
    _require(len(canonical_bytes(value)) <= bounds()["max_policy_bytes"])


def validate(document, catalog):
    """Validate the entire scheduling envelope and return an isolated copy."""
    allowed = _catalog(catalog)
    _shape(document, {"current", "next"})
    _version(document["current"], allowed)
    if document["next"] is not None:
        _version(document["next"], allowed)
        _require(document["next"]["version"] > document["current"]["version"])
        _require(
            document["next"]["effective_from_block"] > document["current"]["effective_from_block"]
        )
    _require(len(canonical_bytes(document)) <= bounds()["max_policy_bytes"])
    return deepcopy(document)


def digest(version, catalog):
    """Bind a complete version, independent of the queued-version envelope."""
    _version(version, _catalog(catalog))
    return hashlib.sha256(DOMAIN + canonical_bytes(version)).hexdigest()


def resolve(document, finalized_block, catalog):
    _integer(finalized_block, 0)
    document = validate(document, catalog)
    current = document["current"]
    _require(finalized_block >= current["effective_from_block"], "price_policy_not_effective")
    pending = document["next"]
    active = (
        pending
        if pending is not None and finalized_block >= pending["effective_from_block"]
        else current
    )
    return {"policy": active, "policy_digest": digest(active, catalog)}
