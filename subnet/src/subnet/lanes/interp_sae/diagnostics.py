# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Closed, versioned SAE diagnostic measurements. Advisory; never an outcome authority.

Only fixed measurement identifiers, finite numbers, bounded strings and enums cross the public
projection. The authoritative referee retains its existing gate ordering and primary reason codes.

Schema 1 (ADR-0227) carries the score-time activation rows. Schema 2 (ADR-0229, gap analysis G22)
adds three groups the reference always printed on rejection: the mismatching Gate 0 pins
(``cfg_<field>``, string ``equals`` rows), the battery components against the floor/baseline/
high-water (``battery_*``), and the paired outcome (``paired_*``, ``throne_outcome``,
``ladder_rank``, ``screen_margin``). The Elixir twin (``CentralApi.Submissions.SaeDiagnostics``)
accepts both versions; roll Central out before validators emit schema 2.
"""

from __future__ import annotations
import math

FAMILIES = ("books", "code", "dialogue", "reference", "scientific", "web")


IDS_V1 = frozenset(
    {
        "l0_mean",
        "l0_p95",
        "l0_max_window",
        "dead_fraction",
        "ce_recovered",
        "ce_clean",
        "ce_ablated",
        "ce_reconstructed",
        "token_count",
        "battery",
        *("l0_family_" + family for family in FAMILIES),
    }
)


CFG_FIELDS = (
    "d_in",
    "d_sae",
    "dtype",
    "normalize_activations",
    "reshape_activations",
    "apply_b_dec_to_input",
    "hook_name",
    "model_name",
    "seqpos_slice",
    "k",
    "rescale_acts_by_decoder_norm",
)


CFG_IDS = frozenset("cfg_" + field for field in CFG_FIELDS)


BATTERY_IDS = frozenset(
    {
        "battery_top1",
        "battery_llm_top1_baseline",
        "battery_absorption",
        "battery_top1_high_water",
        "battery_absorption_high_water",
    }
)


PAIRED_IDS = frozenset(
    {
        "paired_incumbent_ce",
        "paired_gap",
        "paired_epsilon",
        "throne_outcome",
        "ladder_rank",
        "screen_margin",
    }
)


IDS = IDS_V1 | CFG_IDS | BATTERY_IDS | PAIRED_IDS


STRING_IDS = CFG_IDS | {"throne_outcome"}


FIELDS = frozenset({"id", "status", "observed", "minimum", "maximum", "comparison"})


STATUSES = frozenset({"pass", "fail", "measured", "not_run"})


COMPARISONS_V1 = frozenset({"range", "lte", "observe"})


COMPARISONS = COMPARISONS_V1 | {"equals", "gte"}


SCHEMA_VERSIONS = (1, 2)


MAX_STRING_BYTES = 64


_NUMBER_BOUND = 1e12


def _valid_number(number: object) -> bool:
    return (
        number is None
        or type(number) in (int, float)
        and math.isfinite(number)
        and -_NUMBER_BOUND <= number <= _NUMBER_BOUND
    )


def _valid_string(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value.encode("utf-8")) <= MAX_STRING_BYTES


def validate_diagnostics(value: dict) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "checks"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] not in SCHEMA_VERSIONS
    ):
        raise ValueError("invalid diagnostics schema")
    version = value["schema_version"]
    ids = IDS_V1 if version == 1 else IDS
    comparisons = COMPARISONS_V1 if version == 1 else COMPARISONS
    checks = value["checks"]
    if not isinstance(checks, list) or not 1 <= len(checks) <= len(ids):
        raise ValueError("invalid diagnostics check count")
    seen = set()
    for row in checks:
        if (
            not isinstance(row, dict)
            or set(row) != FIELDS
            or not isinstance(row["id"], str)
            or row["id"] not in ids
            or row["id"] in seen
        ):
            raise ValueError("invalid or repeated diagnostic field")
        seen.add(row["id"])
        if (
            not isinstance(row["status"], str)
            or not isinstance(row["comparison"], str)
            or row["status"] not in STATUSES
            or row["comparison"] not in comparisons
        ):
            raise ValueError("invalid diagnostic status/comparison")
        if version == 2 and row["id"] in STRING_IDS:
            _validate_string_row(row)
        else:
            _validate_numeric_row(row)
    return {"schema_version": version, "checks": [dict(row) for row in checks]}


def _validate_numeric_row(row: dict) -> None:
    for name in ("observed", "minimum", "maximum"):
        if not _valid_number(row[name]):
            raise ValueError("diagnostics require bounded finite numbers")
    observed, low, high, comparison = (
        row["observed"],
        row["minimum"],
        row["maximum"],
        row["comparison"],
    )
    if comparison == "equals":
        raise ValueError("numeric diagnostics cannot use string equality")
    if row["status"] == "not_run":
        if observed is not None or low is not None or high is not None or comparison != "observe":
            raise ValueError("unmeasured diagnostic has a value")
    elif observed is None:
        raise ValueError("measured diagnostic is absent")
    elif comparison == "observe":
        if low is not None or high is not None or row["status"] != "measured":
            raise ValueError("observation cannot assert a gate result")
    elif comparison == "gte":
        if low is None or high is not None:
            raise ValueError("invalid diagnostic bounds")
        if row["status"] != ("pass" if observed >= low else "fail"):
            raise ValueError("diagnostic status contradicts measurement")
    else:
        if (
            high is None
            or (comparison == "range" and (low is None or low > high))
            or (comparison == "lte" and low is not None)
        ):
            raise ValueError("invalid diagnostic bounds")
        passed = observed <= high and (low is None or observed >= low)
        if row["status"] != ("pass" if passed else "fail"):
            raise ValueError("diagnostic status contradicts measurement")


def _validate_string_row(row: dict) -> None:
    observed, low, high, comparison = (
        row["observed"],
        row["minimum"],
        row["maximum"],
        row["comparison"],
    )
    if low is not None:
        raise ValueError("string diagnostics carry no minimum")
    if row["status"] == "not_run":
        if observed is not None or high is not None or comparison != "observe":
            raise ValueError("unmeasured diagnostic has a value")
        return
    if not _valid_string(observed):
        raise ValueError("string diagnostics require a bounded observed string")
    if comparison == "observe":
        if high is not None or row["status"] != "measured":
            raise ValueError("observation cannot assert a gate result")
    elif comparison == "equals":
        if not _valid_string(high):
            raise ValueError("string equality requires a bounded expected string")
        if row["status"] != ("pass" if observed == high else "fail"):
            raise ValueError("diagnostic status contradicts measurement")
    else:
        raise ValueError("string diagnostics support only equals or observe")
