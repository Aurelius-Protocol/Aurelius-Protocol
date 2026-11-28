"""This repository is public and is the miner's one source of truth.

Nothing that belongs to the operator's own infrastructure may live in the package or in its
tests: no operator-only module, no retired launch generation or test network, no copied-in
extraction stamp, no pinned image, no sibling repository, no name of the operator's hardware,
and no path into the validator's own records. The rules are written so that this file does not
itself carry any private name.
"""

import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "subnet"
TESTS = PACKAGE.parents[1] / "tests"
REPOSITORY = PACKAGE.parents[2]

FORBIDDEN = {
    "an operator-only package": re.compile(r"subnet[./]operator\b"),
    "the validator's runtime catalog": re.compile(r"subnet[./]runtime_maintenance\b"),
    "a retired launch generation": re.compile(r"(?i)runtime[ _-]?452"),
    "a retired test network's local delivery": re.compile(r"sn455_artifact_delivery"),
    "a copied-in extraction stamp": re.compile(r"Miner dependency subset"),
    "a pinned image": re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@sha256:[0-9a-f]{64}"),
    "a path into the validator's records": re.compile(r"\bdocs/(?:evidence|plan|adr)/"),
    # After the organisation in a GitHub URL, the next path segment is this repository, with or
    # without a ".git" suffix.
    "a sibling repository of this organisation": re.compile(
        r"github(?:usercontent)?\.com/Aurelius-Protocol/"
        r"(?!Aurelius-Protocol(?:\.git)?(?![A-Za-z0-9_.-]))"
    ),
}
HARDWARE = re.compile(r"(?i)gh200")

# The one place the operator's hardware is part of the protocol itself: the validator publishes
# this profile id inside its signed slot policy, and the staging tool checks it. It changes only
# together with the validator. Keyed by package-relative file and the exact text on the line.
ALLOWED = {"lanes/interp_sae/stage.py": ('"gh200-sn37-mainnet"', "the validator's profile id")}


def package_sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        yield path.relative_to(PACKAGE).as_posix(), path


def scanned_test_files():
    """The test files, except this one: a guard states its own rules as text and cannot be
    applied to itself. (Not named `test_...`: pytest would collect a generator as a test.)"""
    for path in sorted(TESTS.rglob("*.py")):
        if path != Path(__file__).resolve():
            yield path.relative_to(TESTS).as_posix(), path


def offenders_in(lines, relative):
    found = []
    allowed_text = ALLOWED.get(relative, (None, None))[0]
    for number, line in enumerate(lines, 1):
        for label, pattern in FORBIDDEN.items():
            if pattern.search(line):
                found.append(f"{relative}:{number}: {label}")
        if HARDWARE.search(line) and not (allowed_text and allowed_text in line):
            found.append(f"{relative}:{number}: the operator's hardware")
    return found


def test_each_pattern_recognises_what_it_names():
    assert FORBIDDEN["a copied-in extraction stamp"].search("# Miner dependency subset of X")
    assert FORBIDDEN["a pinned image"].search("registry.example/team/app@sha256:" + "0" * 64)
    assert not FORBIDDEN["a pinned image"].search("sha256:" + "0" * 64)
    siblings = FORBIDDEN["a sibling repository of this organisation"]
    assert siblings.search("https://github.com/Aurelius-Protocol/some-other-repository/x")
    assert siblings.search("https://github.com/Aurelius-Protocol/Aurelius-Protocol-internal")
    assert not siblings.search("https://github.com/Aurelius-Protocol/Aurelius-Protocol/blob/x.md")
    assert not siblings.search("https://github.com/Aurelius-Protocol/Aurelius-Protocol.git")
    assert not siblings.search(
        "https://raw.githubusercontent.com/Aurelius-Protocol/Aurelius-Protocol/main/x.md"
    )
    assert not siblings.search("https://github.com/decoderesearch/SAELens")
    catalog = FORBIDDEN["the validator's runtime catalog"]
    assert catalog.search("from subnet.runtime_maintenance.catalog import x")
    assert catalog.search("import subnet/runtime_maintenance/catalog.py")
    assert not catalog.search("from subnet.miner.funding_runtime import RUNTIME_PROFILES")
    records = FORBIDDEN["a path into the validator's records"]
    assert records.search("docs/evidence/x")
    assert records.search("docs/plan/x.md")
    assert records.search("docs/adr/0000-template.md")
    assert not records.search("[Gamma funding](docs/gamma-funding.md)")


def test_the_package_names_none_of_the_operator_s_infrastructure():
    offenders = []
    for relative, path in package_sources():
        offenders += offenders_in(path.read_text(encoding="utf-8").splitlines(), relative)
    assert offenders == [], "\n".join(offenders)


def test_the_tests_name_none_of_the_operator_s_infrastructure():
    offenders = []
    for relative, path in scanned_test_files():
        offenders += offenders_in(
            path.read_text(encoding="utf-8").splitlines(), "tests/" + relative
        )
    assert offenders == [], "\n".join(offenders)


def test_each_allowance_still_names_what_it_excuses():
    for relative, (text, reason) in ALLOWED.items():
        assert reason
        source = (PACKAGE / relative).read_text(encoding="utf-8")
        assert text in source, f"{relative} no longer carries {text}; drop the allowance"


@pytest.mark.parametrize(
    "relative",
    [
        "README.md",
        "INCENTIVES.md",
        "docs/gamma-funding.md",
        "lanes/README.md",
        "lanes/lane1.md",
        ".env.example",
    ],
)
def test_the_guides_name_no_private_infrastructure(relative):
    lines = (REPOSITORY / relative).read_text(encoding="utf-8").splitlines()
    package_only = (
        "an operator-only package",
        "the validator's runtime catalog",
        "a retired launch generation",
        "a retired test network's local delivery",
    )
    patterns = [rule for label, rule in FORBIDDEN.items() if label not in package_only]
    hits = [
        f"{relative}:{number}"
        for number, line in enumerate(lines, 1)
        if any(pattern.search(line) for pattern in patterns) or HARDWARE.search(line)
    ]
    assert hits == [], "\n".join(hits)
