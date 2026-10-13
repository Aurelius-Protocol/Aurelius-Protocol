# Miner dependency subset extracted from Aurelius v5 (02375403).
r"""Appendix-F canonicalization + `content_hash` (NORMATIVE implementation, appf-v1).

CONTRACT (capsule header)
- Owns (§12): 37 (single byte-exact canonical form); 2 and 31 jointly with `cost_model`/`windows`
  (identity is locally derivable, server-verified).
- Guarantees: one logical work item -> one byte-identical canonical form -> one content_hash,
  in every language. The Elixir port `CentralApi.Shared.Canonicalization` mirrors this
  byte-for-byte; both are pinned by the golden vectors in `fixtures/canonicalization/` and a
  2000-case live differential (`make canon-diff`).
- Fail-CLOSED: anything outside the canonical domain raises `CanonicalizationError` (with a
  machine-readable `.kind`) rather than coercing — a silent coercion would collapse distinct
  inputs onto one hash (a dedup/novelty/charge bug).

Appendix-F rules implemented here (spec `docs/spec/aurelius-spec-v2-9.md`, App F):
- content_hash = SHA-256(domain_separator || preimage), domain `aurelius/content_hash/v1\n`;
  preimage joins canon(task_type), canon(schema_version), canon(config), canon(cost_model_version),
  canon(cost_class) with 0x1F. 0x1F can never appear inside canon() output because every control
  char < 0x20 is escaped, so field boundaries cannot be forged.
- UTF-8 only; NFC normalization applied to every string (keys and values); invalid UTF-8 and lone
  surrogates rejected; no BOM.
- Object keys: strings only; sorted by code point of the NFC-normalized key; duplicates rejected
  (including two distinct keys that NFC-normalize to the same string).
- Numbers: integers only (arbitrary precision). Decimals travel as schema-validated *strings*.
  Native floats are rejected; the strict parser rejects `-0`, fractions, and exponents.
- null: object keys with a null value are DROPPED before hashing (v1 blanket rule — no v1 schema
  declares null-significant fields; the rule is versioned inside the domain tag). null inside an
  ARRAY is rejected (positional nulls cannot be dropped without changing meaning). A top-level
  bare null encodes as `null`.
- Escaping: exactly `\"`, `\\`, and `\u00xx` (lowercase) for chars < 0x20. Everything else is
  emitted as the literal character (App F: no escaped Unicode where the literal is valid).
"""

from __future__ import annotations
import hashlib
import json
import unicodedata
from bisect import bisect_right
from subnet.shared._unicode_ranges import ASSIGNED_RANGES, PINNED_UNICODE_VERSION

JsonValue = None | bool | int | str | list | dict


CANONICALIZATION_DOMAIN = b"aurelius/content_hash/v1\n"


_FIELD_SEP = b"\x1f"


MAX_NESTING_DEPTH = 128


MAX_INT_DIGITS = 4300


_MAX_INT = 10**MAX_INT_DIGITS


_RANGE_STARTS = [lo for lo, _hi in ASSIGNED_RANGES]


def _assigned(cp: int) -> bool:
    i = bisect_right(_RANGE_STARTS, cp) - 1
    return i >= 0 and cp <= ASSIGNED_RANGES[i][1]


class CanonicalizationError(ValueError):
    """Fail-closed canonicalization rejection. `kind` is a stable, machine-readable label."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(f"{kind}: {message}")


def _nfc(s: str, *, role: str) -> str:
    # Lone surrogates survive unicodedata.normalize and then crash utf-8 encoding with a
    # context-free error; reject them explicitly first. Codepoints unassigned in the PINNED
    # Unicode version reject fail-closed: NFC is only version-stable for assigned codepoints,
    # so an unassigned one normalizes differently across runtimes' Unicode tables and silently
    # forks content_hash (adversarial-review finding; ADR-0003).
    for ch in s:
        cp = ord(ch)
        if 0xD800 <= cp <= 0xDFFF:
            raise CanonicalizationError(
                "lone_surrogate", f"unpaired surrogate U+{cp:04X} in {role}"
            )
        if not _assigned(cp):
            raise CanonicalizationError(
                "unassigned_codepoint",
                f"U+{cp:04X} in {role} is not assigned in pinned Unicode {PINNED_UNICODE_VERSION}",
            )
    return unicodedata.normalize("NFC", s)


def _enc_str(s: str, *, role: str = "string value") -> str:
    out = ['"']
    for ch in _nfc(s, role=role):
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _enc(v: JsonValue, *, in_array: bool, depth: int) -> str:
    # bool must be checked before int (bool is a subclass of int in Python).
    if v is None:
        if in_array:
            raise CanonicalizationError(
                "null_in_array", "null inside an array cannot be dropped; rejected (App F)"
            )
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float):
        raise CanonicalizationError(
            "float", "native floats are forbidden; carry decimals as schema-validated strings"
        )
    if isinstance(v, int):
        if v >= _MAX_INT or v <= -_MAX_INT:
            raise CanonicalizationError(
                "int_too_large", f"integers are capped at {MAX_INT_DIGITS} digits"
            )
        return str(v)
    if isinstance(v, str):
        return _enc_str(v)
    if isinstance(v, list):
        if depth >= MAX_NESTING_DEPTH:
            raise CanonicalizationError(
                "too_deep", f"nesting is capped at {MAX_NESTING_DEPTH} levels"
            )
        return "[" + ",".join(_enc(x, in_array=True, depth=depth + 1) for x in v) + "]"
    if isinstance(v, dict):
        if depth >= MAX_NESTING_DEPTH:
            raise CanonicalizationError(
                "too_deep", f"nesting is capped at {MAX_NESTING_DEPTH} levels"
            )
        normalized: list[tuple[str, JsonValue]] = []
        for k, val in v.items():
            if not isinstance(k, str):
                raise CanonicalizationError(
                    "non_string_key", f"map keys must be strings, got: {type(k).__name__}"
                )
            if val is None:
                continue  # null-valued keys are dropped before hashing (v1 blanket rule)
            normalized.append((_nfc(k, role="object key"), val))
        normalized.sort(key=lambda kv: kv[0])
        for (k1, _), (k2, _) in zip(normalized, normalized[1:]):
            if k1 == k2:
                raise CanonicalizationError(
                    "post_nfc_duplicate_key",
                    f"two distinct keys NFC-normalize to the same string: {k1!r}",
                )
        return (
            "{"
            + ",".join(
                _enc_str(k, role="object key") + ":" + _enc(val, in_array=False, depth=depth + 1)
                for k, val in normalized
            )
            + "}"
        )
    raise CanonicalizationError("unsupported_type", f"non-canonicalizable: {type(v).__name__}")


def canonical_bytes(value: JsonValue) -> bytes:
    """Return the canonical UTF-8 byte string for a JSON-shaped value (App F encoder)."""
    try:
        return _enc(value, in_array=False, depth=0).encode("utf-8")
    except RecursionError:
        # backstop only — the explicit depth cap fires far earlier for well-formed input
        raise CanonicalizationError(
            "too_deep", f"nesting is capped at {MAX_NESTING_DEPTH} levels"
        ) from None


def content_hash(
    task_type: str,
    schema_version: int,
    config: JsonValue,
    cost_model_version: int,
    cost_class: str,
) -> str:
    """Lowercase hex SHA-256 of the domain-separated App-F preimage (spec §3.2 / App F).

    All five fields are folded in so base cost is tamper-evident and a schema/cost-model change
    can never silently collide with an old hash.
    """
    if not isinstance(task_type, str) or not task_type:
        raise CanonicalizationError("bad_field", "task_type must be a non-empty string")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise CanonicalizationError("bad_field", "schema_version must be an integer")
    if isinstance(cost_model_version, bool) or not isinstance(cost_model_version, int):
        raise CanonicalizationError("bad_field", "cost_model_version must be an integer")
    if not isinstance(cost_class, str) or not cost_class:
        raise CanonicalizationError("bad_field", "cost_class must be a non-empty string")

    preimage = CANONICALIZATION_DOMAIN + _FIELD_SEP.join(
        (
            canonical_bytes(task_type),
            canonical_bytes(schema_version),
            canonical_bytes(config),
            canonical_bytes(cost_model_version),
            canonical_bytes(cost_class),
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def _reject_float(lexeme: str) -> None:
    raise CanonicalizationError(
        "float", f"fractions/exponents are forbidden in canonical JSON: {lexeme!r}"
    )


def _reject_constant(lexeme: str) -> None:
    raise CanonicalizationError("float", f"non-finite constants are forbidden: {lexeme!r}")


def _parse_int(lexeme: str) -> int:
    if lexeme == "-0":
        raise CanonicalizationError("bad_number", "-0 is forbidden in canonical JSON")
    digits = lexeme[1:] if lexeme.startswith("-") else lexeme
    if len(digits) > MAX_INT_DIGITS:
        raise CanonicalizationError(
            "int_too_large", f"integers are capped at {MAX_INT_DIGITS} digits"
        )
    return int(lexeme)


def _reject_duplicate_pairs(pairs: list[tuple[str, JsonValue]]) -> dict:
    obj: dict = {}
    for k, v in pairs:
        if k in obj:
            raise CanonicalizationError("duplicate_key", f"duplicate object key: {k!r}")
        obj[k] = v
    return obj


def parse_canonical_json(text: str | bytes) -> JsonValue:
    """Strict boundary parser for canonical-JSON text (App F decode rules).

    Rejects (fail-closed): duplicate keys, `-0`, fractions, exponents, NaN/Infinity, invalid
    UTF-8, a leading BOM, and trailing garbage. The Elixir port is a hand-rolled parser with the
    same acceptance set; the shared `reject_parse_vectors` golden family pins the two together.
    """
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as e:
            raise CanonicalizationError("invalid_utf8", str(e)) from None
    if text.startswith("\ufeff"):
        raise CanonicalizationError("invalid_json", "BOM is forbidden (App F: no BOM)")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_float=_reject_float,
            parse_int=_parse_int,
            parse_constant=_reject_constant,
        )
    except CanonicalizationError:
        raise
    except json.JSONDecodeError as e:
        raise CanonicalizationError("invalid_json", str(e)) from None
    except RecursionError:
        raise CanonicalizationError(
            "too_deep", f"nesting is capped at {MAX_NESTING_DEPTH} levels"
        ) from None
    _reject_surrogates(value)
    return value


def _reject_surrogates(value: JsonValue) -> None:
    r"""json.loads resolves lone \uXXXX surrogate escapes into unpaired surrogates; reject them
    at the parse boundary (the Elixir parser rejects the escape itself) so the parser never
    returns a non-UTF-8-encodable value."""
    if isinstance(value, str):
        for ch in value:
            if 0xD800 <= ord(ch) <= 0xDFFF:
                raise CanonicalizationError(
                    "lone_surrogate", f"unpaired surrogate U+{ord(ch):04X} in parsed string"
                )
    elif isinstance(value, list):
        for item in value:
            _reject_surrogates(item)
    elif isinstance(value, dict):
        for k, v in value.items():
            _reject_surrogates(k)
            _reject_surrogates(v)
