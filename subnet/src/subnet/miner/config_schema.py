"""Miner-side config validation against the PUBLISHED config_schema (plan §1.8, invariant 1 support).

CONTRACT (capsule header)
- The schema is DATA read from the signed operator snapshot (§9), never obtained by importing the
  lane — `platform-not-lane` holds on the miner exactly as on the validator. `load_published_schema`
  verifies the snapshot with the SHARED `verify_snapshot` (identical to the validator's check) and
  pulls `config_schema.<task_type>` out of the body; `validate_config` is a closed top-level
  checker (required keys, types, enum/const, inclusive bounds, pattern/string limits, no unknown
  fields). Unsupported schema rules are errors, never silently ignored.
- ServingConfig owns the ADR-0307 background refresh and bounded last-good window. Serving
  handlers never fetch policy; one-shot tooling retains fresh, uncached verified reads.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
import time
import threading
from copy import deepcopy
from urllib.parse import urlsplit

from subnet.shared.config_snapshot import SnapshotInvalid, verify_snapshot
from subnet.shared.canonicalization import parse_canonical_json
from subnet.miner.trust import config_signing_pubkey

_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}
_FIELD_RULES = frozenset(
    {"type", "enum", "const", "minimum", "maximum", "minLength", "maxLength", "pattern"}
)
_MAX_PATTERN_CHARS = 256
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024
CONFIG_IO_TIMEOUT_S = 2.0
CONFIG_REFRESH_S = 30.0
CONFIG_MAX_AGE_S = 60.0
CONFIG_START_TIMEOUT_S = 3.0


class SchemaUnavailable(Exception):
    """No published schema could be loaded/verified for the task type — validation cannot proceed."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def config_source(environ=None) -> tuple[str, str, str]:
    """Bind source and verification key together; running caches never switch authorities."""
    env = os.environ if environ is None else environ
    try:
        pubkey = config_signing_pubkey(env)
    except ValueError as exc:
        raise SchemaUnavailable(str(exc)) from None
    path = env.get("AURELIUS_CONFIG_SNAPSHOT")
    if path:
        return "file", path, pubkey
    central = env.get("CENTRAL_API_URL", "")
    try:
        parsed = urlsplit(central)
        _ = parsed.port
    except (ValueError, TypeError):
        raise SchemaUnavailable("CENTRAL_API_URL must be an explicit HTTP(S) endpoint") from None
    if (
        not central
        or central != central.strip()
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SchemaUnavailable("CENTRAL_API_URL must be an explicit HTTP(S) endpoint")
    return "http", central.rstrip("/") + "/api/config", pubkey


def _snapshot_doc(source: tuple[str, str, str]) -> dict:
    kind, location, _ = source
    if kind == "file":
        with open(location, "rb") as fh:
            raw = fh.read(MAX_SNAPSHOT_BYTES + 1)
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        deadline = time.monotonic() + CONFIG_IO_TIMEOUT_S
        with opener.open(location, timeout=CONFIG_IO_TIMEOUT_S) as resp:
            raw = bytearray()
            while len(raw) <= MAX_SNAPSHOT_BYTES:
                if time.monotonic() >= deadline:
                    raise SchemaUnavailable("snapshot read deadline exceeded")
                chunk = resp.read1(min(64 * 1024, MAX_SNAPSHOT_BYTES + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
            if time.monotonic() >= deadline:
                raise SchemaUnavailable("snapshot read deadline exceeded")
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise SchemaUnavailable("snapshot exceeds byte limit")
    doc = parse_canonical_json(bytes(raw).decode("utf-8"))
    if not isinstance(doc, dict):
        raise SchemaUnavailable("snapshot must be an object")
    return doc


def _load_entries(source: tuple[str, str, str]) -> dict:
    try:
        body = verify_snapshot(_snapshot_doc(source), source[2])
    except (SnapshotInvalid, OSError, ValueError, TypeError, RecursionError) as e:
        raise SchemaUnavailable(f"snapshot unavailable/invalid: {type(e).__name__}") from None
    entries = body.get("entries")
    if not isinstance(entries, dict):
        raise SchemaUnavailable("verified snapshot has no entries object")
    return entries


class ServingConfig:
    """One refresh worker; query/readiness readers never perform I/O or await a refresh.

    Age measures time since successful local verification, not the signed snapshot's publication
    time. A pinned file is an explicit operator source, reverified with the same age policy.
    """

    def __init__(self, *, source=None, loader=_load_entries, clock=time.monotonic):
        self._source = config_source() if source is None else source
        self._loader = loader
        self._clock = clock
        self._lock = threading.Lock()
        self._entries = None
        self._verified_at = None
        self._error = None
        self._stop = threading.Event()
        self._initialized = threading.Event()
        self._thread = None

    def refresh(self):
        started = self._clock()
        try:
            entries = self._loader(self._source)
            if not isinstance(entries, dict) or not any(
                key.startswith("config_schema.") and isinstance(value, dict)
                for key, value in entries.items()
            ):
                raise SchemaUnavailable("snapshot has no published schemas")
            if self._clock() - started >= CONFIG_START_TIMEOUT_S:
                raise SchemaUnavailable("snapshot refresh deadline exceeded")
        except Exception as exc:  # one failed poll must not kill recovery
            with self._lock:
                self._error = type(exc).__name__
            return False
        with self._lock:
            self._entries = deepcopy(entries)
            self._verified_at = started
            self._error = None
        return True

    def _run(self):
        while not self._stop.is_set():
            self.refresh()
            self._initialized.set()
            if self._stop.wait(CONFIG_REFRESH_S):
                return

    def start(self):
        if self._thread is not None:
            raise RuntimeError("config worker already started")
        self._thread = threading.Thread(target=self._run, name="miner-config", daemon=True)
        self._thread.start()
        if not self._initialized.wait(CONFIG_START_TIMEOUT_S) or self.readiness()[0] != 200:
            self.close()
            raise SchemaUnavailable("no verified serving config available at startup")
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)

    def _fresh(self):
        return (
            self._verified_at is not None and self._clock() - self._verified_at < CONFIG_MAX_AGE_S
        )

    def schema(self, task_type):
        with self._lock:
            if not self._fresh():
                raise SchemaUnavailable("serving config unavailable or stale")
            schema = self._entries.get(f"config_schema.{task_type}")
            if not isinstance(schema, dict):
                raise SchemaUnavailable("no published schema for requested task type")
            return deepcopy(schema)

    def readiness(self):
        with self._lock:
            fresh = self._fresh()
            return (200 if fresh else 503), {
                "status": "ok" if fresh else "config_unavailable",
                "config": {
                    "status": "fresh"
                    if fresh and self._error is None
                    else ("degraded" if fresh else "unavailable"),
                    "age_seconds": None
                    if self._verified_at is None
                    else int(max(0, self._clock() - self._verified_at)),
                    "last_error": self._error,
                },
            }


def load_published_schema(task_type: str, operator_pubkey_hex: str | None = None) -> dict:
    """Return the verified `config_schema.<task_type>` dict from the operator snapshot. Raises
    SchemaUnavailable (never falls back to a lane import) if the snapshot is missing/invalid or
    carries no schema for the type."""
    entries = load_published_entries(operator_pubkey_hex)
    schema = entries.get(f"config_schema.{task_type}")
    if not isinstance(schema, dict):
        raise SchemaUnavailable(f"no published config_schema for task_type {task_type!r}")
    return schema


def load_published_entries(operator_pubkey_hex: str | None = None) -> dict:
    """Return the exact verified entries mapping for lane-owned operator tooling.

    This exposes signed policy data, not executable lane code. Callers must still parse their
    task-specific values with a closed schema and must never interpret an absent key as enabled.
    """

    env = dict(os.environ)
    if operator_pubkey_hex is not None:
        env["CONFIG_SIGNING_PUBKEY"] = operator_pubkey_hex
    return _load_entries(config_source(env))


def main(argv: list[str] | None = None) -> int:
    """Export the VERIFIED signed entries mapping to a file for lane-owned operator tooling.

    `python -m subnet.miner.config_schema --out entries.json` reads the snapshot through the same
    seam every miner command uses (`AURELIUS_CONFIG_SNAPSHOT` file, else `CENTRAL_API_URL`
    `/api/config`), verifies it against the shipped key or `CONFIG_SIGNING_PUBKEY` override,
    and writes only the `entries`
    object (the shape the ADR-0229 drills consume as `--entries`). An unverifiable snapshot exits 2
    and writes nothing; an existing output file is never overwritten.
    """

    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m subnet.miner.config_schema")
    parser.add_argument("--out", required=True, help="absolute path of the entries file to create")
    parser.add_argument("--pubkey", default=None, help="override CONFIG_SIGNING_PUBKEY")
    args = parser.parse_args(argv)
    out = args.out
    if not os.path.isabs(out) or os.path.lexists(out) or not os.path.isdir(os.path.dirname(out)):
        print(
            "--out must be an absolute, non-existing path in an existing directory", file=sys.stderr
        )
        return 2
    try:
        entries = load_published_entries(args.pubkey)
    except SchemaUnavailable as exc:
        print(f"entries export refused: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(out, flags, 0o640)
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"out": out, "entries": len(entries)}))
    return 0


def validate_config(config: dict, schema: dict) -> list[str]:
    """Check one config against the closed published-schema subset.

    The subset intentionally covers only top-level field validation: type, enum/const, numeric
    inclusive bounds, and UTF-8 string length/pattern constraints. Lane-owned typed models still
    own cross-field rules. Unknown or malformed schema rules are errors rather than silently
    weakening validation.
    """
    errors: list[str] = []
    if not isinstance(config, dict):
        return ["config: must be an object"]
    if not isinstance(schema, dict):
        return ["schema: must be an object"]

    fields = schema.get("fields", {})
    if not isinstance(fields, dict) or any(not isinstance(key, str) for key in fields):
        return ["schema: fields must be an object keyed by strings"]
    required = schema.get("required", [])
    if (
        not isinstance(required, list)
        or any(not isinstance(field, str) for field in required)
        or len(required) != len(set(required))
    ):
        return ["schema: required must be a unique string list"]
    if any(field not in fields for field in required):
        errors.append("schema: every required field must have a field rule")
    additional_fields = schema.get("additional_fields", True)
    if not isinstance(additional_fields, bool):
        errors.append("schema: additional_fields must be boolean")

    for req in required:
        if req not in config:
            errors.append(f"required: missing field {req!r}")

    if additional_fields is False:
        for key in config:
            if key not in fields:
                errors.append(f"additional_fields: unknown field {key!r}")

    for key, value in config.items():
        spec = fields.get(key)
        if spec is None:
            continue  # unknown-field handling is above; a field with no spec has no type rule
        if not isinstance(spec, dict):
            errors.append(f"schema: field {key!r} rule must be an object")
            continue
        unknown_rules = set(spec) - _FIELD_RULES
        if unknown_rules:
            errors.append(f"schema: field {key!r} has unsupported rules {sorted(unknown_rules)!r}")
            continue
        expected = spec.get("type")
        py_type = _TYPES.get(expected)
        if expected not in _TYPES:
            errors.append(f"schema: field {key!r} has unsupported type {expected!r}")
            continue
        # bool is a subclass of int — a boolean is never a valid integer/number here.
        if not isinstance(value, py_type) or _is_bad_bool(value, expected):
            errors.append(f"type: field {key!r} must be {expected}")
            continue
        if expected == "number":
            try:
                finite = math.isfinite(float(value))
            except OverflowError:
                finite = False
            if not finite:
                errors.append(f"type: field {key!r} must be a finite number")
                continue

        if "enum" in spec:
            enum = spec["enum"]
            if not isinstance(enum, list) or not enum:
                errors.append(f"schema: field {key!r} enum must be a non-empty list")
            elif not any(_json_equal(value, allowed) for allowed in enum):
                errors.append(f"enum: field {key!r} must be one of {enum}")
        if "const" in spec and not _json_equal(value, spec["const"]):
            errors.append(f"const: field {key!r} must equal {spec['const']!r}")

        if expected in ("integer", "number"):
            errors.extend(_numeric_errors(key, value, spec))
        elif any(rule in spec for rule in ("minimum", "maximum")):
            errors.append(f"schema: field {key!r} numeric bounds require integer/number")

        if expected == "string":
            errors.extend(_string_errors(key, value, spec))
        elif any(rule in spec for rule in ("minLength", "maxLength", "pattern")):
            errors.append(f"schema: field {key!r} string rules require string type")
    return errors


def _json_equal(left, right) -> bool:
    """JSON equality without Python's bool-equals-int aliasing."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    return type(left) is type(right) and left == right


def _numeric_errors(key: str, value: int | float, spec: dict) -> list[str]:
    errors: list[str] = []
    lower = spec.get("minimum")
    upper = spec.get("maximum")
    for name, bound in (("minimum", lower), ("maximum", upper)):
        if name not in spec:
            continue
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            errors.append(f"schema: field {key!r} {name} must be numeric")
        else:
            try:
                finite = math.isfinite(float(bound))
            except OverflowError:
                finite = False
            if not finite:
                errors.append(f"schema: field {key!r} {name} must be finite")
    if errors:
        return errors
    if "minimum" in spec and value < lower:
        errors.append(f"minimum: field {key!r} must be >= {lower}")
    if "maximum" in spec and value > upper:
        errors.append(f"maximum: field {key!r} must be <= {upper}")
    if "minimum" in spec and "maximum" in spec and lower > upper:
        errors.append(f"schema: field {key!r} minimum must not exceed maximum")
    return errors


def _string_errors(key: str, value: str, spec: dict) -> list[str]:
    errors: list[str] = []
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return [f"string: field {key!r} must be valid UTF-8"]

    minimum = spec.get("minLength")
    maximum = spec.get("maxLength")
    for name, bound in (("minLength", minimum), ("maxLength", maximum)):
        if name in spec and (isinstance(bound, bool) or not isinstance(bound, int) or bound < 0):
            errors.append(f"schema: field {key!r} {name} must be a non-negative integer")
    if errors:
        return errors
    if "minLength" in spec and len(value) < minimum:
        errors.append(f"minLength: field {key!r} must have length >= {minimum}")
    if "maxLength" in spec and len(value) > maximum:
        errors.append(f"maxLength: field {key!r} must have length <= {maximum}")
    if "minLength" in spec and "maxLength" in spec and minimum > maximum:
        errors.append(f"schema: field {key!r} minLength must not exceed maxLength")

    if "pattern" in spec:
        pattern = spec["pattern"]
        if not isinstance(pattern, str) or not pattern or len(pattern) > _MAX_PATTERN_CHARS:
            errors.append(f"schema: field {key!r} pattern must be a bounded non-empty string")
        else:
            try:
                matches = re.fullmatch(pattern, value) is not None
            except re.error:
                errors.append(f"schema: field {key!r} pattern is invalid")
            else:
                if not matches:
                    errors.append(f"pattern: field {key!r} does not match")
    return errors


def _is_bad_bool(value, expected: str) -> bool:
    return isinstance(value, bool) and expected in ("integer", "number")


if __name__ == "__main__":
    raise SystemExit(main())
