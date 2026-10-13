"""Miner-owned candidate selection for the §3.4 staging-pool serving path.

CONTRACT (capsule header)
- Every request has one closed v2 metadata shape, including the validator audience, 64-hex
  request identity, and non-negative authoritative fanout window. A ``staging`` request is
  operator policy only: candidate bytes come exclusively from the miner's local ``StagingPool``;
  any validator-supplied ``config`` is ignored.
- Eligibility fails closed against the signed, published ``config_schema.<task_type>``. No lane
  implementation is imported, and schema unavailability never falls back to trusting the item.
- ``inline`` retains testlab/one-shot candidate carriage as an explicit source, but ONLY under
  the exact local testlab posture ([ADR-0292]); outside it an inline request is refused before
  candidate resolution, so a permitted validator can never make a production miner sign
  validator-chosen config. Production staging additionally requires window-scoped rows.
- Window eligibility is miner-local, non-identity metadata. Stale/future candidates are skipped,
  one exact-window candidate is stable for all validators, and ambiguity fails closed.
- This capsule selects content only. It knows nothing about payment, novelty, quality, scoring, or
  adoption; ``WorkIdMinter`` remains the sole envelope/signing step after selection.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from subnet.axon_protocol import (
    CANDIDATE_SOURCE_INLINE,
    CANDIDATE_SOURCE_STAGING,
    PROTOCOL_VERSION,
)
from subnet.miner import config_schema
from subnet.miner.staging import StagingPool


class CandidateUnavailable(ValueError):
    """The request cannot be answered with an eligible miner-owned candidate."""


class NoCandidateAvailable(CandidateUnavailable):
    """A valid request has no eligible miner-owned work in this window."""


class CandidateConfigUnavailable(CandidateUnavailable):
    """Published schema authority is unavailable; retry after configuration recovers."""


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SS58 = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{1,64}$")
_COMMON_REQUEST_FIELDS = frozenset(
    {
        "request_id",
        "validator_hotkey_ss58",
        "fanout_window",
        "validator_version",
        "protocol_version",
        "candidate_source",
        "task_type",
        "schema_version",
    }
)


def _utf8_size(value: str, field: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise CandidateUnavailable(f"{field} must be valid UTF-8") from None


def _inline_allowed() -> bool:
    """Inline candidate carriage belongs to the exact local testlab posture only.

    A permitted-but-malicious validator must never be able to make a production miner
    mint work-ids for validator-chosen config (charge/score sabotage under the miner's
    identity). Lazy import: serve imports this module at load time.
    """
    from subnet.miner.serve import plain_query_allowed

    return plain_query_allowed()


def candidate_for_request(
    request: dict,
    *,
    pool: StagingPool | None = None,
    schema_loader: Callable[[str], dict] = config_schema.load_published_schema,
    allow_inline: bool | None = None,
) -> dict:
    """Resolve the exact task that ``WorkIdMinter`` may sign for this request.

    Both sources share the same closed v2 request metadata. The staging source is the production
    trust claim: the validator chooses the lane/version, but neither chooses nor supplies the
    candidate. Outside the exact local testlab posture, ``inline`` requests are REFUSED and the
    staging pool must carry window-scoped rows (``eligible_fanout_window``), so one stable
    candidate serves every validator in a window — an unscoped legacy pool cannot rotate
    divergent content per validator.
    """
    if not isinstance(request, dict):
        raise CandidateUnavailable("task request must be an object")

    source = request.get("candidate_source")
    if source not in (CANDIDATE_SOURCE_INLINE, CANDIDATE_SOURCE_STAGING):
        raise CandidateUnavailable(f"unknown candidate_source {source!r}")
    fields = frozenset(request)
    allowed_fields = _COMMON_REQUEST_FIELDS | {"config"}
    if not _COMMON_REQUEST_FIELDS <= fields or not fields <= allowed_fields:
        missing = sorted(_COMMON_REQUEST_FIELDS - fields)
        extra = sorted(repr(field) for field in fields - allowed_fields)
        raise CandidateUnavailable(
            f"task request fields are not closed (missing={missing}, extra={extra})"
        )
    if source == CANDIDATE_SOURCE_INLINE:
        if allow_inline is None:
            allow_inline = _inline_allowed()
        if not allow_inline:
            raise CandidateUnavailable(
                "candidate_source=inline is confined to the exact local testlab posture"
            )
        if "config" not in fields:
            raise CandidateUnavailable("inline request missing 'config'")
    request_id = request["request_id"]
    validator_hotkey = request["validator_hotkey_ss58"]
    fanout_window = request["fanout_window"]
    validator_version = request["validator_version"]
    protocol_version = request["protocol_version"]
    task_type = request["task_type"]
    schema_version = request["schema_version"]
    if not isinstance(request_id, str) or _HEX64.fullmatch(request_id) is None:
        raise CandidateUnavailable("request_id must be 64 lowercase hex characters")
    if not isinstance(validator_hotkey, str) or _SS58.fullmatch(validator_hotkey) is None:
        raise CandidateUnavailable("validator_hotkey_ss58 must be a bounded base58 address")
    if isinstance(fanout_window, bool) or not isinstance(fanout_window, int) or fanout_window < 0:
        raise CandidateUnavailable("fanout_window must be a non-negative integer")
    if (
        not isinstance(validator_version, str)
        or not validator_version
        or _utf8_size(validator_version, "validator_version") > 128
    ):
        raise CandidateUnavailable("validator_version must be 1..128 UTF-8 bytes")
    if type(protocol_version) is not int or protocol_version != PROTOCOL_VERSION:
        raise CandidateUnavailable(f"unsupported protocol_version {protocol_version!r}")
    if not isinstance(task_type, str) or not task_type or _utf8_size(task_type, "task_type") > 128:
        raise CandidateUnavailable("task request needs a bounded task_type")
    if type(schema_version) is not int or schema_version < 1:
        raise CandidateUnavailable("task request needs a positive integer schema_version")

    if source == CANDIDATE_SOURCE_INLINE:
        return _inline_candidate(request)

    try:
        schema = schema_loader(task_type)
    except config_schema.SchemaUnavailable as e:
        raise CandidateConfigUnavailable(str(e)) from None
    if not isinstance(schema, dict):
        raise CandidateUnavailable("published config schema must be an object")
    published_version = schema.get("schema_version")
    if type(published_version) is not int or published_version != schema_version:
        raise CandidateUnavailable(
            f"published schema version {published_version!r} does not match request "
            f"version {schema_version}"
        )

    selected_pool = pool or StagingPool()
    try:
        selected = selected_pool.first_valid(
            task_type=task_type,
            schema_version=schema_version,
            fanout_window=fanout_window,
            validate=lambda config: config_schema.validate_config(config, schema),
            require_window_scoped=not _inline_allowed(),
        )
    except (OSError, ValueError) as e:
        raise CandidateUnavailable(f"staging pool refused selection: {e}") from None
    if selected is None:
        raise NoCandidateAvailable(
            f"no valid staged candidate for {task_type!r} schema {schema_version}"
        )
    try:
        selected_pool.pin_selection(selected, fanout_window)
    except (OSError, ValueError) as exc:
        raise CandidateUnavailable(f"staging selection unavailable: {exc}") from None
    return {
        "task_type": selected["task_type"],
        "schema_version": selected["schema_version"],
        "config": selected["config"],
    }


def _inline_candidate(request: dict) -> dict:
    """The testlab validator-supplied candidate path, still carrying exact v2 authorization."""
    return {
        "task_type": request["task_type"],
        "schema_version": request["schema_version"],
        "config": request["config"],
    }
