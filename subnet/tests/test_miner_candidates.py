"""Miner-originated candidate ingress: policy asks for a lane/version; content comes from the
local JSONL pool and must pass the signed published schema before the miner may sign it."""

import pytest

from subnet.axon_protocol import CANDIDATE_SOURCE_STAGING, PROTOCOL_VERSION
from subnet.miner import config_schema
from subnet.miner.candidates import CandidateUnavailable, candidate_for_request
from subnet.miner.serve import WorkIdMinter
from subnet.miner.staging import ELIGIBLE_FANOUT_WINDOW, StagingPool

pytestmark = pytest.mark.miner


def _testlab_posture(monkeypatch):
    """The exact local posture where inline carriage and legacy unscoped pools remain legal."""
    monkeypatch.setenv("AURELIUS_ENV", "testlab")
    monkeypatch.setenv("CHAIN_ID", "localnet")
    monkeypatch.setenv("AURELIUS_AXON_EXPECTED_GENESIS_HASH", "observe-once-testlab")
    monkeypatch.setenv("SUBSTRATE_URL", "ws://subtensor:9944")
    monkeypatch.setenv("AURELIUS_TRANSPORT", "http")


_SCHEMA = {
    "schema_version": 1,
    "required": ["payload"],
    "additional_fields": False,
    "fields": {"payload": {"type": "string"}},
}
_VALIDATOR = WorkIdMinter("validator-hotkey").hotkey_ss58


def _request(**overrides):
    request = {
        "request_id": "ab" * 32,
        "validator_hotkey_ss58": _VALIDATOR,
        "fanout_window": 7,
        "validator_version": "0.0.0",
        "protocol_version": PROTOCOL_VERSION,
        "candidate_source": CANDIDATE_SOURCE_STAGING,
        "task_type": "synthetic",
        "schema_version": 1,
    }
    request.update(overrides)
    return request


def _pool(tmp_path, *items):
    pool = StagingPool(tmp_path / "staging.jsonl")
    for item in items:
        pool.add(item)
    return pool


def test_staging_request_uses_first_valid_miner_item_and_ignores_validator_config(
    tmp_path, monkeypatch
):
    # Legacy append-order pools are confined to the local testlab posture (production requires
    # window-scoped rows); the selection behavior itself is unchanged under that posture.
    _testlab_posture(monkeypatch)
    pool = _pool(
        tmp_path,
        {"task_type": "synthetic", "schema_version": 1, "config": {"invalid": "skip"}},
        {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "miner-first"}},
        {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "miner-second"}},
    )

    candidate = candidate_for_request(
        _request(config={"payload": "validator-must-not-control-this"}),
        pool=pool,
        schema_loader=lambda _task_type: _SCHEMA,
    )

    assert candidate == {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "miner-first"},
    }


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"request_id": ""}, "request_id"),
        ({"request_id": "AB" * 32}, "request_id"),
        ({"validator_hotkey_ss58": "bad|validator"}, "validator_hotkey_ss58"),
        ({"fanout_window": True}, "fanout_window"),
        ({"validator_version": ""}, "validator_version"),
        ({"protocol_version": PROTOCOL_VERSION + 1}, "protocol_version"),
        ({"schema_version": True}, "schema_version"),
        ({"candidate_source": "unknown"}, "candidate_source"),
    ],
)
def test_staging_request_metadata_fails_closed(tmp_path, override, match):
    pool = _pool(
        tmp_path,
        {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "miner"}},
    )
    with pytest.raises(CandidateUnavailable, match=match):
        candidate_for_request(
            _request(**override), pool=pool, schema_loader=lambda _task_type: _SCHEMA
        )


def test_staging_request_requires_matching_published_schema_version(tmp_path):
    pool = _pool(
        tmp_path,
        {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "miner"}},
    )
    wrong_version = {**_SCHEMA, "schema_version": 2}
    with pytest.raises(CandidateUnavailable, match="does not match"):
        candidate_for_request(_request(), pool=pool, schema_loader=lambda _task_type: wrong_version)


def test_staging_request_never_serves_when_schema_is_unavailable(tmp_path):
    pool = _pool(
        tmp_path,
        {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "miner"}},
    )

    def unavailable(_task_type):
        raise config_schema.SchemaUnavailable("signed snapshot unavailable")

    with pytest.raises(CandidateUnavailable, match="signed snapshot unavailable"):
        candidate_for_request(_request(), pool=pool, schema_loader=unavailable)


def test_malformed_jsonl_pool_fails_closed(tmp_path):
    pool = StagingPool(tmp_path / "staging.jsonl")
    pool.path.write_text('{"task_type":', encoding="utf-8")
    with pytest.raises(CandidateUnavailable, match="staging pool refused selection"):
        candidate_for_request(_request(), pool=pool, schema_loader=lambda _task_type: _SCHEMA)


def test_inline_source_preserves_existing_local_and_one_shot_behavior(monkeypatch):
    _testlab_posture(monkeypatch)
    request = _request(candidate_source="inline", config={"payload": "legacy-fixture"})
    assert candidate_for_request(request) == {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "legacy-fixture"},
    }


def test_inline_source_is_refused_outside_the_local_testlab_posture(monkeypatch):
    """A permitted validator must not make a production miner sign validator-chosen config."""
    for key in ("AURELIUS_ENV", "CHAIN_ID", "AURELIUS_AXON_EXPECTED_GENESIS_HASH", "SUBSTRATE_URL"):
        monkeypatch.delenv(key, raising=False)
    request = _request(candidate_source="inline", config={"payload": "smuggled"})
    with pytest.raises(CandidateUnavailable, match="testlab posture"):
        candidate_for_request(request)


def test_unscoped_legacy_pool_is_refused_outside_the_local_testlab_posture(tmp_path, monkeypatch):
    """Production serving requires window-scoped rows so one stable candidate serves a window."""
    for key in ("AURELIUS_ENV", "CHAIN_ID", "AURELIUS_AXON_EXPECTED_GENESIS_HASH", "SUBSTRATE_URL"):
        monkeypatch.delenv(key, raising=False)
    pool = _pool(
        tmp_path,
        {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "legacy"}},
    )
    with pytest.raises(CandidateUnavailable, match="window-scoped"):
        candidate_for_request(_request(), pool=pool, schema_loader=lambda _task_type: _SCHEMA)


def test_request_rejects_unknown_fields_but_staging_may_explicitly_ignore_config(tmp_path):
    pool = _pool(
        tmp_path,
        {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "miner"}},
    )
    with pytest.raises(CandidateUnavailable, match="fields"):
        candidate_for_request(
            _request(attacker_extra="smuggled"),
            pool=pool,
            schema_loader=lambda _task_type: _SCHEMA,
        )


def test_request_window_selects_current_staged_candidate_and_strips_metadata(tmp_path):
    pool = _pool(
        tmp_path,
        {
            "task_type": "synthetic",
            "schema_version": 1,
            "config": {"payload": "stale"},
            ELIGIBLE_FANOUT_WINDOW: 6,
        },
        {
            "task_type": "synthetic",
            "schema_version": 1,
            "config": {"payload": "current"},
            ELIGIBLE_FANOUT_WINDOW: 7,
        },
        {
            "task_type": "synthetic",
            "schema_version": 1,
            "config": {"payload": "future"},
            ELIGIBLE_FANOUT_WINDOW: 8,
        },
    )
    candidate = candidate_for_request(
        _request(fanout_window=7), pool=pool, schema_loader=lambda _task_type: _SCHEMA
    )
    assert candidate == {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "current"},
    }
