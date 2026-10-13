"""Miner staging pool (plan §1.8) — local jsonl, no central, no coldkey, no lane import."""

import pytest

from subnet.miner.staging import (
    ELIGIBLE_FANOUT_WINDOW,
    StagingPool,
    StagingSelectionError,
)

pytestmark = pytest.mark.miner


def test_add_once_survives_restart_without_duplicate_or_replacing_bind_mount_inode(tmp_path):
    path = tmp_path / "staging.jsonl"
    item = {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "a"},
        ELIGIBLE_FANOUT_WINDOW: 7,
    }
    pool = StagingPool(path)
    assert pool.add_once(item) is True
    inode = path.stat().st_ino
    assert StagingPool(path).add_once(item) is False
    assert path.stat().st_ino == inode
    assert pool.items() == [item]
    with pytest.raises(StagingSelectionError, match="occupied"):
        pool.add_once(item | {"config": {"payload": "b"}})
    assert pool.items() == [item]
    assert pool.add_once(item | {ELIGIBLE_FANOUT_WINDOW: 8}) is True


def test_add_once_preserves_and_refuses_partial_existing_write(tmp_path):
    path = tmp_path / "staging.jsonl"
    path.write_text('{"task_type":')
    original = path.read_bytes()
    with pytest.raises(ValueError):
        StagingPool(path).add_once(
            {"task_type": "synthetic", "schema_version": 1, "config": {}, ELIGIBLE_FANOUT_WINDOW: 7}
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("existing", ['{"task_type":"other"}', "[]\n"])
def test_add_once_refuses_unterminated_or_nonobject_rows_without_corrupting_pool(
    tmp_path, existing
):
    path = tmp_path / "staging.jsonl"
    path.write_text(existing)
    with pytest.raises(StagingSelectionError):
        StagingPool(path).add_once(
            {"task_type": "synthetic", "schema_version": 1, "config": {}, ELIGIBLE_FANOUT_WINDOW: 7}
        )
    assert path.read_text() == existing


def test_add_and_read(tmp_path):
    pool = StagingPool(tmp_path / "staging.jsonl")
    pool.add({"task_type": "synthetic", "schema_version": 1, "config": {"payload": "a"}})
    pool.add({"task_type": "synthetic", "schema_version": 1, "config": {"payload": "b"}})
    items = pool.items()
    assert [i["config"]["payload"] for i in items] == ["a", "b"]


def test_missing_field_rejected(tmp_path):
    pool = StagingPool(tmp_path / "s.jsonl")
    with pytest.raises(ValueError):
        pool.add({"task_type": "synthetic", "config": {}})  # no schema_version


def test_empty_pool_is_empty_list(tmp_path):
    assert StagingPool(tmp_path / "none.jsonl").items() == []


def test_clear(tmp_path):
    pool = StagingPool(tmp_path / "s.jsonl")
    pool.add({"task_type": "synthetic", "schema_version": 1, "config": {"payload": "a"}})
    pool.clear()
    assert pool.items() == []


def test_only_identity_fields_persisted(tmp_path):
    pool = StagingPool(tmp_path / "s.jsonl")
    pool.add(
        {"task_type": "synthetic", "schema_version": 1, "config": {"payload": "a"}, "secret": "x"}
    )
    assert set(pool.items()[0]) == {"task_type", "schema_version", "config"}


def test_data_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AURELIUS_MINER_DATA", str(tmp_path))
    pool = StagingPool()  # defaults under AURELIUS_MINER_DATA
    assert str(tmp_path) in str(pool.path)


def test_first_valid_is_miner_local_append_order_and_exact_task_version(tmp_path):
    pool = StagingPool(tmp_path / "s.jsonl")
    pool.add({"task_type": "other", "schema_version": 1, "config": {"payload": "other"}})
    pool.add({"task_type": "synthetic", "schema_version": 2, "config": {"payload": "v2"}})
    pool.add({"task_type": "synthetic", "schema_version": 1, "config": {"bad": True}})
    pool.add({"task_type": "synthetic", "schema_version": 1, "config": {"payload": "first"}})
    pool.add({"task_type": "synthetic", "schema_version": 1, "config": {"payload": "second"}})

    selected = pool.first_valid(
        task_type="synthetic",
        schema_version=1,
        validate=lambda config: [] if "payload" in config else ["missing payload"],
    )

    assert selected == {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "first"},
    }


def test_first_valid_returns_none_when_request_or_pool_has_no_eligible_candidate(tmp_path):
    pool = StagingPool(tmp_path / "s.jsonl")
    pool.add({"task_type": "synthetic", "schema_version": 1, "config": {"bad": True}})

    def invalid(_config):
        return ["invalid"]

    assert pool.first_valid(task_type="synthetic", schema_version=1, validate=invalid) is None
    assert pool.first_valid(task_type="synthetic", schema_version=True, validate=invalid) is None
    assert pool.first_valid(task_type="", schema_version=1, validate=invalid) is None


def test_window_scoped_selection_skips_stale_future_and_legacy_rows(tmp_path):
    pool = StagingPool(tmp_path / "s.jsonl")
    pool.add({"task_type": "synthetic", "schema_version": 1, "config": {"payload": "legacy"}})
    pool.add(
        {
            "task_type": "synthetic",
            "schema_version": 1,
            "config": {"payload": "stale"},
            ELIGIBLE_FANOUT_WINDOW: 6,
        }
    )
    pool.add(
        {
            "task_type": "synthetic",
            "schema_version": 1,
            "config": {"payload": "current"},
            ELIGIBLE_FANOUT_WINDOW: 7,
        }
    )
    pool.add(
        {
            "task_type": "synthetic",
            "schema_version": 1,
            "config": {"payload": "future"},
            ELIGIBLE_FANOUT_WINDOW: 8,
        }
    )

    def select():
        return pool.first_valid(
            task_type="synthetic",
            schema_version=1,
            fanout_window=7,
            validate=lambda config: [] if "payload" in config else ["invalid"],
        )

    assert select()["config"]["payload"] == "current"
    assert select() == select()  # stable for every validator request in the same window
    assert (
        pool.first_valid(
            task_type="synthetic",
            schema_version=1,
            fanout_window=9,
            validate=lambda _config: [],
        )
        is None
    )


def test_multiple_valid_candidates_in_one_window_fail_closed(tmp_path):
    pool = StagingPool(tmp_path / "s.jsonl")
    for payload in ("first", "second"):
        pool.add(
            {
                "task_type": "synthetic",
                "schema_version": 1,
                "config": {"payload": payload},
                ELIGIBLE_FANOUT_WINDOW: 7,
            }
        )
    with pytest.raises(StagingSelectionError, match="multiple valid"):
        pool.first_valid(
            task_type="synthetic",
            schema_version=1,
            fanout_window=7,
            validate=lambda _config: [],
        )


@pytest.mark.parametrize("value", [True, -1, "7"])
def test_invalid_window_metadata_is_rejected(value, tmp_path):
    pool = StagingPool(tmp_path / "s.jsonl")
    with pytest.raises(ValueError, match=ELIGIBLE_FANOUT_WINDOW):
        pool.add(
            {
                "task_type": "synthetic",
                "schema_version": 1,
                "config": {"payload": "x"},
                ELIGIBLE_FANOUT_WINDOW: value,
            }
        )
