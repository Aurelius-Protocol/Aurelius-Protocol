"""Miner preparation and local evidence: retries, identity and honest readiness."""

import json
import urllib.error

import pytest

from subnet.miner import cli, commands
from subnet.miner.staging import StagingPool, StagingSelectionError, item_id

pytestmark = pytest.mark.miner


def candidate(payload="a"):
    return {
        "task_type": "interp_sae",
        "schema_version": 2,
        "config": {"payload": payload},
        "eligible_fanout_window": 7,
    }


def test_serve_reports_configuration_errors_without_a_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("AURELIUS_ENV", raising=False)
    monkeypatch.delenv("AURELIUS_TRANSPORT", raising=False)
    monkeypatch.setenv("AURELIUS_MINER_JOURNAL", str(tmp_path / "attempts.sqlite3"))
    monkeypatch.setenv("AURELIUS_MINER_DATA", str(tmp_path))
    monkeypatch.setenv("AURELIUS_ACCEPT_INCENTIVES", "yes")
    assert cli.main(["serve"]) == 2
    err = capsys.readouterr().err
    assert "[aurelius-miner] cannot serve:" in err
    assert "Traceback" not in err


def test_observe_records_remote_facts_into_the_journal(monkeypatch, tmp_path, capsys):
    from subnet.miner import observation_client
    from subnet.miner.journal import AttemptJournal, identity_for
    from subnet.miner.serve import WorkIdMinter

    hotkey = WorkIdMinter("miner-hotkey").hotkey_ss58
    candidate = {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "a"},
        "eligible_fanout_window": 3,
    }
    identity = identity_for(candidate, hotkey, window=3)
    journal_path = tmp_path / "attempts.sqlite3"
    AttemptJournal(journal_path).record(identity, "envelope_created", "local", {})

    monkeypatch.setenv("CONFIG_SIGNING_PUBKEY", "ab" * 32)
    monkeypatch.setattr(
        observation_client,
        "observe",
        lambda *args, **kwargs: {
            "charge": "recorded",
            "refund": "not_observed",
            "observed_at": "2026-10-01T00:00:00Z",
            "validators": [
                {
                    "id": "v1",
                    "validator_hotkey": "5" * 48,
                    "verdict": "unknown",
                    "inserted_at": "2026-10-01T00:00:00Z",
                },
            ],
        },
    )

    assert (
        cli.main(["observe", "--journal", str(journal_path), "--central", "http://127.0.0.1:4000"])
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["authority"] == "observations_only"
    assert out["refresh"][0]["accounting"]["charge"] == "recorded"
    events = AttemptJournal(journal_path).attempts()[0]["events"]
    assert [event["stage"] for event in events] == ["envelope_created", "charge_observation"]
    assert events[-1]["source"] == "central"


def test_observe_journals_unavailable_central_as_unknown(monkeypatch, tmp_path, capsys):
    from subnet.miner import observation_client
    from subnet.miner.journal import AttemptJournal, identity_for
    from subnet.miner.serve import WorkIdMinter

    hotkey = WorkIdMinter("miner-hotkey").hotkey_ss58
    candidate = {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "a"},
        "eligible_fanout_window": 3,
    }
    identity = identity_for(candidate, hotkey, window=3)
    journal_path = tmp_path / "attempts.sqlite3"
    AttemptJournal(journal_path).record(identity, "envelope_created", "local", {})

    monkeypatch.setenv("CONFIG_SIGNING_PUBKEY", "ab" * 32)

    def unavailable(*args, **kwargs):
        raise observation_client.ReadUnavailable("network_unavailable")

    monkeypatch.setattr(observation_client, "observe", unavailable)
    assert cli.main(["observe", "--journal", str(journal_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["refresh"][0]["accounting"] == {"status": "unavailable"}
    events = AttemptJournal(journal_path).attempts()[0]["events"]
    assert events[-1]["stage"] == "status_unavailable"


def test_observe_rejects_empty_config_signing_pubkey_override(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CONFIG_SIGNING_PUBKEY", "")
    assert cli.main(["observe", "--journal", str(tmp_path / "a.sqlite3")]) == 2
    assert "CONFIG_SIGNING_PUBKEY" in capsys.readouterr().err


def test_repeat_add_and_disable_preserve_history(tmp_path):
    pool = StagingPool(tmp_path / "pool.jsonl")
    item = candidate()
    pool.add(item)
    pool.add(item)
    assert pool.items() == [item]
    assert (
        pool.first_valid(
            task_type="interp_sae", schema_version=2, fanout_window=7, validate=lambda _: []
        )
        == item
    )
    pool.set_enabled(item_id(item), False)
    assert pool.items() == [item]
    assert (
        pool.first_valid(
            task_type="interp_sae", schema_version=2, fanout_window=7, validate=lambda _: []
        )
        is None
    )
    pool.set_enabled(item_id(item), True)
    pool.add(candidate("b"))
    with pytest.raises(StagingSelectionError):
        pool.first_valid(
            task_type="interp_sae", schema_version=2, fanout_window=7, validate=lambda _: []
        )


def test_balance_uses_explicit_lane(monkeypatch):
    seen = []
    monkeypatch.setattr(
        commands, "run_balance", lambda hotkey, task_type: seen.append((hotkey, task_type)) or 0
    )
    assert cli.main(["balance", "--hotkey", "miner", "--task-type", "interp_sae"]) == 0
    assert seen == [("miner", "interp_sae")]


def test_unavailable_balance_is_not_insufficient(monkeypatch, capsys):
    def fail(_):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(commands, "_get", fail)
    assert commands.run_balance("miner", "interp_sae") == 2
    assert "unavailable" in capsys.readouterr().out


def test_empty_pool_is_not_ready(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AURELIUS_MINER_DATA", str(tmp_path))
    assert commands.run_readiness("interp_sae", 7) == 1
    assert json.loads(capsys.readouterr().out)["ready"] is False


def test_controls_cannot_replace_an_already_selected_window(tmp_path):
    from subnet.miner.staging import StagingSelectionError

    pool = StagingPool(tmp_path / "pool")
    first = {
        "task_type": "synthetic",
        "schema_version": 1,
        "config": {"payload": "a"},
        "eligible_fanout_window": 4,
    }
    second = dict(first, config={"payload": "b"})
    pool.add(first)
    pool.pin_selection(first, 4)
    pool.set_enabled(item_id(first), False)
    pool.add(second)
    with pytest.raises(StagingSelectionError, match="already selected"):
        pool.pin_selection(second, 4)


def test_durable_campaign_staging_and_miner_controls_share_identity_after_restart(tmp_path):
    path = tmp_path / "pool.jsonl"
    first = candidate()
    pool = StagingPool(path)
    assert pool.add_once(first) is True
    inode = path.stat().st_ino
    pool.add(first)
    restarted = StagingPool(path)
    assert restarted.add_once(first) is False
    assert restarted.items() == [first]
    assert path.stat().st_ino == inode
    restarted.pin_selection(first, 7)
    restarted.set_enabled(item_id(first), False)
    # Disabling an item does not free its historical campaign window for replacement.
    with pytest.raises(StagingSelectionError, match="occupied"):
        restarted.add_once(candidate("replacement"))
    restarted.add(candidate("replacement"))
    with pytest.raises(StagingSelectionError, match="already selected"):
        StagingPool(path).pin_selection(candidate("replacement"), 7)
    assert path.stat().st_ino == inode
