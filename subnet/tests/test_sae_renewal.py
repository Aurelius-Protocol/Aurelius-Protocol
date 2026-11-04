"""Budget and crash boundaries for opt-in paid renewal staging."""

import json
from copy import deepcopy

import pytest
from subnet.lanes.interp_sae import renewal as r

pytestmark = pytest.mark.miner


def state():
    return {
        "version": 1,
        "enabled": True,
        "max_wt": 20,
        "max_price_wt": 10,
        "lead_blocks": 1200,
        "reservations": [],
        "active": None,
    }


def candidate(window=100):
    return {
        "eligible_fanout_window": window,
        "task_type": "interp_sae",
        "schema_version": 2,
        "config": {"slot_epoch": window + 2},
    }


def test_cap_reserved_before_crash_and_restart(tmp_path):
    path = tmp_path / "state.json"
    s = state()
    assert r.reserve(s, candidate(), 10, 500)
    r.save(path, s)
    restored = json.loads(path.read_text())
    assert r.validate_state(restored) == 10
    assert r.reserve(restored, candidate(101), 10, 600)
    assert not r.reserve(restored, candidate(102), 10, 700)
    assert len(restored["reservations"]) == 2
    assert restored["max_wt"] == 20


def test_duplicate_window_never_reserves_twice():
    s = state()
    r.reserve(s, candidate(), 10, 500)
    with pytest.raises(ValueError, match="already reserved"):
        r.reserve(s, candidate(), 10, 500)
    assert r.validate_state(s) == 10


@pytest.mark.parametrize("cost", [True, 0, -1, 1.5])
def test_malformed_price_cannot_authorize_staging(cost):
    with pytest.raises(ValueError):
        r.reserve(state(), candidate(), cost, 500)


def test_announced_price_over_ceiling_does_not_reserve():
    s = state()
    before = deepcopy(s)
    assert not r.reserve(s, candidate(), 11, 500)
    assert s == before


def test_corrupt_or_overspent_history_fails_closed():
    s = state()
    r.reserve(s, candidate(), 10, 500)
    s["reservations"].append(deepcopy(s["reservations"][0]))
    with pytest.raises(ValueError):
        r.validate_state(s)
    s["reservations"].pop()
    s["max_wt"] = 5
    with pytest.raises(ValueError):
        r.validate_state(s)


def test_missing_state_run_does_not_reset_budget(tmp_path, capsys):
    path = tmp_path / "missing.json"
    assert r.main(["run", "--state", str(path), "--once"]) == 2
    assert not path.exists()
    assert "unavailable" in capsys.readouterr().out


def test_existing_plan_cannot_be_reset(tmp_path):
    path = tmp_path / "state.json"
    r.save(path, state())
    assert r.main(["init", "--state", str(path), "--max-wt", "100"]) == 2
    assert json.loads(path.read_text())["max_wt"] == 20


def test_symlink_state_refused(tmp_path):
    p = tmp_path / "state.json"
    target = tmp_path / "target"
    target.write_text("{}")
    p.symlink_to(target)
    with pytest.raises(ValueError), r.locked(p):
        pass


def test_winner_evidence_requires_agreement_and_exact_artifact():
    identity = {"hotkey": "miner", "artifact_sha256": "abc"}
    row = {
        "rank": 1,
        "champion_hotkey": "miner",
        "transcript": {"accepted": True, "champion": {"artifact_sha256": "abc"}},
    }
    assert r.won(None, identity) is None
    assert r.won({"verdicts": [row]}, identity)
    other = deepcopy(row)
    other["transcript"]["accepted"] = False
    assert not r.won({"verdicts": [row, other]}, identity)
    other = deepcopy(row)
    other["transcript"]["champion"]["artifact_sha256"] = "different"
    assert not r.won({"verdicts": [other]}, identity)


def test_reserve_then_stage_crash_can_repeat_stage_without_duplicate(tmp_path):
    from subnet.miner.staging import StagingPool

    pool = StagingPool(tmp_path / "pool.jsonl")
    c = candidate()
    r.ensure_staged(pool, c)
    r.ensure_staged(pool, c)
    assert len(pool.items()) == 1
    pool.set_enabled(r.item_id(c), False)
    r.ensure_staged(pool, c)
    assert len(pool.active_items()) == 1


def test_conflicting_enabled_candidate_never_overwritten(tmp_path):
    from subnet.miner.staging import StagingPool

    pool = StagingPool(tmp_path / "pool.jsonl")
    c = candidate()
    r.ensure_staged(pool, c)
    other = deepcopy(c)
    other["config"]["slot_epoch"] = 999
    with pytest.raises(ValueError, match="another current/future"):
        r.ensure_staged(pool, other)
    assert pool.items() == [c]


@pytest.fixture
def configured(tmp_path, monkeypatch):
    c = candidate(0)
    c["config"].update(
        slot_id="slot", artifact_sha256="abc", artifact_ref="reference", sparsity_target="0.1"
    )
    s = state()
    s.update(
        seed=c,
        pool=str(tmp_path / "pool.jsonl"),
        journal=str(tmp_path / "journal.db"),
        hotkey="miner",
        bundle=str(tmp_path / "bundle"),
        halt=None,
    )
    path = tmp_path / "renewal.json"
    r.save(path, s)
    monkeypatch.setattr(
        r,
        "identity_for",
        lambda c, h: {
            "hotkey": h,
            "artifact_sha256": c["config"]["artifact_sha256"],
            "slot_id": "slot",
        },
    )
    monkeypatch.setattr(r, "selectable_slot_policy", lambda e, slot: {})
    monkeypatch.setattr(
        r,
        "fetch_replay",
        lambda *a, **k: {
            "verdicts": [
                {
                    "rank": 1,
                    "champion_hotkey": "miner",
                    "transcript": {"accepted": True, "champion": {"artifact_sha256": "abc"}},
                }
            ]
        },
    )
    monkeypatch.setattr(r, "current_winner", lambda *a, **k: True)
    monkeypatch.setattr(r, "fee_ceiling", lambda *a: 10)
    monkeypatch.setattr(
        r, "window_preview", lambda *a, **k: {"slot_epoch": 28, "commit_end_exclusive": 2700}
    )
    new = deepcopy(c)
    new["eligible_fanout_window"] = 26
    new["config"]["slot_epoch"] = 28
    monkeypatch.setattr(r, "build_candidate", lambda *a, **k: new)
    monkeypatch.setattr(r, "validate_config", lambda *a: [])

    class Journal:
        def __init__(self, *a):
            pass

        def record(self, *a):
            pass

    monkeypatch.setattr(r, "AttemptJournal", Journal)
    entries = {
        "window_blocks": 100,
        "pertask.interp_sae.result_freshness_blocks": 3600,
        "pertask.interp_sae.deferred_deadline_blocks": 1800,
        "config_schema.interp_sae": {},
    }

    def run(head=2500, balance=100):
        return r.tick(
            path,
            s,
            head=head,
            entries=entries,
            central="https://example.test",
            economics={"account": {"balance": {"available_real": balance}}},
        )

    return path, s, run


def test_full_renewal_restarts_and_waits_for_result(configured):
    path, s, run = configured
    assert run() == "staged"
    assert json.loads(path.read_text())["reservations"][0]["wt"] == 10
    assert run(2501) == "awaiting_evaluation"
    assert r.validate_state(s) == 10
    assert len(r.StagingPool(s["pool"]).items()) == 1
    assert run(2800) == "fresh"
    assert s["active"] is None
    assert r.validate_state(s) == 10


def test_crash_after_reservation_before_stage_recovers_same_window(configured, monkeypatch):
    path, s, run = configured
    original = r.ensure_staged
    monkeypatch.setattr(r, "ensure_staged", lambda *a: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        run()
    assert json.loads(path.read_text())["active"] == 0
    assert r.validate_state(s) == 10
    monkeypatch.setattr(r, "ensure_staged", original)
    assert run() == "awaiting_evaluation"
    assert len(r.StagingPool(s["pool"]).items()) == 1
    assert r.validate_state(s) == 10


def test_failure_and_ambiguous_timeout_do_not_retry_paid_work(configured, monkeypatch):
    path, s, run = configured
    assert run() == "staged"
    monkeypatch.setattr(r, "fetch_replay", lambda *a, **k: None)
    assert run(2800) == "awaiting_evaluation"
    assert run(4501) == "halted"
    assert run(4600) == "halted"
    assert r.validate_state(s) == 10
    assert json.loads(path.read_text())["halt"]


def test_insufficient_funds_or_lost_title_never_stages(configured, monkeypatch):
    _path, s, run = configured
    assert run(balance=9) == "awaiting_funding"
    assert r.validate_state(s) == 0
    monkeypatch.setattr(r, "current_winner", lambda *a, **k: False)
    assert run() == "halted"
    assert r.validate_state(s) == 0


def test_stop_disables_reserved_candidate_without_freeing_budget(configured):
    path, s, run = configured
    assert run() == "staged"
    assert r.main(["stop", "--state", str(path)]) == 0
    stopped = json.loads(path.read_text())
    assert not stopped["enabled"]
    assert r.validate_state(stopped) == 10
    assert not r.StagingPool(s["pool"]).active_items()


def test_maximum_announced_price_reserved_including_future_activation():
    current = {
        "schema_version": 2,
        "version": 1,
        "effective_from_block": 100,
        "cost_evidence_sha256": "a" * 64,
        "lanes": {"interp_sae": {"gpu_sae_verdict": {"cost_model_version": 2, "work_tokens": 10}}},
        "conversion": {"alpha_rao_per_work_token": 1000000000},
        "capital": {"mode": "none"},
    }
    pending = deepcopy(current)
    pending.update(version=2, effective_from_block=8000)
    pending["lanes"]["interp_sae"]["gpu_sae_verdict"]["work_tokens"] = 15
    entries = {"price_policy": {"current": current, "next": pending}}
    assert r.fee_ceiling(entries, 200) == 15
    assert r.fee_ceiling(entries, 8100) == 15
    assert not r.reserve(state(), candidate(), r.fee_ceiling(entries, 200), 9000)


def test_nonwinning_renewal_stops_without_another_reservation(configured, monkeypatch):
    _path, s, run = configured
    assert run() == "staged"
    monkeypatch.setattr(
        r,
        "fetch_replay",
        lambda *a, **k: {
            "verdicts": [{"rank": 2, "champion_hotkey": "other", "transcript": {"accepted": True}}]
        },
    )
    assert run(2800) == "halted"
    assert run(5000) == "halted"
    assert r.validate_state(s) == 10


def test_uncertain_latest_incumbent_prevents_spending(configured, monkeypatch):
    _path, s, run = configured
    monkeypatch.setattr(r, "current_winner", lambda *a, **k: None)
    assert run() == "awaiting_incumbent_evidence"
    assert not s["reservations"]


def test_stopped_plan_exits_without_wallet_chain_or_network(tmp_path):
    path = tmp_path / "stopped.json"
    s = state()
    s["enabled"] = False
    r.save(path, s)
    assert r.main(["run", "--state", str(path), "--once"]) == 0
    assert json.loads(path.read_text()) == s
