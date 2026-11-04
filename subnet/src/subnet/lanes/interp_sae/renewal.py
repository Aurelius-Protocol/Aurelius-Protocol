"""Opt-in local renewal staging. Never charges, funds, grants eligibility or writes weights."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode

from subnet.miner.config_schema import load_published_entries, validate_config
from subnet.miner.economics_client import read_economics
from subnet.miner.journal import AttemptJournal, identity_for
from subnet.miner.read_client import ReadUnavailable, request_json
from subnet.miner.staging import StagingPool, item_id
from subnet.miner.trust import config_signing_pubkey
from subnet.shared.price_policy import resolve, shipped_catalog

from .miner_checks import window_preview
from .miner_results import fetch_replay
from .stage import build_candidate, selectable_slot_policy


def positive(value):
    if type(value) is not int or not 0 < value <= 2**63 - 1:
        raise ValueError("expected a positive integer")
    return value


@contextmanager
def locked(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("renewal state cannot be a symlink")
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        os.close(fd)


def save(path, state):
    # Reserve durably BEFORE any candidate can become visible to the serving miner.
    temporary = Path(str(path) + ".tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def validate_state(state):
    if (
        type(state["version"]) is not int
        or state["version"] != 1
        or type(state["enabled"]) is not bool
    ):
        raise ValueError("invalid renewal state")
    positive(state["max_wt"])
    positive(state["max_price_wt"])
    positive(state["lead_blocks"])
    if not isinstance(state["reservations"], list):
        raise TypeError("invalid reservations")
    windows = set()
    total = 0
    for row in state["reservations"]:
        positive(row["wt"])
        window = row["candidate"]["eligible_fanout_window"]
        if type(window) is not int or window < 0 or window in windows:
            raise ValueError("duplicate or invalid reserved window")
        windows.add(window)
        total += row["wt"]
    if total > state["max_wt"]:
        raise ValueError("renewal budget exceeded")
    active = state.get("active")
    if active is not None and (
        type(active) is not int or not 0 <= active < len(state["reservations"])
    ):
        raise ValueError("invalid active reservation")
    return total


def reserve(state, candidate, cost, deadline):
    total = validate_state(state)
    positive(cost)
    positive(deadline)
    if cost > state["max_price_wt"] or total + cost > state["max_wt"]:
        return False
    if any(
        r["candidate"]["eligible_fanout_window"] == candidate["eligible_fanout_window"]
        for r in state["reservations"]
    ):
        raise ValueError("window already reserved")
    state["reservations"].append({"candidate": candidate, "wt": cost, "deadline": deadline})
    state["active"] = len(state["reservations"]) - 1
    return True


def won(replay, identity):
    if replay is None or not replay["verdicts"]:
        return None
    # Every configured reporting validator observed for this exact attempt must agree.
    return all(
        r.get("rank") == 1
        and r.get("champion_hotkey") == identity["hotkey"]
        and r.get("transcript", {}).get("accepted") is True
        and r["transcript"].get("champion", {}).get("artifact_sha256")
        == identity["artifact_sha256"]
        for r in replay["verdicts"]
    )


def fee_ceiling(entries, head):
    document = entries["price_policy"]
    current = resolve(document, head, shipped_catalog())["policy"]
    versions = [current] + ([document["next"]] if document["next"] else [])
    # Reserve the maximum announced price, including queued changes. Never assume 10 WT.
    prices = [v["lanes"]["interp_sae"]["gpu_sae_verdict"] for v in versions]
    if any(p["cost_model_version"] != 2 for p in prices):
        raise ValueError("unreviewed cost model")
    return max(positive(p["work_tokens"]) for p in prices)


def current_winner(entries, policy, head, central, identity, fetch=request_json):
    epoch = (head - policy["phase_offset_blocks"]) // policy["cadence_blocks"]
    # A missing latest manifest is an observation hold, not permission to reuse old science.
    page = fetch(
        central.rstrip("/")
        + "/api/verdicts?"
        + urlencode({"slot": identity["slot_id"], "epoch": epoch, "limit": 1})
    )
    manifest = page["manifest"]
    if manifest["slot_id"] != identity["slot_id"] or manifest["epoch"] != epoch:
        raise ValueError("latest manifest identity differs")
    ladder = page["derived_ladder"]
    if ladder["status"] == "retained":
        champion = manifest["previous_incumbent"]
        if champion["status"] != "attested":
            return None
    elif ladder["status"] == "attested" and ladder["entries"]:
        champion = ladder["entries"][0]
        if champion.get("rank") != 1:
            return None
    else:
        return None
    return (champion["hotkey"], champion["artifact_sha256"]) == (
        identity["hotkey"],
        identity["artifact_sha256"],
    )


def tick(path, state, *, head, entries, central, economics, fetch=request_json):
    validate_state(state)
    if not state["enabled"]:
        return "stopped"
    if state.get("halt"):
        return "halted"
    pool = StagingPool(state["pool"])
    candidate = state["seed"]
    active = state["active"]
    if active is not None:
        row = state["reservations"][active]
        candidate = row["candidate"]
        width = positive(entries["window_blocks"])
        start = candidate["eligible_fanout_window"] * width
        # Reconcile after crashes, including reserve-before-stage. Never reserve twice.
        if head < start + width:
            ensure_staged(pool, candidate)
            return "awaiting_evaluation"
        identity = identity_for(candidate, state["hotkey"])
        outcome = won(fetch_replay(central, identity, fetch=fetch), identity)
        if outcome is None:
            if head > row["deadline"]:
                state["halt"] = "evaluation_timeout; reservation retained; inspect before retry"
                save(path, state)
                return "halted"
            return "awaiting_evaluation"
        if not outcome:
            state["halt"] = "renewal did not win; no automatic paid retry"
            save(path, state)
            return "halted"
        state["seed"] = candidate
        state["active"] = None
        save(path, state)
    else:
        identity = identity_for(candidate, state["hotkey"])
        if won(fetch_replay(central, identity, fetch=fetch), identity) is not True:
            return "waiting_for_winning_seed"
    identity = identity_for(candidate, state["hotkey"])
    policy = selectable_slot_policy(entries, identity["slot_id"])
    fresh = positive(entries["pertask.interp_sae.result_freshness_blocks"])
    if state["lead_blocks"] >= fresh:
        raise ValueError("lead must be shorter than result freshness")
    # Conservative lower bound: completion cannot precede the commitment start.
    due = candidate["eligible_fanout_window"] * entries["window_blocks"] + fresh
    if head < due - state["lead_blocks"]:
        return "fresh"
    title = current_winner(entries, policy, head, central, identity, fetch=fetch)
    if title is None:
        return "awaiting_incumbent_evidence"
    if not title:
        state["halt"] = "another incumbent observed; no paid challenge authorized"
        save(path, state)
        return "halted"
    cost = fee_ceiling(entries, head)
    used = validate_state(state)
    if cost > state["max_price_wt"]:
        return "price_limit"
    if used + cost > state["max_wt"]:
        return "budget_exhausted"
    if economics["account"]["balance"]["available_real"] < cost:
        return "awaiting_funding"
    window = window_preview(policy, head=head, window_blocks=entries["window_blocks"])
    # Do not authorize a candidate across a pending price activation.
    end = window["commit_end_exclusive"]
    cost = max(cost, fee_ceiling(entries, end))
    next_candidate = build_candidate(
        state["bundle"],
        artifact_ref=candidate["config"]["artifact_ref"],
        slot_id=identity["slot_id"],
        slot_epoch=window["slot_epoch"],
        sparsity_target=candidate["config"]["sparsity_target"],
        policy=policy,
        window_blocks=entries["window_blocks"],
        ref_hotkey=state["hotkey"],
    )
    if next_candidate["config"]["artifact_sha256"] != identity["artifact_sha256"]:
        raise ValueError("bundle changed; initialize a separate explicit plan")
    errors = validate_config(next_candidate["config"], entries["config_schema.interp_sae"])
    if errors:
        raise ValueError("; ".join(errors))
    check_conflicts(pool, next_candidate)
    deadline = end + positive(entries["pertask.interp_sae.deferred_deadline_blocks"])
    if not reserve(state, next_candidate, cost, deadline):
        return "budget_exhausted"
    save(path, state)
    ensure_staged(pool, next_candidate)
    journal = AttemptJournal(state["journal"])
    journal.record(identity_for(next_candidate, state["hotkey"]), "staged", "local")
    return "staged"


def check_conflicts(pool, candidate):
    window = candidate["eligible_fanout_window"]
    if any(
        x.get("eligible_fanout_window", -1) >= window and item_id(x) != item_id(candidate)
        for x in pool.active_items()
    ):
        raise ValueError("another current/future candidate is enabled")


def ensure_staged(pool, candidate):
    check_conflicts(pool, candidate)
    if item_id(candidate) not in {item_id(x) for x in pool.items()}:
        pool.add(candidate)
    pool.set_enabled(item_id(candidate), True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aurelius-sae renew")
    parser.add_argument("action", choices=["init", "run", "status", "stop"])
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--candidate", type=Path, help="JSON output of aurelius-sae prepare")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--hotkey")
    parser.add_argument("--max-wt", type=int)
    parser.add_argument("--max-price-wt", type=int, default=10)
    parser.add_argument("--lead-blocks", type=int, default=1200)
    parser.add_argument("--pool")
    parser.add_argument("--journal")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        positive(args.poll_seconds)
        with locked(args.state) as path:
            if args.action == "init":
                if path.exists():
                    raise ValueError("state exists; budget cannot be reset by init")
                if not args.candidate or not args.bundle or not args.hotkey:
                    raise ValueError("init requires --candidate, --bundle and --hotkey")
                seed = json.loads(args.candidate.read_text())
                seed = seed.get("candidate", seed)
                identity_for(seed, args.hotkey)
                state = {
                    "version": 1,
                    "enabled": True,
                    "halt": None,
                    "seed": seed,
                    "active": None,
                    "bundle": str(args.bundle.resolve()),
                    "hotkey": args.hotkey,
                    "pool": str(StagingPool(args.pool).path.resolve()),
                    "journal": str(AttemptJournal(args.journal).path.resolve()),
                    "max_wt": positive(args.max_wt),
                    "max_price_wt": positive(args.max_price_wt),
                    "lead_blocks": positive(args.lead_blocks),
                    "reservations": [],
                }
                save(path, state)
                print(json.dumps({"status": "initialized", "reserved_wt": 0}))
                return 0
            state = json.loads(path.read_text())  # Missing state never creates a fresh budget.
            used = validate_state(state)
            if args.action == "stop":
                state["enabled"] = False
                save(path, state)
                if state["active"] is not None:
                    row = state["reservations"][state["active"]]
                    pool = StagingPool(state["pool"])
                    if item_id(row["candidate"]) in {item_id(x) for x in pool.items()}:
                        pool.set_enabled(item_id(row["candidate"]), False)
            if args.action != "run":
                print(
                    json.dumps(
                        {
                            "enabled": state["enabled"],
                            "halt": state["halt"],
                            "reserved_wt": used,
                            "max_wt": state["max_wt"],
                        }
                    )
                )
                return 0
        if not state["enabled"] or state.get("halt"):
            print(json.dumps({"status": "stopped" if not state["enabled"] else "halted"}))
            return 0
        from subnet.bittensor_v11 import FinalizedRpc
        from subnet.miner.serve import minter_from_env

        signer = minter_from_env(os.environ.get("MINER_HOTKEY", "miner-hotkey"))
        central = os.environ["CENTRAL_API_URL"]
        while True:
            try:
                with locked(args.state) as path:
                    state = json.loads(path.read_text())
                    if not state["enabled"] or state.get("halt"):
                        print(
                            json.dumps({"status": "stopped" if not state["enabled"] else "halted"}),
                            flush=True,
                        )
                        return 0
                    if signer.hotkey_ss58 != state["hotkey"]:
                        raise ValueError("serving hotkey differs from renewal plan")
                    rpc = FinalizedRpc(os.environ["SUBSTRATE_URL"])
                    try:
                        snapshot = rpc.snapshot()
                    finally:
                        rpc.close()
                    if snapshot.genesis_hash != os.environ["AURELIUS_AXON_EXPECTED_GENESIS_HASH"]:
                        raise ValueError("chain genesis differs")
                    if snapshot.runtime_spec_version != int(
                        os.environ["AURELIUS_RUNTIME_SPEC_PIN"]
                    ):
                        raise ValueError("runtime pin differs; update the miner")
                    entries = load_published_entries()
                    economics = read_economics(
                        central,
                        hotkey=signer.hotkey_ss58,
                        netuid=37,
                        signer=signer.sign_economics,
                        config_pubkey=config_signing_pubkey(),
                    )
                    result = tick(
                        path,
                        state,
                        head=snapshot.block_number,
                        entries=entries,
                        central=central,
                        economics=economics,
                    )
                    print(
                        json.dumps(
                            {
                                "status": result,
                                "reserved_wt": validate_state(state),
                                "max_wt": state["max_wt"],
                            }
                        ),
                        flush=True,
                    )
            except (ReadUnavailable, OSError, RuntimeError, ValueError, KeyError) as exc:
                # No new staging on missing evidence. Reservations remain consumed.
                print(
                    json.dumps({"status": "waiting_for_evidence", "error": type(exc).__name__}),
                    flush=True,
                )
                if args.once:
                    return 2
            else:
                if args.once or result in {"halted", "stopped", "budget_exhausted"}:
                    return 0
            time.sleep(args.poll_seconds)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)[:200]}))
        return 2
