"""SAE miner tools. Local preparation and measurements never submit or charge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from subnet.miner.config_schema import SchemaUnavailable, load_published_entries, validate_config
from subnet.miner.journal import AttemptJournal, identity_for
from subnet.miner.staging import StagingPool
from subnet.miner.trust import config_signing_pubkey
from .miner_checks import check_bundle, window_preview
from .stage import build_candidate, selectable_slot_policy


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aurelius-sae")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "preview", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--slot-id", required=True)
        if name != "preview":
            command.add_argument("--bundle", required=True, type=Path)
        if name in {"preview", "prepare"}:
            command.add_argument(
                "--head", type=int, required=True, help="observed finalized block height"
            )
            command.add_argument("--slot-epoch", type=int)
        if name == "prepare":
            command.add_argument("--artifact-ref", required=True)
            command.add_argument("--hotkey", required=True)
            command.add_argument("--sparsity-target", required=True)
            command.add_argument("--stage", action="store_true")
            command.add_argument("--pool")
            command.add_argument("--journal")
    status = sub.add_parser("status")
    status.add_argument("--journal")
    status.add_argument("--refresh", action="store_true")
    status.add_argument(
        "--accounting",
        action="store_true",
        help="read exact own charge/refund facts with hotkey proof",
    )
    status.add_argument("--netuid", type=int)
    status.add_argument("--central", default=os.environ.get("CENTRAL_API_URL"))
    status.add_argument(
        "--validator",
        action="append",
        default=[],
        metavar="HOTKEY=HTTPS_URL",
        help="explicit validator progress endpoint; repeat for multiple validators",
    )
    status.add_argument("--attempt-id")
    status.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            report = status_report(args)
        else:
            entries = load_published_entries()
            policy = selectable_slot_policy(entries, args.slot_id)
            if args.command in {"preview", "prepare"}:
                report = window_preview(
                    policy,
                    head=args.head,
                    epoch=args.slot_epoch,
                    window_blocks=entries["window_blocks"],
                )
                report["head_source"] = "caller_supplied_finalized_observation"
            if args.command == "check":
                report = check_bundle(args.bundle, policy)
            elif args.command == "prepare":
                if report["status"] == "expired":
                    raise ValueError("commit window expired; preview the next eligible epoch")
                checks = check_bundle(args.bundle, policy)
                if not checks["compatible"]:
                    print(json.dumps(checks, indent=2, allow_nan=False))
                    return 2
                candidate = build_candidate(
                    args.bundle,
                    artifact_ref=args.artifact_ref,
                    slot_id=args.slot_id,
                    slot_epoch=report["slot_epoch"],
                    sparsity_target=args.sparsity_target,
                    policy=policy,
                    window_blocks=entries["window_blocks"],
                    ref_hotkey=args.hotkey
                    if args.artifact_ref.startswith("aurelius-artifact://")
                    else None,
                )
                errors = validate_config(
                    candidate["config"], entries.get("config_schema.interp_sae")
                )
                if errors:
                    raise ValueError("; ".join(errors))
                identity = identity_for(candidate, args.hotkey)
                journal = AttemptJournal(args.journal)
                identifier = journal.record(
                    identity, "prepared", "local", {"window": report, "checks": checks}
                )
                if args.stage:
                    StagingPool(args.pool).add(candidate)
                    journal.record(identity, "staged", "local")
                report = {
                    "attempt_id": identifier,
                    "identity": identity,
                    "window": report,
                    "staged": args.stage,
                    "candidate": candidate,
                    "checks": checks,
                }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return (
            2 if report.get("compatible") is False or report.get("status") == "unavailable" else 0
        )
    except (OSError, ValueError, RuntimeError, KeyError, SchemaUnavailable) as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)[:500]}))
        return 2


def status_report(args):
    from .miner_results import refresh

    journal = AttemptJournal(args.journal)
    attempts = journal.attempts(limit=args.limit)
    if args.attempt_id:
        attempts = [row for row in attempts if row["attempt_id"] == args.attempt_id]
        if not attempts:
            raise ValueError("attempt not found within the selected journal limit")
    if (args.refresh or args.accounting) and not args.central:
        raise ValueError("--central is required for remote observations")
    minter = None
    if args.accounting or args.validator:
        from subnet.miner.serve import minter_from_env

        if args.netuid is None:
            raise ValueError("accounting requires --netuid")
        public_key = config_signing_pubkey()
        minter = minter_from_env(os.environ.get("MINER_HOTKEY", "miner-hotkey"))
    observations = []
    for attempt in attempts:
        identity = attempt["identity"]
        result = {"attempt_id": attempt["attempt_id"]}
        if args.refresh:
            result["public"] = refresh(journal, identity, args.central)
        if args.accounting and minter is not None:
            from subnet.miner.observation_client import observe

            if identity["hotkey"] != minter.hotkey_ss58:
                result["accounting"] = {"status": "different_hotkey"}
            else:
                try:
                    facts = observe(
                        args.central,
                        identity,
                        netuid=args.netuid,
                        signer=minter.sign_observation,
                        config_pubkey=public_key,
                    )
                    journal.record(identity, "charge_observation", "central", facts)
                    result["accounting"] = facts
                    events = next(
                        row["events"]
                        for row in journal.attempts(limit=args.limit)
                        if row["attempt_id"] == attempt["attempt_id"]
                    )
                    reports = {
                        event["data"]["id"]: event["data"]["report"]
                        for event in events
                        if event["stage"] == "result"
                    }
                    result["validator_results"] = [
                        dict(validator, report=reports.get(validator["id"]))
                        for validator in facts["validators"]
                    ]
                except (ValueError, RuntimeError, OSError) as exc:
                    journal.record(
                        identity, "status_unavailable", "central", {"code": str(exc)[:100]}
                    )
                    result["accounting"] = {"status": "unavailable"}
        if args.validator and identity["hotkey"] == minter.hotkey_ss58:
            from subnet.miner.observation_client import progress

            for endpoint in args.validator:
                validator, separator, url = endpoint.partition("=")
                if not separator:
                    raise ValueError("validator endpoint must be HOTKEY=HTTPS_URL")
                try:
                    facts = progress(
                        url,
                        identity,
                        netuid=args.netuid,
                        validator_hotkey=validator,
                        signer=minter.sign_observation,
                        config_pubkey=public_key,
                    )
                    journal.record(identity, "validator_progress", "validator", facts)
                except (ValueError, RuntimeError, OSError, TypeError) as exc:
                    journal.record(
                        identity,
                        "status_unavailable",
                        "validator",
                        {"validator_hotkey": validator, "code": str(exc)[:100]},
                    )
        observations.append(result)
    identifiers = {row["attempt_id"] for row in attempts}
    return {
        "schema_version": 1,
        "authority": "observations_only",
        "refresh": observations,
        "attempts": [
            row for row in journal.attempts(limit=args.limit) if row["attempt_id"] in identifiers
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
