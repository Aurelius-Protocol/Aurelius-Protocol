"""Miner CLI entry point.

The original v1 commands are: `deposit`, `balance`, `validate`, `serve` (plan §15.1).
- `serve` (testlab long-running `miner` service): a `/v1/query` responder that mints a fresh,
  real-sr25519-signed work_id per request; btauth readiness is exposed by the companion health
  server on `AURELIUS_MINER_HTTP_PORT` (8092 in testlab), signing as `MINER_HOTKEY`.
- `validate`: check staged items against the PUBLISHED config_schema (no lane import).
- `balance`: a public affordability read + the deposit address (no session, no coldkey).
- `deposit`: the sole coldkey unlocker; uses it LOCALLY only (never in a payload).
- `observe`: lane-agnostic refresh of charge/refund + validator-progress facts into the local
  journal (the dashboard's data source; absence of evidence stays unknown).

The miner never authenticates a session and never calls the charge path (invariant 1); the
`charge-path-single-import-site` contract now enumerates `subnet.miner.*` as a forbidden source.
"""

from __future__ import annotations

import argparse
import os
import sys


def _serve_mode() -> str:
    """The shared wire seam: v11 ``btauth`` (publish endpoint + authenticated HTTP) or the
    hermetic plain ``http`` responder."""
    return os.environ.get("AURELIUS_TRANSPORT", "http")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aurelius-miner")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="validate staged items against the published config_schema")

    balance = sub.add_parser("balance", help="report affordability + the deposit address")
    balance.add_argument("--hotkey", default=os.environ.get("MINER_HOTKEY_SS58"))
    balance.add_argument("--task-type", default="synthetic")

    economics = sub.add_parser(
        "economics", help="read own balance, signed prices and recent receipts"
    )
    economics.add_argument("--netuid", type=int, default=int(os.environ.get("NETUID", "2")))

    ready = sub.add_parser("ready", help="check local candidate eligibility, not funding/discovery")
    ready.add_argument("--task-type", required=True)
    ready.add_argument("--fanout-window", type=int, required=True)

    observe = sub.add_parser(
        "observe",
        help="refresh charge/refund and validator-progress observations for recent attempts",
    )
    observe.add_argument("--netuid", type=int, default=int(os.environ.get("NETUID", "2")))
    observe.add_argument(
        "--central", default=os.environ.get("CENTRAL_API_URL", "http://127.0.0.1:4000")
    )
    observe.add_argument("--journal")
    observe.add_argument("--limit", type=int, default=20)
    observe.add_argument(
        "--validator",
        action="append",
        default=[],
        metavar="HOTKEY=URL",
        help="validator progress endpoint (repeatable)",
    )

    pool = sub.add_parser("pool", help="list or enable/disable staged items, preserving history")
    pool.add_argument("action", choices=("list", "enable", "disable"))
    pool.add_argument("item_id", nargs="?")
    pool.add_argument("--pool")

    status = sub.add_parser("status", help="read the local attempt journal")
    status.add_argument("--journal")
    dashboard = sub.add_parser("dashboard", help="read-only localhost observation dashboard")
    dashboard.add_argument("--journal")
    dashboard.add_argument("--pool")
    dashboard.add_argument("--experiments", help="parent directory of local experiment runs")
    dashboard.add_argument("--port", type=int, default=8093)

    deposit = sub.add_parser("deposit", help="resolve the deposit address (unlocks the coldkey)")
    deposit.add_argument(
        "--unlock",
        action="store_true",
        help="explicitly unlock the coldkey for local deposit signing (never sent)",
    )

    serve = sub.add_parser("serve", help="serve /v1/query (mint a fresh work_id per request)")
    serve.add_argument("--netuid", type=int, default=int(os.environ.get("NETUID", "2")))
    serve.add_argument("--port", type=int, default=8091)

    args = parser.parse_args(argv)

    if args.command == "economics":
        import json
        from subnet.miner.economics_client import read_economics
        from subnet.miner.serve import minter_from_env

        minter = minter_from_env(os.environ.get("MINER_HOTKEY", "miner-hotkey"))
        public_key = os.environ.get("CONFIG_SIGNING_PUBKEY")
        if not public_key:
            parser.error("economics requires CONFIG_SIGNING_PUBKEY")
        result = read_economics(
            os.environ.get("CENTRAL_API_URL", "http://127.0.0.1:4000"),
            hotkey=minter.hotkey_ss58,
            netuid=args.netuid,
            signer=minter.sign_economics,
            config_pubkey=public_key,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "status":
        import json
        from subnet.miner.journal import AttemptJournal

        print(json.dumps({"attempts": AttemptJournal(args.journal).attempts()}, indent=2))
        return 0
    if args.command == "dashboard":
        from subnet.miner.dashboard import serve

        serve(args.journal, port=args.port, pool=args.pool, experiments=args.experiments)
        return 0

    if args.command == "serve":
        from subnet.miner.staging import data_dir

        os.environ.setdefault("AURELIUS_MINER_JOURNAL", str(data_dir() / "attempts.sqlite3"))
        hotkey_name = os.environ.get("MINER_HOTKEY", "miner-hotkey")
        from subnet.miner.serve import validate_miner_startup_posture

        try:
            validate_miner_startup_posture()
            if _serve_mode() == "btauth":
                # Bittensor v11: publish the endpoint on-chain, run application-owned btauth HTTP,
                # and keep :8092 for readiness + explicit testlab one-shot scenarios.
                from subnet.miner.axon_serve import serve_axon_forever

                serve_axon_forever(hotkey_name=hotkey_name, netuid=args.netuid, port=args.port)
                return 0
            if _serve_mode() == "http":
                from subnet.miner.serve import serve as serve_miner

                serve_miner(hotkey_name=hotkey_name, port=args.port)
                return 0
            raise ValueError(f"unknown AURELIUS_TRANSPORT {_serve_mode()!r}")
        except KeyboardInterrupt:
            print("[aurelius-miner] serve interrupted", file=sys.stderr, flush=True)
            return 0
        except (ValueError, NotImplementedError, RuntimeError) as e:
            # Expected operator-fixable failures (posture, wallet, guarded publication) surface
            # as one clean line, not a traceback; anything else still crashes loudly.
            print(
                f"[aurelius-miner] cannot serve: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            return 2

    if args.command == "observe":
        from subnet.miner.commands import run_observe

        return run_observe(
            args.central,
            netuid=args.netuid,
            journal=args.journal,
            limit=args.limit,
            validators=args.validator,
        )

    if args.command == "validate":
        from subnet.miner.commands import run_validate

        return run_validate()

    if args.command == "balance":
        from subnet.miner.commands import run_balance
        from subnet.miner.serve import minter_from_env

        hotkey_ss58 = (
            args.hotkey
            or minter_from_env(os.environ.get("MINER_HOTKEY", "miner-hotkey")).hotkey_ss58
        )
        return run_balance(hotkey_ss58, args.task_type)

    if args.command == "ready":
        from subnet.miner.commands import run_readiness

        return run_readiness(args.task_type, args.fanout_window)

    if args.command == "pool":
        import json
        from subnet.miner.staging import StagingPool, item_id

        pool = StagingPool(args.pool)
        if args.action != "list":
            if not args.item_id:
                parser.error("enable/disable requires item_id")
            pool.set_enabled(args.item_id, args.action == "enable")
        states = pool.enabled()
        print(
            json.dumps(
                [
                    {
                        "item_id": item_id(item),
                        "enabled": states.get(item_id(item), True),
                        "candidate": item,
                    }
                    for item in pool.items()
                ],
                sort_keys=True,
            )
        )
        return 0

    if args.command == "deposit":
        from subnet.miner.commands import run_deposit

        coldkey_name = os.environ.get("MINER_COLDKEY", "miner-coldkey")
        unlocked = args.unlock or os.environ.get("AURELIUS_MINER_UNLOCK") == "1"
        return run_deposit(coldkey_name, unlocked=unlocked)

    return 2


if __name__ == "__main__":
    sys.exit(main())
