"""Miner CLI entry point.

The commands are: `balance`, `validate`, `serve` (plan §15.1), and the ones added since.
- `serve`: the long-running miner. It publishes its endpoint on chain and answers authenticated
  `/v1/query` requests, each with a fresh real-sr25519-signed work_id, signing with the hotkey of
  the mounted wallet. Readiness is exposed by the companion health server on
  `AURELIUS_MINER_HTTP_PORT`.
- `validate`: check staged items against the PUBLISHED config_schema (no lane import).
- `balance`: a public affordability read + the deposit address (no session, no coldkey).
- `fund`: separate coldkey funding command, exact native-alpha transfer with durable intent;
  `--method gamma` selects the gamma work-token path (check/prepare/execute/status, §14-G7).
- `observe`: lane-agnostic refresh of charge/refund + validator-progress facts into the local
  journal (the dashboard's data source; absence of evidence stays unknown).

`serve` and `fund` first require agreement to the current incentive terms (`subnet.miner.terms`):
an interactive Y/N, `AURELIUS_ACCEPT_INCENTIVES=yes`, or `--accept-incentives`.

The miner never authenticates a session and never calls the charge path (invariant 1); the
`charge-path-single-import-site` contract now enumerates `subnet.miner.*` as a forbidden source.
"""

from __future__ import annotations

import argparse
import os
import sys


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

    fund = sub.add_parser(
        "fund",
        help="preview or submit one native-alpha WT purchase on subnet 37; "
        "--method gamma selects the gamma funding path",
    )
    # ``--method legacy`` (the default) is the deployed native-alpha transfer, unchanged. Its
    # four mandatory arguments are enforced below with argparse's own message so the legacy
    # behavior stays identical while ``--method gamma`` read-only actions need no wallet.
    fund.add_argument(
        "--method",
        choices=("legacy", "gamma"),
        default="legacy",
        help="funding method; gamma is check-only unless an action flag is given",
    )
    fund.add_argument("--alpha", help="exact alpha amount, at most nine decimal places")
    fund.add_argument("--fee-cap-tao", default="0.01")
    fund.add_argument("--wallet-name", help="separate coldkey funding wallet")
    fund.add_argument("--wallet-path")
    fund.add_argument("--hotkey-name", default="default")
    fund.add_argument(
        "--intent-dir",
        help="durable directory for this one purchase; reuse to observe it",
    )
    fund.add_argument(
        "--execute", action="store_true", help="sign and submit; default is check-only"
    )
    fund.add_argument(
        "--accept-incentives", action="store_true", help="agree to INCENTIVES.md non-interactively"
    )
    fund.add_argument(
        "--endpoint",
        default=os.environ.get(
            "SUBSTRATE_URL", "wss://bittensor-finney.api.onfinality.io/public-ws"
        ),
    )
    fund.add_argument(
        "--central", default=os.environ.get("CENTRAL_API_URL", "https://central.aureliusaligned.ai")
    )
    gamma = fund.add_argument_group("gamma method (--method gamma)")
    gamma.add_argument("--check", action="store_true", help="read-only readiness (the default)")
    gamma.add_argument(
        "--prepare", action="store_true", help="obtain + verify a quote and sign one spend"
    )
    gamma.add_argument("--status", action="store_true", help="read-only observation + receipt")
    gamma.add_argument("--pack", help="approved pack id from the signed purchase policy")
    gamma.add_argument("--gamma-hotkey", help="ss58 of the separate gamma API hotkey (deposit key)")
    gamma.add_argument("--intent", help="expected intent id (consistency check on prepare)")
    gamma.add_argument("--era-period-blocks", type=int, default=64)
    gamma.add_argument(
        "--mint-setup", action="store_true", help="explicit mint/setup helper (never automatic)"
    )
    gamma.add_argument("--mint-tao", help="exact TAO input for the mint")
    gamma.add_argument("--mint-min-gamma-out", help="minimum gamma atoms out (slippage bound)")
    gamma.add_argument("--mint-daily-cap", help="optional daily spend cap in gamma atoms")
    gamma.add_argument(
        "--confirm-mint-burns-tao",
        action="store_true",
        help="separate confirmation required with --mint-setup --execute",
    )

    serve = sub.add_parser("serve", help="serve /v1/query (mint a fresh work_id per request)")
    serve.add_argument("--netuid", type=int, default=int(os.environ.get("NETUID", "2")))
    serve.add_argument("--port", type=int, default=8091)
    serve.add_argument(
        "--accept-incentives", action="store_true", help="agree to INCENTIVES.md non-interactively"
    )

    args = parser.parse_args(argv)

    if args.command == "fund":
        import json
        from subnet.miner.funding import FundingRefused, run
        from subnet.miner.read_client import ReadUnavailable
        from subnet.miner.terms import TermsNotAccepted, ensure_incentives_accepted

        if args.method == "gamma":
            # Arguments are checked first, then the incentive terms, then the wallet is touched:
            # the same order as the legacy method below.
            actions = [
                name
                for name, flag in (
                    ("check", args.check),
                    ("prepare", args.prepare),
                    ("status", args.status),
                    ("mint-setup", args.mint_setup),
                )
                if flag
            ]
            if args.execute and not any(actions):
                actions = ["execute"]
            if len(actions) > 1:
                fund.error("--check/--prepare/--status/--mint-setup are mutually exclusive")
            action = actions[0] if actions else "check"
            if action != "check" and not args.intent_dir:
                fund.error(action + " requires --intent-dir")
            if args.execute and action in {"check", "status"}:
                fund.error(action + " never executes; --execute is for prepare-then-submit")
            # Bound before the try: the except clause names GammaRefused, and the terms check
            # inside the try raises before a later import would have bound it.
            from subnet.miner.gamma_funding import GammaRefused, run as gamma_run

            try:
                ensure_incentives_accepted(accept_flag=args.accept_incentives)
                result = gamma_run(
                    action=action,
                    base_url=args.central,
                    endpoint=args.endpoint,
                    directory=args.intent_dir,
                    pack=args.pack,
                    gamma_hotkey=args.gamma_hotkey,
                    intent=args.intent,
                    fee_cap=args.fee_cap_tao,
                    era_period=args.era_period_blocks,
                    execute=args.execute,
                    wallet_path=args.wallet_path,
                    wallet_name=args.wallet_name,
                    mint_tao=args.mint_tao,
                    mint_min_out=args.mint_min_gamma_out,
                    mint_cap=args.mint_daily_cap,
                    confirm_mint=args.confirm_mint_burns_tao,
                )
            except (GammaRefused, FundingRefused, ReadUnavailable, TermsNotAccepted) as exc:
                print(json.dumps({"status": "refused", "reason": str(exc)}))
                return 2
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0

        missing = [
            flag
            for flag, value in (
                ("--alpha", args.alpha),
                ("--wallet-name", args.wallet_name),
                ("--wallet-path", args.wallet_path),
                ("--intent-dir", args.intent_dir),
            )
            if value is None
        ]
        if missing:
            fund.error("the following arguments are required: " + ", ".join(missing))
        gamma_only = [
            "--" + flag
            for flag, value in (
                ("check", args.check),
                ("prepare", args.prepare),
                ("status", args.status),
                ("mint-setup", args.mint_setup),
                ("confirm-mint-burns-tao", args.confirm_mint_burns_tao),
                ("pack", args.pack),
                ("gamma-hotkey", args.gamma_hotkey),
                ("intent", args.intent),
                ("mint-tao", args.mint_tao),
                ("mint-min-gamma-out", args.mint_min_gamma_out),
                ("mint-daily-cap", args.mint_daily_cap),
                ("era-period-blocks", None if args.era_period_blocks == 64 else "set"),
            )
            if value
        ]
        if gamma_only:
            fund.error(", ".join(gamma_only) + " require --method gamma")
        try:
            ensure_incentives_accepted(accept_flag=args.accept_incentives)
            result = run(
                base_url=args.central,
                endpoint=args.endpoint,
                wallet_path=args.wallet_path,
                wallet_name=args.wallet_name,
                hotkey_name=args.hotkey_name,
                alpha=args.alpha,
                fee_cap=args.fee_cap_tao,
                execute=args.execute,
                directory=args.intent_dir,
            )
        except (FundingRefused, ReadUnavailable, TermsNotAccepted) as exc:
            print(json.dumps({"status": "refused", "executed": False, "reason": str(exc)}))
            return 2
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "economics":
        import json
        from subnet.miner.economics_client import read_economics
        from subnet.miner.serve import minter_from_env
        from subnet.miner.trust import config_signing_pubkey

        try:
            public_key = config_signing_pubkey()
        except ValueError as exc:
            parser.error(str(exc))
        minter = minter_from_env()
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
        from subnet.miner.serve import validate_miner_startup_posture
        from subnet.miner.terms import ensure_incentives_accepted

        try:
            ensure_incentives_accepted(accept_flag=args.accept_incentives)
            validate_miner_startup_posture()
            # Bittensor v11: publish the endpoint on-chain, run application-owned btauth HTTP,
            # and keep the health port for readiness.
            from subnet.miner.axon_serve import serve_axon_forever

            serve_axon_forever(netuid=args.netuid, port=args.port)
            return 0
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

        hotkey_ss58 = args.hotkey or minter_from_env().hotkey_ss58
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

    return 2


if __name__ == "__main__":
    sys.exit(main())
