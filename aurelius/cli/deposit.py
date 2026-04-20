"""CLI tool for work-token deposit address verification and balance queries."""

import argparse
import json
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen


def _api_get(api_url: str, path: str) -> dict:
    req = Request(f"{api_url}{path}")
    req.add_header("Accept", "application/json")
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _verify_multisig_on_chain(address: str, expected_threshold: int, expected_signatories: list[str], network: str):
    """Query the chain to verify multisig configuration matches API claims.

    Returns (verified: bool, reason: str).
    """
    try:
        import bittensor as bt

        subtensor = bt.Subtensor(network=network)
        result = subtensor.substrate.query("Multisig", "Multisigs", [address])

        if result is None or result.value is None:
            return False, "Address is not a multisig on-chain"

        multisig_info = result.value
        on_chain_signatories = list(multisig_info.get("signatories", []))
        on_chain_threshold = multisig_info.get("threshold", 0)

        if on_chain_threshold != expected_threshold:
            return False, f"Threshold mismatch: API says {expected_threshold}, chain says {on_chain_threshold}"

        expected_set = set(expected_signatories)
        chain_set = set(on_chain_signatories)
        if expected_set != chain_set:
            missing = expected_set - chain_set
            extra = chain_set - expected_set
            parts = []
            if missing:
                parts.append(f"missing from chain: {missing}")
            if extra:
                parts.append(f"unexpected on chain: {extra}")
            return False, f"Signatory mismatch: {'; '.join(parts)}"

        return True, "On-chain multisig matches API"
    except ImportError:
        return False, "bittensor package not installed — cannot verify on-chain"
    except Exception as e:
        return False, f"On-chain verification failed: {e}"


def cmd_verify_address(args):
    """Fetch designated address from API and verify on-chain multisig configuration."""
    try:
        data = _api_get(args.api_url, "/work-token/designated-address")
        print(f"Designated deposit address: {data['address']}")
        print(f"Multisig threshold:         {data['multisig_threshold']}-of-{len(data['signatories'])}")
        print("Signatories:")
        for s in data["signatories"]:
            print(f"  - {s}")
        print()

        # On-chain verification
        print("Verifying on-chain multisig configuration...")
        verified, reason = _verify_multisig_on_chain(
            data["address"],
            data["multisig_threshold"],
            data["signatories"],
            args.network,
        )
        if verified:
            print(f"  ✓ {reason}")
        else:
            print(f"  ✗ VERIFICATION FAILED: {reason}", file=sys.stderr)
            print()
            print("DO NOT deposit to this address until the mismatch is resolved.", file=sys.stderr)
            sys.exit(1)
    except URLError as e:
        print(f"Error: Could not reach API at {args.api_url}: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_balance(args):
    """Query work-token balance for a hotkey."""
    try:
        data = _api_get(args.api_url, f"/work-token/balance/{args.hotkey}")
        print(f"Hotkey:  {data['hotkey']}")
        print(f"Balance: {data['balance']}")
        print(f"Active:  {'yes' if data['has_balance'] else 'no'}")
    except URLError as e:
        print(f"Error: Could not reach API at {args.api_url}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    from aurelius.config import ENVIRONMENT, Config

    parser = argparse.ArgumentParser(prog="aurelius-deposit", description="Aurelius work-token deposit tools")
    parser.add_argument(
        "--api-url", default=Config.CENTRAL_API_URL, help="Central API base URL (default: $CENTRAL_API_URL)"
    )
    default_network = "test" if ENVIRONMENT in ("local", "testnet") else "finney"
    parser.add_argument(
        "--network",
        default=default_network,
        choices=["finney", "test"],
        help=f"Bittensor network (default: {default_network}, based on ENVIRONMENT={ENVIRONMENT})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify-address", help="Show and verify the designated deposit address")

    balance_parser = subparsers.add_parser("balance", help="Query work-token balance")
    balance_parser.add_argument("--hotkey", required=True, help="Miner hotkey (SS58 address)")

    args = parser.parse_args()

    if args.command == "verify-address":
        cmd_verify_address(args)
    elif args.command == "balance":
        cmd_balance(args)


if __name__ == "__main__":
    main()
