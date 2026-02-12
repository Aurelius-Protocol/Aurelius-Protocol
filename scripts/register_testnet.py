#!/usr/bin/env python3
"""
Register validator on Bittensor testnet subnet 290.

Usage:
  python scripts/register_testnet.py --check     # Check registration status
  python scripts/register_testnet.py --register  # Register if not registered
  python scripts/register_testnet.py --info      # Show wallet info

Prerequisites:
  - Wallet at ~/.bittensor/wallets/validator/
  - TAO funded on testnet
  - Subnet 290 has available slots

The script is idempotent - safe to run multiple times.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Testnet configuration
NETWORK = "test"
NETUID = 290
MIN_STAKE = 100  # TAO
WALLET_NAME = "validator"
HOTKEY_NAME = "default"


def run_btcli(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Run btcli command and return (exit_code, stdout, stderr)."""
    cmd = ["btcli"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -2, "", "btcli not found. Install with: pip install bittensor"


def get_wallet_path() -> Path:
    """Get path to validator wallet."""
    return Path.home() / ".bittensor" / "wallets" / WALLET_NAME


def check_wallet_exists(wallet_name: str = WALLET_NAME, hotkey_name: str = HOTKEY_NAME) -> bool:
    """Check if wallet directory exists with required keys."""
    wallet_path = Path.home() / ".bittensor" / "wallets" / wallet_name
    coldkey_path = wallet_path / "coldkey"
    hotkey_path = wallet_path / "hotkeys" / hotkey_name

    if not wallet_path.exists():
        print(f"Wallet directory not found: {wallet_path}")
        return False

    if not coldkey_path.exists():
        print(f"Coldkey not found: {coldkey_path}")
        return False

    if not hotkey_path.exists():
        print(f"Hotkey not found: {hotkey_path}")
        return False

    print(f"Wallet found: {wallet_path}")
    return True


def get_wallet_info() -> dict | None:
    """Get wallet balance and registration info."""
    code, stdout, stderr = run_btcli([
        "wallet", "overview",
        "--wallet.name", WALLET_NAME,
        "--network", NETWORK,
        "--subtensor.network", NETWORK,
        "--no_prompt",
    ])

    if code != 0:
        print(f"Failed to get wallet info: {stderr}")
        return None

    print("Wallet Overview:")
    print(stdout)
    return {"raw_output": stdout}


def check_registration() -> dict:
    """
    Check if validator is registered on subnet 290.
    Returns: {registered: bool, uid: int | None, hotkey: str | None}
    """
    print(f"Checking registration on subnet {NETUID}...")

    code, stdout, stderr = run_btcli([
        "subnet", "metagraph",
        "--netuid", str(NETUID),
        "--network", NETWORK,
        "--subtensor.network", NETWORK,
    ], timeout=180)

    if code != 0:
        print(f"Failed to query metagraph: {stderr}")
        return {"registered": False, "uid": None, "hotkey": None, "error": stderr}

    # Parse metagraph output for our hotkey
    # Load our hotkey address to search for it
    hotkey_path = get_wallet_path() / "hotkeys" / HOTKEY_NAME
    if hotkey_path.exists():
        with open(hotkey_path) as f:
            hotkey_data = f.read()
            # Extract SS58 address from hotkey file
            # The format varies, but typically contains the address
            import json
            try:
                data = json.loads(hotkey_data)
                our_hotkey = data.get("ss58Address", "")
            except json.JSONDecodeError:
                # Fallback: search for SS58 format address
                import re
                match = re.search(r"5[A-Za-z0-9]{47}", hotkey_data)
                our_hotkey = match.group(0) if match else ""
    else:
        our_hotkey = ""

    if our_hotkey and our_hotkey in stdout:
        # Find UID from the metagraph output
        lines = stdout.split("\n")
        for line in lines:
            if our_hotkey in line:
                # Parse UID from line (format varies)
                parts = line.split()
                if parts:
                    try:
                        uid = int(parts[0])
                        print(f"Validator registered with UID: {uid}")
                        return {"registered": True, "uid": uid, "hotkey": our_hotkey}
                    except ValueError:
                        pass

        print(f"Hotkey found in metagraph but couldn't parse UID")
        return {"registered": True, "uid": None, "hotkey": our_hotkey}

    print("Validator not registered on subnet")
    return {"registered": False, "uid": None, "hotkey": our_hotkey}


def check_balance() -> float | None:
    """Check wallet balance on testnet."""
    code, stdout, stderr = run_btcli([
        "wallet", "balance",
        "--wallet.name", WALLET_NAME,
        "--network", NETWORK,
        "--subtensor.network", NETWORK,
        "--no_prompt",
    ])

    if code != 0:
        print(f"Failed to check balance: {stderr}")
        return None

    # Parse balance from output
    import re
    # Look for TAO amount in format like "Balance: 123.456 TAO"
    match = re.search(r"(\d+\.?\d*)\s*(?:τ|TAO)", stdout)
    if match:
        balance = float(match.group(1))
        print(f"Wallet balance: {balance} TAO")
        return balance

    print("Could not parse balance from output:")
    print(stdout)
    return None


def register_validator() -> bool:
    """
    Register validator on testnet subnet 290.
    Returns True if successful or already registered.
    """
    # Check if already registered
    status = check_registration()
    if status["registered"]:
        print(f"Already registered with UID {status['uid']}")
        return True

    # Check balance
    balance = check_balance()
    if balance is None:
        print("Could not verify balance")
        return False

    if balance < MIN_STAKE:
        print(f"Insufficient balance: {balance} TAO (need {MIN_STAKE} TAO)")
        return False

    print(f"Registering on subnet {NETUID}...")
    print("This will prompt for your wallet password.")

    # Run registration
    code, stdout, stderr = run_btcli([
        "subnet", "register",
        "--netuid", str(NETUID),
        "--wallet.name", WALLET_NAME,
        "--wallet.hotkey", HOTKEY_NAME,
        "--network", NETWORK,
        "--subtensor.network", NETWORK,
    ], timeout=300)

    if code != 0:
        print(f"Registration failed: {stderr}")
        print(stdout)
        return False

    print("Registration submitted!")
    print(stdout)

    # Wait for confirmation
    return wait_for_inclusion()


def wait_for_inclusion(max_attempts: int = 10, delay: int = 30) -> bool:
    """
    Poll until registration is confirmed on-chain.
    """
    print(f"Waiting for on-chain confirmation (up to {max_attempts * delay}s)...")

    for attempt in range(max_attempts):
        time.sleep(delay)
        status = check_registration()
        if status["registered"]:
            print(f"Registration confirmed! UID: {status['uid']}")
            return True
        print(f"Attempt {attempt + 1}/{max_attempts}: Not yet confirmed...")

    print("Registration not confirmed within timeout")
    return False


def verify_stake() -> bool:
    """Verify minimum stake requirement is met."""
    balance = check_balance()
    if balance is None:
        return False

    if balance < MIN_STAKE:
        print(f"Warning: Balance {balance} TAO is below minimum stake {MIN_STAKE} TAO")
        return False

    print(f"Stake requirement met: {balance} >= {MIN_STAKE} TAO")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Register validator on Bittensor testnet subnet 290"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check current registration status",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register validator if not already registered",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show wallet info and balance",
    )
    parser.add_argument(
        "--wallet",
        default=WALLET_NAME,
        help=f"Wallet name (default: {WALLET_NAME})",
    )
    parser.add_argument(
        "--hotkey",
        default=HOTKEY_NAME,
        help=f"Hotkey name (default: {HOTKEY_NAME})",
    )

    args = parser.parse_args()

    if not any([args.check, args.register, args.info]):
        parser.print_help()
        sys.exit(0)

    # Use command-line args for wallet/hotkey names
    wallet_name = args.wallet
    hotkey_name = args.hotkey

    # Always check wallet exists first
    if not check_wallet_exists(wallet_name, hotkey_name):
        print("\nCreate a wallet first:")
        print(f"  btcli wallet new_coldkey --wallet.name {wallet_name}")
        print(f"  btcli wallet new_hotkey --wallet.name {wallet_name} --wallet.hotkey {hotkey_name}")
        sys.exit(1)

    if args.info:
        get_wallet_info()
        check_balance()

    if args.check:
        status = check_registration()
        if status["registered"]:
            print(f"\nValidator is registered on subnet {NETUID}")
            print(f"  UID: {status['uid']}")
            print(f"  Hotkey: {status['hotkey']}")
        else:
            print(f"\nValidator is NOT registered on subnet {NETUID}")
        verify_stake()

    if args.register:
        if register_validator():
            print("\nRegistration successful!")
            sys.exit(0)
        else:
            print("\nRegistration failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
