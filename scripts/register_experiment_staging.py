#!/usr/bin/env python3
"""Register validator and create experiment on the staging central API.

Usage:
  python scripts/register_experiment_staging.py

Steps:
  1. Load the validator wallet (hotkey signs all requests)
  2. POST /api/validators/register — register this validator (idempotent)
  3. POST /api/experiments — create "moral-reasoning" experiment (idempotent)

Prerequisites:
  - Validator wallet at ~/.bittensor/wallets/validator/
  - Wallet hotkey must be unlocked (no password) or password provided interactively
  - Validator should be registered on-chain (subnet 290) so that UID is known
"""

import argparse
import json
import sys
import time

import bittensor as bt
import requests

# Staging API base URL
STAGING_API = "https://aurelius-data-collector-api-staging.up.railway.app"

# Defaults
WALLET_NAME = "validator"
HOTKEY_NAME = "default"
VALIDATOR_UID = 0  # UID on testnet subnet 290


def load_wallet(wallet_name: str, hotkey_name: str) -> bt.Wallet:
    """Load and return bittensor wallet."""
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    # Access hotkey and coldkeypub to verify they exist
    _ = wallet.hotkey.ss58_address
    _ = wallet.coldkeypub.ss58_address
    return wallet


def register_validator(
    wallet: bt.Wallet, uid: int, api_base: str
) -> bool:
    """Register validator on the central API.

    Signed message: "aurelius-register:{timestamp}:{hotkey}:{uid}:{coldkey}"
    """
    hotkey = wallet.hotkey.ss58_address
    coldkey = wallet.coldkeypub.ss58_address
    # validators.ts passes timestamp directly to verifyTimestamp() which
    # compares against Date.now() (milliseconds), so we must send ms here.
    timestamp = int(time.time() * 1000)

    message = f"aurelius-register:{timestamp}:{hotkey}:{uid}:{coldkey}"
    signature = wallet.hotkey.sign(message.encode()).hex()

    body = {
        "uid": uid,
        "hotkey": hotkey,
        "coldkey": coldkey,
        "signature": signature,
        "timestamp": timestamp,
    }

    url = f"{api_base}/api/validators/register"
    print(f"POST {url}")
    print(f"  hotkey:  {hotkey}")
    print(f"  coldkey: {coldkey}")
    print(f"  uid:     {uid}")

    resp = requests.post(url, json=body, timeout=30)
    data = resp.json() if resp.text else {}

    if resp.status_code == 200:
        print(f"  => Validator registered: {json.dumps(data, indent=2)}")
        return True
    elif resp.status_code == 409:
        print(f"  => Validator already exists (409) — OK")
        return True
    else:
        print(f"  => Failed ({resp.status_code}): {json.dumps(data, indent=2)}")
        return False


def create_experiment(
    wallet: bt.Wallet,
    experiment_id: str,
    description: str,
    is_public: bool,
    api_base: str,
) -> bool:
    """Create an experiment on the central API.

    Auth: Authorization: Signature {sig}
    Signed message: "aurelius-auth:{timestamp}:{hotkey}"
    Body must include hotkey and timestamp for auth middleware.
    """
    hotkey = wallet.hotkey.ss58_address
    timestamp = int(time.time())

    message = f"aurelius-auth:{timestamp}:{hotkey}"
    signature = wallet.hotkey.sign(message.encode()).hex()

    body = {
        "hotkey": hotkey,
        "timestamp": timestamp,
        "experiment_id": experiment_id,
        "description": description,
        "is_public": is_public,
    }

    url = f"{api_base}/api/experiments"
    print(f"\nPOST {url}")
    print(f"  experiment_id: {experiment_id}")
    print(f"  description:   {description}")
    print(f"  is_public:     {is_public}")

    resp = requests.post(
        url,
        json=body,
        headers={"Authorization": f"Signature {signature}"},
        timeout=30,
    )
    data = resp.json() if resp.text else {}

    if resp.status_code == 201:
        print(f"  => Experiment created: {json.dumps(data, indent=2)}")
        return True
    elif resp.status_code == 409:
        print(f"  => Experiment already exists (409) — OK")
        return True
    else:
        print(f"  => Failed ({resp.status_code}): {json.dumps(data, indent=2)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Register validator + create experiment on staging API"
    )
    parser.add_argument(
        "--wallet", default=WALLET_NAME, help=f"Wallet name (default: {WALLET_NAME})"
    )
    parser.add_argument(
        "--hotkey", default=HOTKEY_NAME, help=f"Hotkey name (default: {HOTKEY_NAME})"
    )
    parser.add_argument(
        "--uid", type=int, default=VALIDATOR_UID, help=f"Validator UID (default: {VALIDATOR_UID})"
    )
    parser.add_argument(
        "--api", default=STAGING_API, help=f"API base URL (default: {STAGING_API})"
    )
    args = parser.parse_args()

    # 1. Load wallet
    print(f"Loading wallet: {args.wallet}/{args.hotkey}")
    try:
        wallet = load_wallet(args.wallet, args.hotkey)
    except Exception as e:
        print(f"Failed to load wallet: {e}")
        sys.exit(1)

    print(f"  hotkey:  {wallet.hotkey.ss58_address}")
    print(f"  coldkey: {wallet.coldkeypub.ss58_address}")
    print()

    # 2. Register validator
    print("=== Step 1: Register Validator ===")
    if not register_validator(wallet, args.uid, args.api):
        print("\nValidator registration failed — cannot create experiment")
        sys.exit(1)

    # 3. Create moral-reasoning experiment
    print("\n=== Step 2: Create Experiment ===")
    ok = create_experiment(
        wallet,
        experiment_id="moral-reasoning",
        description="Moral reasoning and ethical decision-making evaluation (MoReBench)",
        is_public=True,
        api_base=args.api,
    )

    if not ok:
        print("\nExperiment creation failed")
        sys.exit(1)

    print("\nDone! Validator registered and experiment created.")


if __name__ == "__main__":
    main()
