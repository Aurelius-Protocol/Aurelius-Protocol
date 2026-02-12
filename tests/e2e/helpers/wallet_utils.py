"""Wallet utilities for E2E tests."""

import json
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import bittensor as bt


# Test wallet tracking for cleanup
_created_test_wallets: list[str] = []


def generate_unique_wallet_name(prefix: str = "e2e_test") -> str:
    """
    Generate a unique wallet name for testing.

    Args:
        prefix: Prefix for the wallet name

    Returns:
        Unique wallet name like "e2e_test_a1b2c3d4"
    """
    suffix = secrets.token_hex(4)
    return f"{prefix}_{suffix}"


def create_test_wallet(
    name: str | None = None,
    hotkey: str = "default",
    overwrite: bool = False,
) -> bt.Wallet:
    """
    Create a new test wallet programmatically.

    Uses bittensor Python API directly (no btcli subprocess needed).

    Args:
        name: Wallet name (auto-generated if None)
        hotkey: Hotkey name
        overwrite: Whether to overwrite existing wallet

    Returns:
        Created wallet instance

    Raises:
        RuntimeError: If wallet creation fails
    """
    if name is None:
        name = generate_unique_wallet_name()

    wallet_path = Path.home() / ".bittensor" / "wallets" / name

    # Check if wallet already exists
    if wallet_path.exists() and not overwrite:
        raise RuntimeError(f"Wallet already exists: {wallet_path}")

    # Clean up existing wallet if overwriting
    if wallet_path.exists() and overwrite:
        cleanup_test_wallet(name)

    try:
        # Create wallet using bittensor Python API
        wallet = bt.Wallet(name=name, hotkey=hotkey)
        wallet.create_if_non_existent(
            coldkey_use_password=False,
            hotkey_use_password=False,
        )

        # Track for cleanup
        _created_test_wallets.append(name)

        return wallet

    except Exception as e:
        # Clean up partial wallet on failure
        cleanup_test_wallet(name)
        raise RuntimeError(f"Failed to create wallet: {e}")


def cleanup_test_wallet(name: str) -> bool:
    """
    Remove a test wallet directory.

    Args:
        name: Wallet name to remove

    Returns:
        True if cleanup succeeded, False otherwise
    """
    wallet_path = Path.home() / ".bittensor" / "wallets" / name

    if not wallet_path.exists():
        return True

    try:
        shutil.rmtree(wallet_path)
        if name in _created_test_wallets:
            _created_test_wallets.remove(name)
        return True
    except Exception:
        return False


def cleanup_all_test_wallets() -> int:
    """
    Clean up all test wallets created during this session.

    Returns:
        Number of wallets cleaned up
    """
    cleaned = 0
    for name in list(_created_test_wallets):
        if cleanup_test_wallet(name):
            cleaned += 1
    return cleaned


def get_test_wallet_count() -> int:
    """
    Get the number of test wallets created in this session.

    Returns:
        Count of tracked test wallets
    """
    return len(_created_test_wallets)


def get_validator_wallet(
    name: str = "validator",
    hotkey: str = "default",
) -> bt.Wallet:
    """
    Load validator wallet for testing.

    Args:
        name: Wallet name
        hotkey: Hotkey name

    Returns:
        Loaded wallet instance

    Raises:
        FileNotFoundError: If wallet doesn't exist
    """
    wallet_path = Path.home() / ".bittensor" / "wallets" / name
    if not wallet_path.exists():
        raise FileNotFoundError(f"Wallet not found: {wallet_path}")

    return bt.Wallet(name=name, hotkey=hotkey)


def verify_wallet_exists(
    name: str = "validator",
    hotkey: str = "default",
) -> dict[str, Any]:
    """
    Verify wallet exists and has required keys.

    Returns:
        Dict with wallet info: {exists, coldkey_exists, hotkey_exists, path}
    """
    wallet_path = Path.home() / ".bittensor" / "wallets" / name
    coldkey_path = wallet_path / "coldkey"
    hotkey_path = wallet_path / "hotkeys" / hotkey

    return {
        "exists": wallet_path.exists(),
        "coldkey_exists": coldkey_path.exists(),
        "hotkey_exists": hotkey_path.exists(),
        "path": str(wallet_path),
    }


def sign_message(wallet: bt.Wallet, message: str) -> str:
    """
    Sign a message with the wallet's hotkey.

    Args:
        wallet: Bittensor wallet
        message: Message to sign

    Returns:
        Hex-encoded signature
    """
    signature = wallet.hotkey.sign(message.encode())
    return signature.hex()


def create_signed_headers(
    wallet: bt.Wallet,
    body: dict | None = None,
) -> dict[str, str]:
    """
    Create authentication headers with SR25519 signature.

    The signature covers: timestamp + sorted JSON body

    Args:
        wallet: Bittensor wallet
        body: Request body to sign (optional)

    Returns:
        Headers dict with X-Timestamp, X-Hotkey, X-Signature
    """
    timestamp = str(int(time.time() * 1000))
    hotkey = wallet.hotkey.ss58_address

    # Create message to sign
    if body:
        body_str = json.dumps(body, sort_keys=True, separators=(",", ":"))
    else:
        body_str = ""

    message = f"{timestamp}{body_str}"
    signature = sign_message(wallet, message)

    return {
        "X-Timestamp": timestamp,
        "X-Hotkey": hotkey,
        "X-Signature": signature,
        "Content-Type": "application/json",
    }


def get_hotkey_address(
    name: str = "validator",
    hotkey: str = "default",
) -> str | None:
    """
    Get the SS58 address of a hotkey without loading the full wallet.

    Returns:
        SS58 address string or None if not found
    """
    hotkey_path = Path.home() / ".bittensor" / "wallets" / name / "hotkeys" / hotkey
    if not hotkey_path.exists():
        return None

    with open(hotkey_path) as f:
        data = f.read()

    try:
        parsed = json.loads(data)
        return parsed.get("ss58Address")
    except json.JSONDecodeError:
        # Fallback: try to find SS58 address in raw content
        import re
        match = re.search(r"5[A-Za-z0-9]{47}", data)
        return match.group(0) if match else None
