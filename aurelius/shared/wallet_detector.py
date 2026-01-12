"""Wallet auto-detection for turnkey validator setup."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectedWallet:
    """Represents a detected wallet with its hotkeys."""

    name: str
    path: Path
    hotkeys: list[str]


@dataclass
class WalletDetectionResult:
    """Result of wallet detection attempt."""

    wallet_name: str | None
    hotkey: str | None
    wallets: list[DetectedWallet]
    error: str | None


def get_wallets_path() -> Path:
    """
    Get the Bittensor wallets directory path.

    Checks BITTENSOR_WALLET_PATH env var first (for Docker/custom setups),
    then falls back to ~/.bittensor/wallets.

    Returns:
        Path to the wallets directory
    """
    custom_path = os.getenv("BITTENSOR_WALLET_PATH")
    if custom_path:
        return Path(custom_path).expanduser()
    return Path.home() / ".bittensor" / "wallets"


def list_wallets(wallets_path: Path) -> tuple[list[DetectedWallet], str | None]:
    """
    List all wallets and their hotkeys from the wallets directory.

    A valid wallet has:
    - A directory under wallets_path
    - A 'coldkey' file in that directory
    - A 'hotkeys' subdirectory with at least one hotkey file

    Args:
        wallets_path: Path to the wallets directory

    Returns:
        Tuple of (list of detected wallets, permission_error message or None)
    """
    wallets = []
    permission_error = None

    if not wallets_path.exists():
        return wallets, None

    try:
        wallet_dirs = list(wallets_path.iterdir())
    except PermissionError:
        return [], f"Permission denied reading wallet directory: {wallets_path}"

    for wallet_dir in wallet_dirs:
        if not wallet_dir.is_dir():
            continue

        # Check for coldkey (required for valid wallet)
        coldkey_path = wallet_dir / "coldkey"
        try:
            if not coldkey_path.exists():
                continue
            # Try to read the coldkey to verify permissions
            coldkey_path.stat()
        except PermissionError:
            permission_error = f"Permission denied reading coldkey: {coldkey_path}"
            continue

        # Find hotkeys
        hotkeys_dir = wallet_dir / "hotkeys"
        hotkeys = []
        if hotkeys_dir.exists():
            try:
                for hotkey_file in hotkeys_dir.iterdir():
                    # Skip .txt files (public key metadata)
                    if hotkey_file.suffix == ".txt":
                        continue
                    if hotkey_file.is_file():
                        # Verify we can read the hotkey
                        try:
                            hotkey_file.stat()
                            hotkeys.append(hotkey_file.name)
                        except PermissionError:
                            permission_error = f"Permission denied reading hotkey: {hotkey_file}"
            except PermissionError:
                permission_error = f"Permission denied reading hotkeys directory: {hotkeys_dir}"
                continue

        # Only include wallets with at least one hotkey
        if hotkeys:
            wallets.append(
                DetectedWallet(
                    name=wallet_dir.name,
                    path=wallet_dir,
                    hotkeys=sorted(hotkeys),
                )
            )

    return sorted(wallets, key=lambda w: w.name), permission_error


def detect_wallet(role: str = "validator") -> WalletDetectionResult:
    """
    Detect wallet to use based on available wallets.

    Auto-detection logic:
    - If exactly one wallet with exactly one hotkey exists → auto-select it
    - If multiple wallets or hotkeys exist → return error with list
    - If no wallets exist → return error with instructions
    - If permission errors occur → return error with fix instructions

    Args:
        role: "validator" or "miner" - affects error message wording

    Returns:
        WalletDetectionResult with detection outcome
    """
    wallets_path = get_wallets_path()
    wallets, permission_error = list_wallets(wallets_path)

    # Case: Permission error reading wallet files
    if permission_error and not wallets:
        return WalletDetectionResult(
            wallet_name=None,
            hotkey=None,
            wallets=[],
            error=(
                f"PERMISSION ERROR: Cannot read wallet files\n\n"
                f"{permission_error}\n\n"
                f"Fix permissions with:\n"
                f"  chmod -R 700 {wallets_path}\n"
                f"  chown -R $(whoami) {wallets_path}\n\n"
                f"If running in Docker, ensure wallet volume is mounted with correct permissions:\n"
                f"  -v ~/.bittensor/wallets:/home/aurelius/.bittensor/wallets:ro"
            ),
        )

    # Case: No wallets found
    if not wallets:
        return WalletDetectionResult(
            wallet_name=None,
            hotkey=None,
            wallets=[],
            error=(
                f"No wallets found in {wallets_path}\n\n"
                f"Create a wallet using:\n"
                f"  btcli wallet create --wallet.name {role}\n\n"
                f"Then register on the subnet:\n"
                f"  btcli subnet register --wallet.name {role} --netuid <NETUID>"
            ),
        )

    # Count total wallet/hotkey combinations
    total_hotkeys = sum(len(w.hotkeys) for w in wallets)

    # Case: Exactly one wallet with exactly one hotkey → auto-select
    if len(wallets) == 1 and len(wallets[0].hotkeys) == 1:
        return WalletDetectionResult(
            wallet_name=wallets[0].name,
            hotkey=wallets[0].hotkeys[0],
            wallets=wallets,
            error=None,
        )

    # Case: Multiple wallets or hotkeys → require explicit config
    wallet_list = []
    for w in wallets:
        for h in w.hotkeys:
            wallet_list.append(f"  - {w.name}/{h}")

    env_var = f"{role.upper()}_WALLET_NAME"
    hotkey_var = f"{role.upper()}_HOTKEY"

    return WalletDetectionResult(
        wallet_name=None,
        hotkey=None,
        wallets=wallets,
        error=(
            f"Multiple wallets/hotkeys found ({total_hotkeys} total). "
            f"Please specify which to use.\n\n"
            f"Available wallets:\n"
            + "\n".join(wallet_list)
            + f"\n\n"
            f"Set these environment variables:\n"
            f"  export {env_var}=<wallet_name>\n"
            f"  export {hotkey_var}=<hotkey_name>\n\n"
            f"Or add to your .env file:\n"
            f"  {env_var}=<wallet_name>\n"
            f"  {hotkey_var}=<hotkey_name>"
        ),
    )
