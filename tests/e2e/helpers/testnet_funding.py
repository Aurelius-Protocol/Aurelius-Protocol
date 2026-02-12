"""Testnet funding utilities for E2E tests."""

import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bittensor as bt


def _find_btcli() -> str:
    """Find btcli executable, checking venv first."""
    # Check if we're in a venv and btcli exists there
    if hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix:
        venv_btcli = Path(sys.prefix) / "bin" / "btcli"
        if venv_btcli.exists():
            return str(venv_btcli)

    # Check common locations
    for path in [
        Path.cwd() / ".venv" / "bin" / "btcli",
        Path.home() / ".local" / "bin" / "btcli",
    ]:
        if path.exists():
            return str(path)

    # Fall back to PATH
    return "btcli"


def request_faucet_funds(
    wallet: "bt.Wallet",
    max_attempts: int = 3,
) -> tuple[bool, str]:
    """
    Request testnet TAO from the faucet.

    Uses btcli wallet faucet command.

    Args:
        wallet: Bittensor wallet to fund
        max_attempts: Maximum faucet request attempts

    Returns:
        Tuple of (success, message)
    """
    btcli = _find_btcli()

    for attempt in range(max_attempts):
        try:
            # Note: Faucet is DISABLED on testnet and mainnet
            # This only works on local chains
            result = subprocess.run(
                [
                    btcli,
                    "wallet",
                    "faucet",
                    "--wallet-name", wallet.name,
                    "--hotkey", wallet.hotkey_str,
                    "--network", "test",
                    "-y",  # No prompts
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                return True, f"Faucet request successful: {result.stdout}"

            # Check for rate limiting
            if "rate limit" in result.stderr.lower():
                if attempt < max_attempts - 1:
                    time.sleep(60)  # Wait before retry
                    continue
                return False, "Faucet rate limited"

            return False, f"Faucet failed: {result.stderr}"

        except subprocess.TimeoutExpired:
            if attempt < max_attempts - 1:
                continue
            return False, "Faucet request timed out"

        except FileNotFoundError:
            return False, "btcli not found in PATH"

        except Exception as e:
            return False, f"Faucet error: {e}"

    return False, "Max attempts reached"


def transfer_tao(
    from_wallet: "bt.Wallet",
    to_address: str,
    amount: float,
    subtensor: "bt.Subtensor | None" = None,
) -> tuple[bool, str]:
    """
    Transfer TAO from one wallet to another.

    Args:
        from_wallet: Source wallet (must have sufficient balance)
        to_address: Destination SS58 address
        amount: Amount of TAO to transfer
        subtensor: Optional subtensor instance (creates new if not provided)

    Returns:
        Tuple of (success, message)
    """
    import bittensor as bt

    try:
        if subtensor is None:
            subtensor = bt.Subtensor(network="test")

        # Check source balance
        balance = subtensor.get_balance(from_wallet.coldkeypub.ss58_address)
        if float(balance) < amount:
            return False, f"Insufficient balance: {balance} TAO (need {amount} TAO)"

        # Perform transfer (bittensor v8+ uses destination_ss58 instead of dest)
        result = subtensor.transfer(
            wallet=from_wallet,
            destination_ss58=to_address,
            amount=bt.Balance.from_tao(amount),
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

        # Result is ExtrinsicResponse object in newer versions
        if hasattr(result, 'success'):
            if result.success:
                return True, f"Transferred {amount} TAO to {to_address}"
            return False, f"Transfer failed: {result.error_message}"
        elif result:
            return True, f"Transferred {amount} TAO to {to_address}"
        return False, "Transfer failed (returned False)"

    except Exception as e:
        return False, f"Transfer error: {e}"


def wait_for_balance(
    subtensor: "bt.Subtensor",
    address: str,
    min_balance: float,
    timeout: int = 120,
) -> tuple[bool, float]:
    """
    Wait for an address to have at least the specified balance.

    Args:
        subtensor: Connected subtensor instance
        address: SS58 address to check
        min_balance: Minimum balance required (TAO)
        timeout: Maximum time to wait in seconds

    Returns:
        Tuple of (success, current_balance)
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            balance = subtensor.get_balance(address)
            current = float(balance)
            if current >= min_balance:
                return True, current
        except Exception:
            pass

        time.sleep(6)  # Check every 6 seconds

    # Final check
    try:
        balance = subtensor.get_balance(address)
        current = float(balance)
        return current >= min_balance, current
    except Exception:
        return False, 0.0


def get_balance(
    subtensor: "bt.Subtensor",
    address: str,
) -> float:
    """
    Get the balance of an address.

    Args:
        subtensor: Connected subtensor instance
        address: SS58 address to check

    Returns:
        Balance in TAO, or 0.0 on error
    """
    try:
        balance = subtensor.get_balance(address)
        return float(balance)
    except Exception:
        return 0.0


def estimate_registration_cost(
    subtensor: "bt.Subtensor",
    netuid: int,
) -> float:
    """
    Estimate the cost to register on a subnet.

    Args:
        subtensor: Connected subtensor instance
        netuid: Subnet ID

    Returns:
        Estimated registration cost in TAO
    """
    try:
        # Get the current burn cost for the subnet
        burn = subtensor.burn(netuid=netuid)
        return float(burn)
    except Exception:
        # Default fallback estimate
        return 0.1


def fund_wallet_for_registration(
    funding_wallet: "bt.Wallet",
    target_wallet: "bt.Wallet",
    subtensor: "bt.Subtensor",
    netuid: int,
    extra_buffer: float = 0.5,
) -> tuple[bool, str]:
    """
    Fund a wallet with enough TAO to register on a subnet.

    Args:
        funding_wallet: Wallet to fund from
        target_wallet: Wallet to fund
        subtensor: Connected subtensor instance
        netuid: Subnet ID to register on
        extra_buffer: Extra TAO beyond registration cost

    Returns:
        Tuple of (success, message)
    """
    # Estimate required amount
    reg_cost = estimate_registration_cost(subtensor, netuid)
    required = reg_cost + extra_buffer

    # Transfer funds
    return transfer_tao(
        from_wallet=funding_wallet,
        to_address=target_wallet.coldkeypub.ss58_address,
        amount=required,
        subtensor=subtensor,
    )
