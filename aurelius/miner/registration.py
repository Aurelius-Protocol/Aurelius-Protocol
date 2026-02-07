"""Miner experiment registration module (T064, T065).

Provides functions for miners to register and withdraw from experiments.
"""

import json
import time
from dataclasses import dataclass

import bittensor as bt
import requests

from aurelius.shared.config import Config


@dataclass
class RegistrationResult:
    """Result of a registration or withdrawal operation.

    Attributes:
        success: Whether the operation succeeded
        experiment_id: The experiment ID
        status: Registration status (active, withdrawn, etc.)
        registered_at: ISO timestamp when registered
        withdrawn_at: ISO timestamp when withdrawn (if applicable)
        message: Additional message from server
        error: Error message if failed
    """

    success: bool
    experiment_id: str | None = None
    status: str | None = None
    registered_at: str | None = None
    withdrawn_at: str | None = None
    message: str | None = None
    error: str | None = None


@dataclass
class MinerRegistrations:
    """List of miner's experiment registrations.

    Attributes:
        miner_hotkey: The miner's hotkey
        registrations: List of registration details
        error: Error message if fetch failed
    """

    miner_hotkey: str
    registrations: list[dict]
    error: str | None = None


def register_for_experiment(
    wallet: bt.Wallet,
    experiment_id: str,
    api_endpoint: str | None = None,
    timeout: int = 30,
) -> RegistrationResult:
    """Register miner for an experiment (T064).

    Signs a registration request with the miner's hotkey and sends it
    to the central API.

    Args:
        wallet: Bittensor wallet with miner hotkey
        experiment_id: Experiment ID to register for
        api_endpoint: API endpoint (defaults to Config.EXPERIMENT_API_ENDPOINT)
        timeout: Request timeout in seconds

    Returns:
        RegistrationResult with success status and registration details
    """
    endpoint = api_endpoint or Config.EXPERIMENT_API_ENDPOINT
    if not endpoint:
        return RegistrationResult(
            success=False,
            experiment_id=experiment_id,
            error="No API endpoint configured",
        )

    # Check for default experiment (all miners auto-registered)
    if experiment_id == "prompt":
        return RegistrationResult(
            success=True,
            experiment_id=experiment_id,
            status="active",
            message="All miners are auto-registered for 'prompt' experiment",
        )

    hotkey = wallet.hotkey.ss58_address
    timestamp = int(time.time())

    # Prepare request body
    body = {
        "miner_hotkey": hotkey,
        "timestamp": timestamp,
    }

    # Sign the request body
    message = json.dumps(body, separators=(",", ":"), sort_keys=True)
    signature = wallet.hotkey.sign(message.encode()).hex()

    try:
        response = requests.post(
            f"{endpoint}/{experiment_id}/register",
            json=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": signature,
                "X-Timestamp": str(timestamp),
            },
            timeout=timeout,
        )

        if response.status_code == 200:
            data = response.json()
            reg = data.get("registration", {})
            return RegistrationResult(
                success=True,
                experiment_id=experiment_id,
                status=reg.get("status"),
                registered_at=reg.get("registered_at"),
                withdrawn_at=reg.get("withdrawn_at"),
                message=data.get("message"),
            )
        else:
            data = response.json() if response.text else {}
            return RegistrationResult(
                success=False,
                experiment_id=experiment_id,
                error=data.get("message", f"HTTP {response.status_code}"),
            )

    except requests.RequestException as e:
        return RegistrationResult(
            success=False,
            experiment_id=experiment_id,
            error=f"Request failed: {e}",
        )
    except Exception as e:
        return RegistrationResult(
            success=False,
            experiment_id=experiment_id,
            error=f"Unexpected error: {e}",
        )


def withdraw_from_experiment(
    wallet: bt.Wallet,
    experiment_id: str,
    api_endpoint: str | None = None,
    timeout: int = 30,
) -> RegistrationResult:
    """Withdraw miner from an experiment.

    Signs a withdrawal request with the miner's hotkey and sends it
    to the central API.

    Args:
        wallet: Bittensor wallet with miner hotkey
        experiment_id: Experiment ID to withdraw from
        api_endpoint: API endpoint (defaults to Config.EXPERIMENT_API_ENDPOINT)
        timeout: Request timeout in seconds

    Returns:
        RegistrationResult with success status and registration details
    """
    endpoint = api_endpoint or Config.EXPERIMENT_API_ENDPOINT
    if not endpoint:
        return RegistrationResult(
            success=False,
            experiment_id=experiment_id,
            error="No API endpoint configured",
        )

    # Cannot withdraw from default experiment
    if experiment_id == "prompt":
        return RegistrationResult(
            success=False,
            experiment_id=experiment_id,
            error="Cannot withdraw from 'prompt' experiment (always active)",
        )

    hotkey = wallet.hotkey.ss58_address
    timestamp = int(time.time())

    # Prepare request body
    body = {
        "miner_hotkey": hotkey,
        "timestamp": timestamp,
    }

    # Sign the request body
    message = json.dumps(body, separators=(",", ":"), sort_keys=True)
    signature = wallet.hotkey.sign(message.encode()).hex()

    try:
        response = requests.post(
            f"{endpoint}/{experiment_id}/withdraw",
            json=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": signature,
                "X-Timestamp": str(timestamp),
            },
            timeout=timeout,
        )

        if response.status_code == 200:
            data = response.json()
            reg = data.get("registration", {})
            return RegistrationResult(
                success=True,
                experiment_id=experiment_id,
                status=reg.get("status", "withdrawn"),
                registered_at=reg.get("registered_at"),
                withdrawn_at=reg.get("withdrawn_at"),
                message=data.get("message"),
            )
        else:
            data = response.json() if response.text else {}
            return RegistrationResult(
                success=False,
                experiment_id=experiment_id,
                error=data.get("message", f"HTTP {response.status_code}"),
            )

    except requests.RequestException as e:
        return RegistrationResult(
            success=False,
            experiment_id=experiment_id,
            error=f"Request failed: {e}",
        )
    except Exception as e:
        return RegistrationResult(
            success=False,
            experiment_id=experiment_id,
            error=f"Unexpected error: {e}",
        )


def list_registrations(
    wallet: bt.Wallet,
    api_endpoint: str | None = None,
    timeout: int = 30,
) -> MinerRegistrations:
    """List all experiments a miner is registered for (T065).

    Signs a request with the miner's hotkey to authenticate the query.

    Args:
        wallet: Bittensor wallet with miner hotkey
        api_endpoint: API endpoint (defaults to Config.EXPERIMENT_API_ENDPOINT)
        timeout: Request timeout in seconds

    Returns:
        MinerRegistrations with list of experiment registrations
    """
    endpoint = api_endpoint or Config.EXPERIMENT_API_ENDPOINT
    hotkey = wallet.hotkey.ss58_address

    if not endpoint:
        return MinerRegistrations(
            miner_hotkey=hotkey,
            registrations=[],
            error="No API endpoint configured",
        )

    # Derive base API URL from experiment endpoint (e.g., .../api/experiments -> .../api)
    base_url = endpoint.rsplit("/", 1)[0]

    timestamp = int(time.time())

    # Sign the timestamp
    message = f"{hotkey}:{timestamp}"
    signature = wallet.hotkey.sign(message.encode()).hex()

    try:
        response = requests.get(
            f"{base_url}/miners/{hotkey}/registrations",
            headers={
                "X-Signature": signature,
                "X-Timestamp": str(timestamp),
            },
            timeout=timeout,
        )

        if response.status_code == 200:
            data = response.json()
            return MinerRegistrations(
                miner_hotkey=hotkey,
                registrations=data.get("registrations", []),
            )
        else:
            data = response.json() if response.text else {}
            return MinerRegistrations(
                miner_hotkey=hotkey,
                registrations=[],
                error=data.get("message", f"HTTP {response.status_code}"),
            )

    except requests.RequestException as e:
        return MinerRegistrations(
            miner_hotkey=hotkey,
            registrations=[],
            error=f"Request failed: {e}",
        )
    except Exception as e:
        return MinerRegistrations(
            miner_hotkey=hotkey,
            registrations=[],
            error=f"Unexpected error: {e}",
        )
