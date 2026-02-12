"""On-chain verification utilities for E2E tests."""

import time
from typing import Any

import bittensor as bt


def verify_registration_onchain(
    subtensor: bt.Subtensor,
    netuid: int,
    hotkey: str,
) -> dict[str, Any]:
    """
    Verify a hotkey is registered on the specified subnet.

    Args:
        subtensor: Connected subtensor instance
        netuid: Subnet ID to check
        hotkey: SS58 hotkey address to verify

    Returns:
        Dict with registration info:
        - registered: bool
        - uid: int or None
        - stake: float (TAO)
        - hotkey: str
    """
    try:
        metagraph = bt.Metagraph(netuid=netuid)
        metagraph.sync(subtensor=subtensor)

        if hotkey in metagraph.hotkeys:
            uid = metagraph.hotkeys.index(hotkey)
            stake = float(metagraph.S[uid])
            return {
                "registered": True,
                "uid": uid,
                "stake": stake,
                "hotkey": hotkey,
            }
    except Exception:
        pass

    return {
        "registered": False,
        "uid": None,
        "stake": 0.0,
        "hotkey": hotkey,
    }


def verify_weights_onchain(
    subtensor: bt.Subtensor,
    netuid: int,
    validator_uid: int,
) -> dict[str, Any]:
    """
    Verify weights set by a validator on-chain.

    Args:
        subtensor: Connected subtensor instance
        netuid: Subnet ID
        validator_uid: UID of the validator whose weights to check

    Returns:
        Dict with weight info:
        - uids: list of UIDs that have weights
        - weights: list of weight values (normalized 0-1)
        - block: block number when weights were read
        - raw_weights: raw weight values from chain
    """
    try:
        metagraph = bt.Metagraph(netuid=netuid)
        metagraph.sync(subtensor=subtensor)

        # Get the weights matrix row for this validator
        weights_row = metagraph.W[validator_uid]

        # Find non-zero weights
        uids = []
        weights = []
        for uid, weight in enumerate(weights_row):
            if weight > 0:
                uids.append(uid)
                weights.append(float(weight))

        return {
            "uids": uids,
            "weights": weights,
            "block": subtensor.block,
            "raw_weights": weights_row.tolist(),
        }
    except Exception as e:
        return {
            "uids": [],
            "weights": [],
            "block": None,
            "error": str(e),
        }


def wait_for_blocks(
    subtensor: bt.Subtensor,
    num_blocks: int,
    timeout: int = 180,
) -> tuple[bool, int]:
    """
    Wait for a specified number of blocks to pass.

    Args:
        subtensor: Connected subtensor instance
        num_blocks: Number of blocks to wait
        timeout: Maximum time to wait in seconds

    Returns:
        Tuple of (success, final_block_number)
    """
    start_block = get_block_with_retry(subtensor)
    target_block = start_block + num_blocks
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_block = get_block_with_retry(subtensor)
        if current_block >= target_block:
            return True, current_block
        time.sleep(6)  # Bittensor blocks are ~12s, check every 6s

    return False, get_block_with_retry(subtensor)


def get_block_with_retry(
    subtensor: bt.Subtensor,
    max_retries: int = 3,
) -> int:
    """
    Get current block number with retry logic.

    Args:
        subtensor: Connected subtensor instance
        max_retries: Maximum number of retry attempts

    Returns:
        Current block number

    Raises:
        Exception: If all retries fail
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return subtensor.block
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    raise last_error or Exception("Failed to get block number")


def get_neuron_info(
    subtensor: bt.Subtensor,
    netuid: int,
    uid: int,
) -> dict[str, Any]:
    """
    Get detailed neuron information from chain.

    Args:
        subtensor: Connected subtensor instance
        netuid: Subnet ID
        uid: Neuron UID

    Returns:
        Dict with neuron info including stake, rank, trust, etc.
    """
    try:
        metagraph = bt.Metagraph(netuid=netuid)
        metagraph.sync(subtensor=subtensor)

        if uid >= len(metagraph.hotkeys):
            return {"error": f"UID {uid} not found in metagraph"}

        return {
            "uid": uid,
            "hotkey": metagraph.hotkeys[uid],
            "coldkey": metagraph.coldkeys[uid],
            "stake": float(metagraph.S[uid]),
            "rank": float(metagraph.R[uid]),
            "trust": float(metagraph.T[uid]),
            "consensus": float(metagraph.C[uid]),
            "incentive": float(metagraph.I[uid]),
            "dividends": float(metagraph.D[uid]),
            "emission": float(metagraph.E[uid]),
            "validator_permit": bool(metagraph.validator_permit[uid]),
            "active": bool(metagraph.active[uid]),
        }
    except Exception as e:
        return {"error": str(e)}


def wait_for_registration(
    subtensor: bt.Subtensor,
    netuid: int,
    hotkey: str,
    timeout: int = 300,
    poll_interval: int = 10,
) -> tuple[bool, dict[str, Any]]:
    """
    Wait for a hotkey to appear in the metagraph.

    Args:
        subtensor: Connected subtensor instance
        netuid: Subnet ID
        hotkey: SS58 hotkey address to wait for
        timeout: Maximum time to wait in seconds
        poll_interval: Time between checks in seconds

    Returns:
        Tuple of (success, registration_info)
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        info = verify_registration_onchain(subtensor, netuid, hotkey)
        if info["registered"]:
            return True, info
        time.sleep(poll_interval)

    return False, {"registered": False, "hotkey": hotkey, "timeout": True}
