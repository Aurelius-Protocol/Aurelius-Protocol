"""
On-chain test fixtures and safety guards.

SAFETY: These tests only run on testnet (subnet 290).
Any attempt to run on mainnet will fail immediately.
"""

import os
from typing import Any, Generator

import pytest

import bittensor as bt

from ..helpers.chain_utils import verify_registration_onchain, wait_for_registration
from ..helpers.testnet_funding import transfer_tao, wait_for_balance, get_balance
from ..helpers.wallet_utils import (
    create_test_wallet,
    cleanup_test_wallet,
    cleanup_all_test_wallets,
    get_test_wallet_count,
)


# Constants
TESTNET_NETWORK = "test"
TESTNET_NETUID = 290
MAINNET_NETWORK = "finney"
MAINNET_NETUID = 37
MAX_TEST_WALLETS = 5
MAX_REGISTRATIONS_PER_SESSION = 3
MIN_FUNDING_AMOUNT = 1.0  # TAO per test wallet

# Track registrations to avoid excessive chain operations
_registrations_this_session = 0


def pytest_configure(config: pytest.Config) -> None:
    """
    Safety check: FAIL if mainnet is detected.

    This runs before any tests and ensures we never
    accidentally interact with mainnet.
    """
    network = os.environ.get("BT_NETWORK", "").lower()
    netuid = os.environ.get("BT_NETUID", "")

    # Fail fast on mainnet indicators
    if network == "finney" or network == "main" or network == "mainnet":
        pytest.exit(
            "SAFETY: BT_NETWORK is set to mainnet. "
            "On-chain tests only run on testnet. "
            "Set BT_NETWORK=test to proceed.",
            returncode=1,
        )

    if netuid == "37":
        pytest.exit(
            "SAFETY: BT_NETUID=37 is mainnet subnet. "
            "On-chain tests only run on subnet 290 (testnet). "
            "Set BT_NETUID=290 to proceed.",
            returncode=1,
        )

    # Register on-chain markers
    config.addinivalue_line(
        "markers", "onchain: tests that interact with real blockchain"
    )
    config.addinivalue_line(
        "markers", "requires_funding: tests that need funded wallets"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Clean up all test wallets at end of session."""
    cleaned = cleanup_all_test_wallets()
    if cleaned > 0:
        print(f"\nCleaned up {cleaned} test wallet(s)")


@pytest.fixture(scope="session", autouse=True)
def verify_testnet() -> dict[str, Any]:
    """
    Double-check we're on testnet before any on-chain operations.

    This fixture is auto-used for all tests in this module.

    Returns:
        Dict with network info

    Raises:
        pytest.fail: If mainnet is detected
    """
    network = os.environ.get("BT_NETWORK", TESTNET_NETWORK)
    netuid = int(os.environ.get("BT_NETUID", TESTNET_NETUID))

    # Safety check
    if network in ["finney", "main", "mainnet"]:
        pytest.fail("SAFETY: Cannot run on-chain tests on mainnet")

    if netuid == MAINNET_NETUID:
        pytest.fail("SAFETY: Cannot run on-chain tests on mainnet subnet 37")

    return {
        "network": network,
        "netuid": netuid,
        "is_testnet": network == TESTNET_NETWORK and netuid == TESTNET_NETUID,
    }


@pytest.fixture(scope="session")
def testnet_subtensor(verify_testnet: dict[str, Any]) -> bt.Subtensor:
    """
    Connect to Bittensor testnet.

    Skips if connection fails.
    """
    try:
        subtensor = bt.Subtensor(network=TESTNET_NETWORK)
        # Verify connection
        _ = subtensor.block
        return subtensor
    except Exception as e:
        pytest.skip(f"Failed to connect to testnet: {e}")


@pytest.fixture(scope="session")
def validator_wallet() -> bt.Wallet:
    """
    Load the validator wallet for funding and operations.

    Uses VALIDATOR_WALLET_NAME and VALIDATOR_HOTKEY from environment.
    """
    wallet_name = os.environ.get("VALIDATOR_WALLET_NAME", "validator")
    hotkey_name = os.environ.get("VALIDATOR_HOTKEY", "default")

    try:
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
        # Verify wallet loads
        _ = wallet.hotkey.ss58_address
        _ = wallet.coldkeypub.ss58_address
        return wallet
    except Exception as e:
        pytest.skip(f"Validator wallet not available: {e}")


@pytest.fixture(scope="session")
def funding_wallet(validator_wallet: bt.Wallet) -> bt.Wallet:
    """
    Wallet used to fund test miner wallets.

    Uses the validator wallet's coldkey to transfer TAO to test wallets.
    """
    return validator_wallet


@pytest.fixture(scope="session")
def validator_registered(
    testnet_subtensor: bt.Subtensor,
    validator_wallet: bt.Wallet,
) -> dict[str, Any]:
    """
    Check if validator is registered on subnet 290.

    Returns registration info dict.
    """
    hotkey = validator_wallet.hotkey.ss58_address
    return verify_registration_onchain(testnet_subtensor, TESTNET_NETUID, hotkey)


@pytest.fixture(scope="session")
def require_validator_registration(
    validator_registered: dict[str, Any],
) -> dict[str, Any]:
    """
    Skip test if validator is not registered.
    """
    if not validator_registered["registered"]:
        pytest.skip(
            f"Validator not registered on subnet {TESTNET_NETUID}. "
            "Register with: btcli subnet register --netuid 290 --subtensor.network test"
        )
    return validator_registered


@pytest.fixture(scope="session")
def require_validator_stake(
    validator_registered: dict[str, Any],
) -> dict[str, Any]:
    """
    Skip test if validator doesn't have minimum stake (100 TAO for testnet).
    """
    if not validator_registered["registered"]:
        pytest.skip("Validator not registered")

    min_stake = 100.0  # Testnet minimum
    if validator_registered["stake"] < min_stake:
        pytest.skip(
            f"Validator has {validator_registered['stake']} TAO staked, "
            f"need {min_stake} TAO for weight setting"
        )
    return validator_registered


@pytest.fixture(scope="function")
def test_miner_wallet(
    funding_wallet: bt.Wallet,
    testnet_subtensor: bt.Subtensor,
) -> Generator[bt.Wallet, None, None]:
    """
    Create a new test miner wallet, fund it, then clean up after test.

    Yields:
        Funded test wallet

    Cleanup:
        Wallet is removed after test completes
    """
    # Check limits
    if get_test_wallet_count() >= MAX_TEST_WALLETS:
        pytest.skip(f"Max test wallets ({MAX_TEST_WALLETS}) reached")

    # Create wallet
    wallet = create_test_wallet()

    try:
        # Fund from validator coldkey
        success, msg = transfer_tao(
            from_wallet=funding_wallet,
            to_address=wallet.coldkeypub.ss58_address,
            amount=MIN_FUNDING_AMOUNT,
            subtensor=testnet_subtensor,
        )

        if not success:
            cleanup_test_wallet(wallet.name)
            pytest.skip(f"Failed to fund test wallet: {msg}")

        # Wait for balance to appear
        success, balance = wait_for_balance(
            testnet_subtensor,
            wallet.coldkeypub.ss58_address,
            MIN_FUNDING_AMOUNT * 0.9,  # Allow for fees
            timeout=60,
        )

        if not success:
            cleanup_test_wallet(wallet.name)
            pytest.skip("Funding transfer did not confirm in time")

        yield wallet

    finally:
        # Always clean up
        cleanup_test_wallet(wallet.name)


@pytest.fixture(scope="function")
def registered_miner(
    test_miner_wallet: bt.Wallet,
    testnet_subtensor: bt.Subtensor,
) -> dict[str, Any]:
    """
    Register a test miner on subnet 290.

    Returns:
        Dict with registration info: {hotkey, uid, wallet}

    Note:
        Registrations persist on-chain even after test cleanup.
        Limited to MAX_REGISTRATIONS_PER_SESSION to avoid spam.
    """
    global _registrations_this_session

    # Check if torch is installed (required for registration POW)
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed - required for registration POW")

    if _registrations_this_session >= MAX_REGISTRATIONS_PER_SESSION:
        pytest.skip(
            f"Max registrations ({MAX_REGISTRATIONS_PER_SESSION}) "
            "reached for this session"
        )

    # Register on subnet
    try:
        success = testnet_subtensor.register(
            wallet=test_miner_wallet,
            netuid=TESTNET_NETUID,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

        if not success:
            pytest.skip("Registration returned False")

        _registrations_this_session += 1

        # Wait for registration to appear in metagraph
        success, info = wait_for_registration(
            testnet_subtensor,
            TESTNET_NETUID,
            test_miner_wallet.hotkey.ss58_address,
            timeout=120,
        )

        if not success:
            pytest.skip("Registration did not appear in metagraph")

        return {
            "hotkey": test_miner_wallet.hotkey.ss58_address,
            "uid": info["uid"],
            "wallet": test_miner_wallet,
            "stake": info.get("stake", 0),
        }

    except Exception as e:
        if "torch" in str(e).lower():
            pytest.skip("torch not installed - required for registration")
        pytest.skip(f"Registration failed: {e}")


@pytest.fixture
def metagraph(testnet_subtensor: bt.Subtensor) -> bt.Metagraph:
    """
    Fresh metagraph synced from testnet.
    """
    metagraph = bt.Metagraph(netuid=TESTNET_NETUID, network=TESTNET_NETWORK)
    metagraph.sync(subtensor=testnet_subtensor)
    return metagraph


@pytest.fixture
def sample_miner_uids(metagraph: bt.Metagraph) -> list[int]:
    """
    Get a list of active miner UIDs for testing.

    Returns up to 5 UIDs with non-zero incentive.
    """
    uids = []
    for uid in range(len(metagraph.hotkeys)):
        # Skip validators (those with stake)
        if metagraph.S[uid] > 0:
            continue
        # Prefer active miners
        if metagraph.active[uid]:
            uids.append(uid)
        if len(uids) >= 5:
            break
    return uids
