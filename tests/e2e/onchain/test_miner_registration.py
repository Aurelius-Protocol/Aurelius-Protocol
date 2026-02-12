"""
Tests for miner registration on testnet (subnet 290).

These tests verify:
- Miner registration on-chain
- Registration appears in metagraph
- Error handling for insufficient funds
"""

import pytest

import bittensor as bt

from ..helpers.chain_utils import (
    verify_registration_onchain,
    wait_for_registration,
    get_neuron_info,
)
from ..helpers.wallet_utils import create_test_wallet, cleanup_test_wallet
from ..helpers.testnet_funding import estimate_registration_cost


TESTNET_NETUID = 290


@pytest.mark.onchain
class TestMinerRegistration:
    """Tests for miner registration on subnet 290."""

    def test_register_miner_on_testnet(
        self,
        registered_miner: dict,
    ):
        """Test that miner registration succeeds on testnet."""
        # registered_miner fixture handles the registration
        assert registered_miner is not None
        assert registered_miner["hotkey"] is not None
        assert registered_miner["uid"] is not None
        assert registered_miner["uid"] >= 0

    def test_registration_appears_in_metagraph(
        self,
        registered_miner: dict,
        metagraph: bt.Metagraph,
    ):
        """Test that registered miner appears in metagraph."""
        hotkey = registered_miner["hotkey"]
        uid = registered_miner["uid"]

        # Verify hotkey is in metagraph
        assert hotkey in metagraph.hotkeys

        # Verify UID matches
        found_uid = metagraph.hotkeys.index(hotkey)
        assert found_uid == uid

    def test_registered_miner_is_active(
        self,
        registered_miner: dict,
        metagraph: bt.Metagraph,
    ):
        """Test that newly registered miner is marked as active."""
        uid = registered_miner["uid"]

        # New registrations should be active
        assert metagraph.active[uid]

    def test_registration_with_insufficient_funds(
        self,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test registration fails gracefully with insufficient funds."""
        # Check if torch is installed (required for registration POW)
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed - required for registration")

        wallet = None
        try:
            # Create unfunded wallet
            wallet = create_test_wallet()

            # Attempt registration (should fail due to no funds)
            with pytest.raises(Exception):
                testnet_subtensor.register(
                    wallet=wallet,
                    netuid=TESTNET_NETUID,
                    wait_for_inclusion=True,
                    wait_for_finalization=True,
                )

        finally:
            if wallet:
                cleanup_test_wallet(wallet.name)

    def test_verify_registration_onchain(
        self,
        registered_miner: dict,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test verify_registration_onchain helper returns correct info."""
        info = verify_registration_onchain(
            testnet_subtensor,
            TESTNET_NETUID,
            registered_miner["hotkey"],
        )

        assert info["registered"] is True
        assert info["uid"] == registered_miner["uid"]
        assert info["hotkey"] == registered_miner["hotkey"]

    def test_verify_unregistered_hotkey(
        self,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test verify_registration_onchain returns False for unregistered hotkey."""
        # Fake hotkey that's definitely not registered
        fake_hotkey = "5FakeHotkey123456789012345678901234567890123456789"

        info = verify_registration_onchain(
            testnet_subtensor,
            TESTNET_NETUID,
            fake_hotkey,
        )

        assert info["registered"] is False
        assert info["uid"] is None
        assert info["stake"] == 0.0

    def test_get_neuron_info(
        self,
        registered_miner: dict,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test getting detailed neuron info from chain."""
        info = get_neuron_info(
            testnet_subtensor,
            TESTNET_NETUID,
            registered_miner["uid"],
        )

        # Basic fields should be present
        assert "uid" in info
        assert "hotkey" in info
        assert "coldkey" in info
        assert "stake" in info

        # Verify hotkey matches
        assert info["hotkey"] == registered_miner["hotkey"]

    def test_estimate_registration_cost(
        self,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test registration cost estimation."""
        cost = estimate_registration_cost(testnet_subtensor, TESTNET_NETUID)

        # Cost should be a reasonable positive number
        assert cost > 0
        assert cost < 10  # Should be less than 10 TAO on testnet


@pytest.mark.onchain
@pytest.mark.slow
class TestRegistrationWaiting:
    """Tests for registration waiting and confirmation."""

    def test_wait_for_registration(
        self,
        test_miner_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test waiting for registration to appear."""
        # Check if torch is installed (required for registration POW)
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed - required for registration")

        # Register the miner
        try:
            success = testnet_subtensor.register(
                wallet=test_miner_wallet,
                netuid=TESTNET_NETUID,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )

            if not success:
                pytest.skip("Registration failed")

            # Use wait_for_registration helper
            success, info = wait_for_registration(
                testnet_subtensor,
                TESTNET_NETUID,
                test_miner_wallet.hotkey.ss58_address,
                timeout=120,
            )

            assert success, "Registration did not appear in metagraph"
            assert info["registered"] is True
            assert info["uid"] is not None

        except Exception as e:
            if "burn" in str(e).lower() or "insufficient" in str(e).lower():
                pytest.skip(f"Registration failed due to funds: {e}")
            if "torch" in str(e).lower():
                pytest.skip("torch not installed - required for registration")
            raise

    def test_wait_for_nonexistent_registration_times_out(
        self,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test that waiting for non-existent registration times out."""
        fake_hotkey = "5NonExistentHotkey123456789012345678901234567890"

        success, info = wait_for_registration(
            testnet_subtensor,
            TESTNET_NETUID,
            fake_hotkey,
            timeout=5,  # Short timeout
            poll_interval=1,
        )

        assert success is False
        assert info["registered"] is False
        assert info.get("timeout") is True


@pytest.mark.onchain
class TestValidatorRegistrationCheck:
    """Tests for validator registration verification."""

    def test_validator_is_registered(
        self,
        require_validator_registration: dict,
    ):
        """Test that validator is registered on subnet 290."""
        assert require_validator_registration["registered"] is True
        assert require_validator_registration["uid"] is not None

    def test_validator_registration_info(
        self,
        validator_registered: dict,
    ):
        """Test validator registration info structure."""
        # Structure should be correct even if not registered
        assert "registered" in validator_registered
        assert "uid" in validator_registered
        assert "hotkey" in validator_registered
        assert "stake" in validator_registered
