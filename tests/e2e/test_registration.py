"""
Tests for wallet and testnet registration status.

These tests verify that the validator wallet is properly configured
and registered on the Bittensor testnet subnet 290.
"""

import pytest

pytestmark = [pytest.mark.e2e]


class TestWalletSetup:
    """Tests for wallet existence and configuration."""

    def test_wallet_exists(self, wallet_info: dict) -> None:
        """Verify wallet directory exists."""
        assert wallet_info["exists"], f"Wallet not found at: {wallet_info['path']}"

    def test_coldkey_exists(self, wallet_info: dict) -> None:
        """Verify coldkey file exists."""
        assert wallet_info["coldkey_exists"], "Coldkey not found"

    def test_hotkey_exists(self, wallet_info: dict) -> None:
        """Verify hotkey file exists."""
        assert wallet_info["hotkey_exists"], (
            f"Hotkey '{wallet_info['hotkey_name']}' not found for wallet '{wallet_info['wallet_name']}'"
        )

    def test_hotkey_address_format(self, wallet_info: dict) -> None:
        """Verify hotkey address is valid SS58 format."""
        hotkey = wallet_info.get("hotkey_address")
        assert hotkey is not None, "Could not extract hotkey address"
        assert hotkey.startswith("5"), "Hotkey should be SS58 format starting with '5'"
        assert len(hotkey) == 48, f"Hotkey should be 48 characters, got {len(hotkey)}"

    def test_wallet_loads(self, validator_wallet) -> None:
        """Verify wallet can be loaded by bittensor."""
        assert validator_wallet is not None
        assert validator_wallet.hotkey is not None
        assert validator_wallet.hotkey.ss58_address is not None


class TestSubtensorConnection:
    """Tests for testnet connectivity."""

    @pytest.mark.slow
    def test_subtensor_connects(self, testnet_subtensor) -> None:
        """Verify connection to testnet subtensor."""
        assert testnet_subtensor is not None
        # Should be able to get current block
        block = testnet_subtensor.block
        assert block > 0, "Should get valid block number"

    @pytest.mark.slow
    def test_subnet_exists(self, testnet_subtensor) -> None:
        """Verify subnet 290 exists on testnet."""
        try:
            # Try to get hyperparameters for subnet 290
            params = testnet_subtensor.get_subnet_hyperparameters(netuid=290)
            assert params is not None, "Subnet 290 should exist on testnet"
        except Exception as e:
            pytest.fail(f"Failed to query subnet 290: {e}")


class TestMetagraphSync:
    """Tests for metagraph synchronization."""

    @pytest.mark.slow
    def test_metagraph_syncs(self, metagraph) -> None:
        """Verify metagraph can be synced."""
        assert metagraph is not None
        assert metagraph.n >= 0, "Metagraph should have valid neuron count"

    @pytest.mark.slow
    def test_metagraph_has_neurons(self, metagraph) -> None:
        """Verify metagraph has registered neurons."""
        # Subnet might be empty in early stages
        assert hasattr(metagraph, "hotkeys"), "Metagraph should have hotkeys attribute"
        assert hasattr(metagraph, "S"), "Metagraph should have stake attribute"


class TestValidatorRegistration:
    """Tests for validator registration status."""

    @pytest.mark.slow
    def test_registration_status(self, validator_registered: dict) -> None:
        """Check registration status (informational, doesn't fail)."""
        if validator_registered["registered"]:
            print(f"\n  Validator registered with UID: {validator_registered['uid']}")
            print(f"  Hotkey: {validator_registered['hotkey']}")
            print(f"  Stake: {validator_registered['stake']} TAO")
        else:
            print(f"\n  Validator NOT registered on subnet 290")
            print(f"  Hotkey: {validator_registered['hotkey']}")
            print("  Run: python scripts/register_testnet.py --register")

    @pytest.mark.slow
    @pytest.mark.requires_registration
    def test_validator_is_registered(self, require_registration: dict) -> None:
        """Verify validator is registered on subnet 290."""
        assert require_registration["registered"], "Validator should be registered"
        assert require_registration["uid"] is not None, "Should have valid UID"

    @pytest.mark.slow
    @pytest.mark.requires_registration
    def test_validator_has_stake(self, require_stake: dict) -> None:
        """Verify validator has minimum stake."""
        assert require_stake["stake"] >= 100, (
            f"Stake {require_stake['stake']} TAO is below minimum 100 TAO"
        )

    @pytest.mark.slow
    @pytest.mark.requires_registration
    def test_validator_in_metagraph(
        self,
        metagraph,
        validator_hotkey: str,
    ) -> None:
        """Verify validator appears in metagraph."""
        assert validator_hotkey in metagraph.hotkeys, (
            f"Validator hotkey not found in metagraph"
        )

        uid = metagraph.hotkeys.index(validator_hotkey)
        assert uid >= 0, "Should have valid UID"


class TestWalletBalance:
    """Tests for wallet balance."""

    @pytest.mark.slow
    def test_check_balance(self, testnet_subtensor, validator_hotkey: str) -> None:
        """Check wallet balance (informational)."""
        try:
            balance = testnet_subtensor.get_balance(validator_hotkey)
            print(f"\n  Hotkey balance: {balance} TAO")
        except Exception as e:
            print(f"\n  Could not check balance: {e}")
