"""
Tests for wallet creation and operations on testnet.

These tests verify:
- Programmatic wallet creation
- Address format validation
- TAO transfers
- Balance verification
"""

import re

import pytest

import bittensor as bt

from ..helpers.wallet_utils import (
    create_test_wallet,
    cleanup_test_wallet,
    generate_unique_wallet_name,
)
from ..helpers.testnet_funding import transfer_tao, wait_for_balance, get_balance


# SS58 address pattern for Bittensor (starts with 5)
SS58_PATTERN = re.compile(r"^5[A-HJ-NP-Za-km-z1-9]{47}$")


@pytest.mark.onchain
class TestWalletCreation:
    """Tests for programmatic wallet creation."""

    def test_generate_unique_wallet_name(self):
        """Test unique wallet name generation."""
        name1 = generate_unique_wallet_name()
        name2 = generate_unique_wallet_name()

        assert name1 != name2
        assert name1.startswith("e2e_test_")
        assert name2.startswith("e2e_test_")
        assert len(name1) == len("e2e_test_") + 8  # prefix + 8 hex chars

    def test_generate_unique_wallet_name_custom_prefix(self):
        """Test unique wallet name with custom prefix."""
        name = generate_unique_wallet_name(prefix="custom")
        assert name.startswith("custom_")

    def test_create_test_wallet(self):
        """Test creating a new test wallet."""
        wallet = None
        try:
            wallet = create_test_wallet()

            # Verify wallet object
            assert wallet is not None
            assert wallet.name.startswith("e2e_test_")

            # Verify keys exist
            assert wallet.hotkey is not None
            assert wallet.coldkeypub is not None

            # Verify addresses are valid SS58
            assert SS58_PATTERN.match(wallet.hotkey.ss58_address)
            assert SS58_PATTERN.match(wallet.coldkeypub.ss58_address)

        finally:
            if wallet:
                cleanup_test_wallet(wallet.name)

    def test_create_wallet_with_custom_name(self):
        """Test creating wallet with specific name."""
        custom_name = f"test_custom_{generate_unique_wallet_name()[-8:]}"
        wallet = None
        try:
            wallet = create_test_wallet(name=custom_name)
            assert wallet.name == custom_name
        finally:
            if wallet:
                cleanup_test_wallet(wallet.name)

    def test_create_wallet_with_custom_hotkey(self):
        """Test creating wallet with custom hotkey name."""
        wallet = None
        try:
            wallet = create_test_wallet(hotkey="miner1")
            assert wallet.hotkey_str == "miner1"
        finally:
            if wallet:
                cleanup_test_wallet(wallet.name)

    def test_cleanup_test_wallet(self):
        """Test wallet cleanup removes files."""
        wallet = create_test_wallet()
        wallet_name = wallet.name

        # Verify wallet exists
        import os
        from pathlib import Path
        wallet_path = Path.home() / ".bittensor" / "wallets" / wallet_name
        assert wallet_path.exists()

        # Clean up
        result = cleanup_test_wallet(wallet_name)
        assert result is True

        # Verify removed
        assert not wallet_path.exists()

    def test_create_wallet_fails_if_exists(self):
        """Test wallet creation fails if wallet already exists."""
        wallet = None
        try:
            wallet = create_test_wallet()

            # Try to create again without overwrite
            with pytest.raises(RuntimeError, match="Wallet already exists"):
                create_test_wallet(name=wallet.name)

        finally:
            if wallet:
                cleanup_test_wallet(wallet.name)

    def test_create_wallet_overwrite(self):
        """Test wallet creation with overwrite flag."""
        wallet = None
        try:
            wallet = create_test_wallet()
            original_hotkey = wallet.hotkey.ss58_address

            # Create again with overwrite
            wallet2 = create_test_wallet(name=wallet.name, overwrite=True)

            # New hotkey should be different
            assert wallet2.hotkey.ss58_address != original_hotkey

            wallet = wallet2  # Update for cleanup

        finally:
            if wallet:
                cleanup_test_wallet(wallet.name)


@pytest.mark.onchain
class TestWalletAddresses:
    """Tests for wallet address validation."""

    def test_wallet_has_valid_addresses(self, test_miner_wallet: bt.Wallet):
        """Test that test wallet has valid SS58 addresses."""
        hotkey = test_miner_wallet.hotkey.ss58_address
        coldkey = test_miner_wallet.coldkeypub.ss58_address

        # Verify format
        assert SS58_PATTERN.match(hotkey), f"Invalid hotkey format: {hotkey}"
        assert SS58_PATTERN.match(coldkey), f"Invalid coldkey format: {coldkey}"

        # Verify they're different
        assert hotkey != coldkey

    def test_hotkey_and_coldkey_are_distinct(self, test_miner_wallet: bt.Wallet):
        """Test that hotkey and coldkey are different addresses."""
        assert (
            test_miner_wallet.hotkey.ss58_address
            != test_miner_wallet.coldkeypub.ss58_address
        )


@pytest.mark.onchain
@pytest.mark.requires_funding
class TestWalletTransfers:
    """Tests for TAO transfers between wallets."""

    def test_transfer_to_test_wallet(
        self,
        funding_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test transferring TAO to a new test wallet."""
        recipient = None
        try:
            recipient = create_test_wallet()

            # Get initial balance (should be 0)
            initial = get_balance(
                testnet_subtensor,
                recipient.coldkeypub.ss58_address,
            )
            assert initial == 0.0

            # Transfer small amount
            amount = 0.5  # TAO
            success, msg = transfer_tao(
                from_wallet=funding_wallet,
                to_address=recipient.coldkeypub.ss58_address,
                amount=amount,
                subtensor=testnet_subtensor,
            )

            if not success and "Insufficient balance" in msg:
                pytest.skip("Funding wallet has insufficient balance")

            assert success, f"Transfer failed: {msg}"

        finally:
            if recipient:
                cleanup_test_wallet(recipient.name)

    def test_check_balance_after_transfer(
        self,
        test_miner_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test that balance appears after transfer."""
        # test_miner_wallet fixture already funds the wallet
        balance = get_balance(
            testnet_subtensor,
            test_miner_wallet.coldkeypub.ss58_address,
        )

        # Should have at least some balance (minus fees)
        assert balance > 0, "Wallet should have balance after funding"

    def test_wait_for_balance(
        self,
        funding_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test waiting for balance to appear."""
        recipient = None
        try:
            recipient = create_test_wallet()

            # Transfer
            amount = 0.5
            success, _ = transfer_tao(
                from_wallet=funding_wallet,
                to_address=recipient.coldkeypub.ss58_address,
                amount=amount,
                subtensor=testnet_subtensor,
            )

            if not success:
                pytest.skip("Transfer failed - insufficient balance?")

            # Wait for balance
            success, balance = wait_for_balance(
                testnet_subtensor,
                recipient.coldkeypub.ss58_address,
                min_balance=amount * 0.9,  # Allow for fees
                timeout=60,
            )

            assert success, f"Balance not received. Current: {balance}"
            assert balance > 0

        finally:
            if recipient:
                cleanup_test_wallet(recipient.name)

    def test_transfer_fails_with_insufficient_balance(
        self,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test transfer fails gracefully with insufficient funds."""
        sender = None
        try:
            # Create unfunded wallet
            sender = create_test_wallet()

            # Try to transfer (should fail)
            success, msg = transfer_tao(
                from_wallet=sender,
                to_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",  # Arbitrary address
                amount=1000.0,  # Large amount
                subtensor=testnet_subtensor,
            )

            assert not success
            assert "Insufficient balance" in msg or "error" in msg.lower()

        finally:
            if sender:
                cleanup_test_wallet(sender.name)
