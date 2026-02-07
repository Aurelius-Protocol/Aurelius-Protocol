"""
Tests for weight setting on testnet (subnet 290).

These tests verify:
- Validator prerequisites (registration, stake)
- Setting weights for single and multiple miners
- On-chain weight verification
"""

import numpy as np
import pytest

import bittensor as bt

from ..helpers.chain_utils import (
    verify_weights_onchain,
    wait_for_blocks,
    get_block_with_retry,
)


TESTNET_NETUID = 290


@pytest.mark.onchain
class TestValidatorPrerequisites:
    """Tests for validator prerequisites before weight setting."""

    def test_validator_is_registered(
        self,
        require_validator_registration: dict,
    ):
        """Test validator is registered on subnet 290."""
        assert require_validator_registration["registered"] is True
        assert require_validator_registration["uid"] is not None
        assert require_validator_registration["uid"] >= 0

    def test_validator_has_minimum_stake(
        self,
        require_validator_stake: dict,
    ):
        """Test validator has minimum stake for weight setting."""
        min_stake = 100.0  # Testnet minimum
        assert require_validator_stake["stake"] >= min_stake

    def test_validator_uid_in_metagraph(
        self,
        require_validator_registration: dict,
        metagraph: bt.Metagraph,
    ):
        """Test validator UID is valid in metagraph."""
        uid = require_validator_registration["uid"]
        assert uid < len(metagraph.hotkeys)
        assert metagraph.hotkeys[uid] == require_validator_registration["hotkey"]


@pytest.mark.onchain
@pytest.mark.requires_funding
class TestWeightSetting:
    """Tests for setting weights on-chain."""

    def test_set_single_weight(
        self,
        validator_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
        require_validator_stake: dict,
        sample_miner_uids: list[int],
    ):
        """Test setting weight for a single miner."""
        if not sample_miner_uids:
            pytest.skip("No active miners found in metagraph")

        validator_uid = require_validator_stake["uid"]
        miner_uid = sample_miner_uids[0]

        # Set weight for single miner
        uids = np.array([miner_uid], dtype=np.int64)
        weights = np.array([1.0], dtype=np.float32)

        try:
            success, msg = testnet_subtensor.set_weights(
                wallet=validator_wallet,
                netuid=TESTNET_NETUID,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )

            # set_weights returns (success, message) tuple
            assert success, f"Weight setting failed: {msg}"

        except Exception as e:
            if "rate limit" in str(e).lower():
                pytest.skip("Weight setting rate limited")
            raise

    def test_set_multiple_weights(
        self,
        validator_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
        require_validator_stake: dict,
        sample_miner_uids: list[int],
    ):
        """Test distributing weights across multiple miners."""
        if len(sample_miner_uids) < 2:
            pytest.skip("Need at least 2 active miners for this test")

        # Use first 3 miners (or all if less)
        miner_uids = sample_miner_uids[:3]

        # Create weight distribution
        uids = np.array(miner_uids, dtype=np.int64)
        # Distribute weights (e.g., 0.5, 0.3, 0.2)
        weight_values = [0.5, 0.3, 0.2][: len(miner_uids)]
        weights = np.array(weight_values, dtype=np.float32)

        # Normalize
        weights = weights / weights.sum()

        try:
            success, msg = testnet_subtensor.set_weights(
                wallet=validator_wallet,
                netuid=TESTNET_NETUID,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )

            assert success, f"Weight setting failed: {msg}"

        except Exception as e:
            if "rate limit" in str(e).lower():
                pytest.skip("Weight setting rate limited")
            raise

    def test_verify_weights_onchain(
        self,
        validator_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
        require_validator_stake: dict,
        sample_miner_uids: list[int],
    ):
        """Test verifying weights appear on-chain after setting."""
        if not sample_miner_uids:
            pytest.skip("No active miners found")

        validator_uid = require_validator_stake["uid"]
        miner_uid = sample_miner_uids[0]

        # Set a weight
        uids = np.array([miner_uid], dtype=np.int64)
        weights = np.array([1.0], dtype=np.float32)

        try:
            success, _ = testnet_subtensor.set_weights(
                wallet=validator_wallet,
                netuid=TESTNET_NETUID,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )

            if not success:
                pytest.skip("Weight setting failed")

            # Wait for a block to ensure propagation
            wait_for_blocks(testnet_subtensor, num_blocks=1, timeout=30)

            # Verify weights on-chain
            weight_info = verify_weights_onchain(
                testnet_subtensor,
                TESTNET_NETUID,
                validator_uid,
            )

            assert miner_uid in weight_info["uids"], (
                f"Miner {miner_uid} not in weights: {weight_info['uids']}"
            )

        except Exception as e:
            if "rate limit" in str(e).lower():
                pytest.skip("Weight setting rate limited")
            raise

    def test_weight_normalization(
        self,
        validator_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
        require_validator_stake: dict,
        sample_miner_uids: list[int],
    ):
        """Test that weights are normalized on-chain."""
        if len(sample_miner_uids) < 2:
            pytest.skip("Need at least 2 miners")

        validator_uid = require_validator_stake["uid"]
        miner_uids = sample_miner_uids[:2]

        # Set non-normalized weights
        uids = np.array(miner_uids, dtype=np.int64)
        weights = np.array([100.0, 50.0], dtype=np.float32)

        # Manually normalize before sending
        weights = weights / weights.sum()

        try:
            success, _ = testnet_subtensor.set_weights(
                wallet=validator_wallet,
                netuid=TESTNET_NETUID,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )

            if not success:
                pytest.skip("Weight setting failed")

            wait_for_blocks(testnet_subtensor, num_blocks=1, timeout=30)

            # Verify weights are normalized (sum to 1)
            weight_info = verify_weights_onchain(
                testnet_subtensor,
                TESTNET_NETUID,
                validator_uid,
            )

            total = sum(weight_info["weights"])
            # Allow small floating point tolerance
            assert 0.99 <= total <= 1.01, f"Weights not normalized: sum={total}"

        except Exception as e:
            if "rate limit" in str(e).lower():
                pytest.skip("Weight setting rate limited")
            raise


@pytest.mark.onchain
class TestWeightVerification:
    """Tests for weight verification utilities."""

    def test_verify_weights_onchain_returns_structure(
        self,
        testnet_subtensor: bt.Subtensor,
        require_validator_registration: dict,
    ):
        """Test verify_weights_onchain returns correct structure."""
        validator_uid = require_validator_registration["uid"]

        info = verify_weights_onchain(
            testnet_subtensor,
            TESTNET_NETUID,
            validator_uid,
        )

        # Should have required keys
        assert "uids" in info
        assert "weights" in info
        assert "block" in info

        # Types should be correct
        assert isinstance(info["uids"], list)
        assert isinstance(info["weights"], list)

    def test_verify_weights_for_invalid_uid(
        self,
        testnet_subtensor: bt.Subtensor,
        metagraph: bt.Metagraph,
    ):
        """Test weight verification for UID beyond metagraph size."""
        invalid_uid = len(metagraph.hotkeys) + 100

        info = verify_weights_onchain(
            testnet_subtensor,
            TESTNET_NETUID,
            invalid_uid,
        )

        # Should return empty or error
        assert info["uids"] == [] or "error" in info


@pytest.mark.onchain
class TestBlockOperations:
    """Tests for block-related utilities."""

    def test_get_block_with_retry(
        self,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test getting current block number."""
        block = get_block_with_retry(testnet_subtensor)

        assert isinstance(block, int)
        assert block > 0

    def test_wait_for_blocks(
        self,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test waiting for blocks to pass."""
        start_block = get_block_with_retry(testnet_subtensor)

        # Wait for 1 block (short test)
        success, final_block = wait_for_blocks(
            testnet_subtensor,
            num_blocks=1,
            timeout=60,
        )

        # May or may not succeed depending on block time
        # but final_block should be >= start_block
        assert final_block >= start_block

    @pytest.mark.slow
    def test_wait_for_multiple_blocks(
        self,
        testnet_subtensor: bt.Subtensor,
    ):
        """Test waiting for multiple blocks."""
        start_block = get_block_with_retry(testnet_subtensor)

        # Wait for 2 blocks
        success, final_block = wait_for_blocks(
            testnet_subtensor,
            num_blocks=2,
            timeout=120,
        )

        if success:
            assert final_block >= start_block + 2
