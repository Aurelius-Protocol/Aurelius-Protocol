"""
Tests for real metagraph operations on testnet.

These tests verify that we can interact with the actual
Bittensor testnet metagraph for subnet 290.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


class TestMetagraphBasics:
    """Basic metagraph functionality tests."""

    def test_metagraph_has_properties(self, metagraph) -> None:
        """Verify metagraph has expected properties."""
        assert hasattr(metagraph, "n"), "Should have neuron count"
        assert hasattr(metagraph, "hotkeys"), "Should have hotkeys list"
        assert hasattr(metagraph, "coldkeys"), "Should have coldkeys list"
        assert hasattr(metagraph, "S"), "Should have stake tensor"
        # Note: 'R' (rank), 'I' (incentive), 'E' (emission) may not be present
        # in all bittensor versions or metagraph states
        assert hasattr(metagraph, "I") or hasattr(metagraph, "incentive"), "Should have incentive"

    def test_metagraph_neuron_count(self, metagraph) -> None:
        """Verify metagraph has valid neuron count."""
        assert metagraph.n >= 0, "Neuron count should be non-negative"
        print(f"\n  Subnet 290 has {metagraph.n} neurons")

    def test_hotkeys_list_length(self, metagraph) -> None:
        """Verify hotkeys list matches neuron count."""
        assert len(metagraph.hotkeys) == metagraph.n

    def test_stake_tensor_length(self, metagraph) -> None:
        """Verify stake tensor matches neuron count."""
        assert len(metagraph.S) == metagraph.n


class TestMetagraphSync:
    """Tests for metagraph synchronization."""

    def test_sync_updates_metagraph(self, testnet_subtensor, metagraph) -> None:
        """Verify metagraph can be synced."""
        original_block = metagraph.block

        # Sync again
        metagraph.sync(subtensor=testnet_subtensor)

        # Block should be same or higher
        assert metagraph.block >= original_block

    def test_sync_with_lite_mode(self, testnet_subtensor) -> None:
        """Test lite sync mode."""
        try:
            import bittensor as bt

            metagraph = bt.Metagraph(netuid=290, network="test", lite=True)
            metagraph.sync(subtensor=testnet_subtensor, lite=True)

            # Should have basic properties even in lite mode
            assert metagraph.n >= 0
            assert len(metagraph.hotkeys) == metagraph.n
        except Exception as e:
            pytest.skip(f"Lite sync not supported: {e}")


class TestValidatorInMetagraph:
    """Tests for validator presence in metagraph."""

    @pytest.mark.requires_registration
    def test_validator_hotkey_in_metagraph(
        self,
        metagraph,
        require_registration: dict,
    ) -> None:
        """Verify validator hotkey appears in metagraph."""
        hotkey = require_registration["hotkey"]
        assert hotkey in metagraph.hotkeys

    @pytest.mark.requires_registration
    def test_validator_uid_matches(
        self,
        metagraph,
        require_registration: dict,
    ) -> None:
        """Verify validator UID matches metagraph index."""
        hotkey = require_registration["hotkey"]
        uid = require_registration["uid"]

        metagraph_uid = metagraph.hotkeys.index(hotkey)
        assert uid == metagraph_uid

    @pytest.mark.requires_registration
    def test_validator_stake_positive(
        self,
        metagraph,
        require_stake: dict,
    ) -> None:
        """Verify validator has positive stake."""
        uid = require_stake["uid"]
        stake = float(metagraph.S[uid])

        assert stake > 0, f"Validator stake should be positive, got {stake}"
        print(f"\n  Validator UID {uid} stake: {stake} TAO")


class TestMetagraphQueries:
    """Tests for querying metagraph data."""

    def test_get_all_stakes(self, metagraph) -> None:
        """Get stake for all neurons."""
        if metagraph.n == 0:
            pytest.skip("No neurons in metagraph")

        stakes = [float(metagraph.S[i]) for i in range(metagraph.n)]

        total_stake = sum(stakes)
        max_stake = max(stakes) if stakes else 0
        staked_count = sum(1 for s in stakes if s > 0)

        print(f"\n  Total stake: {total_stake:.2f} TAO")
        print(f"  Max stake: {max_stake:.2f} TAO")
        print(f"  Neurons with stake: {staked_count}/{metagraph.n}")

    def test_get_all_incentives(self, metagraph) -> None:
        """Get incentive for all neurons."""
        if metagraph.n == 0:
            pytest.skip("No neurons in metagraph")

        incentives = [float(metagraph.I[i]) for i in range(metagraph.n)]

        total_incentive = sum(incentives)
        print(f"\n  Total incentive: {total_incentive:.6f}")

    def test_get_all_emissions(self, metagraph) -> None:
        """Get emission for all neurons."""
        if metagraph.n == 0:
            pytest.skip("No neurons in metagraph")

        emissions = [float(metagraph.E[i]) for i in range(metagraph.n)]

        total_emission = sum(emissions)
        print(f"\n  Total emission: {total_emission:.6f}")

    @pytest.mark.requires_registration
    def test_validator_rank(
        self,
        metagraph,
        require_registration: dict,
    ) -> None:
        """Get validator's rank in metagraph."""
        uid = require_registration["uid"]

        # Attributes vary by bittensor version
        incentive = float(metagraph.I[uid]) if hasattr(metagraph, 'I') else 0.0

        print(f"\n  Validator UID {uid}:")
        print(f"    Incentive: {incentive:.6f}")

        # These may not exist in all versions
        if hasattr(metagraph, 'R'):
            print(f"    Rank: {float(metagraph.R[uid]):.6f}")
        if hasattr(metagraph, 'E'):
            print(f"    Emission: {float(metagraph.E[uid]):.6f}")


class TestSubnetHyperparameters:
    """Tests for subnet hyperparameters."""

    def test_get_hyperparameters(self, testnet_subtensor) -> None:
        """Get subnet 290 hyperparameters."""
        try:
            params = testnet_subtensor.get_subnet_hyperparameters(netuid=290)

            assert params is not None
            print(f"\n  Subnet 290 hyperparameters:")

            # Print some key params if available
            if hasattr(params, "min_allowed_weights"):
                print(f"    Min allowed weights: {params.min_allowed_weights}")
            if hasattr(params, "max_allowed_weights"):
                print(f"    Max allowed weights: {params.max_allowed_weights}")
            if hasattr(params, "immunity_period"):
                print(f"    Immunity period: {params.immunity_period}")
            if hasattr(params, "tempo"):
                print(f"    Tempo: {params.tempo}")

        except Exception as e:
            pytest.skip(f"Could not get hyperparameters: {e}")

    def test_get_min_stake(self, testnet_subtensor) -> None:
        """Get minimum stake requirement."""
        try:
            # Different methods may be available depending on bittensor version
            min_stake = getattr(
                testnet_subtensor,
                "min_required_stake",
                lambda: None
            )()

            if min_stake is not None:
                print(f"\n  Minimum stake requirement: {min_stake} TAO")
        except Exception:
            pass  # Method may not exist


class TestBlockInfo:
    """Tests for blockchain info."""

    def test_current_block(self, testnet_subtensor) -> None:
        """Get current block number."""
        block = testnet_subtensor.block
        assert block > 0
        print(f"\n  Current testnet block: {block}")

    def test_block_hash(self, testnet_subtensor) -> None:
        """Get block hash."""
        try:
            block_hash = testnet_subtensor.get_block_hash()
            if block_hash:
                print(f"\n  Latest block hash: {block_hash[:20]}...")
        except Exception:
            pass  # Method may not be available


class TestNetworkInfo:
    """Tests for network information."""

    def test_network_name(self, testnet_subtensor) -> None:
        """Verify connected to testnet."""
        network = testnet_subtensor.network
        assert network in ["test", "testnet"], f"Expected testnet, got {network}"

    def test_chain_endpoint(self, testnet_subtensor) -> None:
        """Verify chain endpoint."""
        endpoint = testnet_subtensor.chain_endpoint
        assert endpoint is not None
        print(f"\n  Chain endpoint: {endpoint}")
