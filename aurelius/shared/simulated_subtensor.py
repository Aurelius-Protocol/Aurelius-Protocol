"""Simulated subtensor for LOCAL_MODE testing with block height simulation."""

import time

import bittensor as bt


class SimulatedMetagraph:
    """Mock metagraph for LOCAL_MODE testing."""

    def __init__(self, netuid: int):
        """
        Initialize simulated metagraph.

        Args:
            netuid: Network UID
        """
        self.netuid = netuid
        self.hotkeys: list[str] = []
        self.uids: list[int] = []
        self.stakes: list[float] = []
        self.block = 0  # Will be updated by SimulatedSubtensor

    def register_miner(self, hotkey: str, uid: int | None = None, stake: float = 0.0) -> None:
        """
        Register a miner in the simulated metagraph.

        Args:
            hotkey: Miner's hotkey
            uid: Optional UID (auto-assigned if None)
            stake: Miner's stake amount
        """
        if hotkey not in self.hotkeys:
            if uid is None:
                uid = len(self.hotkeys)
            self.hotkeys.append(hotkey)
            self.uids.append(uid)
            self.stakes.append(stake)
            bt.logging.info(f"Registered miner {hotkey[:8]}... as UID {uid} in simulated metagraph")

    def __len__(self):
        """Return number of registered miners."""
        return len(self.hotkeys)


class SimulatedSubtensor:
    """
    Mock subtensor for LOCAL_MODE that simulates block height progression.

    This enables testing of window-based reward calculations without requiring
    an actual Bittensor blockchain connection.
    """

    def __init__(self, start_block: int = 10000, block_time: float = 12.0, real_endpoint: str = None):
        """
        Initialize simulated subtensor.

        Args:
            start_block: Starting block height
            block_time: Seconds per block (default: 12.0 to match Bittensor)
            real_endpoint: Optional real blockchain endpoint for hybrid mode (reads real block heights)
        """
        self._start_time = time.time()
        self._start_block = start_block
        self._block_time = block_time
        self._metagraphs = {}
        self._real_endpoint = real_endpoint
        self._real_subtensor = None

        # Hybrid mode: connect to real blockchain for block heights
        if self._real_endpoint:
            try:
                self._real_subtensor = bt.subtensor(network=self._real_endpoint)
                bt.logging.success(f"✓ Hybrid mode enabled: Reading real block heights from {self._real_endpoint}")
                bt.logging.info(f"  Current real blockchain block: {self._real_subtensor.block}")
            except Exception as e:
                bt.logging.warning(f"⚠ Could not connect to real blockchain: {e}")
                bt.logging.info("  Falling back to simulated block heights")
                self._real_subtensor = None

        mode = "hybrid (real blocks)" if self._real_subtensor else "simulated"
        bt.logging.info(f"SimulatedSubtensor initialized ({mode}): start_block={start_block}, block_time={block_time}s")

    @property
    def block(self) -> int:
        """
        Get current simulated block height.

        Block height increases linearly based on elapsed time and configured block_time.

        Returns:
            Current block height
        """
        elapsed = time.time() - self._start_time
        current_block = self._start_block + int(elapsed / self._block_time)
        return current_block

    def metagraph(self, netuid: int) -> SimulatedMetagraph:
        """
        Get or create simulated metagraph for a network.

        Args:
            netuid: Network UID

        Returns:
            SimulatedMetagraph instance
        """
        if netuid not in self._metagraphs:
            self._metagraphs[netuid] = SimulatedMetagraph(netuid)
            bt.logging.debug(f"Created simulated metagraph for netuid {netuid}")

        # Update metagraph block
        self._metagraphs[netuid].block = self.block

        return self._metagraphs[netuid]

    def set_weights(
        self,
        netuid: int,
        wallet,
        uids: list[int],
        weights: list[float],
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        version_key: int = 0,
    ) -> bool:
        """
        Simulate setting weights on chain.

        In LOCAL_MODE, this just logs the weights but doesn't actually
        interact with a blockchain.

        Args:
            netuid: Network UID
            wallet: Wallet (ignored in simulation)
            uids: List of miner UIDs
            weights: List of weights
            wait_for_inclusion: Ignored in simulation
            wait_for_finalization: Ignored in simulation
            version_key: Version key (ignored in simulation)

        Returns:
            Always True (simulated success)
        """
        bt.logging.info(f"[SIMULATED] set_weights called for netuid {netuid} at block {self.block}")

        # Log non-zero weights
        non_zero = [(uids[i], weights[i]) for i in range(len(uids)) if weights[i] > 0]
        if non_zero:
            bt.logging.info(f"[SIMULATED] Setting {len(non_zero)} non-zero weights:")
            for uid, weight in non_zero[:10]:  # Show top 10
                bt.logging.info(f"  UID {uid}: {weight:.6f}")
        else:
            bt.logging.warning("[SIMULATED] No non-zero weights to set")

        # Always succeed in simulation
        return True

    def register_miner(self, netuid: int, hotkey: str, uid: int | None = None, stake: float = 0.0) -> None:
        """
        Register a miner in the simulated metagraph.

        This is a convenience method for testing that allows manually
        registering miners without actual blockchain transactions.

        Args:
            netuid: Network UID
            hotkey: Miner's hotkey
            uid: Optional UID (auto-assigned if None)
            stake: Miner's stake amount
        """
        metagraph = self.metagraph(netuid)
        metagraph.register_miner(hotkey, uid, stake)

    def get_elapsed_time(self) -> float:
        """
        Get elapsed time since subtensor initialization.

        Returns:
            Elapsed time in seconds
        """
        return time.time() - self._start_time

    def get_elapsed_blocks(self) -> int:
        """
        Get number of blocks elapsed since initialization.

        Returns:
            Number of blocks elapsed
        """
        return self.block - self._start_block

    def __str__(self) -> str:
        """String representation."""
        return (
            f"SimulatedSubtensor(current_block={self.block}, "
            f"elapsed={self.get_elapsed_time():.1f}s, "
            f"networks={len(self._metagraphs)})"
        )
