"""Base classes for the experiment framework."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import bittensor as bt

if TYPE_CHECKING:
    from aurelius.validator.core import ValidatorCore


class ExperimentType(Enum):
    """Type of experiment based on data flow direction."""

    PUSH = "push"  # Miners send requests to validator
    PULL = "pull"  # Validator queries miners


@dataclass
class ExperimentConfig:
    """Self-contained experiment configuration."""

    name: str
    experiment_type: ExperimentType
    weight_allocation: float  # Raw weight (normalized at startup)
    enabled: bool = True
    settings: dict[str, Any] = field(default_factory=dict)
    # Load balancing (T076)
    max_concurrent_requests: int = 10  # Max concurrent requests per experiment

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get experiment-specific setting with optional default."""
        return self.settings.get(key, default)


@dataclass
class ExperimentScores:
    """Normalized scores (0-1) per miner for weight calculation.

    Also includes statistics for per-experiment tracking (T048, T059).
    """

    scores: dict[str, float]  # hotkey -> normalized score
    experiment_name: str
    block_height: int
    # Statistics fields (T048)
    total_submissions: int = 0  # Total submissions received in window
    total_accepted: int = 0  # Total submissions accepted in window
    window_start_block: int = 0  # Block height when current window started
    # Pull experiment statistics (T059)
    queries_sent: int = 0  # Number of queries sent to miners
    responses_received: int = 0  # Number of successful responses
    timeouts: int = 0  # Number of timeout responses


class Experiment(ABC):
    """Base class for all experiments.

    Each experiment implements a specific way of evaluating miners.
    Experiments can be either push-based (miners send requests) or
    pull-based (validator queries miners).

    Subclasses must define:
        NAME: str - Unique identifier for the experiment
        TYPE: ExperimentType - Whether push or pull based
    """

    NAME: str  # Class attribute, e.g., "prompt"
    TYPE: ExperimentType  # Class attribute, e.g., ExperimentType.PUSH

    def __init__(self, core: ValidatorCore, config: ExperimentConfig):
        """Initialize experiment with core infrastructure and config.

        Args:
            core: ValidatorCore providing shared infrastructure
            config: ExperimentConfig with settings for this experiment
        """
        self.core = core
        self.config = config
        self._lock = threading.Lock()
        self._started = False

    def setting(self, key: str, default: Any = None) -> Any:
        """Convenience method to access experiment-specific settings."""
        return self.config.get_setting(key, default)

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return self.config.name

    @property
    def weight_allocation(self) -> float:
        """Return the weight allocation for this experiment."""
        return self.config.weight_allocation

    @property
    def is_enabled(self) -> bool:
        """Return whether this experiment is enabled."""
        return self.config.enabled

    @abstractmethod
    def start(self) -> None:
        """Start the experiment (register handlers or start query loop)."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the experiment gracefully."""
        pass

    @abstractmethod
    def calculate_scores(self, current_block: int) -> ExperimentScores:
        """Return normalized scores (0-1) for all participating miners.

        Args:
            current_block: Current blockchain block height

        Returns:
            ExperimentScores with normalized scores per miner
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Return experiment-specific statistics for logging/monitoring."""
        pass


class PushExperiment(Experiment):
    """Base for experiments where miners send requests to validator.

    Push experiments register synapse handlers with the axon to receive
    and process incoming requests from miners.
    """

    TYPE = ExperimentType.PUSH

    def __init__(self, core: ValidatorCore, config: ExperimentConfig):
        super().__init__(core, config)
        self._handlers_attached = False

    def start(self) -> None:
        """Register synapse handlers with axon."""
        with self._lock:
            if self._started:
                bt.logging.warning(f"Experiment {self.name} already started")
                return

            # Register handlers with axon
            self.core.axon.attach(
                forward_fn=self._create_forward_handler(),
                blacklist_fn=self._create_blacklist_handler(),
                priority_fn=self._create_priority_handler(),
                verify_fn=self._create_verify_handler(),
            )

            self._handlers_attached = True
            self._started = True
        bt.logging.info(f"Push experiment '{self.name}' started - handlers attached to axon")

    def stop(self) -> None:
        """Stop the push experiment."""
        with self._lock:
            self._started = False
            # Note: Bittensor doesn't support detaching handlers
            # Log warning that handlers remain attached
            if self._handlers_attached:
                bt.logging.warning(
                    f"Experiment '{self.name}' stopped but handlers remain attached to axon"
                )
        bt.logging.info(f"Push experiment '{self.name}' stopped")

    @abstractmethod
    def _create_forward_handler(self) -> Callable:
        """Return the forward handler for this experiment's synapse.

        Returns:
            Callable that processes incoming synapses
        """
        pass

    def _create_blacklist_handler(self) -> Callable:
        """Return the blacklist handler. Default accepts all requests.

        Returns:
            Callable that returns (should_blacklist: bool, reason: str)
        """

        def default_blacklist(synapse) -> tuple[bool, str]:
            return False, ""

        return default_blacklist

    def _create_priority_handler(self) -> Callable:
        """Return the priority handler (T074 - dynamic priority based on allocation).

        Priority = 1.0 + (allocation_percentage / 100), so higher allocation
        experiments get higher priority.

        Returns:
            Callable that returns priority value (higher = more priority)
        """
        # Calculate priority based on weight allocation
        allocation_percentage = self.weight_allocation * 100  # Convert to percentage
        base_priority = 1.0 + (allocation_percentage / 100)

        def allocation_based_priority(synapse) -> float:
            return base_priority

        return allocation_based_priority

    def _create_verify_handler(self) -> Callable:
        """Return the verify handler. Default does nothing.

        Returns:
            Callable that verifies incoming requests
        """

        def default_verify(synapse) -> None:
            pass

        return default_verify


class PullExperiment(Experiment):
    """Base for experiments where validator queries miners.

    Pull experiments run a background loop that periodically queries
    miners and processes their responses.

    Features (T055, T056, T058, T059):
    - Registration-aware miner selection
    - Configurable query intervals from ExperimentDefinition
    - Timeout handling and non-response recording
    - Pull experiment statistics tracking
    """

    TYPE = ExperimentType.PULL

    # Configuration (can be overridden via settings or ExperimentDefinition)
    DEFAULT_QUERY_INTERVAL_SECONDS = 300
    DEFAULT_MINERS_PER_ROUND = 10
    DEFAULT_QUERY_TIMEOUT = 30.0

    def __init__(self, core: ValidatorCore, config: ExperimentConfig):
        super().__init__(core, config)
        self._stop_event = threading.Event()
        self._query_thread: threading.Thread | None = None
        # Pull experiment statistics (T059)
        self._queries_sent = 0
        self._responses_received = 0
        self._timeouts = 0
        self._non_responses = 0

    @property
    def query_interval_seconds(self) -> int:
        """How often to run queries."""
        return self.setting("query_interval_seconds", self.DEFAULT_QUERY_INTERVAL_SECONDS)

    @property
    def miners_per_round(self) -> int:
        """How many miners to query per round."""
        return self.setting("miners_per_round", self.DEFAULT_MINERS_PER_ROUND)

    @property
    def query_timeout(self) -> float:
        """Timeout for queries to miners."""
        return self.setting("query_timeout", self.DEFAULT_QUERY_TIMEOUT)

    def start(self) -> None:
        """Start the query loop in a background thread."""
        with self._lock:
            # Check if thread crashed but _started is still True
            if self._started and self._query_thread and not self._query_thread.is_alive():
                bt.logging.warning(f"Experiment '{self.name}' thread crashed, restarting")
                self._started = False
                self._query_thread = None

            if self._started:
                bt.logging.warning(f"Experiment {self.name} already started")
                return

            self._stop_event.clear()
            self._query_thread = threading.Thread(
                target=self._query_loop,
                daemon=True,
                name=f"PullExperiment-{self.name}",
            )
            self._query_thread.start()
            self._started = True
        bt.logging.info(
            f"Pull experiment '{self.name}' started - querying {self.miners_per_round} "
            f"miners every {self.query_interval_seconds}s"
        )

    def stop(self) -> bool:
        """Stop the query loop gracefully.

        Returns:
            True if stopped successfully, False if thread did not stop within timeout
        """
        with self._lock:
            self._stop_event.set()

        success = True
        if self._query_thread is not None:
            self._query_thread.join(timeout=30)
            if self._query_thread.is_alive():
                bt.logging.error(f"Experiment '{self.name}' thread did not stop within timeout")
                success = False

        with self._lock:
            self._query_thread = None
            self._started = False
        bt.logging.info(f"Pull experiment '{self.name}' stopped")
        return success

    def _query_loop(self) -> None:
        """Background loop that queries miners periodically."""
        while not self._stop_event.is_set():
            try:
                miners = self._select_miners()
                if miners:
                    results = self._query_miners(miners)
                    self._process_results(results)
            except Exception as e:
                bt.logging.error(f"Error in {self.name} query loop: {e}")
                import traceback
                bt.logging.debug(f"Traceback: {traceback.format_exc()}")

            # Sleep in small increments to allow quick shutdown
            sleep_remaining = self.query_interval_seconds
            while sleep_remaining > 0 and not self._stop_event.is_set():
                time.sleep(min(1.0, sleep_remaining))
                sleep_remaining -= 1.0

    def _select_miners(self) -> list:
        """Select miners to query this round (T055 - registration-aware).

        Default implementation selects random miners from metagraph,
        filtered by registration status for non-default experiments.

        Returns:
            List of axon info for selected miners
        """
        import random

        miners = self.core.get_miners()
        if not miners:
            return []

        # Filter by registration for non-prompt experiments (T055)
        if self.name != "prompt" and hasattr(self.core, "experiment_client"):
            registered_hotkeys = set(
                self.core.experiment_client.get_registered_miners(self.name)
            )
            if registered_hotkeys:
                # Filter to only registered miners
                miners = [
                    m for m in miners
                    if hasattr(m, "hotkey") and m.hotkey in registered_hotkeys
                ]
                bt.logging.debug(
                    f"Pull experiment '{self.name}': {len(miners)} registered miners available"
                )

        if not miners:
            bt.logging.warning(
                f"Pull experiment '{self.name}': no registered miners available"
            )
            return []

        # Select up to miners_per_round random miners
        count = min(self.miners_per_round, len(miners))
        return random.sample(miners, count)

    def _query_miners(self, miners: list) -> list:
        """Query the selected miners with the experiment's synapse (T058).

        Handles timeouts and records non-responses for scoring.

        Args:
            miners: List of miner axon info to query

        Returns:
            List of synapse responses (includes failed/timeout responses)
        """
        synapse = self._create_query_synapse()
        self._queries_sent += len(miners)

        try:
            responses = self.core.dendrite.query(
                axons=miners,
                synapse=synapse,
                timeout=self.query_timeout,
            )

            # Track responses and timeouts (T058)
            for resp in responses:
                if resp is None:
                    self._non_responses += 1
                elif hasattr(resp, "dendrite") and resp.dendrite:
                    # Check for timeout based on dendrite status
                    status_code = getattr(resp.dendrite, "status_code", None)
                    if status_code == 408:  # Timeout
                        self._timeouts += 1
                    else:
                        self._responses_received += 1
                else:
                    self._responses_received += 1

            return responses

        except Exception as e:
            bt.logging.error(f"Error querying miners in {self.name}: {e}")
            self._non_responses += len(miners)
            return []

    @abstractmethod
    def _create_query_synapse(self) -> bt.Synapse:
        """Create the synapse to send to miners.

        Returns:
            Synapse instance to query miners with
        """
        pass

    @abstractmethod
    def _process_results(self, results: list) -> None:
        """Process responses from miners.

        Args:
            results: List of synapse responses from miners
        """
        pass

    def get_stats(self) -> dict:
        """Return pull experiment statistics (T059).

        Returns:
            Dictionary with pull experiment statistics
        """
        return {
            "experiment_type": "pull",
            "queries_sent": self._queries_sent,
            "responses_received": self._responses_received,
            "timeouts": self._timeouts,
            "non_responses": self._non_responses,
            "response_rate": (
                self._responses_received / self._queries_sent
                if self._queries_sent > 0
                else 0.0
            ),
            "query_interval_seconds": self.query_interval_seconds,
            "miners_per_round": self.miners_per_round,
            "query_timeout": self.query_timeout,
        }
