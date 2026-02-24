"""Experiment sync client for fetching experiment definitions from central API."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt
import requests
from opentelemetry.trace import SpanKind

from aurelius.shared.circuit_breaker import CircuitBreakerConfig, get_circuit_breaker
from aurelius.shared.config import Config
from aurelius.shared.telemetry.otel_setup import get_tracer


@dataclass
class ExperimentDefinition:
    """Experiment definition from central API.

    Attributes:
        id: Unique identifier (e.g., "prompt", "jailbreak-v1")
        name: Human-readable name
        version: Schema version for migrations
        experiment_type: "push" or "pull"
        scoring_type: "danger", "binary", "numeric", "custom"
        status: "active", "inactive", "deprecated"
        deprecated_at: ISO timestamp if deprecated
        thresholds: Scoring thresholds (e.g., {"acceptance": 0.3})
        rate_limit_requests: Max requests per window
        rate_limit_window_hours: Window size in hours
        novelty_threshold: Minimum novelty for acceptance (0.0-1.0)
        pull_interval_seconds: Query interval for pull experiments
        pull_timeout_seconds: Per-miner timeout for pull experiments
        settings: Experiment-specific parameters
        created_at: ISO timestamp
        updated_at: ISO timestamp
    """

    id: str
    name: str
    version: int
    experiment_type: str  # "push" | "pull"
    scoring_type: str  # "danger" | "binary" | "numeric" | "custom"
    status: str  # "active" | "inactive" | "deprecated"
    deprecated_at: str | None
    thresholds: dict[str, float]
    rate_limit_requests: int
    rate_limit_window_hours: int
    novelty_threshold: float
    pull_interval_seconds: int | None
    pull_timeout_seconds: int | None
    settings: dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentDefinition:
        """Create an ExperimentDefinition from a dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            experiment_type=data["experiment_type"],
            scoring_type=data["scoring_type"],
            status=data["status"],
            deprecated_at=data.get("deprecated_at"),
            thresholds=data.get("thresholds", {}),
            rate_limit_requests=data.get("rate_limit_requests", 100),
            rate_limit_window_hours=data.get("rate_limit_window_hours", 1),
            novelty_threshold=data.get("novelty_threshold", 0.02),
            pull_interval_seconds=data.get("pull_interval_seconds"),
            pull_timeout_seconds=data.get("pull_timeout_seconds"),
            settings=data.get("settings", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "experiment_type": self.experiment_type,
            "scoring_type": self.scoring_type,
            "status": self.status,
            "deprecated_at": self.deprecated_at,
            "thresholds": self.thresholds,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_hours": self.rate_limit_window_hours,
            "novelty_threshold": self.novelty_threshold,
            "pull_interval_seconds": self.pull_interval_seconds,
            "pull_timeout_seconds": self.pull_timeout_seconds,
            "settings": self.settings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class MinerRegistration:
    """Miner registration for an experiment.

    Attributes:
        miner_hotkey: SS58 address (48 chars)
        experiment_id: Target experiment
        status: "active" or "withdrawn"
        registered_at: ISO timestamp
        withdrawn_at: ISO timestamp if withdrawn
    """

    miner_hotkey: str
    experiment_id: str
    status: str  # "active" | "withdrawn"
    registered_at: str
    withdrawn_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MinerRegistration:
        """Create a MinerRegistration from a dictionary."""
        return cls(
            miner_hotkey=data["miner_hotkey"],
            experiment_id=data["experiment_id"],
            status=data["status"],
            registered_at=data["registered_at"],
            withdrawn_at=data.get("withdrawn_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "miner_hotkey": self.miner_hotkey,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "registered_at": self.registered_at,
            "withdrawn_at": self.withdrawn_at,
        }


@dataclass
class RewardAllocation:
    """Reward distribution configuration.

    Attributes:
        allocations: Map of experiment_id to percentage (0-100)
        burn_percentage: Percentage to burn address (0-100)
        redistribute_unused: Whether to redistribute inactive allocations
        version: Config version
        updated_at: ISO timestamp
    """

    allocations: dict[str, float]
    burn_percentage: float
    redistribute_unused: bool
    version: int
    updated_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardAllocation:
        """Create a RewardAllocation from a dictionary."""
        return cls(
            allocations=data.get("allocations", {}),
            burn_percentage=data.get("burn_percentage", 0.0),
            redistribute_unused=data.get("redistribute_unused", True),
            version=data.get("version", 1),
            updated_at=data.get("updated_at", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "allocations": self.allocations,
            "burn_percentage": self.burn_percentage,
            "redistribute_unused": self.redistribute_unused,
            "version": self.version,
            "updated_at": self.updated_at,
        }

    def validate(self) -> tuple[bool, str]:
        """Validate that allocations sum to 100%.

        Returns:
            Tuple of (is_valid, error_message)
        """
        total = sum(self.allocations.values()) + self.burn_percentage
        if abs(total - 100.0) > 0.01:
            return False, f"Allocations sum to {total}%, must be 100%"
        return True, ""


@dataclass
class ExperimentSyncResponse:
    """Response from GET /api/experiments.

    Attributes:
        experiments: List of experiment definitions
        registrations: List of miner registrations
        reward_allocation: Reward distribution configuration
        prompts: Per-experiment prompt overrides {experiment_id: {prompt_key: prompt_text}}
        sync_version: Monotonic version for change detection
        server_time: Server timestamp
    """

    experiments: list[ExperimentDefinition]
    registrations: list[MinerRegistration]
    reward_allocation: RewardAllocation
    prompts: dict[str, dict[str, str]]
    sync_version: int
    server_time: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentSyncResponse:
        """Create an ExperimentSyncResponse from a dictionary."""
        experiments = [
            ExperimentDefinition.from_dict(e) for e in data.get("experiments", [])
        ]
        registrations = [
            MinerRegistration.from_dict(r) for r in data.get("registrations", [])
        ]
        reward_allocation = RewardAllocation.from_dict(
            data.get("reward_allocation", {})
        )
        prompts = data.get("prompts", {})
        return cls(
            experiments=experiments,
            registrations=registrations,
            reward_allocation=reward_allocation,
            prompts=prompts,
            sync_version=data.get("sync_version", 0),
            server_time=data.get("server_time", ""),
        )


# Default experiment definition for backward compatibility
# Used as fallback when API is unavailable or cache is empty
DEFAULT_PROMPT_EXPERIMENT = ExperimentDefinition(
    id="prompt",
    name="Dangerous Prompt Detection",
    version=1,
    experiment_type="push",
    scoring_type="danger",
    status="active",
    deprecated_at=None,
    thresholds={
        "acceptance": Config.DANGER_THRESHOLD,
        "single_category": Config.SINGLE_CATEGORY_THRESHOLD,
    },
    rate_limit_requests=Config.RATE_LIMIT_REQUESTS,
    rate_limit_window_hours=int(Config.RATE_LIMIT_WINDOW_HOURS),
    novelty_threshold=Config.MIN_NOVELTY_THRESHOLD,
    pull_interval_seconds=None,
    pull_timeout_seconds=None,
    settings={},
    created_at="2026-01-01T00:00:00Z",
    updated_at="2026-01-01T00:00:00Z",
)

# Default moral-reasoning experiment definition for backward compatibility
# Used as fallback when API is unavailable or cache is empty
DEFAULT_MORAL_REASONING_EXPERIMENT = ExperimentDefinition(
    id="moral-reasoning",
    name="Moral Reasoning Evaluation",
    version=1,
    experiment_type="push",
    scoring_type="numeric",
    status="active",
    deprecated_at=None,
    thresholds={},
    rate_limit_requests=Config.RATE_LIMIT_REQUESTS,
    rate_limit_window_hours=int(Config.RATE_LIMIT_WINDOW_HOURS),
    novelty_threshold=Config.MIN_NOVELTY_THRESHOLD,
    pull_interval_seconds=None,
    pull_timeout_seconds=None,
    settings={},
    created_at="2026-01-01T00:00:00Z",
    updated_at="2026-01-01T00:00:00Z",
)

# Core experiment defaults - always available even without central API
CORE_EXPERIMENT_DEFAULTS: dict[str, ExperimentDefinition] = {
    DEFAULT_PROMPT_EXPERIMENT.id: DEFAULT_PROMPT_EXPERIMENT,
    DEFAULT_MORAL_REASONING_EXPERIMENT.id: DEFAULT_MORAL_REASONING_EXPERIMENT,
}


class ExperimentClient:
    """Client for syncing experiment definitions from central API.

    This client follows the same pattern as NoveltyClient:
    - Circuit breaker for API resilience
    - Local JSON cache for offline operation
    - Periodic background sync
    """

    def __init__(
        self,
        api_endpoint: str | None = None,
        cache_path: str | None = None,
        timeout: int = 30,
        wallet: bt.Wallet | None = None,
    ):
        """Initialize experiment client.

        Args:
            api_endpoint: Base URL for experiment API
            cache_path: Path to local cache file
            timeout: Request timeout in seconds
            wallet: Bittensor wallet for SR25519 header-based signing (set after init if needed)
        """
        self.api_endpoint = api_endpoint or Config.EXPERIMENT_API_ENDPOINT
        self.cache_path = cache_path or Config.EXPERIMENT_CACHE_PATH
        self.timeout = timeout
        self.wallet: bt.Wallet | None = wallet
        self._tracer = get_tracer("aurelius.experiments") if Config.TELEMETRY_ENABLED else None

        # Initialize circuit breaker for API resilience
        self._circuit_breaker = get_circuit_breaker(
            "experiment-api",
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60.0,
                half_open_max_calls=1,
                success_threshold=2,
            ),
        )

        # Internal state
        self._cache: dict[str, ExperimentDefinition] = {}
        self._registrations: dict[str, list[str]] = {}  # experiment_id -> list of hotkeys
        self._reward_allocation: RewardAllocation | None = None
        self._prompts: dict[str, dict[str, str]] = {}  # experiment_id -> {prompt_key: prompt_text}
        self._sync_version: int = 0
        self._synced_at: str | None = None
        self._lock = threading.Lock()

        # Sync loop state
        self._sync_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Load cache on init
        self._load_cache()

        if self.api_endpoint:
            bt.logging.info(f"Experiment client: API endpoint at {self.api_endpoint}")
            if not self.wallet:
                bt.logging.warning(
                    "Experiment client: No wallet set — experiment sync disabled. "
                    "Attach a wallet for SR25519-authenticated experiment sync."
                )
        else:
            bt.logging.warning("Experiment client: No API endpoint configured")

    def _build_auth_headers(self) -> dict[str, str]:
        """Build authentication headers with SR25519 signing.

        Follows the same pattern as NoveltyClient._build_auth_headers().
        """
        headers: dict[str, str] = {}

        if self.wallet:
            try:
                timestamp = int(time.time())
                hotkey = self.wallet.hotkey.ss58_address
                message = f"aurelius-submission:{timestamp}:{hotkey}"
                signature = self.wallet.hotkey.sign(message.encode()).hex()
                headers["X-Validator-Hotkey"] = hotkey
                headers["X-Signature"] = signature
                headers["X-Timestamp"] = str(timestamp)
            except Exception as e:
                bt.logging.error(
                    f"Failed to sign experiment sync request: {e} "
                    f"(wallet path: {self.wallet.path if hasattr(self.wallet, 'path') else 'unknown'})"
                )

        return headers

    def _load_cache(self) -> bool:
        """Load experiment definitions from local cache file.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not self.cache_path or not os.path.exists(self.cache_path):
            bt.logging.debug("No experiment cache file found, using defaults")
            self._apply_defaults()
            return False

        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)

            with self._lock:
                # Load experiments
                self._cache = {}
                for exp_id, exp_data in data.get("experiments", {}).items():
                    self._cache[exp_id] = ExperimentDefinition.from_dict(exp_data)

                # Load registrations
                self._registrations = data.get("registrations", {})

                # Load reward allocation
                if "reward_allocation" in data:
                    self._reward_allocation = RewardAllocation.from_dict(
                        data["reward_allocation"]
                    )

                # Load prompts
                self._prompts = data.get("prompts", {})

                self._sync_version = data.get("sync_version", 0)
                self._synced_at = data.get("synced_at")

            # Record cache hit in telemetry
            if self._tracer:
                with self._tracer.start_as_current_span(
                    "experiment.cache_hit",
                    attributes={
                        "experiment.cache_path": self.cache_path,
                        "experiment.cached_count": len(self._cache),
                        "experiment.sync_version": self._sync_version,
                    },
                ):
                    pass

            bt.logging.info(
                f"Loaded {len(self._cache)} experiments from cache "
                f"(version {self._sync_version}, synced {self._synced_at})"
            )
            return True

        except Exception as e:
            bt.logging.warning(f"Failed to load experiment cache: {e}")
            self._apply_defaults()
            return False

    def _apply_defaults(self) -> None:
        """Apply default experiment definition when cache is unavailable."""
        with self._lock:
            self._cache = dict(CORE_EXPERIMENT_DEFAULTS)
            self._registrations = {}  # Empty - all miners auto-registered for core experiments
            self._reward_allocation = RewardAllocation(
                allocations={"moral-reasoning": 100.0, "prompt": 0.0},
                burn_percentage=0.0,
                redistribute_unused=True,
                version=1,
                updated_at="",
            )
        bt.logging.info("Applied default experiment configuration (moral-reasoning + prompt)")

    def _save_cache(self) -> bool:
        """Save experiment definitions to local cache file.

        Returns:
            True if cache was saved successfully, False otherwise
        """
        if not self.cache_path:
            return False

        try:
            with self._lock:
                data = {
                    "sync_version": self._sync_version,
                    "synced_at": self._synced_at,
                    "experiments": {
                        exp_id: exp.to_dict() for exp_id, exp in self._cache.items()
                    },
                    "registrations": self._registrations,
                    "reward_allocation": (
                        self._reward_allocation.to_dict()
                        if self._reward_allocation
                        else None
                    ),
                    "prompts": self._prompts,
                }

            # Write atomically using temp file
            temp_path = f"{self.cache_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.cache_path)

            bt.logging.debug(f"Saved experiment cache to {self.cache_path}")
            return True

        except Exception as e:
            bt.logging.error(f"Failed to save experiment cache: {e}")
            return False

    def sync(self) -> bool:
        """Sync experiment definitions from central API.

        Returns:
            True if sync was successful, False otherwise
        """
        if not self.api_endpoint:
            bt.logging.debug("Experiment sync skipped: no API endpoint configured")
            return False

        # Check circuit breaker first
        if not self._circuit_breaker.can_execute():
            bt.logging.debug(
                f"Experiment circuit breaker OPEN - skipping sync "
                f"(retry in {self._circuit_breaker.get_time_until_retry():.1f}s)"
            )
            return False

        start_time = time.time()

        # Wrap with tracing span if enabled
        if self._tracer:
            with self._tracer.start_as_current_span(
                "experiment.sync",
                kind=SpanKind.CLIENT,
                attributes={
                    "experiment.endpoint": self.api_endpoint,
                    "experiment.sync_version": self._sync_version,
                },
            ) as span:
                result = self._do_sync()
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", round(duration_ms, 2))
                span.set_attribute("experiment.success", result)
                if result:
                    span.set_attribute("experiment.new_sync_version", self._sync_version)
                return result
        else:
            return self._do_sync()

    def _do_sync(self) -> bool:
        """Internal method to perform the sync API call."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            headers.update(self._build_auth_headers())

            # Use conditional fetch if we have a sync version
            if self._sync_version > 0:
                headers["If-None-Match"] = f'"{self._sync_version}"'

            response = requests.get(
                self.api_endpoint,
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 304:
                # Not modified - cache is current
                self._circuit_breaker.record_success()
                bt.logging.debug("Experiment sync: no changes (304)")
                return True

            if response.status_code == 200:
                data = response.json()

                # Full sync response format (API always returns sync_version)
                sync_response = ExperimentSyncResponse.from_dict(data)
                with self._lock:
                    self._cache = {exp.id: exp for exp in sync_response.experiments}
                    self._registrations = {}
                    for reg in sync_response.registrations:
                        if reg.status == "active":
                            if reg.experiment_id not in self._registrations:
                                self._registrations[reg.experiment_id] = []
                            self._registrations[reg.experiment_id].append(
                                reg.miner_hotkey
                            )
                    self._reward_allocation = sync_response.reward_allocation
                    self._prompts = sync_response.prompts
                    self._sync_version = sync_response.sync_version
                    self._synced_at = sync_response.server_time

                # Save to cache
                self._save_cache()

                self._circuit_breaker.record_success()
                bt.logging.info(
                    f"Experiment sync complete: {len(self._cache)} experiments, "
                    f"version {self._sync_version}"
                )
                return True

            elif response.status_code == 401:
                bt.logging.warning("Experiment sync: unauthorized (check wallet signing)")
                return False

            elif response.status_code in {502, 503, 504}:
                self._circuit_breaker.record_failure()
                bt.logging.debug("Experiment sync: service not available")
                return False

            else:
                if response.status_code >= 500:
                    self._circuit_breaker.record_failure()
                bt.logging.warning(
                    f"Experiment sync failed: HTTP {response.status_code} - {response.text}"
                )
                return False

        except requests.Timeout:
            self._circuit_breaker.record_failure()
            bt.logging.warning(f"Experiment sync timed out after {self.timeout}s")
            return False
        except requests.RequestException as e:
            self._circuit_breaker.record_failure()
            bt.logging.warning(f"Experiment sync request failed: {e}")
            return False
        except Exception as e:
            self._circuit_breaker.record_failure()
            bt.logging.error(f"Experiment sync unexpected error: {e}")
            return False

    def get_experiment(self, experiment_id: str) -> ExperimentDefinition | None:
        """Get an experiment definition by ID.

        Falls back to CORE_EXPERIMENT_DEFAULTS for core experiments
        even if the cache was replaced by a sync response without them.

        Args:
            experiment_id: The experiment ID to look up

        Returns:
            ExperimentDefinition if found, None otherwise
        """
        with self._lock:
            result = self._cache.get(experiment_id)
        if result is None:
            result = CORE_EXPERIMENT_DEFAULTS.get(experiment_id)
        return result

    def get_active_experiments(self) -> list[ExperimentDefinition]:
        """Get all active experiment definitions.

        Returns:
            List of active ExperimentDefinition instances
        """
        with self._lock:
            return [exp for exp in self._cache.values() if exp.status == "active"]

    def get_registered_miners(self, experiment_id: str) -> list[str]:
        """Get list of registered miner hotkeys for an experiment.

        Note: The "prompt" experiment has implicit registration for all miners.

        Args:
            experiment_id: The experiment ID

        Returns:
            List of miner hotkey strings
        """
        if experiment_id == "prompt":
            # All miners are auto-registered for the default experiment
            return []  # Empty list means "all miners"

        with self._lock:
            return self._registrations.get(experiment_id, [])

    def is_miner_registered(self, experiment_id: str, miner_hotkey: str) -> bool:
        """Check if a miner is registered for an experiment.

        Note: The "prompt" experiment has implicit registration for all miners.

        Args:
            experiment_id: The experiment ID
            miner_hotkey: The miner's hotkey

        Returns:
            True if registered, False otherwise
        """
        if experiment_id in ("prompt", "moral-reasoning"):
            # All miners are auto-registered for default experiments
            return True

        with self._lock:
            registered = self._registrations.get(experiment_id, [])
            return miner_hotkey in registered

    def get_reward_allocation(self) -> RewardAllocation | None:
        """Get the current reward allocation configuration.

        Returns:
            RewardAllocation if available, None otherwise
        """
        with self._lock:
            return self._reward_allocation

    def get_prompt(
        self, experiment_id: str, prompt_key: str, default: str | None = None
    ) -> str | None:
        """Get a dynamic prompt override for an experiment.

        Returns the override prompt text if it exists, otherwise the default.

        Args:
            experiment_id: The experiment to get the prompt for.
            prompt_key: The prompt key (e.g. "response_system_prompt").
            default: Value to return if no override exists.

        Returns:
            The prompt text, or default if no override found.
        """
        with self._lock:
            return self._prompts.get(experiment_id, {}).get(prompt_key, default)

    def start_sync_loop(self) -> None:
        """Start the background sync loop.

        The sync loop runs every EXPERIMENT_SYNC_INTERVAL seconds and
        fetches the latest experiment definitions from the central API.
        """
        with self._lock:
            if self._sync_thread is not None and self._sync_thread.is_alive():
                bt.logging.warning("Experiment sync loop already running")
                return

            self._stop_event.clear()
            self._sync_thread = threading.Thread(
                target=self._sync_loop,
                daemon=True,
                name="ExperimentSyncLoop",
            )
            self._sync_thread.start()
        bt.logging.info(
            f"Started experiment sync loop (interval: {Config.EXPERIMENT_SYNC_INTERVAL}s)"
        )

    def stop_sync_loop(self) -> bool:
        """Stop the background sync loop.

        Returns:
            True if stopped successfully, False if thread did not stop within timeout
        """
        self._stop_event.set()

        if self._sync_thread is not None:
            self._sync_thread.join(timeout=30)
            if self._sync_thread.is_alive():
                bt.logging.error("Experiment sync loop did not stop within timeout")
                return False

        with self._lock:
            self._sync_thread = None

        bt.logging.info("Stopped experiment sync loop")
        return True

    def _sync_loop(self) -> None:
        """Background loop that syncs experiment definitions periodically."""
        # Initial delay with jitter to prevent thundering herd
        initial_delay = random.uniform(0, 30)
        bt.logging.debug(f"Experiment sync: initial delay {initial_delay:.1f}s")

        sleep_remaining = initial_delay
        while sleep_remaining > 0 and not self._stop_event.is_set():
            time.sleep(min(1.0, sleep_remaining))
            sleep_remaining -= 1.0

        while not self._stop_event.is_set():
            try:
                self.sync()
            except Exception as e:
                bt.logging.error(f"Error in experiment sync loop: {e}")
                import traceback
                bt.logging.debug(f"Traceback: {traceback.format_exc()}")

            # Sleep in small increments to allow quick shutdown
            sleep_remaining = Config.EXPERIMENT_SYNC_INTERVAL
            while sleep_remaining > 0 and not self._stop_event.is_set():
                time.sleep(min(1.0, sleep_remaining))
                sleep_remaining -= 1.0


# Singleton instance
_experiment_client: ExperimentClient | None = None


def get_experiment_client() -> ExperimentClient:
    """Get singleton experiment client instance."""
    global _experiment_client
    if _experiment_client is None:
        _experiment_client = ExperimentClient()
    return _experiment_client
