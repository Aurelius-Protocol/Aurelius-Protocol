"""Experiment manager for coordinating multiple experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import bittensor as bt

from aurelius.shared.config import Config, ConfigurationError
from aurelius.shared.experiment_client import ExperimentDefinition, get_experiment_client
from aurelius.shared.protocol import PromptSynapse
from aurelius.validator.experiments.base import Experiment, ExperimentScores

if TYPE_CHECKING:
    from aurelius.validator.core import ValidatorCore


@dataclass
class RoutingResult:
    """Result of experiment routing decision.

    Attributes:
        experiment: The experiment to route to, or None if rejected
        rejection_reason: Reason for rejection if experiment is None
        available_experiments: List of available experiment IDs (on rejection)
        registration_required: True if miner needs to register
    """

    experiment: Experiment | None
    rejection_reason: str | None = None
    available_experiments: list[str] | None = None
    registration_required: bool | None = None


class ExperimentManager:
    """Manages experiment lifecycle and score collection.

    The ExperimentManager is responsible for:
    - Registering experiments
    - Starting and stopping all experiments
    - Collecting scores from enabled experiments for weight calculation
    - Per-experiment rate limiting (T046)
    """

    def __init__(self, core: ValidatorCore):
        """Initialize the experiment manager.

        Args:
            core: ValidatorCore providing shared infrastructure
        """
        self.core = core
        self.experiments: dict[str, Experiment] = {}
        self._experiment_client = None  # Lazy init
        # Per-experiment rate limiting (T046)
        # Structure: {experiment_id: {hotkey: [(timestamp1, ...), ...]}}
        self._rate_limit_windows: dict[str, dict[str, list[float]]] = {}
        import threading
        self._rate_limit_lock = threading.Lock()
        # Per-experiment concurrency tracking (T075)
        # Structure: {experiment_id: current_count}
        self._concurrent_requests: dict[str, int] = {}
        self._concurrency_lock = threading.Lock()
        # Monopolization prevention constants (T077)
        self._max_thread_pool_percentage = 0.8  # Cap at 80% of thread pool
        self._default_max_concurrent = 10  # Default max concurrent per experiment

    @property
    def experiment_client(self):
        """Get the experiment client (lazy initialization)."""
        if self._experiment_client is None:
            # Try to get from core if available, otherwise use singleton
            if hasattr(self.core, "experiment_client") and self.core.experiment_client:
                self._experiment_client = self.core.experiment_client
            else:
                self._experiment_client = get_experiment_client()
        return self._experiment_client

    def register(self, experiment: Experiment) -> None:
        """Register an experiment.

        Args:
            experiment: Experiment instance to register

        Raises:
            ValueError: If an experiment with the same name is already registered
        """
        if experiment.name in self.experiments:
            raise ValueError(f"Experiment '{experiment.name}' already registered")

        self.experiments[experiment.name] = experiment
        bt.logging.info(
            f"Registered experiment '{experiment.name}' "
            f"(type={experiment.config.experiment_type.value}, "
            f"weight={experiment.weight_allocation:.2f}, "
            f"enabled={experiment.is_enabled})"
        )

    def unregister(self, name: str) -> None:
        """Unregister an experiment by name.

        Args:
            name: Name of the experiment to unregister

        Raises:
            ValueError: If no experiment with that name exists
        """
        if name not in self.experiments:
            raise ValueError(f"Experiment '{name}' not registered")

        experiment = self.experiments.pop(name)
        if experiment._started:
            experiment.stop()
        bt.logging.info(f"Unregistered experiment '{name}'")

    def get_experiment(self, name: str) -> Experiment | None:
        """Get an experiment by name.

        Args:
            name: Name of the experiment

        Returns:
            Experiment instance if found, None otherwise
        """
        return self.experiments.get(name)

    def get_enabled_experiments(self) -> list[Experiment]:
        """Get all enabled experiments.

        Returns:
            List of enabled Experiment instances
        """
        return [exp for exp in self.experiments.values() if exp.is_enabled]

    def start_all(self) -> dict[str, bool]:
        """Start all enabled experiments (T060).

        Handles both push and pull experiments:
        - Push experiments: Register handlers with axon
        - Pull experiments: Start background query thread

        Returns:
            Dictionary mapping experiment names to start success status
        """
        enabled = self.get_enabled_experiments()
        if not enabled:
            bt.logging.warning("No enabled experiments to start")
            return {}

        # Count experiment types
        push_count = sum(1 for e in enabled if getattr(e, "TYPE", None) and e.TYPE.value == "push")
        pull_count = sum(1 for e in enabled if getattr(e, "TYPE", None) and e.TYPE.value == "pull")

        bt.logging.info(
            f"Starting {len(enabled)} enabled experiments "
            f"({push_count} push, {pull_count} pull)..."
        )

        results = {}
        for exp in enabled:
            try:
                exp.start()
                results[exp.name] = True
                # Log type-specific info (T060)
                exp_type = getattr(exp, "TYPE", None)
                if exp_type and exp_type.value == "pull":
                    bt.logging.info(
                        f"  Started pull experiment '{exp.name}' "
                        f"(query thread active)"
                    )
                else:
                    bt.logging.info(f"  Started experiment '{exp.name}'")
            except Exception as e:
                bt.logging.error(f"Failed to start experiment '{exp.name}': {e}")
                results[exp.name] = False

        succeeded = sum(1 for v in results.values() if v)
        bt.logging.info(f"Started {succeeded}/{len(enabled)} experiments successfully")
        return results

    def stop_all(self) -> dict[str, bool]:
        """Stop all experiments gracefully.

        Returns:
            Dictionary mapping experiment names to stop success status
        """
        bt.logging.info(f"Stopping {len(self.experiments)} experiments...")
        results = {}
        for exp in self.experiments.values():
            try:
                result = exp.stop()
                # stop() may return bool (PullExperiment) or None (PushExperiment)
                results[exp.name] = result if isinstance(result, bool) else True
                bt.logging.info(f"  Stopped experiment '{exp.name}'")
            except Exception as e:
                bt.logging.error(f"Error stopping experiment '{exp.name}': {e}")
                results[exp.name] = False

        succeeded = sum(1 for v in results.values() if v)
        bt.logging.info(f"Stopped {succeeded}/{len(self.experiments)} experiments successfully")
        return results

    def collect_scores(self, current_block: int) -> list[ExperimentScores]:
        """Collect scores from all enabled experiments.

        Args:
            current_block: Current blockchain block height

        Returns:
            List of ExperimentScores from each enabled experiment
        """
        scores = []
        for exp in self.get_enabled_experiments():
            try:
                exp_scores = exp.calculate_scores(current_block)
                scores.append(exp_scores)
            except Exception as e:
                bt.logging.error(f"Failed to collect scores from '{exp.name}': {e}")
        return scores

    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics from all experiments.

        Returns:
            Dictionary mapping experiment names to their stats
        """
        stats = {}
        for name, exp in self.experiments.items():
            try:
                stats[name] = exp.get_stats()
            except Exception as e:
                bt.logging.error(f"Failed to get stats from '{name}': {e}")
                stats[name] = {"error": str(e)}
        return stats

    def get_experiment_stats(self, experiment_id: str) -> dict[str, Any] | None:
        """Get statistics for a specific experiment (T050).

        Returns experiment stats including rate limit information for operator dashboards.

        Args:
            experiment_id: The experiment ID to get stats for

        Returns:
            Dictionary with experiment stats, or None if experiment not found
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return None

        try:
            # Get base stats from experiment
            if hasattr(exp, "get_stats"):
                stats = exp.get_stats()
            else:
                stats = {}

            # Add rate limit statistics
            with self._rate_limit_lock:
                if experiment_id in self._rate_limit_windows:
                    exp_windows = self._rate_limit_windows[experiment_id]
                    stats["rate_limit_stats"] = {
                        "tracked_miners": len(exp_windows),
                        "total_tracked_requests": sum(len(ts) for ts in exp_windows.values()),
                    }
                else:
                    stats["rate_limit_stats"] = {
                        "tracked_miners": 0,
                        "total_tracked_requests": 0,
                    }

            # Add experiment definition info if available
            exp_def = self.experiment_client.get_experiment(experiment_id)
            if exp_def:
                stats["config"] = {
                    "rate_limit_requests": exp_def.rate_limit_requests,
                    "rate_limit_window_hours": exp_def.rate_limit_window_hours,
                    "status": exp_def.status,
                    "version": exp_def.version,
                }

            return stats

        except Exception as e:
            bt.logging.error(f"Failed to get stats for '{experiment_id}': {e}")
            return {"error": str(e)}

    def get_total_weight_allocation(self) -> float:
        """Get total weight allocation across enabled experiments.

        Returns:
            Sum of weight allocations for enabled experiments
        """
        return sum(exp.weight_allocation for exp in self.get_enabled_experiments())

    def validate_allocations(self) -> tuple[bool, str]:
        """Validate that weight allocations are reasonable.

        Returns:
            Tuple of (is_valid, message)
        """
        enabled = self.get_enabled_experiments()
        if not enabled:
            return False, "No enabled experiments"

        total = self.get_total_weight_allocation()

        # Check if total is zero
        if total == 0:
            return False, "Total weight allocation is zero"

        # Allow some tolerance for floating point
        if abs(total - 1.0) > 0.01:
            bt.logging.warning(
                f"Weight allocations sum to {total:.3f}, not 1.0. "
                "Weights will be normalized."
            )

        return True, f"Valid: {len(enabled)} experiments, total allocation {total:.3f}"

    def acquire_concurrency_slot(self, experiment_id: str) -> bool:
        """Try to acquire a concurrency slot for an experiment (T075).

        Implements per-experiment concurrency limiting with monopolization
        prevention (T077).

        Args:
            experiment_id: The experiment ID

        Returns:
            True if slot acquired, False if at capacity
        """
        # Get max concurrent for this experiment
        exp = self.experiments.get(experiment_id)
        if exp and hasattr(exp, "config") and hasattr(exp.config, "max_concurrent_requests"):
            max_concurrent = exp.config.max_concurrent_requests
        else:
            max_concurrent = self._default_max_concurrent

        # Calculate absolute cap based on thread pool (T077 monopolization prevention)
        # Assume thread pool of ~10 workers, cap at 80% = 8
        absolute_cap = int(10 * self._max_thread_pool_percentage)
        effective_max = min(max_concurrent, absolute_cap)

        with self._concurrency_lock:
            current = self._concurrent_requests.get(experiment_id, 0)

            if current >= effective_max:
                # Check if approaching cap (T077)
                if current >= absolute_cap - 1:
                    bt.logging.warning(
                        f"Experiment '{experiment_id}' approaching thread pool cap "
                        f"({current}/{absolute_cap})"
                    )
                bt.logging.debug(
                    f"Concurrency limit reached for '{experiment_id}': {current}/{effective_max}"
                )
                return False

            self._concurrent_requests[experiment_id] = current + 1
            return True

    def release_concurrency_slot(self, experiment_id: str) -> None:
        """Release a concurrency slot for an experiment (T075).

        Args:
            experiment_id: The experiment ID
        """
        with self._concurrency_lock:
            current = self._concurrent_requests.get(experiment_id, 0)
            if current > 0:
                self._concurrent_requests[experiment_id] = current - 1

    def get_concurrency_stats(self) -> dict[str, dict[str, int]]:
        """Get current concurrency statistics for all experiments (T075).

        Returns:
            Dictionary mapping experiment_id to {current, max}
        """
        with self._concurrency_lock:
            stats = {}
            for exp_id, current in self._concurrent_requests.items():
                exp = self.experiments.get(exp_id)
                if exp and hasattr(exp, "config") and hasattr(exp.config, "max_concurrent_requests"):
                    max_concurrent = exp.config.max_concurrent_requests
                else:
                    max_concurrent = self._default_max_concurrent

                stats[exp_id] = {
                    "current": current,
                    "max": max_concurrent,
                    "utilization": current / max_concurrent if max_concurrent > 0 else 0,
                }
            return stats

    def check_rate_limits(self, hotkey: str, experiment_id: str) -> bool:
        """Check if a miner is within rate limits for an experiment (T047).

        Rate limits are per (hotkey, experiment_id) pair. Each experiment
        can have different rate limits configured via ExperimentDefinition.

        Args:
            hotkey: The miner's hotkey
            experiment_id: The experiment ID to check

        Returns:
            True if request is allowed, False if rate limited
        """
        import time

        # Get rate limit config from experiment definition
        exp_def = self.experiment_client.get_experiment(experiment_id)
        if not exp_def:
            # No experiment definition = no rate limit
            return True

        max_requests = exp_def.rate_limit_requests
        window_hours = exp_def.rate_limit_window_hours

        if not max_requests or max_requests <= 0:
            return True  # No rate limit configured

        window_seconds = window_hours * 3600
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        with self._rate_limit_lock:
            # Initialize structures if needed
            if experiment_id not in self._rate_limit_windows:
                self._rate_limit_windows[experiment_id] = {}
            if hotkey not in self._rate_limit_windows[experiment_id]:
                self._rate_limit_windows[experiment_id][hotkey] = []

            # Get timestamps for this (hotkey, experiment) pair
            timestamps = self._rate_limit_windows[experiment_id][hotkey]

            # Remove expired timestamps
            timestamps = [t for t in timestamps if t > cutoff_time]
            self._rate_limit_windows[experiment_id][hotkey] = timestamps

            # Check if under limit
            if len(timestamps) >= max_requests:
                bt.logging.debug(
                    f"Rate limit exceeded: {hotkey[:16]}... has {len(timestamps)}/{max_requests} "
                    f"requests for experiment '{experiment_id}'"
                )
                # Record rate limit metric (T052)
                self.record_experiment_metrics(
                    experiment_id=experiment_id,
                    event_type="rate_limited",
                    miner_hotkey=hotkey,
                    rate_limit_requests=max_requests,
                    current_requests=len(timestamps),
                )
                return False

            # Record this request
            timestamps.append(current_time)
            return True

    def route_submission(self, synapse: PromptSynapse) -> RoutingResult:
        """Route a submission to the appropriate experiment.

        This method validates the experiment_id, checks experiment status,
        verifies miner registration, and checks rate limits for non-default experiments.

        Args:
            synapse: The incoming synapse with optional experiment_id

        Returns:
            RoutingResult with experiment to route to or rejection details
        """
        from aurelius.shared.telemetry.otel_setup import get_tracer

        tracer = get_tracer("aurelius.experiments") if Config.TELEMETRY_ENABLED else None

        # Default to "prompt" experiment for backward compatibility (FR-022)
        experiment_id = synapse.experiment_id or "prompt"

        bt.logging.debug(
            f"Routing submission: experiment_id={experiment_id}, "
            f"miner_hotkey={synapse.miner_hotkey}"
        )

        # Get list of available experiments for rejection messages
        available = self._get_available_experiment_ids()

        # Check if experiment exists in our registered experiments
        experiment = self.experiments.get(experiment_id)

        if experiment is None:
            rejection = f"Unknown experiment '{experiment_id}'. Available: {', '.join(available)}"
            bt.logging.debug(f"Routing rejected: {rejection}")

            if tracer:
                with tracer.start_as_current_span(
                    "experiment.route",
                    attributes={
                        "experiment.id": experiment_id,
                        "experiment.route.success": False,
                        "experiment.route.reason": "unknown_experiment",
                    },
                ):
                    pass

            return RoutingResult(
                experiment=None,
                rejection_reason=rejection,
                available_experiments=available,
            )

        # Check if experiment is enabled
        if not experiment.is_enabled:
            rejection = f"Experiment '{experiment_id}' is not active/enabled."
            bt.logging.debug(f"Routing rejected: {rejection}")

            if tracer:
                with tracer.start_as_current_span(
                    "experiment.route",
                    attributes={
                        "experiment.id": experiment_id,
                        "experiment.route.success": False,
                        "experiment.route.reason": "experiment_disabled",
                    },
                ):
                    pass

            return RoutingResult(
                experiment=None,
                rejection_reason=rejection,
                available_experiments=available,
            )

        # Check miner registration (skip for "prompt" - all miners auto-registered)
        # On testnet, skip registration check (no central API registration endpoint yet)
        if experiment_id != "prompt" and Config.BT_NETWORK != "test":
            miner_hotkey = synapse.miner_hotkey or ""
            if not self.experiment_client.is_miner_registered(experiment_id, miner_hotkey):
                rejection = (
                    f"Registration required for experiment '{experiment_id}'. "
                    f"Miner {miner_hotkey[:16]}... is not registered."
                )
                bt.logging.debug(f"Routing rejected: {rejection}")

                if tracer:
                    with tracer.start_as_current_span(
                        "experiment.route",
                        attributes={
                            "experiment.id": experiment_id,
                            "experiment.route.success": False,
                            "experiment.route.reason": "registration_required",
                        },
                    ):
                        pass

                return RoutingResult(
                    experiment=None,
                    rejection_reason=rejection,
                    available_experiments=available,
                    registration_required=True,
                )

        # Check rate limits (T051)
        miner_hotkey = synapse.miner_hotkey or ""
        if not self.check_rate_limits(miner_hotkey, experiment_id):
            rejection = (
                f"Rate limit exceeded for experiment '{experiment_id}'. "
                f"Please try again later."
            )
            bt.logging.debug(f"Routing rejected: {rejection}")

            if tracer:
                with tracer.start_as_current_span(
                    "experiment.route",
                    attributes={
                        "experiment.id": experiment_id,
                        "experiment.route.success": False,
                        "experiment.route.reason": "rate_limited",
                    },
                ):
                    pass

            return RoutingResult(
                experiment=None,
                rejection_reason=rejection,
                available_experiments=available,
            )

        # Check for deprecated experiment (T042) - process with warning
        if self.is_experiment_deprecated(experiment_id):
            bt.logging.warning(
                f"Routing to DEPRECATED experiment '{experiment_id}' "
                f"(miner: {synapse.miner_hotkey[:16] if synapse.miner_hotkey else 'unknown'}...). "
                "Submission will be processed but experiment may be removed soon."
            )

        # Successfully routed
        bt.logging.debug(f"Routing success: {experiment_id} -> {experiment.name}")

        if tracer:
            is_deprecated = self.is_experiment_deprecated(experiment_id)
            with tracer.start_as_current_span(
                "experiment.route",
                attributes={
                    "experiment.id": experiment_id,
                    "experiment.name": experiment.name,
                    "experiment.route.success": True,
                    "experiment.deprecated": is_deprecated,
                },
            ):
                pass

        # Record submission metric (T052)
        self.record_experiment_metrics(
            experiment_id=experiment_id,
            event_type="submission",
            miner_hotkey=synapse.miner_hotkey,
        )

        return RoutingResult(experiment=experiment)

    def _get_available_experiment_ids(self) -> list[str]:
        """Get list of available (enabled) experiment IDs.

        Returns:
            List of experiment ID strings
        """
        return [exp.name for exp in self.experiments.values() if exp.is_enabled]

    def apply_routing_rejection(self, synapse: PromptSynapse, result: RoutingResult) -> None:
        """Apply routing rejection details to synapse for miner feedback.

        Args:
            synapse: The synapse to update
            result: The routing result with rejection details
        """
        synapse.accepted = False
        synapse.rejection_reason = result.rejection_reason
        synapse.available_experiments = result.available_experiments
        synapse.registration_required = result.registration_required

    def calculate_merged_weights(
        self,
        experiment_scores: list[ExperimentScores],
        allocations: dict[str, float],
        burn_percentage: float = 0.0,
        redistribute_unused: bool = True,
    ) -> dict[str, float]:
        """Merge per-experiment scores into final weights.

        This function implements the reward allocation algorithm per research.md section 4:
        1. Validate allocations sum to 100%
        2. Calculate per-experiment normalized weights (sum to allocation %)
        3. Redistribute inactive experiment allocations proportionally (if enabled)
        4. Apply burn percentage
        5. Return final {hotkey: weight} dict

        Args:
            experiment_scores: List of ExperimentScores from each experiment
            allocations: Map of experiment_id to allocation percentage (0-100)
            burn_percentage: Percentage to send to burn address (0-100)
            redistribute_unused: If True, redistribute unused allocation to active experiments;
                                 if False, add unused to burn

        Returns:
            Dictionary mapping hotkey (or "burn") to weight percentage (0-100)

        Raises:
            ConfigurationError: If allocations + burn don't sum to 100%
        """
        from collections import defaultdict

        # Step 0: Validate allocations sum to 100%
        total_configured = sum(allocations.values()) + burn_percentage
        if abs(total_configured - 100.0) > 0.01:
            raise ConfigurationError(
                f"Allocations must sum to 100%. "
                f"Got {sum(allocations.values()):.2f}% allocations + {burn_percentage:.2f}% burn = {total_configured:.2f}%"
            )

        # Step 1: Normalize each experiment's scores to its allocation
        experiment_weights: dict[str, dict[str, float]] = {}
        active_allocation = 0.0

        for exp_scores in experiment_scores:
            exp_id = exp_scores.experiment_name
            allocation = allocations.get(exp_id, 0.0)

            if exp_scores.scores:  # Has activity
                # Treat negative scores as zero
                positive_scores = {k: max(0.0, v) for k, v in exp_scores.scores.items()}
                total = sum(positive_scores.values())

                if total > 0:
                    experiment_weights[exp_id] = {
                        hotkey: (score / total) * allocation
                        for hotkey, score in positive_scores.items()
                        if score > 0  # Don't include zero-score miners
                    }
                    active_allocation += allocation

        # Step 2: Handle unused allocation
        unused = sum(allocations.values()) - active_allocation

        if unused > 0:
            if redistribute_unused and active_allocation > 0:
                # Redistribute to active experiments proportionally
                redistribution_factor = (active_allocation + unused) / active_allocation
                for exp_id in experiment_weights:
                    for hotkey in experiment_weights[exp_id]:
                        experiment_weights[exp_id][hotkey] *= redistribution_factor
                bt.logging.info(
                    f"Redistributed {unused:.2f}% unused allocation "
                    f"(factor: {redistribution_factor:.4f})"
                )
            else:
                # Add unused to burn
                burn_percentage += unused
                bt.logging.info(f"Added {unused:.2f}% unused allocation to burn")

        # Step 3: Merge all experiments into final weights
        merged: dict[str, float] = defaultdict(float)
        for exp_id, exp_weights in experiment_weights.items():
            for hotkey, weight in exp_weights.items():
                merged[hotkey] += weight

        # Step 4: Apply burn
        if burn_percentage > 0:
            merged["burn"] = burn_percentage

        # Log reward distribution breakdown (T034)
        bt.logging.info("Reward distribution breakdown:")
        for exp_id, exp_weights in experiment_weights.items():
            exp_total = sum(exp_weights.values())
            top_miners = sorted(exp_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            top_summary = ", ".join(f"{h[:8]}...={w:.2f}%" for h, w in top_miners)
            bt.logging.info(f"  {exp_id}: {exp_total:.2f}% ({len(exp_weights)} miners) - top: {top_summary}")
        if burn_percentage > 0:
            bt.logging.info(f"  burn: {burn_percentage:.2f}%")

        return dict(merged)

    def refresh_from_client(self) -> dict[str, str]:
        """Refresh experiment registry from ExperimentClient.

        This method synchronizes the manager's experiment registry with the
        definitions from the ExperimentClient. It handles:
        - Adding new experiments
        - Updating existing experiments (version changes)
        - Removing experiments no longer in the client
        - Status transitions (active <-> inactive <-> deprecated)

        Returns:
            Dictionary of changes: experiment_id -> change_type
            (e.g., {"exp1": "added", "exp2": "updated", "exp3": "removed"})
        """
        from aurelius.validator.experiments.base import ExperimentConfig, ExperimentType

        changes: dict[str, str] = {}
        client = self.experiment_client

        # Get all experiment definitions from client (including non-active)
        client_experiments: dict[str, ExperimentDefinition] = {}
        for exp in client.get_active_experiments():
            client_experiments[exp.id] = exp

        # Also check for experiments that might have become inactive/deprecated
        # by iterating over what we have registered
        for exp_id in list(self.experiments.keys()):
            if exp_id not in client_experiments:
                # Check if experiment still exists but is now inactive/deprecated
                exp_def = client.get_experiment(exp_id)
                if exp_def:
                    client_experiments[exp_id] = exp_def

        # Track current experiment IDs
        current_ids = set(self.experiments.keys())
        new_ids = set(client_experiments.keys())

        # Process updates and additions
        for exp_id, exp_def in client_experiments.items():
            if exp_id in self.experiments:
                # Existing experiment - check for updates
                existing = self.experiments[exp_id]

                # Check version for updates (T093)
                existing_version = getattr(existing, "_definition_version", 0)
                if exp_def.version > existing_version:
                    bt.logging.info(
                        f"Experiment '{exp_id}' version updated: {existing_version} -> {exp_def.version}"
                    )
                    changes[exp_id] = "version_updated"

                # Check status transitions (T041)
                old_status = "active" if existing.is_enabled else "inactive"
                new_status = exp_def.status

                if old_status != new_status:
                    self._handle_status_transition(exp_id, old_status, new_status)
                    changes[exp_id] = f"status_{old_status}_to_{new_status}"

                # Update the experiment wrapper with new definition
                self._update_experiment_wrapper(exp_id, exp_def)

            else:
                # New experiment - create wrapper
                self._create_experiment_wrapper(exp_id, exp_def)
                changes[exp_id] = "added"
                bt.logging.info(f"Added new experiment '{exp_id}' (status: {exp_def.status})")

        # Process removals (experiments no longer in client)
        removed_ids = current_ids - new_ids
        for exp_id in removed_ids:
            if exp_id == "prompt":
                # Never remove the default prompt experiment
                continue
            del self.experiments[exp_id]
            changes[exp_id] = "removed"
            bt.logging.info(f"Removed experiment '{exp_id}' (no longer in API)")

        if changes:
            bt.logging.info(f"Experiment registry refresh: {len(changes)} changes applied")
        else:
            bt.logging.debug("Experiment registry refresh: no changes")

        return changes

    def _handle_status_transition(
        self, exp_id: str, old_status: str, new_status: str
    ) -> None:
        """Handle experiment status transition.

        Args:
            exp_id: Experiment ID
            old_status: Previous status
            new_status: New status
        """
        bt.logging.info(f"Experiment '{exp_id}' status transition: {old_status} -> {new_status}")

        if new_status == "deprecated":
            bt.logging.warning(
                f"Experiment '{exp_id}' is now DEPRECATED. "
                "Existing submissions will be processed with warning, new submissions rejected."
            )
        elif new_status == "inactive":
            bt.logging.info(f"Experiment '{exp_id}' is now INACTIVE. Submissions will be rejected.")
        elif new_status == "active" and old_status in ("inactive", "deprecated"):
            bt.logging.info(f"Experiment '{exp_id}' is now ACTIVE. Accepting submissions.")

    def _create_experiment_wrapper(
        self, exp_id: str, exp_def: ExperimentDefinition
    ) -> None:
        """Create an experiment wrapper from a definition.

        Args:
            exp_id: Experiment ID
            exp_def: Experiment definition
        """
        from aurelius.validator.experiments.base import ExperimentConfig, ExperimentType

        class DynamicExperimentWrapper:
            """Wrapper for dynamically created experiments from API definitions."""

            def __init__(self, definition: ExperimentDefinition):
                self.name = definition.id
                self._definition = definition
                self._definition_version = definition.version
                self.config = ExperimentConfig(
                    name=definition.id,
                    experiment_type=(
                        ExperimentType.PUSH if definition.experiment_type == "push"
                        else ExperimentType.PULL
                    ),
                    weight_allocation=0.0,  # Set from reward allocation
                    enabled=definition.status == "active",
                    settings=definition.settings,
                )

            @property
            def is_enabled(self) -> bool:
                return self.config.enabled

            @property
            def weight_allocation(self) -> float:
                return self.config.weight_allocation

            @property
            def is_deprecated(self) -> bool:
                return self._definition.status == "deprecated"

        self.experiments[exp_id] = DynamicExperimentWrapper(exp_def)

    def _update_experiment_wrapper(
        self, exp_id: str, exp_def: ExperimentDefinition
    ) -> None:
        """Update an existing experiment wrapper with new definition.

        Args:
            exp_id: Experiment ID
            exp_def: New experiment definition
        """
        exp = self.experiments.get(exp_id)
        if not exp:
            return

        # Update version tracking
        if hasattr(exp, "_definition_version"):
            exp._definition_version = exp_def.version
        if hasattr(exp, "_definition"):
            exp._definition = exp_def

        # Update enabled status
        if hasattr(exp, "config"):
            exp.config.enabled = exp_def.status == "active"

    def is_experiment_deprecated(self, experiment_id: str) -> bool:
        """Check if an experiment is deprecated.

        Args:
            experiment_id: The experiment ID

        Returns:
            True if deprecated, False otherwise
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return False

        if hasattr(exp, "is_deprecated"):
            return exp.is_deprecated
        if hasattr(exp, "_definition"):
            return exp._definition.status == "deprecated"

        return False

    def record_load_balancing_metrics(
        self,
        experiment_id: str,
        metric_type: str,
        value: float,
        **extra_attributes: Any,
    ) -> None:
        """Record load balancing metrics via telemetry (T078).

        Creates telemetry spans for:
        - experiment.queue_depth - current concurrent requests
        - experiment.wait_time - time spent waiting for slot
        - experiment.rejection_cap - rejections due to capacity

        Args:
            experiment_id: The experiment ID
            metric_type: Type of metric (queue_depth, wait_time, rejection_cap)
            value: The metric value
            **extra_attributes: Additional span attributes
        """
        if not Config.TELEMETRY_ENABLED:
            return

        from aurelius.shared.telemetry.otel_setup import get_tracer

        tracer = get_tracer("aurelius.experiments.loadbalancing")

        attributes = {
            "experiment.id": experiment_id,
            "loadbalancing.metric_type": metric_type,
            "loadbalancing.value": value,
        }

        # Add concurrency stats
        with self._concurrency_lock:
            current = self._concurrent_requests.get(experiment_id, 0)
            attributes["loadbalancing.current_concurrent"] = current

        attributes.update(extra_attributes)

        with tracer.start_as_current_span(
            f"experiment.loadbalancing.{metric_type}",
            attributes=attributes,
        ):
            pass  # Span is just for metrics recording

    def record_experiment_metrics(
        self,
        experiment_id: str,
        event_type: str,
        miner_hotkey: str | None = None,
        **extra_attributes: Any,
    ) -> None:
        """Record experiment-level metrics via telemetry (T052).

        Creates a telemetry span for experiment events like submissions,
        acceptances, rejections, rate limits, etc.

        Args:
            experiment_id: The experiment ID
            event_type: Type of event (submission, accepted, rejected, rate_limited)
            miner_hotkey: Optional miner hotkey
            **extra_attributes: Additional span attributes
        """
        if not Config.TELEMETRY_ENABLED:
            return

        from aurelius.shared.telemetry.otel_setup import get_tracer

        tracer = get_tracer("aurelius.experiments")

        attributes = {
            "experiment.id": experiment_id,
            "experiment.event_type": event_type,
        }

        if miner_hotkey:
            attributes["miner.hotkey"] = miner_hotkey[:16] + "..."

        # Add experiment info if available
        exp_def = self.experiment_client.get_experiment(experiment_id)
        if exp_def:
            attributes["experiment.type"] = exp_def.experiment_type
            attributes["experiment.scoring_type"] = exp_def.scoring_type

        attributes.update(extra_attributes)

        with tracer.start_as_current_span(
            f"experiment.{event_type}",
            attributes=attributes,
        ):
            pass  # Span is just for metrics recording

    def handle_deprecated_submission(
        self, synapse: PromptSynapse, experiment_id: str
    ) -> bool:
        """Handle a submission to a deprecated experiment.

        For deprecated experiments (T042):
        - Process existing/pending submissions with a warning
        - Reject new registrations
        - Log deprecation warning

        Args:
            synapse: The submission synapse
            experiment_id: The experiment ID

        Returns:
            True if submission should be processed (with warning), False to reject
        """
        if not self.is_experiment_deprecated(experiment_id):
            return True  # Not deprecated, process normally

        bt.logging.warning(
            f"Processing submission to DEPRECATED experiment '{experiment_id}' "
            f"(miner: {synapse.miner_hotkey[:16] if synapse.miner_hotkey else 'unknown'}...). "
            "This experiment will be removed in a future update."
        )

        # Allow processing but log warning
        return True
