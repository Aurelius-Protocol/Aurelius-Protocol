"""Experiment manager for coordinating multiple experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import bittensor as bt

from aurelius.validator.experiments.base import Experiment, ExperimentScores

if TYPE_CHECKING:
    from aurelius.validator.core import ValidatorCore


class ExperimentManager:
    """Manages experiment lifecycle and score collection.

    The ExperimentManager is responsible for:
    - Registering experiments
    - Starting and stopping all experiments
    - Collecting scores from enabled experiments for weight calculation
    """

    def __init__(self, core: ValidatorCore):
        """Initialize the experiment manager.

        Args:
            core: ValidatorCore providing shared infrastructure
        """
        self.core = core
        self.experiments: dict[str, Experiment] = {}

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
        """Start all enabled experiments.

        Returns:
            Dictionary mapping experiment names to start success status
        """
        enabled = self.get_enabled_experiments()
        if not enabled:
            bt.logging.warning("No enabled experiments to start")
            return {}

        bt.logging.info(f"Starting {len(enabled)} enabled experiments...")
        results = {}
        for exp in enabled:
            try:
                exp.start()
                results[exp.name] = True
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
