"""Scoring system for miner performance tracking."""

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import bittensor as bt


@dataclass
class MinerScore:
    """Score record for a single miner.

    Attributes:
        hotkey: Miner's hotkey

        # All-time statistics
        total_submissions: Total number of submissions
        accepted_submissions: Number that met the danger threshold
        total_danger_score: Sum of all danger scores
        average_danger_score: Average danger score across all submissions
        acceptance_rate: Percentage of submissions that were accepted

        # Window-based statistics (for reward calculation)
        window_submissions: Number of submissions in current window
        window_accepted: Number of accepted submissions in window
        window_total_danger: Sum of danger scores in window
        window_avg_danger: Average danger score in window
        window_acceptance_rate: Acceptance rate in window

        last_submission_time: Unix timestamp of last submission
    """

    hotkey: str

    # All-time stats
    total_submissions: int = 0
    accepted_submissions: int = 0
    total_danger_score: float = 0.0
    average_danger_score: float = 0.0
    acceptance_rate: float = 0.0

    # Window stats
    window_submissions: int = 0
    window_accepted: int = 0
    window_total_danger: float = 0.0
    window_avg_danger: float = 0.0
    window_acceptance_rate: float = 0.0

    last_submission_time: float = 0.0


class ScoringSystem:
    """Track and calculate miner scores for weight setting."""

    def __init__(
        self,
        danger_threshold: float = 0.5,
        persistence_path: str = "./miner_scores.json",
        window_blocks: int = 1000,
        history_retention_blocks: int = 10000,
    ):
        """
        Initialize the scoring system.

        Args:
            danger_threshold: Threshold for accepting a submission
            persistence_path: Path to save/load scoring data
            window_blocks: Number of blocks for reward calculation window
            history_retention_blocks: Number of blocks to retain in history
        """
        self.danger_threshold = danger_threshold
        self.persistence_path = persistence_path
        self.window_blocks = window_blocks
        self.history_retention_blocks = history_retention_blocks

        self.miner_scores: dict[str, MinerScore] = {}
        self.score_history: dict[str, list] = defaultdict(list)  # For tracking over time

        # Load existing data if available
        self._load()

        bt.logging.info(f"Scoring system initialized with threshold: {danger_threshold}")
        bt.logging.info(f"Window size: {window_blocks} blocks (~{window_blocks * 12 / 3600:.1f} hours)")
        bt.logging.info(f"History retention: {history_retention_blocks} blocks")

    def record_submission(
        self,
        hotkey: str,
        danger_score: float,
        accepted: bool,
        block: int | None = None,
    ) -> MinerScore:
        """
        Record a miner submission and update their score.

        Args:
            hotkey: Miner's hotkey
            danger_score: The danger score for this submission
            accepted: Whether the submission was accepted
            block: Current block height (for windowed rewards)

        Returns:
            Updated MinerScore for this miner
        """
        # Initialize if first submission from this miner
        if hotkey not in self.miner_scores:
            self.miner_scores[hotkey] = MinerScore(hotkey=hotkey)

        score = self.miner_scores[hotkey]

        # Update all-time statistics
        score.total_submissions += 1
        if accepted:
            score.accepted_submissions += 1

        score.total_danger_score += danger_score
        score.average_danger_score = score.total_danger_score / score.total_submissions
        score.acceptance_rate = (score.accepted_submissions / score.total_submissions) * 100
        score.last_submission_time = time.time()

        # Add to history with block height
        self.score_history[hotkey].append(
            {
                "timestamp": score.last_submission_time,
                "block": block,
                "danger_score": danger_score,
                "accepted": accepted,
            }
        )

        # Prune old submissions if block height is available
        if block is not None:
            self._prune_old_submissions(block)

        # Update window stats if block height is available
        if block is not None:
            self._update_window_stats(hotkey, block)

        bt.logging.info(
            f"Miner {hotkey[:8]}... - "
            f"All-time: {score.total_submissions} submissions, "
            f"{score.accepted_submissions} accepted, "
            f"Avg Danger: {score.average_danger_score:.3f}, "
            f"Accept Rate: {score.acceptance_rate:.1f}% | "
            f"Window: {score.window_submissions} submissions, "
            f"Danger: {score.window_total_danger:.2f}"
        )

        # Persist after each submission
        self._save()

        return score

    def get_miner_score(self, hotkey: str) -> MinerScore | None:
        """
        Get the score record for a specific miner.

        Args:
            hotkey: Miner's hotkey

        Returns:
            MinerScore if miner has submissions, None otherwise
        """
        return self.miner_scores.get(hotkey)

    def get_all_scores(self) -> dict[str, MinerScore]:
        """
        Get all miner scores.

        Returns:
            Dictionary mapping hotkeys to MinerScores
        """
        return self.miner_scores.copy()

    def calculate_weights(
        self,
        uids: list,
        hotkeys: list,
        min_submissions: int = 1,
    ) -> list:
        """
        Calculate Bittensor weights for miners based on their performance.

        The weight calculation rewards miners who:
        1. Successfully submit prompts that meet the danger threshold
        2. Have higher average danger scores
        3. Have higher acceptance rates

        Args:
            uids: List of miner UIDs
            hotkeys: List of miner hotkeys (same order as uids)
            min_submissions: Minimum submissions required to get non-zero weight

        Returns:
            List of weights (same order as uids), normalized to sum to 1
        """
        if len(uids) != len(hotkeys):
            bt.logging.error("UIDs and hotkeys lists must have same length")
            return [0.0] * len(uids)

        weights = []

        for _uid, hotkey in zip(uids, hotkeys, strict=False):
            score = self.miner_scores.get(hotkey)

            if not score or score.total_submissions < min_submissions:
                # No score or not enough submissions
                weights.append(0.0)
                continue

            # Weight calculation formula:
            # Base weight from acceptance rate (0-100)
            # Bonus for high average danger scores
            # This encourages miners to submit dangerous prompts that are accepted

            base_weight = score.acceptance_rate  # 0-100

            # Bonus: multiply by average danger score (0-1+)
            # This rewards miners who consistently get high danger scores
            danger_bonus = score.average_danger_score

            # Final weight
            weight = base_weight * (1.0 + danger_bonus)

            weights.append(weight)

        # Normalize weights to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            # No dangerous prompts in window - all miners get 0 (zero-sum game)
            weights = [0.0] * len(weights)

        return weights

    def get_top_miners(self, n: int = 10, metric: str = "acceptance_rate") -> list:
        """
        Get top N miners by a specific metric.

        Args:
            n: Number of top miners to return
            metric: Metric to sort by ("acceptance_rate", "average_danger_score", "total_submissions")

        Returns:
            List of (hotkey, MinerScore) tuples, sorted by metric
        """
        if metric not in ["acceptance_rate", "average_danger_score", "total_submissions"]:
            bt.logging.error(f"Invalid metric: {metric}")
            return []

        sorted_miners = sorted(self.miner_scores.items(), key=lambda x: getattr(x[1], metric), reverse=True)

        return sorted_miners[:n]

    def get_stats(self) -> dict:
        """
        Get overall scoring system statistics.

        Returns:
            Dictionary with system-wide stats
        """
        if not self.miner_scores:
            return {
                "total_miners": 0,
                "total_submissions": 0,
                "total_accepted": 0,
                "overall_acceptance_rate": 0.0,
                "average_danger_score": 0.0,
            }

        total_submissions = sum(s.total_submissions for s in self.miner_scores.values())
        total_accepted = sum(s.accepted_submissions for s in self.miner_scores.values())
        total_danger = sum(s.total_danger_score for s in self.miner_scores.values())

        return {
            "total_miners": len(self.miner_scores),
            "total_submissions": total_submissions,
            "total_accepted": total_accepted,
            "overall_acceptance_rate": (total_accepted / total_submissions * 100) if total_submissions > 0 else 0.0,
            "average_danger_score": (total_danger / total_submissions) if total_submissions > 0 else 0.0,
            "danger_threshold": self.danger_threshold,
        }

    def reset_miner_score(self, hotkey: str) -> None:
        """
        Reset scores for a specific miner.

        Args:
            hotkey: Miner's hotkey
        """
        if hotkey in self.miner_scores:
            del self.miner_scores[hotkey]
            bt.logging.info(f"Reset scores for miner {hotkey[:8]}...")

    def reset_all_scores(self) -> None:
        """Reset all miner scores."""
        self.miner_scores.clear()
        self.score_history.clear()
        bt.logging.info("Reset all miner scores")

    def calculate_weights_windowed(
        self,
        uids: list,
        hotkeys: list,
        current_block: int,
        min_submissions: int = 1,
    ) -> list:
        """
        Calculate Bittensor weights based on windowed performance.

        Weights are calculated proportional to total danger score of accepted
        submissions within the look-back window. This implements a pool-based
        reward system where all successful miners split the pool proportionally.

        Args:
            uids: List of miner UIDs
            hotkeys: List of miner hotkeys (same order as uids)
            current_block: Current block height
            min_submissions: Minimum accepted submissions required to get non-zero weight

        Returns:
            List of weights (same order as uids), normalized to sum to 1
        """
        if len(uids) != len(hotkeys):
            bt.logging.error("UIDs and hotkeys lists must have same length")
            return [0.0] * len(uids)

        # Update window stats for all miners first
        for hotkey in hotkeys:
            if hotkey in self.miner_scores:
                self._update_window_stats(hotkey, current_block)

        weights = []
        window_start = current_block - self.window_blocks

        for _uid, hotkey in zip(uids, hotkeys, strict=False):
            score = self.miner_scores.get(hotkey)

            if not score:
                weights.append(0.0)
                continue

            # Get accepted submissions within window
            submissions_in_window = [
                sub
                for sub in self.score_history.get(hotkey, [])
                if sub.get("block") is not None
                and sub["block"] >= window_start
                and sub["accepted"]  # Only accepted submissions
            ]

            if len(submissions_in_window) < min_submissions:
                # Not enough accepted submissions in window
                weights.append(0.0)
                continue

            # Anti-spam: Cap submissions per window to prevent gaming via volume
            # Sort by danger score (descending) and take top N
            from aurelius.shared.config import Config

            if len(submissions_in_window) > Config.MAX_SUBMISSIONS_PER_WINDOW:
                submissions_in_window = sorted(
                    submissions_in_window, key=lambda x: x["danger_score"], reverse=True
                )[: Config.MAX_SUBMISSIONS_PER_WINDOW]
                bt.logging.debug(
                    f"Miner {hotkey[:8]}... had {len(submissions_in_window)} submissions in window, "
                    f"capped to top {Config.MAX_SUBMISSIONS_PER_WINDOW} by danger score"
                )

            # Weight = sum of danger scores of accepted submissions
            # This splits the pool proportionally based on total danger contribution
            weight = sum(sub["danger_score"] for sub in submissions_in_window)

            weights.append(weight)

        # Normalize weights to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            # No dangerous prompts in window - all miners get 0 (zero-sum game)
            weights = [0.0] * len(weights)

        bt.logging.info(
            f"Calculated windowed weights for {len(weights)} miners "
            f"(window: blocks {window_start}-{current_block}, "
            f"non-zero weights: {sum(1 for w in weights if w > 0)})"
        )

        return weights

    def _update_window_stats(self, hotkey: str, current_block: int) -> None:
        """
        Update window-based statistics for a miner.

        Args:
            hotkey: Miner's hotkey
            current_block: Current block height
        """
        if hotkey not in self.miner_scores:
            return

        score = self.miner_scores[hotkey]
        window_start = current_block - self.window_blocks

        # Filter submissions in window
        submissions_in_window = [
            sub
            for sub in self.score_history.get(hotkey, [])
            if sub.get("block") is not None and sub["block"] >= window_start
        ]

        # Calculate window stats
        score.window_submissions = len(submissions_in_window)
        score.window_accepted = sum(1 for s in submissions_in_window if s["accepted"])
        score.window_total_danger = sum(s["danger_score"] for s in submissions_in_window)

        if score.window_submissions > 0:
            score.window_avg_danger = score.window_total_danger / score.window_submissions
            score.window_acceptance_rate = (score.window_accepted / score.window_submissions) * 100
        else:
            score.window_avg_danger = 0.0
            score.window_acceptance_rate = 0.0

    def _prune_old_submissions(self, current_block: int) -> None:
        """
        Remove submissions older than retention period from history.

        Args:
            current_block: Current block height
        """
        cutoff_block = current_block - self.history_retention_blocks
        pruned_count = 0

        for hotkey in list(self.score_history.keys()):
            original_count = len(self.score_history[hotkey])
            self.score_history[hotkey] = [
                sub for sub in self.score_history[hotkey] if sub.get("block") is None or sub["block"] >= cutoff_block
            ]
            pruned_count += original_count - len(self.score_history[hotkey])

        if pruned_count > 0:
            bt.logging.debug(f"Pruned {pruned_count} old submissions (before block {cutoff_block})")

    def _save(self) -> None:
        """Save scores and history to disk."""
        try:
            data = {
                "scores": {hotkey: asdict(score) for hotkey, score in self.miner_scores.items()},
                "history": dict(self.score_history),
                "config": {
                    "danger_threshold": self.danger_threshold,
                    "window_blocks": self.window_blocks,
                    "history_retention_blocks": self.history_retention_blocks,
                },
            }
            Path(self.persistence_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            bt.logging.error(f"Error saving scoring data: {e}")

    def _load(self) -> None:
        """Load scores and history from disk."""
        if not os.path.exists(self.persistence_path):
            bt.logging.info("No existing scoring data found, starting fresh")
            return

        try:
            with open(self.persistence_path) as f:
                data = json.load(f)

            # Load scores
            for hotkey, score_dict in data.get("scores", {}).items():
                self.miner_scores[hotkey] = MinerScore(**score_dict)

            # Load history
            self.score_history = defaultdict(list, data.get("history", {}))

            bt.logging.info(
                f"Loaded scoring data: {len(self.miner_scores)} miners, "
                f"{sum(len(h) for h in self.score_history.values())} submissions"
            )
        except Exception as e:
            bt.logging.error(f"Error loading scoring data: {e}")
            bt.logging.warning("Starting with fresh scoring data")
