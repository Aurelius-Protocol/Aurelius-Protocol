"""Scoring system for the prompt experiment."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import bittensor as bt

from aurelius.shared.config import Config

if TYPE_CHECKING:
    pass


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

        # Novelty statistics
        total_novelty_score: Sum of all novelty scores
        average_novelty_score: Average novelty score across all submissions
        novelty_submissions: Number of submissions with novelty scores

        # Window-based statistics (for reward calculation)
        window_submissions: Number of submissions in current window
        window_accepted: Number of accepted submissions in window
        window_total_danger: Sum of danger scores in window
        window_avg_danger: Average danger score in window
        window_acceptance_rate: Acceptance rate in window
        window_total_novelty: Sum of novelty scores in window
        window_avg_novelty: Average novelty score in window

        last_submission_time: Unix timestamp of last submission
    """

    hotkey: str

    # All-time stats
    total_submissions: int = 0
    accepted_submissions: int = 0
    total_danger_score: float = 0.0
    average_danger_score: float = 0.0
    acceptance_rate: float = 0.0

    # Novelty stats
    total_novelty_score: float = 0.0
    average_novelty_score: float = 1.0
    novelty_submissions: int = 0

    # Window stats
    window_submissions: int = 0
    window_accepted: int = 0
    window_total_danger: float = 0.0
    window_avg_danger: float = 0.0
    window_acceptance_rate: float = 0.0
    window_total_novelty: float = 0.0
    window_avg_novelty: float = 1.0

    last_submission_time: float = 0.0


class PromptScoringSystem:
    """Track and calculate miner scores for the prompt experiment.

    This is the experiment-specific scoring system. It tracks miner performance
    and calculates normalized scores for weight allocation.
    """

    def __init__(
        self,
        danger_threshold: float = 0.5,
        persistence_path: str = "./prompt_experiment_scores.json",
        window_blocks: int = 1000,
        history_retention_blocks: int = 10000,
    ):
        """Initialize the scoring system.

        Args:
            danger_threshold: Threshold for accepting a submission
            persistence_path: Path to save/load scoring data
            window_blocks: Number of blocks for reward calculation window
            history_retention_blocks: Number of blocks to retain in history
        """
        self._lock = threading.RLock()  # RLock for nested calls (e.g., record_submission -> _save)
        self.danger_threshold = danger_threshold
        self.persistence_path = persistence_path
        self.window_blocks = window_blocks
        self.history_retention_blocks = history_retention_blocks

        self.miner_scores: dict[str, MinerScore] = {}
        self.score_history: dict[str, list] = defaultdict(list)

        self._load()

        bt.logging.info(f"PromptScoringSystem initialized (threshold: {danger_threshold})")
        bt.logging.info(f"  Window: {window_blocks} blocks (~{window_blocks * 12 / 3600:.1f} hours)")

    def record_submission(
        self,
        hotkey: str,
        danger_score: float,
        accepted: bool,
        block: int | None = None,
        novelty_score: float | None = None,
    ) -> MinerScore:
        """Record a miner submission and update their score.

        Args:
            hotkey: Miner's hotkey
            danger_score: The danger score for this submission
            accepted: Whether the submission was accepted
            block: Current block height (for windowed rewards)
            novelty_score: Novelty score from central API (0-1, 1=completely novel)

        Returns:
            Updated MinerScore for this miner
        """
        # Validate inputs
        if not isinstance(danger_score, (int, float)) or danger_score < 0:
            bt.logging.warning(f"Invalid danger_score: {danger_score}, clamping to 0")
            danger_score = max(0.0, float(danger_score) if danger_score else 0.0)

        if novelty_score is not None and not 0.0 <= novelty_score <= 1.0:
            bt.logging.warning(f"Invalid novelty_score: {novelty_score}, clamping")
            novelty_score = max(0.0, min(1.0, novelty_score))

        if block is not None and block < 0:
            bt.logging.warning(f"Invalid block: {block}, setting to None")
            block = None

        with self._lock:
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

            # Update novelty statistics if score provided
            if novelty_score is not None:
                score.novelty_submissions += 1
                score.total_novelty_score += novelty_score
                score.average_novelty_score = score.total_novelty_score / score.novelty_submissions

            # Add to history with block height and novelty
            self.score_history[hotkey].append({
                "timestamp": score.last_submission_time,
                "block": block,
                "danger_score": danger_score,
                "accepted": accepted,
                "novelty_score": novelty_score,
            })

            # Prune old submissions if block height is available
            if block is not None:
                self._prune_old_submissions(block)
                self._update_window_stats(hotkey, block)

            self._save()
            return score

    def update_novelty(self, hotkey: str, novelty_score: float) -> None:
        """Update novelty stats for an existing miner score.

        Args:
            hotkey: Miner's hotkey
            novelty_score: Novelty score from central API (0-1, 1=completely novel)
        """
        # Validate novelty_score
        if not 0.0 <= novelty_score <= 1.0:
            bt.logging.warning(f"Invalid novelty_score: {novelty_score}, clamping")
            novelty_score = max(0.0, min(1.0, novelty_score))

        with self._lock:
            if hotkey not in self.miner_scores:
                bt.logging.warning(f"Cannot update novelty: no score record for {hotkey[:8]}...")
                return

            score = self.miner_scores[hotkey]

            score.novelty_submissions += 1
            score.total_novelty_score += novelty_score
            score.average_novelty_score = score.total_novelty_score / score.novelty_submissions

            # Update the most recent history entry
            history = self.score_history.get(hotkey, [])
            if history:
                for entry in reversed(history):
                    if entry.get("novelty_score") is None:
                        entry["novelty_score"] = novelty_score
                        break

            self._save()

    def get_miner_score(self, hotkey: str) -> MinerScore | None:
        """Get the score record for a specific miner."""
        return self.miner_scores.get(hotkey)

    def get_all_scores(self) -> dict[str, MinerScore]:
        """Get all miner scores."""
        return self.miner_scores.copy()

    def calculate_normalized_scores(
        self,
        current_block: int,
        min_submissions: int = 1,
    ) -> dict[str, float]:
        """Calculate normalized scores (0-1) for all miners.

        This is the main scoring method for experiments. It returns normalized
        scores that can be combined with other experiments.

        Args:
            current_block: Current block height
            min_submissions: Minimum accepted submissions required

        Returns:
            Dictionary mapping hotkeys to normalized scores (0-1)
        """
        with self._lock:
            # Update window stats for all miners
            for hotkey in self.miner_scores:
                self._update_window_stats(hotkey, current_block)

            raw_scores: dict[str, float] = {}
            window_start = current_block - self.window_blocks

            for hotkey, score in self.miner_scores.items():
                # Get accepted submissions within window
                submissions_in_window = [
                    sub
                    for sub in self.score_history.get(hotkey, [])
                    if sub.get("block") is not None
                    and sub["block"] >= window_start
                    and sub["accepted"]
                ]

                if len(submissions_in_window) < min_submissions:
                    continue

                # Hit Rate Filter
                hit_rate = score.window_acceptance_rate / 100.0 if score.window_submissions > 0 else 0.0
                if hit_rate < Config.MIN_HIT_RATE_THRESHOLD:
                    bt.logging.debug(f"Miner {hotkey[:8]}... filtered: hit rate {hit_rate:.2%}")
                    continue

                # Anti-spam: Cap submissions per window
                if len(submissions_in_window) > Config.MAX_SUBMISSIONS_PER_WINDOW:
                    submissions_in_window = sorted(
                        submissions_in_window,
                        key=lambda x: x["danger_score"],
                        reverse=True,
                    )[: Config.MAX_SUBMISSIONS_PER_WINDOW]

                # Calculate novelty average
                novelty_scores = [
                    s.get("novelty_score")
                    for s in submissions_in_window
                    if s.get("novelty_score") is not None
                ]

                if novelty_scores:
                    avg_novelty = sum(novelty_scores) / len(novelty_scores)
                    if avg_novelty < Config.MIN_NOVELTY_THRESHOLD:
                        bt.logging.debug(f"Miner {hotkey[:8]}... filtered: novelty {avg_novelty:.3f}")
                        continue
                else:
                    avg_novelty = 1.0

                # Calculate raw score using formula:
                # score = severity_avg x novelty_avg^NOVELTY_WEIGHT
                severity_avg = sum(sub["danger_score"] for sub in submissions_in_window) / len(submissions_in_window) if submissions_in_window else 0.0
                novelty_multiplier = avg_novelty ** Config.NOVELTY_WEIGHT

                raw_score = severity_avg * novelty_multiplier
                raw_scores[hotkey] = raw_score

            # Normalize scores to 0-1 range
            if not raw_scores:
                return {}

            max_score = max(raw_scores.values())
            if max_score == 0:
                return {}

            normalized = {hotkey: score / max_score for hotkey, score in raw_scores.items()}
            return normalized

    def get_stats(self) -> dict:
        """Get overall scoring system statistics."""
        if not self.miner_scores:
            return {
                "total_miners": 0,
                "total_submissions": 0,
                "total_accepted": 0,
                "miners": [],
            }

        total_submissions = sum(s.total_submissions for s in self.miner_scores.values())
        total_accepted = sum(s.accepted_submissions for s in self.miner_scores.values())

        return {
            "total_miners": len(self.miner_scores),
            "total_submissions": total_submissions,
            "total_accepted": total_accepted,
            "overall_acceptance_rate": (total_accepted / total_submissions * 100) if total_submissions > 0 else 0.0,
            "danger_threshold": self.danger_threshold,
            "miners": list(self.miner_scores.keys()),
        }

    def get_top_miners(
        self,
        n: int = 10,
        metric: str = "acceptance_rate",
    ) -> list[tuple[str, MinerScore]]:
        """Get top N miners by a specific metric.

        Args:
            n: Number of top miners to return
            metric: Metric to sort by. One of:
                - "acceptance_rate": Percentage of accepted submissions
                - "average_danger_score": Average danger score across all submissions
                - "total_submissions": Total number of submissions

        Returns:
            List of (hotkey, MinerScore) tuples sorted by the specified metric
        """
        valid_metrics = ["acceptance_rate", "average_danger_score", "total_submissions"]
        if metric not in valid_metrics:
            bt.logging.error(f"Invalid metric: {metric}. Valid metrics: {valid_metrics}")
            return []

        with self._lock:
            sorted_miners = sorted(
                self.miner_scores.items(),
                key=lambda x: getattr(x[1], metric),
                reverse=True,
            )
            return sorted_miners[:n]

    def reset_miner_score(self, hotkey: str) -> bool:
        """Reset scores for a specific miner.

        Args:
            hotkey: Miner's hotkey

        Returns:
            True if miner was found and reset, False otherwise
        """
        with self._lock:
            if hotkey not in self.miner_scores:
                bt.logging.warning(f"Cannot reset: no score record for {hotkey[:8]}...")
                return False

            del self.miner_scores[hotkey]
            if hotkey in self.score_history:
                del self.score_history[hotkey]
            self._save()
            bt.logging.info(f"Reset scores for miner {hotkey[:8]}...")
            return True

    def reset_all_scores(self) -> int:
        """Reset all miner scores.

        Returns:
            Number of miners that were reset
        """
        with self._lock:
            count = len(self.miner_scores)
            self.miner_scores.clear()
            self.score_history.clear()
            self._save()
            bt.logging.info(f"Reset all scores ({count} miners)")
            return count

    def _update_window_stats(self, hotkey: str, current_block: int) -> None:
        """Update window-based statistics for a miner."""
        if hotkey not in self.miner_scores:
            return

        score = self.miner_scores[hotkey]
        window_start = current_block - self.window_blocks

        submissions_in_window = [
            sub
            for sub in self.score_history.get(hotkey, [])
            if sub.get("block") is not None and sub["block"] >= window_start
        ]

        score.window_submissions = len(submissions_in_window)
        score.window_accepted = sum(1 for s in submissions_in_window if s["accepted"])
        score.window_total_danger = sum(s["danger_score"] for s in submissions_in_window)

        novelty_scores = [
            s.get("novelty_score")
            for s in submissions_in_window
            if s.get("novelty_score") is not None
        ]
        score.window_total_novelty = sum(novelty_scores) if novelty_scores else 0.0

        if score.window_submissions > 0:
            score.window_avg_danger = score.window_total_danger / score.window_submissions
            score.window_acceptance_rate = (score.window_accepted / score.window_submissions) * 100
        else:
            score.window_avg_danger = 0.0
            score.window_acceptance_rate = 0.0

        if novelty_scores:
            score.window_avg_novelty = score.window_total_novelty / len(novelty_scores)
        else:
            score.window_avg_novelty = 1.0

    def _prune_old_submissions(self, current_block: int) -> None:
        """Remove submissions older than retention period."""
        cutoff_block = current_block - self.history_retention_blocks

        for hotkey in list(self.score_history.keys()):
            self.score_history[hotkey] = [
                sub
                for sub in self.score_history[hotkey]
                if sub.get("block") is None or sub["block"] >= cutoff_block
            ]

    def _save(self) -> None:
        """Save scores and history to disk atomically.

        Uses a temporary file and atomic rename to prevent data corruption
        if the process is interrupted during write.
        """
        temp_path = None
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

            # Create parent directory
            path = Path(self.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(
                dir=path.parent,
                prefix=".scoring_",
                suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                # Atomic rename (on POSIX systems)
                shutil.move(temp_path, self.persistence_path)
                temp_path = None  # Successfully moved, don't delete
            except Exception:
                # Clean up temp file on error
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        except Exception as e:
            bt.logging.error(f"Error saving scoring data: {e}")
            import traceback
            bt.logging.debug(f"Traceback: {traceback.format_exc()}")

    def _load(self) -> None:
        """Load scores and history from disk."""
        if not os.path.exists(self.persistence_path):
            bt.logging.info("No existing scoring data found, starting fresh")
            return

        try:
            with open(self.persistence_path) as f:
                data = json.load(f)

            for hotkey, score_dict in data.get("scores", {}).items():
                self.miner_scores[hotkey] = MinerScore(**score_dict)

            self.score_history = defaultdict(list, data.get("history", {}))

            bt.logging.info(
                f"Loaded scoring data: {len(self.miner_scores)} miners, "
                f"{sum(len(h) for h in self.score_history.values())} submissions"
            )
        except Exception as e:
            bt.logging.error(f"Error loading scoring data: {e}")
