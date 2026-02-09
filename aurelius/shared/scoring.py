"""Scoring system for miner performance tracking."""

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import bittensor as bt

from aurelius.shared.config import Config


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
    average_novelty_score: float = 1.0  # Default to 1.0 (fully novel)
    novelty_submissions: int = 0

    # Window stats
    window_submissions: int = 0
    window_accepted: int = 0
    window_total_danger: float = 0.0
    window_avg_danger: float = 0.0
    window_acceptance_rate: float = 0.0
    window_total_novelty: float = 0.0
    window_avg_novelty: float = 1.0  # Default to 1.0 (fully novel)

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

        self._lock = threading.Lock()
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
        novelty_score: float | None = None,
    ) -> MinerScore:
        """
        Record a miner submission and update their score.

        Args:
            hotkey: Miner's hotkey
            danger_score: The danger score for this submission
            accepted: Whether the submission was accepted
            block: Current block height (for windowed rewards)
            novelty_score: Novelty score from central API (0-1, 1=completely novel)

        Returns:
            Updated MinerScore for this miner
        """
        with self._lock:
            return self._record_submission_unlocked(hotkey, danger_score, accepted, block, novelty_score)

    def _record_submission_unlocked(
        self,
        hotkey: str,
        danger_score: float,
        accepted: bool,
        block: int | None = None,
        novelty_score: float | None = None,
    ) -> MinerScore:
        """Internal unlocked implementation of record_submission."""
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

        # Update novelty statistics if score provided
        if novelty_score is not None:
            score.novelty_submissions += 1
            score.total_novelty_score += novelty_score
            score.average_novelty_score = score.total_novelty_score / score.novelty_submissions

        # Add to history with block height and novelty
        self.score_history[hotkey].append(
            {
                "timestamp": score.last_submission_time,
                "block": block,
                "danger_score": danger_score,
                "accepted": accepted,
                "novelty_score": novelty_score,
            }
        )

        # Enforce per-miner cap to prevent unbounded memory growth
        max_per_miner = Config.MAX_SUBMISSIONS_PER_WINDOW * 2
        if len(self.score_history[hotkey]) > max_per_miner:
            self.score_history[hotkey] = self.score_history[hotkey][-max_per_miner:]

        # Prune old submissions if block height is available
        if block is not None:
            self._prune_old_submissions(block)

        # Update window stats if block height is available
        if block is not None:
            self._update_window_stats(hotkey, block)

        novelty_str = f", Novelty: {novelty_score:.3f}" if novelty_score is not None else ""
        bt.logging.info(
            f"Miner {hotkey[:8]}... - "
            f"All-time: {score.total_submissions} submissions, "
            f"{score.accepted_submissions} accepted, "
            f"Avg Danger: {score.average_danger_score:.3f}{novelty_str}, "
            f"Accept Rate: {score.acceptance_rate:.1f}% | "
            f"Window: {score.window_submissions} submissions, "
            f"Danger: {score.window_total_danger:.2f}"
        )

        # Persist after each submission
        self._save()

        return score

    def update_novelty(self, hotkey: str, novelty_score: float) -> None:
        """
        Update novelty stats for an existing miner score.

        This is called from the background thread after the submission
        has already been recorded (to provide immediate feedback to miners).

        Args:
            hotkey: Miner's hotkey
            novelty_score: Novelty score from central API (0-1, 1=completely novel)
        """
        with self._lock:
            self._update_novelty_unlocked(hotkey, novelty_score)

    def _update_novelty_unlocked(self, hotkey: str, novelty_score: float) -> None:
        """Internal unlocked implementation of update_novelty."""
        if hotkey not in self.miner_scores:
            bt.logging.warning(f"Cannot update novelty: no score record for {hotkey[:8]}...")
            return

        score = self.miner_scores[hotkey]

        # Update novelty statistics
        score.novelty_submissions += 1
        score.total_novelty_score += novelty_score
        score.average_novelty_score = score.total_novelty_score / score.novelty_submissions

        # Update the most recent history entry with the novelty score
        history = self.score_history.get(hotkey, [])
        if history:
            # Find the most recent entry without a novelty score and update it
            for entry in reversed(history):
                if entry.get("novelty_score") is None:
                    entry["novelty_score"] = novelty_score
                    break

        bt.logging.debug(
            f"Updated novelty for {hotkey[:8]}...: "
            f"novelty={novelty_score:.3f}, avg={score.average_novelty_score:.3f}"
        )

        # Persist after update
        self._save()

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
        with self._lock:
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
        with self._lock:
            if hotkey in self.miner_scores:
                del self.miner_scores[hotkey]
                bt.logging.info(f"Reset scores for miner {hotkey[:8]}...")

    def reset_all_scores(self) -> None:
        """Reset all miner scores."""
        with self._lock:
            self.miner_scores.clear()
            self.score_history.clear()
            bt.logging.info("Reset all miner scores")

    def calculate_normalized_scores(
        self,
        current_block: int,
        min_submissions: int = 1,
    ) -> dict[str, float]:
        """Calculate normalized scores (0-1) for all miners.

        Returns a hotkey→score dict suitable for ExperimentScores. Uses the
        same windowed scoring logic as calculate_weights_windowed but without
        needing a UID/hotkey list from the metagraph.

        Args:
            current_block: Current block height
            min_submissions: Minimum accepted submissions required

        Returns:
            Dictionary mapping hotkeys to normalized scores (0-1)
        """
        with self._lock:
            for hotkey in self.miner_scores:
                self._update_window_stats(hotkey, current_block)

            raw_scores: dict[str, float] = {}
            window_start = current_block - self.window_blocks

            for hotkey, score in self.miner_scores.items():
                submissions_in_window = [
                    sub
                    for sub in self.score_history.get(hotkey, [])
                    if sub.get("block") is not None
                    and sub["block"] >= window_start
                    and sub["accepted"]
                ]

                if len(submissions_in_window) < min_submissions:
                    continue

                hit_rate = score.window_acceptance_rate / 100.0 if score.window_submissions > 0 else 0.0
                if hit_rate < Config.MIN_HIT_RATE_THRESHOLD:
                    continue

                if len(submissions_in_window) > Config.MAX_SUBMISSIONS_PER_WINDOW:
                    submissions_in_window = sorted(
                        submissions_in_window, key=lambda x: x["danger_score"], reverse=True
                    )[: Config.MAX_SUBMISSIONS_PER_WINDOW]

                novelty_scores = [
                    s.get("novelty_score")
                    for s in submissions_in_window
                    if s.get("novelty_score") is not None
                ]

                if novelty_scores:
                    avg_novelty = sum(novelty_scores) / len(novelty_scores)
                    if avg_novelty < Config.MIN_NOVELTY_THRESHOLD:
                        continue
                else:
                    avg_novelty = 1.0

                danger_sum = sum(sub["danger_score"] for sub in submissions_in_window)
                severity_avg = danger_sum / len(submissions_in_window) if submissions_in_window else 0.0
                novelty_multiplier = avg_novelty ** Config.NOVELTY_WEIGHT

                raw_score = danger_sum * severity_avg * novelty_multiplier
                raw_scores[hotkey] = raw_score

            if not raw_scores:
                return {}

            max_score = max(raw_scores.values())
            if max_score == 0:
                return {}

            return {hotkey: score / max_score for hotkey, score in raw_scores.items()}

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

        Miners must meet a minimum hit rate threshold (MIN_HIT_RATE_THRESHOLD) to
        receive any rewards. This filters out miners who spam low-quality prompts.

        Args:
            uids: List of miner UIDs
            hotkeys: List of miner hotkeys (same order as uids)
            current_block: Current block height
            min_submissions: Minimum accepted submissions required to get non-zero weight

        Returns:
            List of weights (same order as uids), normalized to sum to 1
        """
        with self._lock:
            return self._calculate_weights_windowed_unlocked(
                uids, hotkeys, current_block, min_submissions
            )

    def _calculate_weights_windowed_unlocked(
        self,
        uids: list,
        hotkeys: list,
        current_block: int,
        min_submissions: int = 1,
    ) -> list:
        """Internal unlocked implementation of calculate_weights_windowed."""
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

            # Hit Rate Filter: Check if miner meets minimum acceptance rate threshold
            # This ensures miners are rewarded for reliability, not just volume
            from aurelius.shared.config import Config

            hit_rate = score.window_acceptance_rate / 100.0 if score.window_submissions > 0 else 0.0

            if hit_rate < Config.MIN_HIT_RATE_THRESHOLD:
                # Miner's hit rate is below threshold - gets zero weight
                bt.logging.debug(
                    f"Miner {hotkey[:8]}... filtered out: hit rate {hit_rate:.2%} < "
                    f"threshold {Config.MIN_HIT_RATE_THRESHOLD:.2%} "
                    f"(accepted: {score.window_accepted}/{score.window_submissions})"
                )
                weights.append(0.0)
                continue

            # Anti-spam: Cap submissions per window to prevent gaming via volume
            # Sort by danger score (descending) and take top N
            if len(submissions_in_window) > Config.MAX_SUBMISSIONS_PER_WINDOW:
                submissions_in_window = sorted(
                    submissions_in_window, key=lambda x: x["danger_score"], reverse=True
                )[: Config.MAX_SUBMISSIONS_PER_WINDOW]
                bt.logging.debug(
                    f"Miner {hotkey[:8]}... had {len(submissions_in_window)} submissions in window, "
                    f"capped to top {Config.MAX_SUBMISSIONS_PER_WINDOW} by danger score"
                )

            # Calculate novelty average for this miner's submissions
            novelty_scores = [
                s.get("novelty_score") for s in submissions_in_window
                if s.get("novelty_score") is not None
            ]

            if novelty_scores:
                avg_novelty = sum(novelty_scores) / len(novelty_scores)

                # Filter out submissions below minimum novelty threshold
                if avg_novelty < Config.MIN_NOVELTY_THRESHOLD:
                    bt.logging.debug(
                        f"Miner {hotkey[:8]}... filtered out: avg novelty {avg_novelty:.3f} < "
                        f"threshold {Config.MIN_NOVELTY_THRESHOLD}"
                    )
                    weights.append(0.0)
                    continue
            else:
                # No novelty scores available — fail closed (consistent with moderation fail-closed)
                # Miners are not penalized permanently; scores update when novelty API returns
                avg_novelty = 0.0
                bt.logging.warning(
                    f"Miner {hotkey[:8]}... has no novelty scores — defaulting to 0.0 (fail-closed)"
                )

            # Calculate final weight using the formula:
            # score = danger_sum × severity_avg × novelty_avg^NOVELTY_WEIGHT
            danger_sum = sum(sub["danger_score"] for sub in submissions_in_window)
            severity_avg = danger_sum / len(submissions_in_window) if submissions_in_window else 0.0
            novelty_multiplier = avg_novelty ** Config.NOVELTY_WEIGHT

            weight = danger_sum * severity_avg * novelty_multiplier

            bt.logging.debug(
                f"Miner {hotkey[:8]}... weight calculation: "
                f"danger_sum={danger_sum:.3f} × severity_avg={severity_avg:.3f} × "
                f"novelty_avg^{Config.NOVELTY_WEIGHT}={novelty_multiplier:.3f} = {weight:.3f}"
            )

            weights.append(weight)

        # Top-N Winners Take All: Only top N miners by contribution get rewards (equal split)
        from aurelius.shared.config import Config

        # Create list of (index, weight) pairs and sort by weight descending
        indexed_weights = [(i, w) for i, w in enumerate(weights)]
        sorted_by_weight = sorted(indexed_weights, key=lambda x: x[1], reverse=True)

        # Filter to only those with non-zero contribution, take top N
        top_n = [x for x in sorted_by_weight if x[1] > 0][:Config.TOP_REWARDED_MINERS]

        # Weighted distribution among winners (proportional to contribution)
        # This ensures high-quality miners get proportionally more rewards
        final_weights = [0.0] * len(weights)
        if top_n:
            total_weight = sum(w for _, w in top_n)
            for idx, original_weight in top_n:
                final_weights[idx] = original_weight / total_weight if total_weight > 0 else 0.0

            bt.logging.info(
                f"Top-{len(top_n)} miners receive weighted rewards: "
                + ", ".join(
                    f"idx={idx} (score={original_weight:.3f}, share={final_weights[idx]:.2%})"
                    for idx, original_weight in top_n
                )
            )
        else:
            # Changed from warning to info - this is normal when there are no recent submissions
            bt.logging.info("📊 No qualifying miners in current window (normal if no recent activity)")

        if Config.LOG_WEIGHT_CALCULATIONS:
            bt.logging.info(
                f"📊 Weight calculation summary: "
                f"{sum(1 for w in final_weights if w > 0)}/{Config.TOP_REWARDED_MINERS} miners rewarded "
                f"(window: blocks {window_start}-{current_block})"
            )
        else:
            bt.logging.info(
                f"Calculated windowed weights for {len(final_weights)} miners "
                f"(window: blocks {window_start}-{current_block}, "
                f"rewarded: {sum(1 for w in final_weights if w > 0)}/{Config.TOP_REWARDED_MINERS} max)"
            )

        # Apply miner burn if enabled
        if Config.MINER_BURN_ENABLED:
            burn_percentage = Config.MINER_BURN_PERCENTAGE

            # Ensure burn UID exists in the weights array
            if Config.BURN_UID < len(final_weights):
                # Scale down all miner weights by (1 - burn_percentage)
                scaled_weights = [w * (1 - burn_percentage) for w in final_weights]

                # Allocate burn_percentage to the burn UID
                scaled_weights[Config.BURN_UID] = burn_percentage
                final_weights = scaled_weights

                bt.logging.info(
                    f"Applied {burn_percentage:.0%} miner burn to UID {Config.BURN_UID}"
                )
            else:
                bt.logging.warning(
                    f"BURN_UID {Config.BURN_UID} not in metagraph (size: {len(final_weights)}). "
                    f"Burn not applied - ensure burn hotkey is registered."
                )

        return final_weights

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

        # Calculate novelty stats for window
        novelty_scores = [s.get("novelty_score") for s in submissions_in_window if s.get("novelty_score") is not None]
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
            score.window_avg_novelty = 1.0  # Default to fully novel if no scores

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
        """Save scores and history to disk using atomic write."""
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
            tmp_path = self.persistence_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.persistence_path)
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
