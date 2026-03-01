"""Scoring system for the Moral Reasoning Experiment.

Tracks per-miner quality scores from binary signal aggregation,
with window-based normalization for weight calculation.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import bittensor as bt


@dataclass
class MinerMoralScore:
    """Per-miner moral reasoning score tracking."""

    hotkey: str

    # All-time stats
    total_submissions: int = 0
    accepted_submissions: int = 0  # Passed screening
    total_quality_score: float = 0.0
    average_quality: float = 0.0

    # Novelty stats
    total_novelty_score: float = 0.0
    average_novelty: float = 1.0
    novelty_submissions: int = 0

    # Window stats
    window_submissions: int = 0
    window_accepted: int = 0
    window_quality_sum: float = 0.0
    window_avg_quality: float = 0.0

    last_submission_time: float = 0.0

    # Signal velocity tracking — sliding window of recent true-signal counts
    signal_velocity_window: list[int] | None = None  # Recent true-signal counts
    flagged_for_review: bool = False
    flag_count: int = 0  # Total times flagged (persisted, never resets)
    flag_block: int | None = None  # Block height when last flagged


class MoralReasoningScoringSystem:
    """Track and calculate miner scores for the moral reasoning experiment.

    Follows the same pattern as PromptScoringSystem but uses quality scores
    (from binary signal aggregation) instead of danger scores.
    """

    def __init__(
        self,
        persistence_path: str = "./moral_reasoning_scores.json",
        window_blocks: int = 1000,
        history_retention_blocks: int = 10000,
        novelty_unavailable_default: float = 0.0,
        top_rewarded_miners: int = 3,
    ):
        self._lock = threading.RLock()
        self.persistence_path = persistence_path
        self.window_blocks = window_blocks
        self.history_retention_blocks = history_retention_blocks
        self.novelty_unavailable_default = novelty_unavailable_default
        self.top_rewarded_miners = top_rewarded_miners

        self.miner_scores: dict[str, MinerMoralScore] = {}
        self.score_history: dict[str, list] = defaultdict(list)

        # Escalating cooldown: 1st flag = 300 blocks (~1h), 2nd = 1000 (~3.3h), 3rd = 3000 (~10h), 4th+ = permanent
        self.flag_cooldowns = [300, 1000, 3000]

        self._load()
        bt.logging.info(f"MoralReasoningScoringSystem initialized (path: {persistence_path})")

    def record_submission(
        self,
        hotkey: str,
        final_score: float,
        passed_screening: bool,
        block: int | None = None,
        novelty_score: float | None = None,
    ) -> tuple[MinerMoralScore, str]:
        """Record a moral reasoning submission.

        Args:
            hotkey: Miner's hotkey.
            final_score: Final score (quality if screening passed, else 0.0).
            passed_screening: Whether the response passed the screening gate.
            block: Current block height.
            novelty_score: Novelty score from central API (0-1).

        Returns:
            Tuple of (Updated MinerMoralScore, submission_id).
        """
        submission_id = str(uuid.uuid4())

        with self._lock:
            if hotkey not in self.miner_scores:
                self.miner_scores[hotkey] = MinerMoralScore(hotkey=hotkey)

            score = self.miner_scores[hotkey]
            score.total_submissions += 1
            if passed_screening:
                score.accepted_submissions += 1

            score.total_quality_score += final_score
            score.average_quality = score.total_quality_score / score.total_submissions
            score.last_submission_time = time.time()

            if novelty_score is not None:
                score.novelty_submissions += 1
                score.total_novelty_score += novelty_score
                score.average_novelty = score.total_novelty_score / score.novelty_submissions

            self.score_history[hotkey].append({
                "submission_id": submission_id,
                "timestamp": score.last_submission_time,
                "block": block,
                "final_score": final_score,
                "passed_screening": passed_screening,
                "novelty_score": novelty_score,
            })

            if block is not None:
                self._prune_old_submissions(block)
                self._update_window_stats(hotkey, block)

            self._save()
            return score, submission_id

    # Default sliding window size for signal velocity tracking
    VELOCITY_WINDOW_SIZE = 20

    def record_signal_velocity(
        self,
        hotkey: str,
        true_signal_count: int,
        high_signal_threshold: int = 20,
        window_size: int | None = None,
        flag_ratio: float = 0.5,
        current_block: int | None = None,
    ) -> bool:
        """Track signal velocity using a sliding window — flag miners with high ratios.

        Only the most recent *window_size* submissions are considered, so old
        legitimate submissions cannot dilute a burst of injected ones (F4 fix).

        Flagging uses escalating cooldowns (V2 fix):
        - 1st flag: 1000-block cooldown before possible unflag
        - 2nd flag: 5000-block cooldown
        - 3rd+ flag: permanent (no auto-unflag)

        Args:
            hotkey: Miner's hotkey.
            true_signal_count: Number of true signals in this submission.
            high_signal_threshold: Threshold for "high signal" (default 20).
            window_size: Number of recent submissions to track (default 20).
            flag_ratio: Fraction of high-signal submissions that triggers flagging (default 0.5).
            current_block: Current block height (needed for cooldown tracking).

        Returns:
            True if the miner is flagged for review.
        """
        if window_size is None:
            window_size = self.VELOCITY_WINDOW_SIZE

        with self._lock:
            if hotkey not in self.miner_scores:
                return False

            score = self.miner_scores[hotkey]

            # Initialise window list on first use (backward-compat with old data)
            if score.signal_velocity_window is None:
                score.signal_velocity_window = []

            score.signal_velocity_window.append(true_signal_count)

            # Trim to sliding window
            if len(score.signal_velocity_window) > window_size:
                score.signal_velocity_window = score.signal_velocity_window[-window_size:]

            window = score.signal_velocity_window
            recent_count = len(window)
            high_count = sum(1 for c in window if c >= high_signal_threshold)

            # Flag if ratio of high-signal submissions exceeds threshold (min 4 submissions)
            if recent_count >= 4 and high_count / recent_count > flag_ratio:
                if not score.flagged_for_review:
                    score.flagged_for_review = True
                    score.flag_count += 1
                    score.flag_block = current_block
                    bt.logging.warning(
                        f"Miner {hotkey} flagged for review (flag #{score.flag_count}): "
                        f"{high_count}/{recent_count} recent submissions "
                        f"have {high_signal_threshold}+ signals"
                    )
                    self._save()
                return True

            # Only consider unflagging if cooldown has elapsed (V2 fix)
            if score.flagged_for_review:
                cooldown = self._get_cooldown_blocks(score.flag_count)
                if cooldown is None:
                    # Permanent flag (3rd+ offense) — no auto-unflag
                    return False
                if current_block is None or score.flag_block is None:
                    return False  # Can't check cooldown without block heights
                blocks_since_flag = current_block - score.flag_block
                if blocks_since_flag < cooldown:
                    return False  # Still in cooldown
                # Cooldown elapsed AND window is now below threshold → unflag
                if recent_count < 4 or high_count / recent_count <= flag_ratio:
                    score.flagged_for_review = False
                    bt.logging.info(
                        f"Miner {hotkey} un-flagged after cooldown ({blocks_since_flag} blocks, "
                        f"flag #{score.flag_count}): {high_count}/{recent_count} below threshold"
                    )
                    self._save()

            return False

    def _get_cooldown_blocks(self, flag_count: int) -> int | None:
        """Return cooldown in blocks for this flag offense, or None for permanent."""
        if flag_count <= 0:
            return 0
        idx = flag_count - 1
        if idx >= len(self.flag_cooldowns):
            return None  # Permanent
        return self.flag_cooldowns[idx]

    def update_novelty(
        self, hotkey: str, novelty_score: float, submission_id: str | None = None
    ) -> None:
        """Update novelty stats for an existing miner.

        Args:
            hotkey: Miner's hotkey.
            novelty_score: Novelty score from central API.
            submission_id: Target a specific submission entry. Falls back to
                searching for the first None entry (backward compat).
        """
        with self._lock:
            if hotkey not in self.miner_scores:
                return

            score = self.miner_scores[hotkey]
            score.novelty_submissions += 1
            score.total_novelty_score += novelty_score
            score.average_novelty = score.total_novelty_score / score.novelty_submissions

            history = self.score_history.get(hotkey, [])
            if history:
                if submission_id:
                    # Target the exact entry by submission_id
                    for entry in history:
                        if entry.get("submission_id") == submission_id:
                            entry["novelty_score"] = novelty_score
                            break
                else:
                    # Backward-compat fallback: find first None entry
                    for entry in reversed(history):
                        if entry.get("novelty_score") is None:
                            entry["novelty_score"] = novelty_score
                            break

            self._save()

    def get_miner_score(self, hotkey: str) -> MinerMoralScore | None:
        return self.miner_scores.get(hotkey)

    def calculate_normalized_scores(
        self,
        current_block: int,
        min_submissions: int = 1,
    ) -> dict[str, float]:
        """Calculate normalized scores (0-1) for all miners.

        Args:
            current_block: Current block height.
            min_submissions: Minimum accepted submissions required.

        Returns:
            Dict mapping hotkeys to normalized scores (0-1).
        """
        with self._lock:
            for hotkey in self.miner_scores:
                self._update_window_stats(hotkey, current_block)

            raw_scores: dict[str, float] = {}
            window_start = current_block - self.window_blocks

            for hotkey, score in self.miner_scores.items():
                accepted_in_window = [
                    sub for sub in self.score_history.get(hotkey, [])
                    if sub.get("block") is not None
                    and sub["block"] >= window_start
                    and sub["passed_screening"]
                ]

                if len(accepted_in_window) < min_submissions:
                    continue

                # Calculate average quality in window
                quality_scores = [sub["final_score"] for sub in accepted_in_window]
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

                # Apply novelty multiplier
                novelty_scores = [
                    s.get("novelty_score")
                    for s in accepted_in_window
                    if s.get("novelty_score") is not None
                ]

                if novelty_scores:
                    avg_novelty = sum(novelty_scores) / len(novelty_scores)
                else:
                    avg_novelty = self.novelty_unavailable_default

                raw_score = avg_quality * avg_novelty

                # F3: Zero out score for miners flagged by signal velocity check
                if score.flagged_for_review:
                    bt.logging.info(
                        f"Miner {hotkey[:16]}... flagged_for_review — score zeroed"
                    )
                    raw_score = 0.0

                if raw_score > 0:
                    raw_scores[hotkey] = raw_score

            if not raw_scores:
                return {}

            # Top-N equal split: top miners share weight equally
            sorted_miners = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
            top_n = sorted_miners[:self.top_rewarded_miners]
            equal_weight = 1.0 / len(top_n)
            return {hotkey: equal_weight for hotkey, _ in top_n}

    def get_stats(self) -> dict:
        """Get scoring system statistics."""
        if not self.miner_scores:
            return {
                "total_miners": 0,
                "total_submissions": 0,
                "total_accepted": 0,
            }

        total_submissions = sum(s.total_submissions for s in self.miner_scores.values())
        total_accepted = sum(s.accepted_submissions for s in self.miner_scores.values())

        return {
            "total_miners": len(self.miner_scores),
            "total_submissions": total_submissions,
            "total_accepted": total_accepted,
            "screening_pass_rate": (total_accepted / total_submissions * 100) if total_submissions > 0 else 0.0,
        }

    def _update_window_stats(self, hotkey: str, current_block: int) -> None:
        if hotkey not in self.miner_scores:
            return

        score = self.miner_scores[hotkey]
        window_start = current_block - self.window_blocks

        subs = [
            sub for sub in self.score_history.get(hotkey, [])
            if sub.get("block") is not None and sub["block"] >= window_start
        ]

        score.window_submissions = len(subs)
        score.window_accepted = sum(1 for s in subs if s["passed_screening"])
        score.window_quality_sum = sum(s["final_score"] for s in subs)
        score.window_avg_quality = score.window_quality_sum / len(subs) if subs else 0.0

    def _prune_old_submissions(self, current_block: int) -> None:
        cutoff = current_block - self.history_retention_blocks
        for hotkey in list(self.score_history.keys()):
            self.score_history[hotkey] = [
                sub for sub in self.score_history[hotkey]
                if sub.get("block") is None or sub["block"] >= cutoff
            ]

    def _save(self) -> None:
        temp_path = None
        try:
            data = {
                "scores": {hk: asdict(s) for hk, s in self.miner_scores.items()},
                "history": dict(self.score_history),
            }
            path = Path(self.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=".moral_scoring_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                shutil.move(temp_path, self.persistence_path)
                temp_path = None
            except Exception:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        except Exception as e:
            bt.logging.error(f"Error saving moral reasoning scores: {e}")

    def _load(self) -> None:
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path) as f:
                data = json.load(f)
            for hk, sd in data.get("scores", {}).items():
                self.miner_scores[hk] = MinerMoralScore(**sd)
            self.score_history = defaultdict(list, data.get("history", {}))
            bt.logging.info(f"Loaded moral reasoning scores: {len(self.miner_scores)} miners")
        except Exception as e:
            bt.logging.error(f"Error loading moral reasoning scores: {e}")
