"""Submission history logging for miner operations.

Records every submission result to daily-rotated JSONL files under ~/.aurelius/history/,
providing persistent visibility into past performance and trends.
"""

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class SubmissionRecord:
    """A single submission result."""

    timestamp: str  # ISO 8601
    experiment_id: str
    validator_uid: int
    prompt_hash: str  # SHA-256 of prompt text
    danger_score: float | None = None
    accepted: bool | None = None
    rejection_reason: str | None = None
    rejection_code: str | None = None
    novelty_avg: float | None = None
    hit_rate: float | None = None
    submission_count: int | None = None
    latency_ms: float = 0.0
    model_used: str | None = None
    success: bool = True
    error_message: str | None = None

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Create a SHA-256 hash of the prompt text."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class SubmissionHistory:
    """Manages persistent submission history as daily-rotated JSONL files.

    Args:
        data_dir: Base directory for history files (default: ~/.aurelius)
    """

    def __init__(self, data_dir: str | None = None, retention_days: int = 30):
        if data_dir is None:
            data_dir = os.path.expanduser("~/.aurelius")
        self.history_dir = os.path.join(data_dir, "history")
        self._lock = threading.Lock()
        self._retention_days = retention_days
        self._cleanup_old_files()

    def _ensure_dir(self) -> None:
        """Create history directory if it doesn't exist."""
        os.makedirs(self.history_dir, exist_ok=True)

    def _date_path(self, date: datetime) -> str:
        """Get the JSONL file path for a given date."""
        return os.path.join(self.history_dir, f"{date.strftime('%Y-%m-%d')}.jsonl")

    def _cleanup_old_files(self) -> int:
        """Remove JSONL files older than retention period.

        Returns:
            Number of files removed.
        """
        if not os.path.isdir(self.history_dir):
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        removed = 0

        for filename in os.listdir(self.history_dir):
            if not filename.endswith(".jsonl"):
                continue
            date_str = filename[:-6]  # strip .jsonl
            if date_str < cutoff_str:
                try:
                    os.remove(os.path.join(self.history_dir, filename))
                    removed += 1
                except OSError:
                    pass

        if removed > 0:
            logger.info("Cleaned up %d old history file(s)", removed)
        return removed

    def record(self, record: SubmissionRecord) -> None:
        """Append a submission record to today's history file.

        Thread-safe via lock. Each record is a single JSON line.
        """
        self._ensure_dir()
        today = datetime.now(timezone.utc)
        path = self._date_path(today)

        line = json.dumps(asdict(record), separators=(",", ":")) + "\n"

        with self._lock:
            with open(path, "a") as f:
                f.write(line)

    def load(self, days: int = 7) -> list[SubmissionRecord]:
        """Load submission records from the last N days.

        Args:
            days: Number of days of history to load (default: 7)

        Returns:
            List of SubmissionRecord objects, oldest first.
        """
        records: list[SubmissionRecord] = []
        now = datetime.now(timezone.utc)

        for day_offset in range(days - 1, -1, -1):
            date = now - timedelta(days=day_offset)
            path = self._date_path(date)

            if not os.path.exists(path):
                continue

            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            records.append(SubmissionRecord(**data))
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning("Corrupt history record in %s: %s", path, e)
                            continue
            except OSError:
                continue

        return records

    def summary(self, days: int = 1) -> dict:
        """Compute aggregate stats over the last N days.

        Args:
            days: Number of days to summarize (default: 1)

        Returns:
            Dict with summary statistics including:
            - total: Total submissions
            - accepted: Number accepted
            - acceptance_rate: Fraction accepted
            - avg_danger: Average danger score (accepted only)
            - avg_novelty: Average novelty score
            - avg_latency_ms: Average latency
            - per_experiment: Breakdown by experiment_id
            - rejections: Breakdown by rejection_reason
        """
        records = self.load(days=days)

        if not records:
            return {
                "total": 0,
                "accepted": 0,
                "acceptance_rate": 0.0,
                "avg_danger": 0.0,
                "avg_novelty": 0.0,
                "avg_latency_ms": 0.0,
                "per_experiment": {},
                "rejections": {},
            }

        total = len(records)
        successful = [r for r in records if r.success]
        accepted = [r for r in successful if r.accepted]
        rejected = [r for r in successful if r.accepted is False]

        # Danger scores from accepted submissions
        danger_scores = [r.danger_score for r in accepted if r.danger_score is not None]
        avg_danger = sum(danger_scores) / len(danger_scores) if danger_scores else 0.0

        # Novelty scores
        novelty_scores = [r.novelty_avg for r in successful if r.novelty_avg is not None]
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        # Latency
        latencies = [r.latency_ms for r in successful if r.latency_ms > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Per-experiment breakdown
        experiments: dict[str, dict] = {}
        for r in records:
            exp = r.experiment_id
            if exp not in experiments:
                experiments[exp] = {"total": 0, "accepted": 0, "danger_scores": [], "novelty_scores": []}
            experiments[exp]["total"] += 1
            if r.accepted:
                experiments[exp]["accepted"] += 1
                if r.danger_score is not None:
                    experiments[exp]["danger_scores"].append(r.danger_score)
            if r.novelty_avg is not None:
                experiments[exp]["novelty_scores"].append(r.novelty_avg)

        per_experiment = {}
        for exp, stats in experiments.items():
            ds = stats["danger_scores"]
            ns = stats["novelty_scores"]
            per_experiment[exp] = {
                "total": stats["total"],
                "accepted": stats["accepted"],
                "acceptance_rate": stats["accepted"] / stats["total"] if stats["total"] > 0 else 0.0,
                "avg_danger": sum(ds) / len(ds) if ds else 0.0,
                "avg_novelty": sum(ns) / len(ns) if ns else 0.0,
            }

        # Rejection breakdown
        rejections: dict[str, int] = {}
        for r in rejected:
            reason = r.rejection_reason or "unknown"
            rejections[reason] = rejections.get(reason, 0) + 1

        # Count errors too
        errors = [r for r in records if not r.success]
        if errors:
            rejections["error"] = len(errors)

        return {
            "total": total,
            "accepted": len(accepted),
            "acceptance_rate": len(accepted) / total if total > 0 else 0.0,
            "avg_danger": avg_danger,
            "avg_novelty": avg_novelty,
            "avg_latency_ms": avg_latency,
            "per_experiment": per_experiment,
            "rejections": rejections,
        }

    def format_table(self, count: int = 20, use_colors: bool = True) -> str:
        """Format recent submissions as a readable table.

        Args:
            count: Number of recent submissions to show
            use_colors: Whether to use ANSI colors

        Returns:
            Formatted string with table and summary header.
        """
        GREEN = "\033[92m" if use_colors else ""
        RED = "\033[91m" if use_colors else ""
        BOLD = "\033[1m" if use_colors else ""
        RESET = "\033[0m" if use_colors else ""

        records = self.load(days=7)
        if not records:
            return "No submission history found."

        # Get most recent N
        recent = records[-count:]
        recent.reverse()  # Most recent first

        # 24h summary
        summary = self.summary(days=1)
        total = summary["total"]
        accepted = summary["accepted"]
        rate = summary["acceptance_rate"]
        avg_danger = summary["avg_danger"]

        lines: list[str] = []
        lines.append(
            f"{BOLD}Last {len(recent)} submissions{RESET} "
            f"(24h: {accepted}/{total} accepted, {rate:.0%} rate, avg danger {avg_danger:.4f})"
        )
        lines.append("")

        # Header
        header = f"{'TIME':>8}  {'EXP':>8}  {'UID':>4}  {'DANGER':>7}  {'ACCEPTED':>8}  {'NOVELTY':>7}  {'HIT_RATE':>8}  {'LATENCY':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for r in recent:
            # Parse time from ISO timestamp
            try:
                ts = datetime.fromisoformat(r.timestamp)
                time_str = ts.strftime("%H:%M:%S")
            except (ValueError, TypeError):
                time_str = "??:??:??"

            exp = r.experiment_id[:8] if r.experiment_id else "?"
            uid_str = str(r.validator_uid)

            # Build accepted column — pad BEFORE adding color to avoid
            # ANSI escape codes breaking alignment
            if not r.success:
                danger_str = f"{'ERR':>7}"
                accepted_str = f"{RED}{'FAIL':>8}{RESET}"
            elif r.danger_score is not None:
                danger_str = f"{r.danger_score:>7.4f}"
                if r.accepted:
                    accepted_str = f"{GREEN}{'YES':>8}{RESET}"
                else:
                    accepted_str = f"{RED}{'NO':>8}{RESET}"
            else:
                danger_str = f"{'-':>7}"
                accepted_str = f"{'-':>8}"

            novelty_str = f"{r.novelty_avg:.2f}" if r.novelty_avg is not None else "   -"
            hit_str = f"{r.hit_rate:.1%}" if r.hit_rate is not None else "   -"
            latency_str = f"{r.latency_ms:.0f}ms" if r.latency_ms > 0 else "   -"

            lines.append(
                f"{time_str:>8}  {exp:>8}  {uid_str:>4}  {danger_str}  {accepted_str}  "
                f"{novelty_str:>7}  {hit_str:>8}  {latency_str:>8}"
            )

        return "\n".join(lines)
