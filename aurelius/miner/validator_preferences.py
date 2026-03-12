"""Validator selection intelligence for miners.

Tracks per-validator performance (acceptance rate, latency, reliability) and
re-ranks validators using a composite score that blends stake with historical outcomes.
"""

import json
import os
import time
from dataclasses import asdict, dataclass


@dataclass
class ValidatorPerformance:
    """Tracked performance metrics for a single validator."""

    uid: int
    total_queries: int = 0
    successful_queries: int = 0
    total_latency_ms: float = 0.0
    accepted_count: int = 0
    rejected_count: int = 0
    error_count: int = 0
    last_seen: float = 0.0  # Unix timestamp

    @property
    def acceptance_rate(self) -> float:
        """Fraction of successful queries that were accepted."""
        if self.successful_queries == 0:
            return 0.0
        return self.accepted_count / self.successful_queries

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all queries."""
        if self.total_queries == 0:
            return 0.0
        return self.total_latency_ms / self.total_queries

    @property
    def success_rate(self) -> float:
        """Fraction of queries that got a response (success or rejection, not error)."""
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries


class ValidatorPreferences:
    """Manages per-validator performance data and intelligent ranking.

    Args:
        data_dir: Base directory for persistence (default: ~/.aurelius)
        stake_weight: Weight for stake in composite score (default: 0.6)
        acceptance_weight: Weight for acceptance rate (default: 0.2)
        reliability_weight: Weight for success rate (default: 0.15)
        latency_penalty: Penalty weight for latency (default: 0.05)
        decay_rate: Daily decay rate for stale data (default: 0.95)
        stale_days: Days after which stats are reset (default: 7)
    """

    def __init__(
        self,
        data_dir: str | None = None,
        stake_weight: float = 0.6,
        acceptance_weight: float = 0.2,
        reliability_weight: float = 0.15,
        latency_penalty: float = 0.05,
        decay_rate: float = 0.95,
        stale_days: int = 7,
    ):
        if data_dir is None:
            data_dir = os.path.expanduser("~/.aurelius")
        self._path = os.path.join(data_dir, "validator_prefs.json")
        self.validators: dict[int, ValidatorPerformance] = {}

        self.stake_weight = stake_weight
        self.acceptance_weight = acceptance_weight
        self.reliability_weight = reliability_weight
        self.latency_penalty = latency_penalty
        self.decay_rate = decay_rate
        self.stale_days = stale_days

        self.load()

    def update(self, uid: int, success: bool, accepted: bool | None = None, latency_ms: float = 0.0) -> None:
        """Update stats for a validator based on a query result.

        Does NOT auto-save. Call save() after a batch of updates to persist.

        Args:
            uid: Validator UID
            success: Whether the query got a response (not an error)
            accepted: Whether the submission was accepted (None if unknown/error)
            latency_ms: Query latency in milliseconds
        """
        if uid not in self.validators:
            self.validators[uid] = ValidatorPerformance(uid=uid)

        perf = self.validators[uid]
        perf.total_queries += 1
        perf.total_latency_ms += latency_ms
        perf.last_seen = time.time()

        if success:
            perf.successful_queries += 1
            if accepted is True:
                perf.accepted_count += 1
            elif accepted is False:
                perf.rejected_count += 1
        else:
            perf.error_count += 1

    def score(self, uid: int, stake_normalized: float) -> float:
        """Compute a composite selection score for a validator.

        Args:
            uid: Validator UID
            stake_normalized: Stake value normalized to [0, 1]

        Returns:
            Composite score (higher = better)
        """
        perf = self.validators.get(uid)
        if perf is None or perf.total_queries == 0:
            # Cold start: use stake only
            return self.stake_weight * stake_normalized

        # Time-based decay (clamp to 0 to handle clock skew)
        days_since = max(0.0, (time.time() - perf.last_seen) / 86400.0)
        weight = self.decay_rate**days_since

        # Normalize latency: map to [0, 1] where lower is better
        # Use 10000ms as the "worst" reference latency
        norm_latency = min(perf.avg_latency_ms / 10000.0, 1.0)

        return weight * (
            self.stake_weight * stake_normalized
            + self.acceptance_weight * perf.acceptance_rate
            + self.reliability_weight * perf.success_rate
            - self.latency_penalty * norm_latency
        )

    def rank(self, candidates: list[tuple[int, float]]) -> list[int]:
        """Re-rank validator candidates using composite scores.

        Args:
            candidates: List of (uid, stake) tuples

        Returns:
            List of UIDs sorted by composite score (highest first)
        """
        if not candidates:
            return []

        # Normalize stakes to [0, 1]
        max_stake = max(stake for _, stake in candidates)
        if max_stake <= 0:
            max_stake = 1.0

        scored = []
        for uid, stake in candidates:
            stake_norm = stake / max_stake
            s = self.score(uid, stake_norm)
            scored.append((uid, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in scored]

    def save(self) -> None:
        """Persist preferences to disk."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        data = {str(uid): asdict(perf) for uid, perf in self.validators.items()}
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
            os.replace(tmp_path, self._path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def load(self) -> None:
        """Load preferences from disk."""
        if not os.path.exists(self._path):
            return

        try:
            with open(self._path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.validators = {}
            return

        now = time.time()
        stale_cutoff = now - (self.stale_days * 86400)

        for _uid_str, perf_data in data.items():
            try:
                perf = ValidatorPerformance(**perf_data)
            except TypeError:
                continue  # Skip entries with unexpected fields
            # Skip stale entries
            if perf.last_seen < stale_cutoff:
                continue
            self.validators[perf.uid] = perf

    def get_stats_table(self, use_colors: bool = True) -> str:
        """Format per-validator stats as a readable table.

        Args:
            use_colors: Whether to use ANSI colors

        Returns:
            Formatted string with validator performance table.
        """
        GREEN = "\033[92m" if use_colors else ""
        RED = "\033[91m" if use_colors else ""
        YELLOW = "\033[93m" if use_colors else ""
        BOLD = "\033[1m" if use_colors else ""
        RESET = "\033[0m" if use_colors else ""

        if not self.validators:
            return "No validator performance data yet."

        lines: list[str] = []
        lines.append(f"{BOLD}Validator Performance Stats{RESET}")
        lines.append("")

        header = f"{'UID':>5}  {'QUERIES':>7}  {'SUCCESS':>7}  {'ACCEPTED':>8}  {'ACC_RATE':>8}  {'AVG_LAT':>8}  {'LAST_SEEN':>10}"
        lines.append(header)
        lines.append("-" * len(header))

        # Sort by total queries descending
        sorted_perfs = sorted(self.validators.values(), key=lambda p: p.total_queries, reverse=True)

        for perf in sorted_perfs:
            acc_rate = perf.acceptance_rate
            if acc_rate >= 0.7:
                rate_color = GREEN
            elif acc_rate >= 0.4:
                rate_color = YELLOW
            else:
                rate_color = RED

            rate_str = f"{rate_color}{acc_rate:.0%}{RESET}" if use_colors else f"{acc_rate:.0%}"
            lat_str = f"{perf.avg_latency_ms:.0f}ms" if perf.total_queries > 0 else "-"

            # Time since last seen
            if perf.last_seen > 0:
                hours_ago = (time.time() - perf.last_seen) / 3600
                if hours_ago < 1:
                    seen_str = f"{hours_ago * 60:.0f}m ago"
                elif hours_ago < 24:
                    seen_str = f"{hours_ago:.0f}h ago"
                else:
                    seen_str = f"{hours_ago / 24:.0f}d ago"
            else:
                seen_str = "never"

            lines.append(
                f"{perf.uid:>5}  {perf.total_queries:>7}  {perf.successful_queries:>7}  "
                f"{perf.accepted_count:>8}  {rate_str:>8}  {lat_str:>8}  {seen_str:>10}"
            )

        return "\n".join(lines)
