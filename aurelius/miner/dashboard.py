"""Performance dashboard for miner operators.

Aggregates submission history into rolling statistics, per-experiment breakdowns,
per-validator analysis, and trend detection. Renders as text or JSON.
"""

import json
from datetime import datetime, timedelta, timezone

from aurelius.miner.history import SubmissionHistory, SubmissionRecord


def _parse_period(period: str) -> int:
    """Parse a period string like '1h', '24h', '7d' into days (minimum 1 day for loading).

    Returns the number of days to load from history.
    """
    period = period.strip().lower()
    try:
        if period.endswith("h"):
            hours = int(period[:-1])
            return max(1, (hours + 23) // 24)  # Round up to nearest day
        elif period.endswith("d"):
            return max(1, int(period[:-1]))
    except (ValueError, IndexError):
        pass
    return 1


def _period_hours(period: str) -> float:
    """Parse period string into hours."""
    period = period.strip().lower()
    try:
        if period.endswith("h"):
            return max(1.0, float(period[:-1]))
        elif period.endswith("d"):
            return max(24.0, float(period[:-1]) * 24)
    except (ValueError, IndexError):
        pass
    return 24.0


def _filter_by_period(records: list[SubmissionRecord], period: str) -> list[SubmissionRecord]:
    """Filter records to only those within the given period."""
    hours = _period_hours(period)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    filtered = []
    for r in records:
        try:
            ts = datetime.fromisoformat(r.timestamp)
            if ts >= cutoff:
                filtered.append(r)
        except (ValueError, TypeError):
            continue
    return filtered


class Dashboard:
    """Performance dashboard built from submission history.

    Args:
        history: SubmissionHistory instance to read data from
    """

    def __init__(self, history: SubmissionHistory):
        self.history = history

    def compute_stats(
        self, period: str = "24h", experiment: str | None = None
    ) -> dict:
        """Compute rolling statistics for the given period.

        Args:
            period: Time period string (e.g., '1h', '24h', '7d')
            experiment: Optional experiment filter

        Returns:
            Dict with all computed statistics.
        """
        days = _parse_period(period)
        # Load extra days for trend comparison
        all_records = self.history.load(days=days * 2 + 1)

        current = _filter_by_period(all_records, period)
        if experiment:
            current = [r for r in current if r.experiment_id == experiment]

        # Previous period for trends
        hours = _period_hours(period)
        now = datetime.now(timezone.utc)
        prev_cutoff = now - timedelta(hours=hours * 2)
        curr_cutoff = now - timedelta(hours=hours)

        previous = []
        for r in all_records:
            try:
                ts = datetime.fromisoformat(r.timestamp)
                if prev_cutoff <= ts < curr_cutoff:
                    previous.append(r)
            except (ValueError, TypeError):
                continue
        if experiment:
            previous = [r for r in previous if r.experiment_id == experiment]

        stats = self._compute_period_stats(current)
        prev_stats = self._compute_period_stats(previous)
        stats["trends"] = self._compute_trends(stats, prev_stats)
        stats["per_experiment"] = self._per_experiment_stats(current)
        stats["per_validator"] = self._per_validator_stats(current)
        stats["period"] = period
        stats["record_count"] = len(current)

        return stats

    def _compute_period_stats(self, records: list[SubmissionRecord]) -> dict:
        """Compute aggregate stats for a set of records."""
        total = len(records)
        if total == 0:
            return {
                "total": 0,
                "accepted": 0,
                "acceptance_rate": 0.0,
                "avg_danger": 0.0,
                "avg_novelty": 0.0,
                "avg_latency_ms": 0.0,
                "errors": 0,
                "submissions_per_hour": 0.0,
            }

        successful = [r for r in records if r.success]
        accepted = [r for r in successful if r.accepted]
        errors = [r for r in records if not r.success]

        danger_scores = [r.danger_score for r in accepted if r.danger_score is not None]
        novelty_scores = [r.novelty_avg for r in successful if r.novelty_avg is not None]
        latencies = [r.latency_ms for r in successful if r.latency_ms > 0]

        # Time span
        if len(records) >= 2:
            try:
                first_ts = datetime.fromisoformat(records[0].timestamp)
                last_ts = datetime.fromisoformat(records[-1].timestamp)
                span_hours = max((last_ts - first_ts).total_seconds() / 3600, 0.01)
            except (ValueError, TypeError):
                span_hours = 1.0
        else:
            span_hours = 1.0

        return {
            "total": total,
            "accepted": len(accepted),
            "acceptance_rate": len(accepted) / total if total > 0 else 0.0,
            "avg_danger": sum(danger_scores) / len(danger_scores) if danger_scores else 0.0,
            "avg_novelty": sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "errors": len(errors),
            "submissions_per_hour": total / span_hours,
        }

    def _per_experiment_stats(self, records: list[SubmissionRecord]) -> dict[str, dict]:
        """Compute stats broken down by experiment."""
        by_exp: dict[str, list[SubmissionRecord]] = {}
        for r in records:
            by_exp.setdefault(r.experiment_id, []).append(r)

        result = {}
        for exp, recs in by_exp.items():
            result[exp] = self._compute_period_stats(recs)
        return result

    def _per_validator_stats(self, records: list[SubmissionRecord]) -> dict[int, dict]:
        """Compute stats broken down by validator UID."""
        by_val: dict[int, list[SubmissionRecord]] = {}
        for r in records:
            by_val.setdefault(r.validator_uid, []).append(r)

        result = {}
        for uid, recs in by_val.items():
            stats = self._compute_period_stats(recs)
            result[uid] = stats
        return result

    def _compute_trends(self, current: dict, previous: dict) -> dict:
        """Compare current vs previous period for trend detection."""
        trends = {}
        for key in ("acceptance_rate", "avg_danger", "avg_novelty"):
            curr_val = current.get(key, 0.0)
            prev_val = previous.get(key, 0.0)

            if previous.get("total", 0) == 0:
                trends[key] = {"direction": "new", "current": curr_val, "previous": prev_val}
            elif abs(curr_val - prev_val) < 0.02:
                trends[key] = {"direction": "stable", "current": curr_val, "previous": prev_val}
            elif curr_val > prev_val:
                trends[key] = {"direction": "up", "current": curr_val, "previous": prev_val}
            else:
                trends[key] = {"direction": "down", "current": curr_val, "previous": prev_val}

        return trends

    def render(
        self,
        period: str = "24h",
        experiment: str | None = None,
        json_output: bool = False,
        use_colors: bool = True,
    ) -> str:
        """Render the dashboard.

        Args:
            period: Time period (e.g., '1h', '24h', '7d')
            experiment: Optional experiment filter
            json_output: If True, return JSON instead of formatted text
            use_colors: Whether to use ANSI colors in text output

        Returns:
            Formatted dashboard string.
        """
        stats = self.compute_stats(period=period, experiment=experiment)

        if json_output:
            # Convert int keys to strings for JSON serialization
            if "per_validator" in stats:
                stats["per_validator"] = {str(k): v for k, v in stats["per_validator"].items()}
            return json.dumps(stats, indent=2)

        return self._render_text(stats, use_colors=use_colors)

    def _render_text(self, stats: dict, use_colors: bool = True) -> str:
        """Render dashboard as formatted text."""
        GREEN = "\033[92m" if use_colors else ""
        RED = "\033[91m" if use_colors else ""
        YELLOW = "\033[93m" if use_colors else ""
        CYAN = "\033[96m" if use_colors else ""
        BOLD = "\033[1m" if use_colors else ""
        RESET = "\033[0m" if use_colors else ""

        lines: list[str] = []
        period = stats.get("period", "24h")

        # Header
        lines.append(f"{BOLD}{'=' * 60}{RESET}")
        lines.append(f"{BOLD}MINER PERFORMANCE DASHBOARD ({period}){RESET}")
        lines.append(f"{BOLD}{'=' * 60}{RESET}")
        lines.append("")

        # Overview
        total = stats["total"]
        accepted = stats["accepted"]
        rate = stats["acceptance_rate"]
        errors = stats["errors"]

        lines.append(f"{BOLD}Overview{RESET}")
        lines.append(f"  Submissions:    {total}")
        lines.append(f"  Accepted:       {accepted} ({rate:.0%})")
        if errors > 0:
            lines.append(f"  Errors:         {RED}{errors}{RESET}")
        lines.append(f"  Avg Danger:     {stats['avg_danger']:.4f}")
        if stats["avg_novelty"] > 0:
            lines.append(f"  Avg Novelty:    {stats['avg_novelty']:.2f}")
        lines.append(f"  Avg Latency:    {stats['avg_latency_ms']:.0f}ms")
        lines.append(f"  Rate:           {stats['submissions_per_hour']:.1f}/hour")
        lines.append("")

        # Trends
        trends = stats.get("trends", {})
        if trends and any(t.get("direction") != "new" for t in trends.values()):
            lines.append(f"{BOLD}Trends (vs previous {period}){RESET}")
            for key, trend in trends.items():
                direction = trend["direction"]
                curr = trend["current"]
                prev = trend["previous"]
                label = key.replace("_", " ").title()

                if direction == "up":
                    arrow = f"{GREEN}^{RESET}" if use_colors else "^"
                    lines.append(f"  {label}: {curr:.2f} ({arrow} from {prev:.2f})")
                elif direction == "down":
                    arrow = f"{RED}v{RESET}" if use_colors else "v"
                    lines.append(f"  {label}: {curr:.2f} ({arrow} from {prev:.2f})")
                elif direction == "stable":
                    arrow = f"{YELLOW}={RESET}" if use_colors else "="
                    lines.append(f"  {label}: {curr:.2f} ({arrow} stable)")
                else:
                    lines.append(f"  {label}: {curr:.2f} (no prior data)")
            lines.append("")

        # Per-experiment breakdown
        per_exp = stats.get("per_experiment", {})
        if len(per_exp) > 1:
            lines.append(f"{BOLD}Per-Experiment Breakdown{RESET}")
            for exp, exp_stats in sorted(per_exp.items()):
                exp_rate = exp_stats["acceptance_rate"]
                exp_danger = exp_stats["avg_danger"]
                lines.append(
                    f"  {CYAN}{exp}{RESET}: {exp_stats['total']} submissions, "
                    f"{exp_rate:.0%} accepted, avg danger {exp_danger:.4f}"
                )
            lines.append("")

        # Per-validator breakdown
        per_val = stats.get("per_validator", {})
        if per_val:
            lines.append(f"{BOLD}Per-Validator Breakdown{RESET}")
            # Sort by total submissions descending
            sorted_vals = sorted(per_val.items(), key=lambda x: x[1]["total"], reverse=True)

            # Bar chart for acceptance rates
            max_bar = 20
            for uid, val_stats in sorted_vals[:10]:  # Top 10
                val_rate = val_stats["acceptance_rate"]
                val_total = val_stats["total"]
                bar_len = int(val_rate * max_bar)
                bar = "\u2588" * bar_len + "\u2591" * (max_bar - bar_len)

                if val_rate >= 0.7:
                    color = GREEN
                elif val_rate >= 0.4:
                    color = YELLOW
                else:
                    color = RED

                lines.append(
                    f"  UID {uid:>4}: {color}{bar}{RESET} {val_rate:.0%} ({val_total} queries)"
                )
            lines.append("")

        lines.append(f"{BOLD}{'=' * 60}{RESET}")
        return "\n".join(lines)
