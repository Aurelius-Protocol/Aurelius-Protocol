"""Rate limiting for validator request management."""

import json
import os
import time
from dataclasses import dataclass
from threading import Lock

import bittensor as bt

from aurelius.shared.config import Config


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        max_requests: Maximum number of requests allowed in the time window
        window_hours: Time window in hours
    """

    max_requests: int
    window_hours: float


@dataclass
class RequestRecord:
    """Record of a request for rate limiting.

    Attributes:
        timestamp: Unix timestamp when the request was made
        hotkey: Hotkey of the requester (optional, for per-miner tracking)
    """

    timestamp: float
    hotkey: str | None = None


class RateLimiter:
    """Global rate limiter using sliding window algorithm."""

    def __init__(self, config: RateLimitConfig):
        """
        Initialize the rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.requests: list = []
        self.lock = Lock()

        bt.logging.info(f"Rate limiter initialized: {config.max_requests} requests per {config.window_hours} hour(s)")

    def _clean_old_requests(self, current_time: float) -> None:
        """
        Remove requests outside the current time window.

        Args:
            current_time: Current Unix timestamp
        """
        window_seconds = self.config.window_hours * 3600
        cutoff_time = current_time - window_seconds

        # Remove requests older than the cutoff
        self.requests = [r for r in self.requests if r.timestamp > cutoff_time]

    def check_rate_limit(self, hotkey: str | None = None) -> tuple[bool, str, int]:
        """
        Check if a request should be allowed based on rate limits.

        Args:
            hotkey: Optional hotkey of the requester for tracking

        Returns:
            Tuple of (allowed, reason, remaining_quota)
            - allowed: True if request should be allowed
            - reason: Explanation if not allowed
            - remaining_quota: Number of requests remaining in window
        """
        with self.lock:
            current_time = time.time()

            # Clean up old requests
            self._clean_old_requests(current_time)

            # Check if we're at the limit
            current_count = len(self.requests)
            remaining = self.config.max_requests - current_count

            if current_count >= self.config.max_requests:
                reason = (
                    f"Rate limit exceeded: {current_count}/{self.config.max_requests} "
                    f"requests in the last {self.config.window_hours} hour(s)"
                )
                return False, reason, 0

            return True, "", remaining

    def record_request(self, hotkey: str | None = None) -> None:
        """
        Record a new request for rate limiting.

        This should be called after check_rate_limit() confirms
        the request is allowed.

        Args:
            hotkey: Optional hotkey of the requester
        """
        with self.lock:
            current_time = time.time()
            self.requests.append(RequestRecord(timestamp=current_time, hotkey=hotkey))

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current stats
        """
        with self.lock:
            current_time = time.time()
            self._clean_old_requests(current_time)

            current_count = len(self.requests)
            remaining = self.config.max_requests - current_count

            # Calculate time until oldest request expires
            if self.requests:
                oldest_timestamp = min(r.timestamp for r in self.requests)
                window_seconds = self.config.window_hours * 3600
                reset_time = oldest_timestamp + window_seconds
                time_until_reset = max(0, reset_time - current_time)
            else:
                time_until_reset = 0

            return {
                "current_count": current_count,
                "max_requests": self.config.max_requests,
                "remaining": remaining,
                "window_hours": self.config.window_hours,
                "time_until_reset_seconds": time_until_reset,
            }


class PerMinerRateLimiter:
    """Rate limiter with per-miner quotas.

    This can be used for more sophisticated rate limiting where
    each miner gets their own quota.
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize per-miner rate limiter.

        Args:
            config: Rate limit configuration (applied per miner)
        """
        self.config = config
        self.miner_requests: dict = {}
        self.lock = Lock()

        bt.logging.info(
            f"Per-miner rate limiter initialized: {config.max_requests} requests "
            f"per {config.window_hours} hour(s) per miner"
        )

    def _clean_old_requests(self, current_time: float, hotkey: str) -> None:
        """
        Remove old requests for a specific miner.

        Args:
            current_time: Current Unix timestamp
            hotkey: Miner's hotkey
        """
        if hotkey not in self.miner_requests:
            return

        window_seconds = self.config.window_hours * 3600
        cutoff_time = current_time - window_seconds

        self.miner_requests[hotkey] = [r for r in self.miner_requests[hotkey] if r.timestamp > cutoff_time]

    def check_rate_limit(self, hotkey: str) -> tuple[bool, str, int]:
        """
        Check if a request from a specific miner should be allowed.

        Args:
            hotkey: Miner's hotkey (must be valid non-empty string)

        Returns:
            Tuple of (allowed, reason, remaining_quota)
        """
        # SECURITY: Validate hotkey to prevent bypass via None/empty values
        if not hotkey or not isinstance(hotkey, str) or len(hotkey.strip()) == 0:
            bt.logging.warning(f"🚫 Rate limiter: Invalid hotkey provided: {repr(hotkey)}")
            return False, "Invalid or missing hotkey", 0

        # Basic SS58 address format validation (Bittensor hotkeys are 48 chars)
        if len(hotkey) < 46 or len(hotkey) > 50:
            bt.logging.warning(f"🚫 Rate limiter: Suspicious hotkey format: {hotkey[:20]}...")
            return False, "Invalid hotkey format", 0

        with self.lock:
            current_time = time.time()

            # Initialize if first request from this miner
            is_new_miner = hotkey not in self.miner_requests
            if is_new_miner:
                self.miner_requests[hotkey] = []
                if Config.LOG_RATE_LIMIT_DETAILS:
                    bt.logging.info(f"📝 New miner connected: {hotkey[:16]}...")

            # Clean up old requests
            self._clean_old_requests(current_time, hotkey)

            # Check limit
            current_count = len(self.miner_requests[hotkey])
            remaining = self.config.max_requests - current_count

            if current_count >= self.config.max_requests:
                reason = (
                    f"Per-miner rate limit exceeded: {current_count}/{self.config.max_requests} "
                    f"requests in the last {self.config.window_hours} hour(s)"
                )
                if Config.LOG_RATE_LIMIT_DETAILS:
                    bt.logging.info(f"🚫 Rate limit EXCEEDED for miner {hotkey[:16]}... ({current_count}/{self.config.max_requests})")
                return False, reason, 0

            if Config.LOG_RATE_LIMIT_DETAILS:
                bt.logging.info(f"✅ Rate limit check PASSED for miner {hotkey[:16]}... ({current_count}/{self.config.max_requests}, {remaining} remaining)")

            return True, "", remaining

    def record_request(self, hotkey: str) -> None:
        """
        Record a request from a miner.

        Args:
            hotkey: Miner's hotkey (must be valid non-empty string)
        """
        # SECURITY: Validate hotkey to prevent tracking under invalid keys
        if not hotkey or not isinstance(hotkey, str) or len(hotkey.strip()) == 0:
            bt.logging.warning(f"Rate limiter: Refusing to record request for invalid hotkey: {repr(hotkey)}")
            return

        if len(hotkey) < 46 or len(hotkey) > 50:
            bt.logging.warning(f"Rate limiter: Refusing to record request for suspicious hotkey: {hotkey[:20]}...")
            return

        with self.lock:
            current_time = time.time()

            if hotkey not in self.miner_requests:
                self.miner_requests[hotkey] = []

            self.miner_requests[hotkey].append(RequestRecord(timestamp=current_time, hotkey=hotkey))

            if Config.LOG_RATE_LIMIT_DETAILS:
                count = len(self.miner_requests[hotkey])
                bt.logging.info(f"📥 Request recorded for miner {hotkey[:16]}... ({count}/{self.config.max_requests})")

    def save_state(self, filepath: str) -> None:
        """Persist rate limiter state to disk."""
        with self.lock:
            current_time = time.time()
            window_seconds = self.config.window_hours * 3600
            cutoff = current_time - window_seconds
            state: dict = {}
            for hotkey, requests in self.miner_requests.items():
                recent = [r.timestamp for r in requests if r.timestamp > cutoff]
                if recent:
                    state[hotkey] = recent
            try:
                tmp_path = filepath + ".tmp"
                with open(tmp_path, 'w') as f:
                    json.dump({"saved_at": current_time, "requests": state}, f)
                os.replace(tmp_path, filepath)
            except Exception as e:
                bt.logging.warning(f"Failed to save rate limiter state: {e}")

    def load_state(self, filepath: str) -> None:
        """Restore rate limiter state from disk."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            current_time = time.time()
            window_seconds = self.config.window_hours * 3600
            cutoff = current_time - window_seconds
            with self.lock:
                for hotkey, timestamps in data.get("requests", {}).items():
                    recent = [RequestRecord(timestamp=t, hotkey=hotkey) for t in timestamps if t > cutoff]
                    if recent:
                        self.miner_requests[hotkey] = recent
            bt.logging.info(f"Rate limiter: restored state ({len(self.miner_requests)} miners)")
        except FileNotFoundError:
            pass
        except Exception as e:
            bt.logging.warning(f"Failed to load rate limiter state: {e}")

    def get_active_miners_count(self) -> int:
        """
        Get the number of miners with requests in the current window.

        Returns:
            Number of active miners
        """
        with self.lock:
            current_time = time.time()
            window_seconds = self.config.window_hours * 3600
            cutoff_time = current_time - window_seconds

            active_count = 0
            for hotkey in self.miner_requests:
                recent_requests = [r for r in self.miner_requests[hotkey] if r.timestamp > cutoff_time]
                if recent_requests:
                    active_count += 1

            return active_count

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current stats
        """
        with self.lock:
            current_time = time.time()
            window_seconds = self.config.window_hours * 3600
            cutoff_time = current_time - window_seconds

            total_requests = 0
            active_miners = 0

            for hotkey in self.miner_requests:
                recent_requests = [r for r in self.miner_requests[hotkey] if r.timestamp > cutoff_time]
                if recent_requests:
                    active_miners += 1
                    total_requests += len(recent_requests)

            return {
                "active_miners": active_miners,
                "total_miners_seen": len(self.miner_requests),
                "total_requests_in_window": total_requests,
                "max_requests_per_miner": self.config.max_requests,
                "window_hours": self.config.window_hours,
            }


@dataclass
class ExperimentRateLimitConfig:
    """Configuration for per-experiment rate limiting (T072).

    Attributes:
        max_requests: Maximum requests allowed per miner in the window
        window_seconds: Time window in seconds
    """

    max_requests: int
    window_seconds: float


class PerExperimentRateLimiter:
    """Rate limiter with per (hotkey, experiment_id) tracking (T072).

    This extends PerMinerRateLimiter with an experiment_id dimension,
    allowing different experiments to have independent rate limits
    for each miner.

    Features:
    - Isolation between experiments (hitting limit on exp-A doesn't affect exp-B)
    - Per-miner limits within each experiment
    - Configurable limits per experiment
    - Default limits for unconfigured experiments
    """

    # Default limit if experiment not configured
    DEFAULT_MAX_REQUESTS = 1000
    DEFAULT_WINDOW_SECONDS = 3600  # 1 hour

    def __init__(self):
        """Initialize per-experiment rate limiter."""
        # {experiment_id: {hotkey: [timestamp, ...]}}
        self._windows: dict[str, dict[str, list[float]]] = {}
        # {experiment_id: ExperimentRateLimitConfig}
        self._limits: dict[str, ExperimentRateLimitConfig] = {}
        self._lock = Lock()

        bt.logging.debug("Per-experiment rate limiter initialized")

    def set_experiment_limit(
        self,
        experiment_id: str,
        max_requests: int,
        window_seconds: float,
    ) -> None:
        """Set rate limit configuration for an experiment.

        Args:
            experiment_id: The experiment ID
            max_requests: Maximum requests per miner in window
            window_seconds: Time window in seconds
        """
        with self._lock:
            self._limits[experiment_id] = ExperimentRateLimitConfig(
                max_requests=max_requests,
                window_seconds=window_seconds,
            )
            bt.logging.debug(
                f"Set rate limit for '{experiment_id}': "
                f"{max_requests} requests per {window_seconds}s"
            )

    def _get_limit(self, experiment_id: str) -> ExperimentRateLimitConfig:
        """Get rate limit config for an experiment (uses default if not set)."""
        return self._limits.get(
            experiment_id,
            ExperimentRateLimitConfig(
                max_requests=self.DEFAULT_MAX_REQUESTS,
                window_seconds=self.DEFAULT_WINDOW_SECONDS,
            ),
        )

    def _clean_old_requests(
        self,
        experiment_id: str,
        hotkey: str,
        current_time: float,
        window_seconds: float,
    ) -> None:
        """Remove expired timestamps for a (hotkey, experiment) pair."""
        if experiment_id not in self._windows:
            return
        if hotkey not in self._windows[experiment_id]:
            return

        cutoff = current_time - window_seconds
        self._windows[experiment_id][hotkey] = [
            t for t in self._windows[experiment_id][hotkey] if t > cutoff
        ]

    def check(self, hotkey: str, experiment_id: str) -> bool:
        """Check if a request is allowed and record it if so.

        This is a convenience method that combines check and record.

        Args:
            hotkey: The miner's hotkey
            experiment_id: The experiment ID

        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        limit = self._get_limit(experiment_id)

        with self._lock:
            # Initialize structures
            if experiment_id not in self._windows:
                self._windows[experiment_id] = {}
            if hotkey not in self._windows[experiment_id]:
                self._windows[experiment_id][hotkey] = []

            # Clean old requests
            self._clean_old_requests(
                experiment_id, hotkey, current_time, limit.window_seconds
            )

            # Check limit
            timestamps = self._windows[experiment_id][hotkey]
            if len(timestamps) >= limit.max_requests:
                bt.logging.debug(
                    f"Rate limited: {hotkey[:16]}... on '{experiment_id}' "
                    f"({len(timestamps)}/{limit.max_requests})"
                )
                return False

            # Record this request
            timestamps.append(current_time)
            return True

    def get_experiment_stats(self, experiment_id: str) -> dict:
        """Get statistics for a specific experiment.

        Args:
            experiment_id: The experiment ID

        Returns:
            Dictionary with tracked_miners and total_requests
        """
        with self._lock:
            if experiment_id not in self._windows:
                return {"tracked_miners": 0, "total_requests": 0}

            exp_data = self._windows[experiment_id]
            current_time = time.time()
            limit = self._get_limit(experiment_id)

            # Count active miners and requests
            active_miners = 0
            total_requests = 0

            for hotkey, timestamps in exp_data.items():
                # Filter to current window
                cutoff = current_time - limit.window_seconds
                recent = [t for t in timestamps if t > cutoff]
                if recent:
                    active_miners += 1
                    total_requests += len(recent)

            return {
                "tracked_miners": active_miners,
                "total_requests": total_requests,
                "max_requests": limit.max_requests,
                "window_seconds": limit.window_seconds,
            }

    def get_all_stats(self) -> dict:
        """Get statistics for all experiments.

        Returns:
            Dictionary mapping experiment_id to stats
        """
        with self._lock:
            return {
                exp_id: self.get_experiment_stats(exp_id)
                for exp_id in self._windows
            }
