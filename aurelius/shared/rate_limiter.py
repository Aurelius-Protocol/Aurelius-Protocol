"""Rate limiting for validator request management."""

import time
from dataclasses import dataclass
from threading import Lock

import bittensor as bt


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
            hotkey: Miner's hotkey

        Returns:
            Tuple of (allowed, reason, remaining_quota)
        """
        with self.lock:
            current_time = time.time()

            # Initialize if first request from this miner
            if hotkey not in self.miner_requests:
                self.miner_requests[hotkey] = []

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
                return False, reason, 0

            return True, "", remaining

    def record_request(self, hotkey: str) -> None:
        """
        Record a request from a miner.

        Args:
            hotkey: Miner's hotkey
        """
        with self.lock:
            current_time = time.time()

            if hotkey not in self.miner_requests:
                self.miner_requests[hotkey] = []

            self.miner_requests[hotkey].append(RequestRecord(timestamp=current_time, hotkey=hotkey))
