"""Rate limiting for validator request management."""

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
                bt.logging.debug(f"✅ Rate limit check PASSED for miner {hotkey[:16]}... ({current_count}/{self.config.max_requests}, {remaining} remaining)")

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
                bt.logging.debug(f"📥 Request recorded for miner {hotkey[:16]}... ({count}/{self.config.max_requests})")

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
