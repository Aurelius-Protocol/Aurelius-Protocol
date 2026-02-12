"""Pre-flight health checks for validator connectivity."""

import socket
import threading
import time
from dataclasses import dataclass

from aurelius.miner.errors import (
    ErrorCategory,
    MinerError,
    classify_error,
    create_error,
)


@dataclass
class HealthCheckResult:
    """Result of a validator health check."""

    is_healthy: bool
    latency_ms: float | None = None
    error: MinerError | None = None
    details: dict | None = None


class ValidatorHealthChecker:
    """Pre-flight validator connectivity checker."""

    def __init__(self, timeout: float = 5.0):
        """
        Initialize health checker.

        Args:
            timeout: Timeout in seconds for health checks
        """
        self.timeout = timeout

    def check_dns_resolution(self, host: str) -> HealthCheckResult:
        """
        Verify DNS resolution works for the host.

        Args:
            host: Hostname to resolve

        Returns:
            HealthCheckResult with resolution status
        """
        # Skip DNS check for IP addresses
        if self._is_ip_address(host):
            return HealthCheckResult(
                is_healthy=True,
                details={"host": host, "type": "ip_address"},
            )

        context = {"host": host}
        start_time = time.time()

        try:
            socket.getaddrinfo(host, None, socket.AF_INET)
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                is_healthy=True,
                latency_ms=latency_ms,
                details={"host": host, "type": "dns_resolved"},
            )
        except socket.gaierror as e:
            return HealthCheckResult(
                is_healthy=False,
                error=create_error(ErrorCategory.DNS_RESOLUTION, context, e),
                details={"host": host},
            )
        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                error=classify_error(e, context),
                details={"host": host},
            )

    def check_tcp_connectivity(self, host: str, port: int) -> HealthCheckResult:
        """
        Test TCP connection to host:port.

        Args:
            host: Host to connect to
            port: Port to connect to

        Returns:
            HealthCheckResult with connectivity status
        """
        context = {"host": host, "port": port}
        start_time = time.time()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)

            result = sock.connect_ex((host, port))
            latency_ms = (time.time() - start_time) * 1000
            sock.close()

            if result == 0:
                return HealthCheckResult(
                    is_healthy=True,
                    latency_ms=latency_ms,
                    details={"host": host, "port": port},
                )
            else:
                # Connection failed with error code
                error_msg = self._get_socket_error_message(result)
                context["error"] = error_msg

                if result in (111, 10061):  # Connection refused (Linux/Windows)
                    error = create_error(
                        ErrorCategory.CONNECTION_REFUSED,
                        context,
                        OSError(result, error_msg),
                    )
                elif result in (110, 10060):  # Connection timed out
                    context["timeout"] = self.timeout
                    error = create_error(
                        ErrorCategory.NETWORK_TIMEOUT,
                        context,
                        OSError(result, error_msg),
                    )
                else:
                    error = create_error(
                        ErrorCategory.CONNECTION_REFUSED,
                        context,
                        OSError(result, error_msg),
                    )

                return HealthCheckResult(
                    is_healthy=False,
                    error=error,
                    details={"host": host, "port": port, "errno": result},
                )

        except TimeoutError:
            context["timeout"] = self.timeout
            return HealthCheckResult(
                is_healthy=False,
                error=create_error(
                    ErrorCategory.NETWORK_TIMEOUT,
                    context,
                    TimeoutError(f"Connection timed out after {self.timeout}s"),
                ),
                details={"host": host, "port": port},
            )
        except ConnectionRefusedError as e:
            return HealthCheckResult(
                is_healthy=False,
                error=create_error(ErrorCategory.CONNECTION_REFUSED, context, e),
                details={"host": host, "port": port},
            )
        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                error=classify_error(e, context),
                details={"host": host, "port": port},
            )

    def check_validator_reachable(
        self,
        host: str,
        port: int,
    ) -> HealthCheckResult:
        """
        Perform full validator reachability check.

        This performs:
        1. DNS resolution (if hostname)
        2. TCP connectivity test
        3. HTTP readiness probe (best-effort)

        Args:
            host: Validator host (IP or hostname)
            port: Validator port

        Returns:
            HealthCheckResult with overall status
        """
        # Step 1: DNS resolution (if hostname)
        if not self._is_ip_address(host):
            dns_result = self.check_dns_resolution(host)
            if not dns_result.is_healthy:
                return dns_result

        # Step 2: TCP connectivity
        tcp_result = self.check_tcp_connectivity(host, port)
        if not tcp_result.is_healthy:
            return tcp_result

        # Step 3: HTTP readiness probe (best-effort)
        http_result = self._check_http_readiness(host, port)
        if http_result is not None and not http_result.is_healthy:
            return http_result

        return tcp_result

    def _check_http_readiness(self, host: str, port: int) -> HealthCheckResult | None:
        """Best-effort HTTP readiness check. Returns None if inconclusive."""
        import urllib.error
        import urllib.request

        try:
            url = f"http://{host}:{port}/"
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 500:
                    return HealthCheckResult(
                        is_healthy=False,
                        error=create_error(
                            ErrorCategory.API_ERROR,
                            {"host": host, "port": port},
                            Exception(f"Validator returned HTTP {resp.status}"),
                        ),
                        details={"host": host, "port": port, "http_status": resp.status},
                    )
        except urllib.error.HTTPError as e:
            # 4xx errors (405 Method Not Allowed, etc.) mean the server is up
            if e.code >= 500:
                return HealthCheckResult(
                    is_healthy=False,
                    error=create_error(
                        ErrorCategory.API_ERROR,
                        {"host": host, "port": port},
                        e,
                    ),
                    details={"host": host, "port": port, "http_status": e.code},
                )
        except Exception:
            # HTTP check is best-effort; TCP already passed, so don't fail
            pass
        return None

    def _is_ip_address(self, host: str) -> bool:
        """Check if host is an IP address (not hostname)."""
        try:
            socket.inet_aton(host)
            return True
        except OSError:
            pass

        # Check IPv6
        try:
            socket.inet_pton(socket.AF_INET6, host)
            return True
        except OSError:
            pass

        return False

    def _get_socket_error_message(self, errno: int) -> str:
        """Get human-readable error message for socket error code."""
        error_messages = {
            # Linux/macOS error codes
            111: "Connection refused",
            110: "Connection timed out",
            113: "No route to host",
            101: "Network is unreachable",
            # Windows error codes
            10061: "Connection refused",
            10060: "Connection timed out",
            10065: "No route to host",
            10051: "Network is unreachable",
        }
        return error_messages.get(errno, f"Socket error {errno}")


@dataclass
class ValidatorStatus:
    """Status of a validator in the cache."""

    uid: int
    unhealthy_until: float  # Unix timestamp when validator can be retried
    failure_count: int = 1
    last_error: str | None = None


class ValidatorStatusCache:
    """
    Cache validator health status to avoid repeated failed connections.

    When a validator connection fails, it's marked as unhealthy for a cooldown
    period. Subsequent requests check this cache first to skip known-bad validators.

    Uses exponential backoff for repeated failures:
    - 1st failure: 60s cooldown
    - 2nd failure: 120s cooldown
    - 3rd+ failure: 300s cooldown (max)
    """

    # Cooldown durations for exponential backoff (seconds)
    COOLDOWN_BASE = 60.0
    COOLDOWN_MULTIPLIER = 2.0
    COOLDOWN_MAX = 300.0  # 5 minutes max

    def __init__(self):
        """Initialize the validator status cache."""
        self._unhealthy: dict[int, ValidatorStatus] = {}
        self._lock = threading.RLock()

    def mark_unhealthy(
        self,
        uid: int,
        error: str | None = None,
    ) -> float:
        """
        Mark a validator as unhealthy.

        Uses exponential backoff for repeated failures.

        Args:
            uid: Validator UID
            error: Optional error message for logging

        Returns:
            The cooldown duration in seconds
        """
        with self._lock:
            now = time.time()

            # Get existing status if any
            existing = self._unhealthy.get(uid)

            if existing:
                # Increment failure count for exponential backoff
                failure_count = existing.failure_count + 1
            else:
                failure_count = 1

            # Calculate cooldown with exponential backoff
            cooldown = min(
                self.COOLDOWN_BASE * (self.COOLDOWN_MULTIPLIER ** (failure_count - 1)),
                self.COOLDOWN_MAX,
            )

            self._unhealthy[uid] = ValidatorStatus(
                uid=uid,
                unhealthy_until=now + cooldown,
                failure_count=failure_count,
                last_error=error,
            )

            return cooldown

    def mark_healthy(self, uid: int) -> None:
        """
        Mark a validator as healthy, removing it from the unhealthy cache.

        Args:
            uid: Validator UID
        """
        with self._lock:
            if uid in self._unhealthy:
                del self._unhealthy[uid]

    def is_available(self, uid: int) -> bool:
        """
        Check if a validator is available (not in cooldown).

        Args:
            uid: Validator UID

        Returns:
            True if validator is available, False if in cooldown
        """
        with self._lock:
            status = self._unhealthy.get(uid)
            if status is None:
                return True

            now = time.time()
            if now >= status.unhealthy_until:
                # Cooldown expired, allow retry
                # Don't remove from cache yet - let success remove it
                return True

            return False

    def get_cooldown_remaining(self, uid: int) -> float:
        """
        Get remaining cooldown time for a validator.

        Args:
            uid: Validator UID

        Returns:
            Remaining cooldown seconds, or 0 if not in cooldown
        """
        with self._lock:
            status = self._unhealthy.get(uid)
            if status is None:
                return 0.0

            remaining = status.unhealthy_until - time.time()
            return max(0.0, remaining)

    def get_status(self, uid: int) -> ValidatorStatus | None:
        """
        Get the status of a specific validator.

        Args:
            uid: Validator UID

        Returns:
            ValidatorStatus if in cache, None otherwise
        """
        with self._lock:
            return self._unhealthy.get(uid)

    def get_unhealthy_count(self) -> int:
        """Get count of validators currently in cooldown."""
        with self._lock:
            now = time.time()
            return sum(
                1 for status in self._unhealthy.values()
                if status.unhealthy_until > now
            )

    def get_all_unhealthy(self) -> list[ValidatorStatus]:
        """
        Get all validators currently in cooldown.

        Returns:
            List of ValidatorStatus for unhealthy validators
        """
        with self._lock:
            now = time.time()
            return [
                status for status in self._unhealthy.values()
                if status.unhealthy_until > now
            ]

    def clear(self) -> None:
        """Clear all cached statuses."""
        with self._lock:
            self._unhealthy.clear()

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            expired = [
                uid for uid, status in self._unhealthy.items()
                if status.unhealthy_until <= now
            ]
            for uid in expired:
                del self._unhealthy[uid]
            return len(expired)


# Global singleton instance
_validator_status_cache: ValidatorStatusCache | None = None


def get_validator_status_cache() -> ValidatorStatusCache:
    """Get the global validator status cache singleton."""
    global _validator_status_cache
    if _validator_status_cache is None:
        _validator_status_cache = ValidatorStatusCache()
    return _validator_status_cache
