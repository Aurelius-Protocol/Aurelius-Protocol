"""HTTP client with connection pooling and circuit breaker for telemetry.

Provides a singleton HTTP session with:
- Connection pooling via HTTPAdapter
- Automatic retry with exponential backoff
- Circuit breaker to prevent wasted retries when API is down
"""

from __future__ import annotations

import enum
import threading
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests allowed
    OPEN = "open"          # Failures exceeded threshold, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures.

    When failures exceed the threshold, the circuit opens and blocks
    requests for a recovery timeout period. After the timeout, it
    enters half-open state and allows a test request through.

    Thread-safe implementation.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before half-open state
            half_open_max_calls: Max test calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if self._last_failure_time and \
                   time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if circuit is open
        """
        state = self.state  # This also updates state if timeout elapsed

        with self._lock:
            if state == CircuitState.CLOSED:
                return True
            elif state == CircuitState.HALF_OPEN:
                # Allow limited test requests in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Success in half-open state closes the circuit
                self._state = CircuitState.CLOSED
            # Reset failure count on success
            self._failure_count = 0
            self._last_failure_time = None

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open state reopens the circuit
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                # Too many failures, open the circuit
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def is_open(self) -> bool:
        """Check if circuit is currently blocking requests."""
        return self.state == CircuitState.OPEN


class TelemetryHTTPClient:
    """Singleton HTTP client with connection pooling for telemetry.

    Provides efficient HTTP connection reuse across all telemetry
    requests instead of creating new connections for each request.
    """

    _instance: Optional["TelemetryHTTPClient"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TelemetryHTTPClient":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialize()
                    cls._instance = instance
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the HTTP session with connection pooling."""
        self._session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,  # 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
            raise_on_status=False,  # Don't raise, let caller handle
        )

        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,  # Number of connection pools
            pool_maxsize=20,      # Max connections per pool
            max_retries=retry_strategy,
        )

        # Mount for both HTTP and HTTPS
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Circuit breaker for API availability
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
        )

    @property
    def session(self) -> requests.Session:
        """Get the shared HTTP session."""
        return self._session

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker instance."""
        return self._circuit_breaker

    def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            self._session.close()


def get_telemetry_session() -> requests.Session:
    """Get the shared telemetry HTTP session.

    Returns:
        requests.Session with connection pooling configured
    """
    return TelemetryHTTPClient().session


def get_circuit_breaker() -> CircuitBreaker:
    """Get the shared circuit breaker.

    Returns:
        CircuitBreaker instance for telemetry requests
    """
    return TelemetryHTTPClient().circuit_breaker
