"""Circuit breaker pattern for external service resilience.

When an external service fails repeatedly, the circuit breaker "opens" to prevent
further requests from being sent, allowing the service time to recover. After a
recovery period, it allows a test request through to probe recovery.

States:
- CLOSED: Normal operation, requests flow through
- OPEN: Circuit is open, requests fail immediately
- HALF_OPEN: Testing if service has recovered
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

import bittensor as bt

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, not attempting requests
    HALF_OPEN = "half_open"  # Testing recovery with limited requests


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of consecutive failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery (half-open)
        half_open_max_calls: Max concurrent calls allowed in half-open state
        success_threshold: Successes needed in half-open to close circuit
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 1
    success_threshold: int = 1


class CircuitBreakerOpen(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, name: str, time_until_retry: float):
        self.name = name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Retry in {time_until_retry:.1f}s"
        )


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    Usage:
        breaker = CircuitBreaker("openai-moderation")

        def call_api():
            return breaker.call(lambda: client.moderate(text))

        # Or with decorator:
        @breaker.protect
        def call_api():
            return client.moderate(text)
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker (for logging)
            config: Configuration options
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, transitioning if needed."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self._last_failure_time is None:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state for recovery testing."""
        bt.logging.info(
            f"Circuit breaker '{self.name}': OPEN -> HALF_OPEN (testing recovery)"
        )
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0

    def _transition_to_open(self) -> None:
        """Transition to open state after failures."""
        bt.logging.warning(
            f"Circuit breaker '{self.name}': -> OPEN "
            f"(failures: {self._failure_count}, recovery in {self.config.recovery_timeout}s)"
        )
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()

    def _transition_to_closed(self) -> None:
        """Transition to closed state after successful recovery."""
        bt.logging.info(f"Circuit breaker '{self.name}': -> CLOSED (service recovered)")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                bt.logging.warning(
                    f"Circuit breaker '{self.name}': HALF_OPEN -> OPEN (recovery failed)"
                )
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        with self._lock:
            current_state = self.state  # May trigger state transition

            if current_state == CircuitState.CLOSED:
                return True
            elif current_state == CircuitState.OPEN:
                return False
            elif current_state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            return False

    def get_time_until_retry(self) -> float:
        """Get seconds until circuit might allow retry."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0
            if self._last_failure_time is None:
                return 0.0
            elapsed = time.time() - self._last_failure_time
            remaining = self.config.recovery_timeout - elapsed
            return max(0.0, remaining)

    def call(self, func: Callable[[], T], fallback: Callable[[], T] | None = None) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            fallback: Optional fallback function if circuit is open

        Returns:
            Result from func or fallback

        Raises:
            CircuitBreakerOpen: If circuit is open and no fallback provided
        """
        if not self.can_execute():
            if fallback is not None:
                bt.logging.debug(
                    f"Circuit breaker '{self.name}' open, using fallback"
                )
                return fallback()
            raise CircuitBreakerOpen(self.name, self.get_time_until_retry())

        try:
            result = func()
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise

    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to protect a function with circuit breaker.

        Usage:
            @breaker.protect
            def call_api():
                return client.moderate(text)
        """
        def wrapper(*args, **kwargs):
            return self.call(lambda: func(*args, **kwargs))
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            bt.logging.info(f"Circuit breaker '{self.name}': manual reset to CLOSED")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "half_open_max_calls": self.config.half_open_max_calls,
                    "success_threshold": self.config.success_threshold,
                },
            }


# Global registry of circuit breakers for monitoring
_circuit_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """
    Get or create a named circuit breaker.

    This ensures only one circuit breaker exists per name.

    Args:
        name: Unique identifier for the circuit breaker
        config: Configuration (only used when creating new breaker)

    Returns:
        CircuitBreaker instance
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all registered circuit breakers for monitoring."""
    with _registry_lock:
        return dict(_circuit_breakers)
