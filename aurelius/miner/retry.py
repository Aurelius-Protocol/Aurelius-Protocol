"""Retry logic with exponential backoff for miner operations."""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aurelius.miner.errors import MinerError, classify_error


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1  # Add randomness to prevent thundering herd


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    result: Any = None
    attempts: int = 0
    total_time_seconds: float = 0.0
    last_error: MinerError | None = None
    all_errors: list[MinerError] = field(default_factory=list)


class RetryHandler:
    """Handles retry logic with exponential backoff."""

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay with exponential backoff and jitter.

        Args:
            attempt: The attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        base_delay = self.config.initial_delay_seconds * (
            self.config.exponential_base**attempt
        )
        delay = min(base_delay, self.config.max_delay_seconds)

        # Add jitter to prevent thundering herd
        jitter = delay * self.config.jitter_factor * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)  # Minimum 100ms delay

    def should_retry(self, error: MinerError, attempt: int) -> bool:
        """
        Determine if operation should be retried.

        Args:
            error: The error that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry, False otherwise
        """
        if attempt >= self.config.max_attempts - 1:
            return False
        return error.is_retryable

    def execute_with_retry(
        self,
        operation: Callable[[], Any],
        context: dict | None = None,
        on_retry: Callable[[int, MinerError, float], None] | None = None,
    ) -> RetryResult:
        """
        Execute operation with retry logic.

        Args:
            operation: The callable to execute
            context: Context dict for error classification
            on_retry: Optional callback called before each retry with
                     (attempt_number, error, delay_seconds)

        Returns:
            RetryResult with success status and result or errors
        """
        context = context or {}
        all_errors: list[MinerError] = []
        start_time = time.time()

        for attempt in range(self.config.max_attempts):
            try:
                result = operation()
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_time_seconds=time.time() - start_time,
                    all_errors=all_errors,
                )
            except Exception as e:
                error = classify_error(e, context)
                all_errors.append(error)

                if not self.should_retry(error, attempt):
                    # Not retryable or out of attempts
                    return RetryResult(
                        success=False,
                        attempts=attempt + 1,
                        total_time_seconds=time.time() - start_time,
                        last_error=error,
                        all_errors=all_errors,
                    )

                # Calculate delay and notify callback
                delay = self.calculate_delay(attempt)
                if on_retry:
                    on_retry(attempt + 1, error, delay)

                time.sleep(delay)

        # Should not reach here, but handle edge case
        return RetryResult(
            success=False,
            attempts=self.config.max_attempts,
            total_time_seconds=time.time() - start_time,
            last_error=all_errors[-1] if all_errors else None,
            all_errors=all_errors,
        )
