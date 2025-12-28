"""Aurelius Miner Module."""

from aurelius.miner.errors import (
    DiagnosticsFormatter,
    ErrorCategory,
    ErrorSeverity,
    MinerError,
    classify_error,
    create_error,
)
from aurelius.miner.health import HealthCheckResult, ValidatorHealthChecker
from aurelius.miner.retry import RetryConfig, RetryHandler, RetryResult


# Lazy import for miner to avoid bittensor import at module load time
def send_prompt(*args, **kwargs):
    """Send a prompt to a validator. See miner.send_prompt for full docs."""
    from aurelius.miner.miner import send_prompt as _send_prompt
    return _send_prompt(*args, **kwargs)


def main():
    """Main entry point for the miner."""
    from aurelius.miner.miner import main as _main
    return _main()


__all__ = [
    # Main functionality
    "send_prompt",
    "main",
    # Error handling
    "ErrorCategory",
    "ErrorSeverity",
    "MinerError",
    "classify_error",
    "create_error",
    "DiagnosticsFormatter",
    # Health checks
    "HealthCheckResult",
    "ValidatorHealthChecker",
    # Retry logic
    "RetryConfig",
    "RetryHandler",
    "RetryResult",
]
