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
    """Send a prompt to a single validator. See miner.send_prompt for full docs."""
    from aurelius.miner.miner import send_prompt as _send_prompt
    return _send_prompt(*args, **kwargs)


def send_prompt_multi(*args, **kwargs):
    """Send a prompt to multiple validators in parallel. See miner.send_prompt_multi for full docs."""
    from aurelius.miner.miner import send_prompt_multi as _send_prompt_multi
    return _send_prompt_multi(*args, **kwargs)


def discover_validators(*args, **kwargs):
    """Discover eligible validators from metagraph. See miner.discover_validators for full docs."""
    from aurelius.miner.miner import discover_validators as _discover_validators
    return _discover_validators(*args, **kwargs)


def display_multi_results(*args, **kwargs):
    """Display results from multiple validators. See miner.display_multi_results for full docs."""
    from aurelius.miner.miner import display_multi_results as _display_multi_results
    return _display_multi_results(*args, **kwargs)


def main():
    """Main entry point for the miner."""
    from aurelius.miner.miner import main as _main
    return _main()


__all__ = [
    # Main functionality
    "send_prompt",
    "send_prompt_multi",
    "discover_validators",
    "display_multi_results",
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
