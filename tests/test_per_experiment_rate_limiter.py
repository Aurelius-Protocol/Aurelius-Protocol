"""Tests for PerExperimentRateLimiter (T073).

Tests isolation between experiments, per-experiment limits, and backward compatibility.
"""

from unittest.mock import MagicMock

import pytest


class TestPerExperimentRateLimiter:
    """Tests for PerExperimentRateLimiter class."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a PerExperimentRateLimiter instance."""
        from aurelius.shared.rate_limiter import PerExperimentRateLimiter

        return PerExperimentRateLimiter()

    def test_rate_limiter_creation(self, rate_limiter):
        """Test that rate limiter can be instantiated."""
        assert rate_limiter is not None

    def test_isolation_between_experiments(self, rate_limiter):
        """Test that rate limits are isolated between experiments."""
        # Set different limits for different experiments
        rate_limiter.set_experiment_limit("exp-a", max_requests=5, window_seconds=3600)
        rate_limiter.set_experiment_limit("exp-b", max_requests=10, window_seconds=3600)

        # Exhaust exp-a limit
        for _ in range(5):
            assert rate_limiter.check("miner-1", "exp-a") is True

        # exp-a should be rate limited
        assert rate_limiter.check("miner-1", "exp-a") is False

        # exp-b should still allow (different experiment)
        assert rate_limiter.check("miner-1", "exp-b") is True

    def test_per_experiment_limits_independent(self, rate_limiter):
        """Test that per-experiment limits don't affect other experiments."""
        rate_limiter.set_experiment_limit("exp-strict", max_requests=2, window_seconds=3600)
        rate_limiter.set_experiment_limit("exp-loose", max_requests=100, window_seconds=3600)

        # Hit strict limit
        rate_limiter.check("miner-1", "exp-strict")
        rate_limiter.check("miner-1", "exp-strict")

        # Strict is rate limited
        assert rate_limiter.check("miner-1", "exp-strict") is False

        # Loose is unaffected
        for _ in range(50):
            assert rate_limiter.check("miner-1", "exp-loose") is True

    def test_per_miner_isolation(self, rate_limiter):
        """Test that different miners have independent limits."""
        rate_limiter.set_experiment_limit("exp-a", max_requests=3, window_seconds=3600)

        # miner-1 hits limit
        for _ in range(3):
            rate_limiter.check("miner-1", "exp-a")

        assert rate_limiter.check("miner-1", "exp-a") is False

        # miner-2 is unaffected
        assert rate_limiter.check("miner-2", "exp-a") is True

    def test_default_experiment_uses_default_limit(self, rate_limiter):
        """Test that unconfigured experiments use default limits."""
        # No explicit limit set for "unknown-exp"
        # Should use default limit (or no limit)
        result = rate_limiter.check("miner-1", "unknown-exp")
        assert result is True  # Default should allow

    def test_backward_compatibility_with_per_miner_rate_limiter(self, rate_limiter):
        """Test backward compatibility with existing PerMinerRateLimiter interface."""
        # Should support check_submission-like method
        rate_limiter.set_experiment_limit("prompt", max_requests=100, window_seconds=3600)

        # Basic check should work
        assert rate_limiter.check("miner-1", "prompt") is True


class TestPerExperimentRateLimiterStats:
    """Tests for rate limiter statistics."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a PerExperimentRateLimiter instance."""
        from aurelius.shared.rate_limiter import PerExperimentRateLimiter

        return PerExperimentRateLimiter()

    def test_get_stats_for_experiment(self, rate_limiter):
        """Test getting statistics for a specific experiment."""
        rate_limiter.set_experiment_limit("exp-a", max_requests=10, window_seconds=3600)

        # Make some requests
        for _ in range(5):
            rate_limiter.check("miner-1", "exp-a")
        for _ in range(3):
            rate_limiter.check("miner-2", "exp-a")

        stats = rate_limiter.get_experiment_stats("exp-a")

        assert stats["tracked_miners"] == 2
        assert stats["total_requests"] == 8

    def test_get_stats_returns_none_for_unknown(self, rate_limiter):
        """Test that unknown experiment returns empty stats."""
        stats = rate_limiter.get_experiment_stats("nonexistent")
        assert stats["tracked_miners"] == 0


class TestPerExperimentRateLimiterWindowExpiry:
    """Tests for rate limit window expiration."""

    @pytest.fixture
    def rate_limiter(self):
        """Create a PerExperimentRateLimiter instance."""
        from aurelius.shared.rate_limiter import PerExperimentRateLimiter

        return PerExperimentRateLimiter()

    def test_window_expiry_resets_limit(self, rate_limiter):
        """Test that expired windows reset the limit."""
        import time

        # Set very short window (1 second)
        rate_limiter.set_experiment_limit("exp-a", max_requests=2, window_seconds=1)

        # Hit limit
        rate_limiter.check("miner-1", "exp-a")
        rate_limiter.check("miner-1", "exp-a")
        assert rate_limiter.check("miner-1", "exp-a") is False

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        assert rate_limiter.check("miner-1", "exp-a") is True
