"""Contract tests for experiment routing in ExperimentManager."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from aurelius.shared.experiment_client import (
    ExperimentDefinition,
    get_experiment_client,
)
from aurelius.shared.protocol import PromptSynapse


@dataclass
class MockExperimentConfig:
    """Mock experiment config for testing."""
    name: str
    enabled: bool = True


class MockExperiment:
    """Mock experiment for testing routing."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.config = MockExperimentConfig(name=name, enabled=enabled)

    @property
    def is_enabled(self) -> bool:
        return self.config.enabled


class TestRouteSubmission:
    """Tests for ExperimentManager.route_submission() method."""

    @pytest.fixture
    def mock_experiment_client(self):
        """Create a mock experiment client."""
        client = MagicMock()
        client.get_experiment.return_value = ExperimentDefinition(
            id="prompt",
            name="Dangerous Prompt Detection",
            version=1,
            experiment_type="push",
            scoring_type="danger",
            status="active",
            deprecated_at=None,
            thresholds={"acceptance": 0.3},
            rate_limit_requests=100,
            rate_limit_window_hours=1,
            novelty_threshold=0.02,
            pull_interval_seconds=None,
            pull_timeout_seconds=None,
            settings={},
            created_at="",
            updated_at="",
        )
        client.get_active_experiments.return_value = [
            client.get_experiment.return_value
        ]
        client.is_miner_registered.return_value = True
        return client

    @pytest.fixture
    def manager_with_experiments(self, mock_experiment_client):
        """Create an ExperimentManager with mock experiments."""
        from aurelius.validator.experiments.manager import ExperimentManager

        # Create a mock core
        mock_core = MagicMock()
        mock_core.experiment_client = mock_experiment_client

        manager = ExperimentManager(mock_core)

        # Add mock experiments
        mock_prompt_exp = MockExperiment("prompt", enabled=True)
        mock_other_exp = MockExperiment("jailbreak-v1", enabled=True)
        mock_inactive_exp = MockExperiment("deprecated-exp", enabled=False)

        manager.experiments["prompt"] = mock_prompt_exp
        manager.experiments["jailbreak-v1"] = mock_other_exp
        manager.experiments["deprecated-exp"] = mock_inactive_exp

        return manager

    def test_route_valid_experiment(self, manager_with_experiments):
        """Test routing to a valid, active experiment."""
        synapse = PromptSynapse(
            prompt="test prompt",
            experiment_id="prompt",
        )

        result = manager_with_experiments.route_submission(synapse)

        assert result is not None
        assert result.experiment is not None
        assert result.experiment.name == "prompt"
        assert result.rejection_reason is None

    def test_route_default_experiment_when_none(self, manager_with_experiments):
        """Test that None experiment_id defaults to 'prompt'."""
        synapse = PromptSynapse(
            prompt="test prompt",
            experiment_id=None,  # No experiment specified
        )

        result = manager_with_experiments.route_submission(synapse)

        assert result is not None
        assert result.experiment is not None
        assert result.experiment.name == "prompt"

    def test_route_invalid_experiment_rejected(self, manager_with_experiments):
        """Test that invalid experiment_id is rejected with available list."""
        synapse = PromptSynapse(
            prompt="test prompt",
            experiment_id="nonexistent-experiment",
        )

        result = manager_with_experiments.route_submission(synapse)

        assert result is not None
        assert result.experiment is None
        assert result.rejection_reason is not None
        assert "nonexistent-experiment" in result.rejection_reason
        assert result.available_experiments is not None
        assert len(result.available_experiments) > 0

    def test_route_inactive_experiment_rejected(self, manager_with_experiments):
        """Test that inactive experiment is rejected."""
        synapse = PromptSynapse(
            prompt="test prompt",
            experiment_id="deprecated-exp",
        )

        result = manager_with_experiments.route_submission(synapse)

        assert result is not None
        assert result.experiment is None
        assert result.rejection_reason is not None
        reason = result.rejection_reason.lower()
        assert "inactive" in reason or "disabled" in reason or "not active" in reason

    def test_route_provides_available_experiments_on_rejection(self, manager_with_experiments):
        """Test that rejection includes list of available experiments."""
        synapse = PromptSynapse(
            prompt="test prompt",
            experiment_id="bad-id",
        )

        result = manager_with_experiments.route_submission(synapse)

        assert result.available_experiments is not None
        # Should include enabled experiments
        assert "prompt" in result.available_experiments
        assert "jailbreak-v1" in result.available_experiments
        # Should not include disabled experiments
        assert "deprecated-exp" not in result.available_experiments


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing miners."""

    def test_synapse_without_experiment_id_works(self):
        """Test that synapse without experiment_id field still works."""
        # Create synapse with only required fields (like old miners would)
        synapse = PromptSynapse(prompt="test prompt")

        # Should default to None which gets routed to "prompt"
        assert synapse.experiment_id is None

    def test_synapse_with_experiment_id_preserves_value(self):
        """Test that experiment_id is preserved when set."""
        synapse = PromptSynapse(
            prompt="test prompt",
            experiment_id="jailbreak-v1",
        )

        assert synapse.experiment_id == "jailbreak-v1"

    def test_synapse_rejection_fields_default_to_none(self):
        """Test that rejection fields default to None."""
        synapse = PromptSynapse(prompt="test prompt")

        assert synapse.registration_required is None
        assert synapse.available_experiments is None


class TestRegistrationCheck:
    """Tests for miner registration validation in routing."""

    @pytest.fixture
    def manager_with_registration_check(self):
        """Create a manager that checks registration."""
        from aurelius.validator.experiments.manager import ExperimentManager

        mock_client = MagicMock()

        # prompt experiment - all miners auto-registered
        prompt_def = ExperimentDefinition(
            id="prompt",
            name="Dangerous Prompt Detection",
            version=1,
            experiment_type="push",
            scoring_type="danger",
            status="active",
            deprecated_at=None,
            thresholds={},
            rate_limit_requests=100,
            rate_limit_window_hours=1,
            novelty_threshold=0.02,
            pull_interval_seconds=None,
            pull_timeout_seconds=None,
            settings={},
            created_at="",
            updated_at="",
        )

        # other experiment - requires registration
        other_def = ExperimentDefinition(
            id="jailbreak-v1",
            name="Jailbreak Detection",
            version=1,
            experiment_type="push",
            scoring_type="binary",
            status="active",
            deprecated_at=None,
            thresholds={},
            rate_limit_requests=50,
            rate_limit_window_hours=1,
            novelty_threshold=0.1,
            pull_interval_seconds=None,
            pull_timeout_seconds=None,
            settings={},
            created_at="",
            updated_at="",
        )

        def get_experiment(exp_id):
            if exp_id == "prompt":
                return prompt_def
            elif exp_id == "jailbreak-v1":
                return other_def
            return None

        mock_client.get_experiment.side_effect = get_experiment
        mock_client.get_active_experiments.return_value = [prompt_def, other_def]

        # Registration check - only registered-hotkey is registered for jailbreak
        def is_registered(exp_id, hotkey):
            if exp_id == "prompt":
                return True  # All miners auto-registered for prompt
            return hotkey == "registered-hotkey"

        mock_client.is_miner_registered.side_effect = is_registered

        mock_core = MagicMock()
        mock_core.experiment_client = mock_client

        manager = ExperimentManager(mock_core)
        manager.experiments["prompt"] = MockExperiment("prompt")
        manager.experiments["jailbreak-v1"] = MockExperiment("jailbreak-v1")

        return manager

    def test_prompt_experiment_all_miners_auto_registered(self, manager_with_registration_check):
        """Test that all miners are auto-registered for 'prompt' experiment."""
        synapse = PromptSynapse(
            prompt="test",
            experiment_id="prompt",
            miner_hotkey="any-hotkey-works",
        )

        result = manager_with_registration_check.route_submission(synapse)

        # Should route successfully - all miners auto-registered
        assert result.experiment is not None
        assert result.rejection_reason is None

    def test_other_experiment_registered_miner_accepted(self, manager_with_registration_check):
        """Test that registered miner can submit to non-prompt experiment."""
        synapse = PromptSynapse(
            prompt="test",
            experiment_id="jailbreak-v1",
            miner_hotkey="registered-hotkey",
        )

        result = manager_with_registration_check.route_submission(synapse)

        assert result.experiment is not None
        assert result.rejection_reason is None

    def test_other_experiment_unregistered_miner_rejected(self, manager_with_registration_check):
        """Test that unregistered miner is rejected for non-prompt experiment."""
        synapse = PromptSynapse(
            prompt="test",
            experiment_id="jailbreak-v1",
            miner_hotkey="unregistered-hotkey",
        )

        result = manager_with_registration_check.route_submission(synapse)

        assert result.experiment is None
        assert result.rejection_reason is not None
        assert result.registration_required is True
        assert "register" in result.rejection_reason.lower()


class TestPerExperimentRateLimiting:
    """Tests for per-experiment rate limiting in routing (T045)."""

    @pytest.fixture
    def manager_with_rate_limits(self):
        """Create a manager with per-experiment rate limiting configured."""
        from aurelius.validator.experiments.manager import ExperimentManager

        mock_client = MagicMock()
        mock_client.is_miner_registered.return_value = True

        # Experiments with different rate limits
        prompt_def = ExperimentDefinition(
            id="prompt",
            name="Prompt Detection",
            version=1,
            experiment_type="push",
            scoring_type="danger",
            status="active",
            deprecated_at=None,
            thresholds={},
            rate_limit_requests=100,  # 100 requests
            rate_limit_window_hours=1,  # per hour
            novelty_threshold=0.02,
            pull_interval_seconds=None,
            pull_timeout_seconds=None,
            settings={},
            created_at="",
            updated_at="",
        )

        jailbreak_def = ExperimentDefinition(
            id="jailbreak-v1",
            name="Jailbreak Detection",
            version=1,
            experiment_type="push",
            scoring_type="binary",
            status="active",
            deprecated_at=None,
            thresholds={},
            rate_limit_requests=10,  # Much lower limit
            rate_limit_window_hours=1,
            novelty_threshold=0.1,
            pull_interval_seconds=None,
            pull_timeout_seconds=None,
            settings={},
            created_at="",
            updated_at="",
        )

        def get_experiment(exp_id):
            if exp_id == "prompt":
                return prompt_def
            elif exp_id == "jailbreak-v1":
                return jailbreak_def
            return None

        mock_client.get_experiment.side_effect = get_experiment
        mock_client.get_active_experiments.return_value = [prompt_def, jailbreak_def]

        mock_core = MagicMock()
        mock_core.experiment_client = mock_client

        manager = ExperimentManager(mock_core)
        manager.experiments["prompt"] = MockExperiment("prompt")
        manager.experiments["jailbreak-v1"] = MockExperiment("jailbreak-v1")

        return manager

    def test_check_rate_limits_method_exists(self, manager_with_rate_limits):
        """Test that check_rate_limits method exists on ExperimentManager."""
        assert hasattr(manager_with_rate_limits, "check_rate_limits")
        assert callable(getattr(manager_with_rate_limits, "check_rate_limits"))

    def test_rate_limit_passes_for_first_request(self, manager_with_rate_limits):
        """Test that first request passes rate limiting."""
        is_allowed = manager_with_rate_limits.check_rate_limits(
            hotkey="miner-1",
            experiment_id="prompt",
        )
        assert is_allowed is True

    def test_rate_limit_per_experiment_independence(self, manager_with_rate_limits):
        """Test that rate limits are independent per experiment."""
        # Hit rate limit on jailbreak (limit=10)
        for _ in range(10):
            manager_with_rate_limits.check_rate_limits("miner-1", "jailbreak-v1")

        # jailbreak should be rate limited
        is_jailbreak_allowed = manager_with_rate_limits.check_rate_limits("miner-1", "jailbreak-v1")
        assert is_jailbreak_allowed is False

        # prompt should still allow (different experiment, limit=100)
        is_prompt_allowed = manager_with_rate_limits.check_rate_limits("miner-1", "prompt")
        assert is_prompt_allowed is True

    def test_rate_limit_per_miner_independence(self, manager_with_rate_limits):
        """Test that rate limits are independent per miner."""
        # Hit rate limit for miner-1
        for _ in range(10):
            manager_with_rate_limits.check_rate_limits("miner-1", "jailbreak-v1")

        # miner-1 should be rate limited
        assert manager_with_rate_limits.check_rate_limits("miner-1", "jailbreak-v1") is False

        # miner-2 should not be affected
        assert manager_with_rate_limits.check_rate_limits("miner-2", "jailbreak-v1") is True

    def test_rate_limit_integrated_into_routing(self, manager_with_rate_limits):
        """Test that rate limiting is checked during routing."""
        # Exceed rate limit
        for _ in range(10):
            manager_with_rate_limits.check_rate_limits("rate-limited-miner", "jailbreak-v1")

        synapse = PromptSynapse(
            prompt="test",
            experiment_id="jailbreak-v1",
            miner_hotkey="rate-limited-miner",
        )

        result = manager_with_rate_limits.route_submission(synapse)

        # Should be rejected due to rate limiting
        assert result.experiment is None
        assert result.rejection_reason is not None
        assert "rate" in result.rejection_reason.lower()


class TestExperimentScoresExtended:
    """Tests for extended ExperimentScores with statistics (T048)."""

    def test_experiment_scores_has_statistics_fields(self):
        """Test that ExperimentScores includes statistics fields."""
        from aurelius.validator.experiments.base import ExperimentScores

        scores = ExperimentScores(
            experiment_name="test",
            scores={"miner1": 0.5},
            block_height=1000,
            total_submissions=100,
            total_accepted=85,
            window_start_block=900,
        )

        assert scores.total_submissions == 100
        assert scores.total_accepted == 85
        assert scores.window_start_block == 900

    def test_experiment_scores_default_statistics(self):
        """Test that statistics fields have sensible defaults."""
        from aurelius.validator.experiments.base import ExperimentScores

        scores = ExperimentScores(
            experiment_name="test",
            scores={},
            block_height=1000,
        )

        # Should have default values (0 or None depending on implementation)
        assert hasattr(scores, "total_submissions")
        assert hasattr(scores, "total_accepted")
        assert hasattr(scores, "window_start_block")
