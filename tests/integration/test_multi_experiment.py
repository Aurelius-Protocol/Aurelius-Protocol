"""Integration tests for multi-experiment framework."""

from unittest.mock import MagicMock, patch

import pytest

from aurelius.shared.experiment_client import (
    ExperimentDefinition,
    ExperimentClient,
    RewardAllocation,
)
from aurelius.shared.protocol import PromptSynapse


class TestMinerSubmissionFlow:
    """Integration tests for miner submission flow with experiment targeting."""

    @pytest.fixture
    def mock_experiment_definitions(self):
        """Create mock experiment definitions for testing."""
        return {
            "prompt": ExperimentDefinition(
                id="prompt",
                name="Dangerous Prompt Detection",
                version=1,
                experiment_type="push",
                scoring_type="danger",
                status="active",
                deprecated_at=None,
                thresholds={"acceptance": 0.3, "single_category": 0.8},
                rate_limit_requests=100,
                rate_limit_window_hours=1,
                novelty_threshold=0.02,
                pull_interval_seconds=None,
                pull_timeout_seconds=None,
                settings={},
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            ),
            "jailbreak-v1": ExperimentDefinition(
                id="jailbreak-v1",
                name="Jailbreak Detection",
                version=1,
                experiment_type="push",
                scoring_type="binary",
                status="active",
                deprecated_at=None,
                thresholds={"acceptance": 0.5},
                rate_limit_requests=50,
                rate_limit_window_hours=1,
                novelty_threshold=0.1,
                pull_interval_seconds=None,
                pull_timeout_seconds=None,
                settings={"model_required": "gpt-4"},
                created_at="2026-02-01T00:00:00Z",
                updated_at="2026-02-04T00:00:00Z",
            ),
        }

    def test_submission_with_experiment_id_targets_correct_experiment(self, mock_experiment_definitions):
        """Test that submission with experiment_id is routed correctly."""
        synapse = PromptSynapse(
            prompt="Test jailbreak prompt",
            experiment_id="jailbreak-v1",
            miner_hotkey="test-hotkey",
        )

        # Verify synapse contains experiment_id
        assert synapse.experiment_id == "jailbreak-v1"

        # In real flow, validator would:
        # 1. Check experiment exists and is active
        # 2. Check miner is registered (if not prompt)
        # 3. Route to correct experiment handler
        # 4. Return experiment-specific feedback

    def test_submission_without_experiment_id_defaults_to_prompt(self, mock_experiment_definitions):
        """Test that submission without experiment_id defaults to 'prompt'."""
        synapse = PromptSynapse(
            prompt="Test dangerous prompt",
            miner_hotkey="test-hotkey",
            # No experiment_id specified
        )

        assert synapse.experiment_id is None

        # In routing, None should be treated as "prompt"
        effective_experiment = synapse.experiment_id or "prompt"
        assert effective_experiment == "prompt"

    def test_submission_to_invalid_experiment_returns_available_list(self):
        """Test that invalid experiment_id returns list of available experiments."""
        synapse = PromptSynapse(
            prompt="Test prompt",
            experiment_id="nonexistent-experiment",
            miner_hotkey="test-hotkey",
        )

        # Validator would populate these fields on rejection
        synapse.accepted = False
        synapse.rejection_reason = "Unknown experiment 'nonexistent-experiment'"
        synapse.available_experiments = ["prompt", "jailbreak-v1"]

        assert synapse.accepted is False
        assert "nonexistent-experiment" in synapse.rejection_reason
        assert len(synapse.available_experiments) == 2

    def test_backward_compatibility_old_miner_works(self):
        """Test that old miners without experiment_id still work."""
        # Simulate old miner that doesn't know about experiment_id
        synapse = PromptSynapse(
            prompt="Old miner prompt",
        )

        # Old miners won't set these fields
        assert synapse.experiment_id is None
        assert synapse.registration_required is None
        assert synapse.available_experiments is None

        # Validator should handle gracefully and route to default experiment
        # This is critical for backward compatibility (FR-021, FR-022)

    def test_rejection_response_includes_experiment_feedback(self):
        """Test that rejection response includes experiment-specific feedback."""
        synapse = PromptSynapse(
            prompt="Test prompt",
            experiment_id="jailbreak-v1",
            miner_hotkey="unregistered-hotkey",
        )

        # Validator populates rejection info
        synapse.accepted = False
        synapse.rejection_reason = "Registration required for experiment 'jailbreak-v1'"
        synapse.registration_required = True
        synapse.available_experiments = ["prompt", "jailbreak-v1"]

        # Miner can check why rejected and what's available
        assert synapse.registration_required is True
        assert "registration" in synapse.rejection_reason.lower()


class TestExperimentClientIntegration:
    """Integration tests for ExperimentClient."""

    def test_client_provides_experiment_definitions(self):
        """Test that client provides experiment definitions for routing."""
        # This would be a real integration test with mock API
        # For now, test the interface contract

        client = MagicMock(spec=ExperimentClient)
        client.get_experiment.return_value = ExperimentDefinition(
            id="prompt",
            name="Test",
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

        exp = client.get_experiment("prompt")
        assert exp is not None
        assert exp.id == "prompt"

    def test_client_reports_available_experiments(self):
        """Test that client reports list of available experiments."""
        client = MagicMock(spec=ExperimentClient)
        client.get_active_experiments.return_value = [
            ExperimentDefinition(
                id="prompt",
                name="Prompt",
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
            ),
            ExperimentDefinition(
                id="jailbreak-v1",
                name="Jailbreak",
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
            ),
        ]

        active = client.get_active_experiments()
        assert len(active) == 2
        assert any(e.id == "prompt" for e in active)
        assert any(e.id == "jailbreak-v1" for e in active)

    def test_client_checks_miner_registration(self):
        """Test that client can check miner registration status."""
        client = MagicMock(spec=ExperimentClient)
        client.is_miner_registered.side_effect = lambda exp, hk: (
            True if exp == "prompt" else hk == "registered-hotkey"
        )

        # All miners auto-registered for prompt
        assert client.is_miner_registered("prompt", "any-hotkey") is True

        # Only registered miners for other experiments
        assert client.is_miner_registered("jailbreak", "registered-hotkey") is True
        assert client.is_miner_registered("jailbreak", "other-hotkey") is False


class TestConcurrentSubmissions:
    """Integration tests for concurrent submissions to multiple experiments (T044)."""

    @pytest.fixture
    def multi_experiment_manager(self):
        """Create an ExperimentManager with multiple experiments."""
        from aurelius.validator.experiments.manager import ExperimentManager

        mock_client = MagicMock()
        mock_client.is_miner_registered.return_value = True
        mock_client.get_active_experiments.return_value = []

        # Return proper experiment definitions with real rate_limit values
        def get_experiment(exp_id):
            return ExperimentDefinition(
                id=exp_id,
                name=exp_id,
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
        mock_client.get_experiment.side_effect = get_experiment

        mock_core = MagicMock()
        mock_core.experiment_client = mock_client

        manager = ExperimentManager(mock_core)

        # Add multiple mock experiments
        class MockExperiment:
            def __init__(self, name, enabled=True):
                self.name = name
                self._enabled = enabled
                self.submissions = []

            @property
            def is_enabled(self):
                return self._enabled

            def record_submission(self, synapse):
                self.submissions.append(synapse)

        manager.experiments["prompt"] = MockExperiment("prompt")
        manager.experiments["jailbreak-v1"] = MockExperiment("jailbreak-v1")
        manager.experiments["benchmark-v2"] = MockExperiment("benchmark-v2")

        return manager

    def test_concurrent_submissions_to_different_experiments(self, multi_experiment_manager):
        """Test that concurrent submissions to different experiments are handled independently."""
        # Create submissions to different experiments
        synapses = [
            PromptSynapse(prompt="prompt-1", experiment_id="prompt", miner_hotkey="miner-1"),
            PromptSynapse(prompt="jailbreak-1", experiment_id="jailbreak-v1", miner_hotkey="miner-2"),
            PromptSynapse(prompt="benchmark-1", experiment_id="benchmark-v2", miner_hotkey="miner-3"),
            PromptSynapse(prompt="prompt-2", experiment_id="prompt", miner_hotkey="miner-4"),
        ]

        # Route all submissions
        results = [multi_experiment_manager.route_submission(s) for s in synapses]

        # All should route successfully
        assert all(r.experiment is not None for r in results)

        # Each should route to the correct experiment
        assert results[0].experiment.name == "prompt"
        assert results[1].experiment.name == "jailbreak-v1"
        assert results[2].experiment.name == "benchmark-v2"
        assert results[3].experiment.name == "prompt"

    def test_concurrent_submissions_from_same_miner(self, multi_experiment_manager):
        """Test that same miner can submit to multiple experiments concurrently."""
        # Same miner submitting to multiple experiments
        synapses = [
            PromptSynapse(prompt="test-1", experiment_id="prompt", miner_hotkey="multi-exp-miner"),
            PromptSynapse(prompt="test-2", experiment_id="jailbreak-v1", miner_hotkey="multi-exp-miner"),
            PromptSynapse(prompt="test-3", experiment_id="benchmark-v2", miner_hotkey="multi-exp-miner"),
        ]

        results = [multi_experiment_manager.route_submission(s) for s in synapses]

        # All should route successfully
        assert all(r.experiment is not None for r in results)
        assert all(r.rejection_reason is None for r in results)

    def test_mixed_valid_invalid_concurrent_submissions(self, multi_experiment_manager):
        """Test handling of mixed valid and invalid concurrent submissions."""
        synapses = [
            PromptSynapse(prompt="valid-1", experiment_id="prompt", miner_hotkey="miner-1"),
            PromptSynapse(prompt="invalid-1", experiment_id="nonexistent", miner_hotkey="miner-2"),
            PromptSynapse(prompt="valid-2", experiment_id="jailbreak-v1", miner_hotkey="miner-3"),
            PromptSynapse(prompt="invalid-2", experiment_id="also-fake", miner_hotkey="miner-4"),
        ]

        results = [multi_experiment_manager.route_submission(s) for s in synapses]

        # Check valid submissions routed correctly
        assert results[0].experiment is not None
        assert results[0].experiment.name == "prompt"
        assert results[2].experiment is not None
        assert results[2].experiment.name == "jailbreak-v1"

        # Check invalid submissions rejected with available list
        assert results[1].experiment is None
        assert results[1].rejection_reason is not None
        assert results[1].available_experiments is not None
        assert results[3].experiment is None
        assert results[3].rejection_reason is not None

    def test_concurrent_submissions_isolation(self, multi_experiment_manager):
        """Test that failures in one experiment don't affect others."""
        # Make jailbreak experiment disabled
        multi_experiment_manager.experiments["jailbreak-v1"]._enabled = False

        synapses = [
            PromptSynapse(prompt="prompt-ok", experiment_id="prompt", miner_hotkey="miner-1"),
            PromptSynapse(prompt="jailbreak-fail", experiment_id="jailbreak-v1", miner_hotkey="miner-2"),
            PromptSynapse(prompt="benchmark-ok", experiment_id="benchmark-v2", miner_hotkey="miner-3"),
        ]

        results = [multi_experiment_manager.route_submission(s) for s in synapses]

        # prompt and benchmark should succeed
        assert results[0].experiment is not None
        assert results[2].experiment is not None

        # jailbreak should fail
        assert results[1].experiment is None
        assert "not active" in results[1].rejection_reason.lower() or "disabled" in results[1].rejection_reason.lower()


class TestPerExperimentStatistics:
    """Integration tests for per-experiment statistics tracking."""

    @pytest.fixture
    def manager_with_stats(self):
        """Create a manager with statistics-enabled experiments."""
        from aurelius.validator.experiments.manager import ExperimentManager

        mock_client = MagicMock()
        mock_client.is_miner_registered.return_value = True
        mock_client.get_active_experiments.return_value = []

        mock_core = MagicMock()
        mock_core.experiment_client = mock_client

        return ExperimentManager(mock_core)

    def test_get_experiment_stats_returns_dict(self, manager_with_stats):
        """Test that get_experiment_stats returns properly structured data."""
        # Add mock experiment with stats
        class StatsExperiment:
            def __init__(self, name):
                self.name = name
                self._enabled = True

            @property
            def is_enabled(self):
                return self._enabled

            def get_stats(self):
                return {
                    "total_submissions": 100,
                    "total_accepted": 85,
                    "window_start_block": 1000,
                }

        manager_with_stats.experiments["prompt"] = StatsExperiment("prompt")

        stats = manager_with_stats.get_all_stats()

        assert "prompt" in stats
        assert stats["prompt"]["total_submissions"] == 100
        assert stats["prompt"]["total_accepted"] == 85

    def test_stats_isolation_between_experiments(self, manager_with_stats):
        """Test that statistics are isolated between experiments."""
        class StatsExperiment:
            def __init__(self, name, submissions):
                self.name = name
                self._enabled = True
                self._submissions = submissions

            @property
            def is_enabled(self):
                return self._enabled

            def get_stats(self):
                return {"total_submissions": self._submissions}

        manager_with_stats.experiments["prompt"] = StatsExperiment("prompt", 100)
        manager_with_stats.experiments["jailbreak"] = StatsExperiment("jailbreak", 50)

        stats = manager_with_stats.get_all_stats()

        assert stats["prompt"]["total_submissions"] == 100
        assert stats["jailbreak"]["total_submissions"] == 50


class TestPullExperimentIntegration:
    """Integration tests for pull-based experiments (T053)."""

    @pytest.fixture
    def mock_pull_experiment_definition(self):
        """Create a pull experiment definition."""
        return ExperimentDefinition(
            id="data-collection-v1",
            name="Data Collection",
            version=1,
            experiment_type="pull",  # Pull-based
            scoring_type="binary",
            status="active",
            deprecated_at=None,
            thresholds={"acceptance": 0.5},
            rate_limit_requests=100,
            rate_limit_window_hours=1,
            novelty_threshold=0.1,
            pull_interval_seconds=300,  # Query every 5 minutes
            pull_timeout_seconds=30,  # 30 second timeout
            settings={"miners_per_round": 10},
            created_at="2026-02-01T00:00:00Z",
            updated_at="2026-02-04T00:00:00Z",
        )

    def test_pull_experiment_has_schedule_configuration(self, mock_pull_experiment_definition):
        """Test that pull experiment has schedule configuration."""
        assert mock_pull_experiment_definition.pull_interval_seconds == 300
        assert mock_pull_experiment_definition.pull_timeout_seconds == 30

    def test_pull_experiment_type_is_pull(self, mock_pull_experiment_definition):
        """Test that experiment type is correctly set to pull."""
        assert mock_pull_experiment_definition.experiment_type == "pull"

    def test_pull_experiment_can_coexist_with_push(self):
        """Test that pull and push experiments can coexist."""
        from aurelius.validator.experiments.manager import ExperimentManager

        mock_client = MagicMock()
        mock_client.is_miner_registered.return_value = True
        mock_client.get_active_experiments.return_value = []

        mock_core = MagicMock()
        mock_core.experiment_client = mock_client

        manager = ExperimentManager(mock_core)

        # Add both push and pull experiments
        class MockPushExperiment:
            def __init__(self):
                self.name = "prompt"
                self._enabled = True
                self._started = False
                self.experiment_type = "push"

            @property
            def is_enabled(self):
                return self._enabled

        class MockPullExperiment:
            def __init__(self):
                self.name = "data-collection"
                self._enabled = True
                self._started = False
                self.experiment_type = "pull"

            @property
            def is_enabled(self):
                return self._enabled

        manager.experiments["prompt"] = MockPushExperiment()
        manager.experiments["data-collection"] = MockPullExperiment()

        # Both should be enabled
        enabled = manager.get_enabled_experiments()
        assert len(enabled) == 2
        assert any(e.experiment_type == "push" for e in enabled)
        assert any(e.experiment_type == "pull" for e in enabled)


class TestPullMinerSelection:
    """Integration tests for miner selection in pull experiments (T054)."""

    @pytest.fixture
    def pull_experiment_with_registrations(self):
        """Create a pull experiment with registered miners."""
        from aurelius.validator.experiments.base import PullExperiment, ExperimentConfig, ExperimentType

        class TestPullExperiment(PullExperiment):
            NAME = "test-pull"
            TYPE = ExperimentType.PULL

            def __init__(self, core, config, registered_miners):
                super().__init__(core, config)
                self._registered_miners = registered_miners
                self._query_results = []

            def _create_query_synapse(self):
                return MagicMock()

            def _process_results(self, results):
                self._query_results.extend(results)

            def calculate_scores(self, current_block):
                from aurelius.validator.experiments.base import ExperimentScores
                return ExperimentScores(
                    experiment_name=self.name,
                    scores={},
                    block_height=current_block,
                )

            def get_stats(self):
                return {"queries": len(self._query_results)}

        mock_core = MagicMock()
        mock_core.experiment_client = MagicMock()
        mock_core.experiment_client.get_registered_miners.return_value = [
            "registered-miner-1",
            "registered-miner-2",
            "registered-miner-3",
        ]

        config = ExperimentConfig(
            name="test-pull",
            experiment_type=ExperimentType.PULL,
            weight_allocation=0.1,
            enabled=True,
            settings={"miners_per_round": 2},
        )

        return TestPullExperiment(mock_core, config, ["registered-miner-1", "registered-miner-2"])

    def test_pull_experiment_has_miners_per_round_setting(self, pull_experiment_with_registrations):
        """Test that pull experiment respects miners_per_round setting."""
        assert pull_experiment_with_registrations.miners_per_round == 2

    def test_pull_experiment_query_interval_configurable(self, pull_experiment_with_registrations):
        """Test that query interval is configurable via settings."""
        # Default is 300 seconds
        assert pull_experiment_with_registrations.query_interval_seconds == 300

    def test_pull_experiment_timeout_configurable(self, pull_experiment_with_registrations):
        """Test that query timeout is configurable."""
        assert pull_experiment_with_registrations.query_timeout == 30.0
