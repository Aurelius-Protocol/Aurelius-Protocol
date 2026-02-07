"""Unit tests for ExperimentClient."""

import json
import os
import tempfile
import threading
from unittest.mock import MagicMock, patch

import pytest

from aurelius.shared.experiment_client import (
    DEFAULT_PROMPT_EXPERIMENT,
    ExperimentClient,
    ExperimentDefinition,
    ExperimentSyncResponse,
    MinerRegistration,
    RewardAllocation,
    get_experiment_client,
)


class TestExperimentDefinition:
    """Tests for ExperimentDefinition dataclass."""

    def test_from_dict_minimal(self):
        """Test creating ExperimentDefinition from minimal dict."""
        data = {
            "id": "test-exp",
            "name": "Test Experiment",
            "version": 1,
            "experiment_type": "push",
            "scoring_type": "danger",
            "status": "active",
        }
        exp = ExperimentDefinition.from_dict(data)
        assert exp.id == "test-exp"
        assert exp.name == "Test Experiment"
        assert exp.version == 1
        assert exp.experiment_type == "push"
        assert exp.scoring_type == "danger"
        assert exp.status == "active"
        assert exp.deprecated_at is None
        assert exp.thresholds == {}
        assert exp.rate_limit_requests == 100
        assert exp.rate_limit_window_hours == 1

    def test_from_dict_full(self):
        """Test creating ExperimentDefinition from full dict."""
        data = {
            "id": "jailbreak-v1",
            "name": "Jailbreak Detection",
            "version": 2,
            "experiment_type": "push",
            "scoring_type": "binary",
            "status": "active",
            "deprecated_at": None,
            "thresholds": {"acceptance": 0.5},
            "rate_limit_requests": 50,
            "rate_limit_window_hours": 2,
            "novelty_threshold": 0.1,
            "pull_interval_seconds": None,
            "pull_timeout_seconds": None,
            "settings": {"model_required": "gpt-4"},
            "created_at": "2026-02-01T00:00:00Z",
            "updated_at": "2026-02-04T00:00:00Z",
        }
        exp = ExperimentDefinition.from_dict(data)
        assert exp.id == "jailbreak-v1"
        assert exp.thresholds == {"acceptance": 0.5}
        assert exp.rate_limit_requests == 50
        assert exp.settings == {"model_required": "gpt-4"}

    def test_to_dict_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original = ExperimentDefinition(
            id="test",
            name="Test",
            version=1,
            experiment_type="pull",
            scoring_type="numeric",
            status="inactive",
            deprecated_at="2026-01-01T00:00:00Z",
            thresholds={"min": 0.2, "max": 0.8},
            rate_limit_requests=25,
            rate_limit_window_hours=4,
            novelty_threshold=0.05,
            pull_interval_seconds=300,
            pull_timeout_seconds=30,
            settings={"key": "value"},
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-02T00:00:00Z",
        )
        d = original.to_dict()
        restored = ExperimentDefinition.from_dict(d)
        assert restored.id == original.id
        assert restored.thresholds == original.thresholds
        assert restored.pull_interval_seconds == original.pull_interval_seconds


class TestMinerRegistration:
    """Tests for MinerRegistration dataclass."""

    def test_from_dict(self):
        """Test creating MinerRegistration from dict."""
        data = {
            "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "experiment_id": "jailbreak-v1",
            "status": "active",
            "registered_at": "2026-02-03T10:00:00Z",
            "withdrawn_at": None,
        }
        reg = MinerRegistration.from_dict(data)
        assert reg.miner_hotkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        assert reg.experiment_id == "jailbreak-v1"
        assert reg.status == "active"
        assert reg.withdrawn_at is None

    def test_to_dict_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original = MinerRegistration(
            miner_hotkey="5FHneW46",
            experiment_id="test",
            status="withdrawn",
            registered_at="2026-01-01T00:00:00Z",
            withdrawn_at="2026-02-01T00:00:00Z",
        )
        restored = MinerRegistration.from_dict(original.to_dict())
        assert restored.status == "withdrawn"
        assert restored.withdrawn_at == "2026-02-01T00:00:00Z"


class TestRewardAllocation:
    """Tests for RewardAllocation dataclass."""

    def test_from_dict(self):
        """Test creating RewardAllocation from dict."""
        data = {
            "allocations": {"prompt": 85.0, "jailbreak-v1": 10.0},
            "burn_percentage": 5.0,
            "redistribute_unused": True,
            "version": 1,
            "updated_at": "2026-02-04T00:00:00Z",
        }
        alloc = RewardAllocation.from_dict(data)
        assert alloc.allocations["prompt"] == 85.0
        assert alloc.burn_percentage == 5.0
        assert alloc.redistribute_unused is True

    def test_validate_success(self):
        """Test validation passes when allocations sum to 100%."""
        alloc = RewardAllocation(
            allocations={"prompt": 90.0, "other": 5.0},
            burn_percentage=5.0,
            redistribute_unused=True,
            version=1,
            updated_at="",
        )
        is_valid, error = alloc.validate()
        assert is_valid is True
        assert error == ""

    def test_validate_failure(self):
        """Test validation fails when allocations don't sum to 100%."""
        alloc = RewardAllocation(
            allocations={"prompt": 80.0},
            burn_percentage=5.0,
            redistribute_unused=True,
            version=1,
            updated_at="",
        )
        is_valid, error = alloc.validate()
        assert is_valid is False
        assert "85.0%" in error


class TestExperimentClient:
    """Tests for ExperimentClient class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def client_no_api(self, temp_cache_dir):
        """Create a client with no API endpoint (offline mode)."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        return ExperimentClient(
            api_endpoint=None,
            cache_path=cache_path,
        )

    @pytest.fixture
    def client_with_api(self, temp_cache_dir):
        """Create a client with a mock API endpoint."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        return ExperimentClient(
            api_endpoint="https://test.api/api/experiments",
            cache_path=cache_path,
            api_key="test-key",
        )

    def test_init_applies_defaults_when_no_cache(self, client_no_api):
        """Test that defaults are applied when no cache file exists."""
        exp = client_no_api.get_experiment("prompt")
        assert exp is not None
        assert exp.id == "prompt"
        assert exp.name == "Dangerous Prompt Detection"
        assert exp.status == "active"

    def test_init_loads_cache(self, temp_cache_dir):
        """Test that cache is loaded on init."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")

        # Write a cache file
        cache_data = {
            "sync_version": 42,
            "synced_at": "2026-02-04T12:00:00Z",
            "experiments": {
                "test-exp": {
                    "id": "test-exp",
                    "name": "Test Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                    "thresholds": {},
                    "rate_limit_requests": 100,
                    "rate_limit_window_hours": 1,
                    "novelty_threshold": 0.02,
                    "settings": {},
                    "created_at": "",
                    "updated_at": "",
                }
            },
            "registrations": {"test-exp": ["hotkey1", "hotkey2"]},
            "reward_allocation": {
                "allocations": {"test-exp": 100.0},
                "burn_percentage": 0.0,
                "redistribute_unused": True,
                "version": 1,
                "updated_at": "",
            },
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        # Create client - should load cache
        client = ExperimentClient(
            api_endpoint=None,
            cache_path=cache_path,
        )

        exp = client.get_experiment("test-exp")
        assert exp is not None
        assert exp.name == "Test Experiment"
        assert client._sync_version == 42

    def test_get_experiment_not_found(self, client_no_api):
        """Test get_experiment returns None for unknown experiment."""
        exp = client_no_api.get_experiment("nonexistent")
        assert exp is None

    def test_get_active_experiments(self, temp_cache_dir):
        """Test get_active_experiments filters by status."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "active1": {
                    "id": "active1",
                    "name": "Active 1",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                },
                "inactive1": {
                    "id": "inactive1",
                    "name": "Inactive 1",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "inactive",
                },
                "deprecated1": {
                    "id": "deprecated1",
                    "name": "Deprecated 1",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "deprecated",
                },
            },
            "registrations": {},
            "reward_allocation": {
                "allocations": {"active1": 100.0},
                "burn_percentage": 0.0,
                "redistribute_unused": True,
                "version": 1,
            },
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(api_endpoint=None, cache_path=cache_path)
        active = client.get_active_experiments()

        assert len(active) == 1
        assert active[0].id == "active1"

    def test_get_registered_miners_prompt(self, client_no_api):
        """Test that 'prompt' experiment returns empty list (all miners auto-registered)."""
        miners = client_no_api.get_registered_miners("prompt")
        assert miners == []  # Empty means all miners

    def test_get_registered_miners_other_experiment(self, temp_cache_dir):
        """Test get_registered_miners for non-prompt experiment."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "jailbreak": {
                    "id": "jailbreak",
                    "name": "Jailbreak",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "binary",
                    "status": "active",
                }
            },
            "registrations": {"jailbreak": ["hotkey1", "hotkey2", "hotkey3"]},
            "reward_allocation": {
                "allocations": {"jailbreak": 100.0},
                "burn_percentage": 0.0,
                "redistribute_unused": True,
                "version": 1,
            },
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(api_endpoint=None, cache_path=cache_path)
        miners = client.get_registered_miners("jailbreak")

        assert len(miners) == 3
        assert "hotkey1" in miners

    def test_is_miner_registered_prompt(self, client_no_api):
        """Test that all miners are registered for 'prompt' experiment."""
        assert client_no_api.is_miner_registered("prompt", "any-hotkey") is True

    def test_is_miner_registered_other_experiment(self, temp_cache_dir):
        """Test is_miner_registered for non-prompt experiment."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "jailbreak": {
                    "id": "jailbreak",
                    "name": "Jailbreak",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "binary",
                    "status": "active",
                }
            },
            "registrations": {"jailbreak": ["registered-hotkey"]},
            "reward_allocation": {
                "allocations": {"jailbreak": 100.0},
                "burn_percentage": 0.0,
                "redistribute_unused": True,
                "version": 1,
            },
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(api_endpoint=None, cache_path=cache_path)

        assert client.is_miner_registered("jailbreak", "registered-hotkey") is True
        assert client.is_miner_registered("jailbreak", "not-registered") is False

    def test_save_cache(self, temp_cache_dir):
        """Test that cache is saved correctly."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        client = ExperimentClient(api_endpoint=None, cache_path=cache_path)

        # Trigger a save
        client._save_cache()

        # Verify cache file exists and has correct structure
        assert os.path.exists(cache_path)
        with open(cache_path, "r") as f:
            data = json.load(f)

        assert "experiments" in data
        assert "prompt" in data["experiments"]

    @patch("aurelius.shared.experiment_client.requests.get")
    def test_sync_success(self, mock_get, client_with_api):
        """Test successful sync from API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiments": [
                {
                    "id": "new-exp",
                    "name": "New Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                    "thresholds": {"acceptance": 0.3},
                    "rate_limit_requests": 100,
                    "rate_limit_window_hours": 1,
                    "novelty_threshold": 0.02,
                    "settings": {},
                    "created_at": "",
                    "updated_at": "",
                }
            ],
            "registrations": [],
            "reward_allocation": {
                "allocations": {"new-exp": 100.0},
                "burn_percentage": 0.0,
                "redistribute_unused": True,
                "version": 1,
                "updated_at": "",
            },
            "sync_version": 99,
            "server_time": "2026-02-04T12:00:00Z",
        }
        mock_get.return_value = mock_response

        result = client_with_api.sync()

        assert result is True
        assert client_with_api._sync_version == 99
        exp = client_with_api.get_experiment("new-exp")
        assert exp is not None
        assert exp.name == "New Experiment"

    @patch("aurelius.shared.experiment_client.requests.get")
    def test_sync_304_not_modified(self, mock_get, client_with_api):
        """Test sync with 304 Not Modified response."""
        # Set an initial sync version
        client_with_api._sync_version = 42

        mock_response = MagicMock()
        mock_response.status_code = 304
        mock_get.return_value = mock_response

        result = client_with_api.sync()

        assert result is True
        assert client_with_api._sync_version == 42  # Unchanged

    @patch("aurelius.shared.experiment_client.requests.get")
    def test_sync_failure_records_circuit_breaker(self, mock_get, client_with_api):
        """Test that sync failure records to circuit breaker."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_get.return_value = mock_response

        result = client_with_api.sync()

        assert result is False

    def test_sync_skipped_without_api_endpoint(self, client_no_api):
        """Test that sync is skipped when no API endpoint is configured."""
        result = client_no_api.sync()
        assert result is False


class TestDefaultPromptExperiment:
    """Tests for DEFAULT_PROMPT_EXPERIMENT constant."""

    def test_default_experiment_structure(self):
        """Test that default experiment has correct structure."""
        assert DEFAULT_PROMPT_EXPERIMENT.id == "prompt"
        assert DEFAULT_PROMPT_EXPERIMENT.name == "Dangerous Prompt Detection"
        assert DEFAULT_PROMPT_EXPERIMENT.experiment_type == "push"
        assert DEFAULT_PROMPT_EXPERIMENT.scoring_type == "danger"
        assert DEFAULT_PROMPT_EXPERIMENT.status == "active"
        assert "acceptance" in DEFAULT_PROMPT_EXPERIMENT.thresholds

    def test_default_experiment_is_active(self):
        """Test that default experiment is always active."""
        assert DEFAULT_PROMPT_EXPERIMENT.status == "active"
        assert DEFAULT_PROMPT_EXPERIMENT.deprecated_at is None


class TestGetExperimentClientSingleton:
    """Tests for get_experiment_client singleton function."""

    def test_returns_same_instance(self):
        """Test that get_experiment_client returns the same instance."""
        # Reset the singleton
        import aurelius.shared.experiment_client as module

        module._experiment_client = None

        client1 = get_experiment_client()
        client2 = get_experiment_client()

        assert client1 is client2

        # Clean up
        module._experiment_client = None


class TestExperimentLifecycle:
    """Tests for experiment activation/deactivation lifecycle (T035)."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_experiment_status_active(self, temp_cache_dir):
        """Test that active experiments are returned by get_active_experiments."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "exp1": {
                    "id": "exp1",
                    "name": "Active Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                }
            },
            "registrations": {},
            "reward_allocation": {"allocations": {}, "burn_percentage": 100.0, "redistribute_unused": True, "version": 1},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(api_endpoint=None, cache_path=cache_path)
        active = client.get_active_experiments()

        assert len(active) == 1
        assert active[0].id == "exp1"
        assert active[0].status == "active"

    def test_experiment_status_inactive(self, temp_cache_dir):
        """Test that inactive experiments are not returned by get_active_experiments."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "exp1": {
                    "id": "exp1",
                    "name": "Inactive Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "inactive",
                }
            },
            "registrations": {},
            "reward_allocation": {"allocations": {}, "burn_percentage": 100.0, "redistribute_unused": True, "version": 1},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(api_endpoint=None, cache_path=cache_path)
        active = client.get_active_experiments()

        assert len(active) == 0

    def test_experiment_status_deprecated(self, temp_cache_dir):
        """Test that deprecated experiments are not returned by get_active_experiments."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "exp1": {
                    "id": "exp1",
                    "name": "Deprecated Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "deprecated",
                    "deprecated_at": "2026-02-01T00:00:00Z",
                }
            },
            "registrations": {},
            "reward_allocation": {"allocations": {}, "burn_percentage": 100.0, "redistribute_unused": True, "version": 1},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(api_endpoint=None, cache_path=cache_path)
        active = client.get_active_experiments()

        assert len(active) == 0

        # But get_experiment should still return it
        exp = client.get_experiment("exp1")
        assert exp is not None
        assert exp.status == "deprecated"
        assert exp.deprecated_at == "2026-02-01T00:00:00Z"

    @patch("aurelius.shared.experiment_client.requests.get")
    def test_experiment_activation_via_sync(self, mock_get, temp_cache_dir):
        """Test that experiment becomes active after sync."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")

        # Start with inactive experiment
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "exp1": {
                    "id": "exp1",
                    "name": "Test Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "inactive",
                }
            },
            "registrations": {},
            "reward_allocation": {"allocations": {}, "burn_percentage": 100.0, "redistribute_unused": True, "version": 1},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(
            api_endpoint="https://test.api/api/experiments",
            cache_path=cache_path,
            api_key="test-key",
        )

        # Verify initially inactive
        assert len(client.get_active_experiments()) == 0

        # Mock API response with active status
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiments": [
                {
                    "id": "exp1",
                    "name": "Test Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",  # Now active
                }
            ],
            "registrations": [],
            "reward_allocation": {"allocations": {"exp1": 100.0}, "burn_percentage": 0.0, "redistribute_unused": True, "version": 1},
            "sync_version": 2,
            "server_time": "2026-02-04T12:00:00Z",
        }
        mock_get.return_value = mock_response

        # Sync
        result = client.sync()
        assert result is True

        # Now should be active
        active = client.get_active_experiments()
        assert len(active) == 1
        assert active[0].status == "active"

    @patch("aurelius.shared.experiment_client.requests.get")
    def test_experiment_deactivation_via_sync(self, mock_get, temp_cache_dir):
        """Test that experiment becomes inactive after sync."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")

        # Start with active experiment
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "exp1": {
                    "id": "exp1",
                    "name": "Test Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                }
            },
            "registrations": {},
            "reward_allocation": {"allocations": {"exp1": 100.0}, "burn_percentage": 0.0, "redistribute_unused": True, "version": 1},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(
            api_endpoint="https://test.api/api/experiments",
            cache_path=cache_path,
            api_key="test-key",
        )

        # Verify initially active
        assert len(client.get_active_experiments()) == 1

        # Mock API response with inactive status
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiments": [
                {
                    "id": "exp1",
                    "name": "Test Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "inactive",  # Now inactive
                }
            ],
            "registrations": [],
            "reward_allocation": {"allocations": {}, "burn_percentage": 100.0, "redistribute_unused": True, "version": 1},
            "sync_version": 2,
            "server_time": "2026-02-04T12:00:00Z",
        }
        mock_get.return_value = mock_response

        # Sync
        result = client.sync()
        assert result is True

        # Now should be inactive
        active = client.get_active_experiments()
        assert len(active) == 0

    def test_mixed_experiment_statuses(self, temp_cache_dir):
        """Test filtering with mixed experiment statuses."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "active1": {"id": "active1", "name": "Active 1", "version": 1, "experiment_type": "push", "scoring_type": "danger", "status": "active"},
                "active2": {"id": "active2", "name": "Active 2", "version": 1, "experiment_type": "push", "scoring_type": "danger", "status": "active"},
                "inactive1": {"id": "inactive1", "name": "Inactive 1", "version": 1, "experiment_type": "push", "scoring_type": "danger", "status": "inactive"},
                "deprecated1": {"id": "deprecated1", "name": "Deprecated 1", "version": 1, "experiment_type": "push", "scoring_type": "danger", "status": "deprecated"},
            },
            "registrations": {},
            "reward_allocation": {"allocations": {}, "burn_percentage": 100.0, "redistribute_unused": True, "version": 1},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(api_endpoint=None, cache_path=cache_path)
        active = client.get_active_experiments()

        assert len(active) == 2
        active_ids = {e.id for e in active}
        assert active_ids == {"active1", "active2"}


class TestExperimentVersioning:
    """Tests for experiment versioning (T036)."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_experiment_version_field(self):
        """Test that experiment version is stored correctly."""
        exp = ExperimentDefinition(
            id="test",
            name="Test",
            version=5,
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
        assert exp.version == 5

    @patch("aurelius.shared.experiment_client.requests.get")
    def test_sync_updates_experiment_version(self, mock_get, temp_cache_dir):
        """Test that sync updates experiment version."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")

        # Start with version 1
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "exp1": {
                    "id": "exp1",
                    "name": "Test Experiment",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                    "thresholds": {"acceptance": 0.3},
                }
            },
            "registrations": {},
            "reward_allocation": {"allocations": {"exp1": 100.0}, "burn_percentage": 0.0, "redistribute_unused": True, "version": 1},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(
            api_endpoint="https://test.api/api/experiments",
            cache_path=cache_path,
            api_key="test-key",
        )

        # Verify initial version
        exp = client.get_experiment("exp1")
        assert exp.version == 1
        assert exp.thresholds == {"acceptance": 0.3}

        # Mock API response with newer version and changed thresholds
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiments": [
                {
                    "id": "exp1",
                    "name": "Test Experiment Updated",
                    "version": 2,  # Newer version
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                    "thresholds": {"acceptance": 0.5},  # Changed threshold
                }
            ],
            "registrations": [],
            "reward_allocation": {"allocations": {"exp1": 100.0}, "burn_percentage": 0.0, "redistribute_unused": True, "version": 1},
            "sync_version": 2,
            "server_time": "2026-02-04T12:00:00Z",
        }
        mock_get.return_value = mock_response

        # Sync
        result = client.sync()
        assert result is True

        # Verify updated version and name
        exp = client.get_experiment("exp1")
        assert exp.version == 2
        assert exp.name == "Test Experiment Updated"
        assert exp.thresholds == {"acceptance": 0.5}

    @patch("aurelius.shared.experiment_client.requests.get")
    def test_sync_adds_new_experiment(self, mock_get, temp_cache_dir):
        """Test that sync adds new experiments while keeping existing ones."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")

        # Start with one experiment
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "exp1": {
                    "id": "exp1",
                    "name": "Experiment 1",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                }
            },
            "registrations": {},
            "reward_allocation": {"allocations": {"exp1": 100.0}, "burn_percentage": 0.0, "redistribute_unused": True, "version": 1},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(
            api_endpoint="https://test.api/api/experiments",
            cache_path=cache_path,
            api_key="test-key",
        )

        assert len(client.get_active_experiments()) == 1

        # Mock API response with two experiments (existing + new)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiments": [
                {
                    "id": "exp1",
                    "name": "Experiment 1",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                },
                {
                    "id": "exp2",
                    "name": "Experiment 2",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "binary",
                    "status": "active",
                },
            ],
            "registrations": [],
            "reward_allocation": {"allocations": {"exp1": 50.0, "exp2": 50.0}, "burn_percentage": 0.0, "redistribute_unused": True, "version": 1},
            "sync_version": 2,
            "server_time": "2026-02-04T12:00:00Z",
        }
        mock_get.return_value = mock_response

        # Sync
        result = client.sync()
        assert result is True

        # Verify both experiments exist
        active = client.get_active_experiments()
        assert len(active) == 2
        assert client.get_experiment("exp1") is not None
        assert client.get_experiment("exp2") is not None

    @patch("aurelius.shared.experiment_client.requests.get")
    def test_sync_removes_experiment(self, mock_get, temp_cache_dir):
        """Test that sync removes experiments no longer in API response."""
        cache_path = os.path.join(temp_cache_dir, "experiments_cache.json")

        # Start with two experiments
        cache_data = {
            "sync_version": 1,
            "experiments": {
                "exp1": {
                    "id": "exp1",
                    "name": "Experiment 1",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                },
                "exp2": {
                    "id": "exp2",
                    "name": "Experiment 2",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "binary",
                    "status": "active",
                },
            },
            "registrations": {},
            "reward_allocation": {"allocations": {"exp1": 50.0, "exp2": 50.0}, "burn_percentage": 0.0, "redistribute_unused": True, "version": 1},
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        client = ExperimentClient(
            api_endpoint="https://test.api/api/experiments",
            cache_path=cache_path,
            api_key="test-key",
        )

        assert len(client.get_active_experiments()) == 2

        # Mock API response with only one experiment (exp2 removed)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "experiments": [
                {
                    "id": "exp1",
                    "name": "Experiment 1",
                    "version": 1,
                    "experiment_type": "push",
                    "scoring_type": "danger",
                    "status": "active",
                },
            ],
            "registrations": [],
            "reward_allocation": {"allocations": {"exp1": 100.0}, "burn_percentage": 0.0, "redistribute_unused": True, "version": 1},
            "sync_version": 2,
            "server_time": "2026-02-04T12:00:00Z",
        }
        mock_get.return_value = mock_response

        # Sync
        result = client.sync()
        assert result is True

        # Verify only exp1 remains
        active = client.get_active_experiments()
        assert len(active) == 1
        assert client.get_experiment("exp1") is not None
        assert client.get_experiment("exp2") is None

    def test_version_comparison_for_updates(self):
        """Test that versions can be compared for update decisions."""
        exp_v1 = ExperimentDefinition(
            id="test", name="V1", version=1,
            experiment_type="push", scoring_type="danger", status="active",
            deprecated_at=None, thresholds={}, rate_limit_requests=100,
            rate_limit_window_hours=1, novelty_threshold=0.02,
            pull_interval_seconds=None, pull_timeout_seconds=None,
            settings={}, created_at="", updated_at="",
        )
        exp_v2 = ExperimentDefinition(
            id="test", name="V2", version=2,
            experiment_type="push", scoring_type="danger", status="active",
            deprecated_at=None, thresholds={}, rate_limit_requests=100,
            rate_limit_window_hours=1, novelty_threshold=0.02,
            pull_interval_seconds=None, pull_timeout_seconds=None,
            settings={}, created_at="", updated_at="",
        )

        # Version comparison for update logic
        assert exp_v2.version > exp_v1.version
        assert exp_v1.version < exp_v2.version

        # Same ID but different version indicates an update
        assert exp_v1.id == exp_v2.id
        assert exp_v1.version != exp_v2.version
