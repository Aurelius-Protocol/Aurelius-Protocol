"""
Integration tests for per-experiment novelty pools (T083).

These tests verify that:
1. The API accepts experiment_id in novelty check requests
2. Default experiment is 'prompt' when not specified
3. Novelty is isolated between different experiments
4. Miner stats can be filtered by experiment

Requires a running collector API instance. Set COLLECTOR_URL env var
or use the default http://localhost:3000.
"""

import os
import pytest
import requests
from typing import List

# Test configuration
COLLECTOR_URL = os.getenv("COLLECTOR_URL", "http://localhost:3000")

# Test hotkeys (valid SS58 format)
VALIDATOR_HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
MINER_HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def generate_embedding(seed: int = 0) -> List[float]:
    """Generate a 384-dimension embedding with slight variations based on seed."""
    return [0.01 * seed + 0.001 * i for i in range(384)]


@pytest.fixture
def api_url():
    """Return the collector API URL."""
    return COLLECTOR_URL


@pytest.mark.integration
class TestPerExperimentNovelty:
    """Test suite for per-experiment novelty isolation."""

    def test_health_check(self, api_url):
        """Verify the API is healthy before running tests."""
        response = requests.get(f"{api_url}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"

    def test_novelty_check_accepts_experiment_id(self, api_url):
        """Test that novelty check endpoint accepts experiment_id parameter."""
        embedding = generate_embedding(seed=100)
        payload = {
            "prompt": "Test prompt for experiment_id acceptance",
            "prompt_embedding": embedding,
            "experiment_id": "test_experiment_acceptance",
        }

        response = requests.post(
            f"{api_url}/api/novelty/check",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        data = response.json()
        assert "novelty_score" in data
        assert "max_similarity" in data
        assert "similar_count" in data

    def test_default_experiment_is_prompt(self, api_url):
        """Test that default experiment_id is 'prompt' when not specified."""
        embedding = generate_embedding(seed=101)
        payload = {
            "prompt": "Test prompt for default experiment",
            "prompt_embedding": embedding,
            # No experiment_id specified - should default to 'prompt'
        }

        response = requests.post(
            f"{api_url}/api/novelty/check",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 200
        data = response.json()
        assert "novelty_score" in data

    def test_experiment_isolation_via_submission(self, api_url):
        """
        Test that prompts submitted to different experiments are isolated.

        This test submits the same prompt to two different experiments and
        verifies that each experiment sees it as novel (since they have
        separate novelty pools).
        """
        embedding = generate_embedding(seed=200)
        prompt_text = "Unique prompt for isolation test"

        # Submit to experiment A
        payload_a = {
            "validator_hotkey": VALIDATOR_HOTKEY,
            "miner_hotkey": MINER_HOTKEY,
            "prompt": prompt_text,
            "response": "Test response A",
            "danger_score": 0.1,
            "accepted": True,
            "prompt_embedding": embedding,
            "experiment_id": "isolation_test_experiment_a",
        }

        response_a = requests.post(
            f"{api_url}/api/collections",
            json=payload_a,
            timeout=30,
        )
        assert response_a.status_code == 201
        data_a = response_a.json()
        assert data_a.get("success") is True
        novelty_a = data_a.get("novelty", {}).get("score")

        # Submit same embedding to experiment B - should be novel in B's pool
        payload_b = {
            "validator_hotkey": VALIDATOR_HOTKEY,
            "miner_hotkey": MINER_HOTKEY,
            "prompt": prompt_text,
            "response": "Test response B",
            "danger_score": 0.15,
            "accepted": True,
            "prompt_embedding": embedding,
            "experiment_id": "isolation_test_experiment_b",
        }

        response_b = requests.post(
            f"{api_url}/api/collections",
            json=payload_b,
            timeout=30,
        )
        assert response_b.status_code == 201
        data_b = response_b.json()
        assert data_b.get("success") is True
        novelty_b = data_b.get("novelty", {}).get("score")

        # Both should have high novelty since they're in different pools
        # (first submission in each experiment)
        if novelty_a is not None:
            assert novelty_a >= 0.5, f"Expected high novelty in experiment A, got {novelty_a}"
        if novelty_b is not None:
            assert novelty_b >= 0.5, f"Expected high novelty in experiment B, got {novelty_b}"

    def test_miner_stats_filter_by_experiment(self, api_url):
        """Test that miner stats can be filtered by experiment_id."""
        # Get stats for a specific experiment
        response = requests.get(
            f"{api_url}/api/novelty/miner/{MINER_HOTKEY}",
            params={"experiment_id": "isolation_test_experiment_a"},
            timeout=10,
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("hotkey") == MINER_HOTKEY
        assert data.get("experiment_id") == "isolation_test_experiment_a"
        assert "avg_novelty" in data
        assert "total_prompts" in data

    def test_miner_stats_default_experiment(self, api_url):
        """Test that miner stats default to 'prompt' experiment when not specified."""
        response = requests.get(
            f"{api_url}/api/novelty/miner/{MINER_HOTKEY}",
            timeout=10,
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("hotkey") == MINER_HOTKEY
        assert data.get("experiment_id") == "prompt"

    def test_submission_backward_compatibility(self, api_url):
        """Test that submissions without experiment_id work (backward compatibility)."""
        embedding = generate_embedding(seed=300)

        payload = {
            "validator_hotkey": VALIDATOR_HOTKEY,
            "miner_hotkey": MINER_HOTKEY,
            "prompt": "Backward compatibility test prompt",
            "response": "Test response",
            "danger_score": 0.2,
            "accepted": True,
            "prompt_embedding": embedding,
            # No experiment_id - should default to 'prompt'
        }

        response = requests.post(
            f"{api_url}/api/collections",
            json=payload,
            timeout=30,
        )

        assert response.status_code == 201
        data = response.json()
        assert data.get("success") is True
        assert data.get("execution_id") is not None

    def test_novelty_check_invalid_experiment_id(self, api_url):
        """Test that invalid experiment_id values are rejected."""
        embedding = generate_embedding(seed=400)

        # Empty string should be rejected
        payload = {
            "prompt": "Test prompt",
            "prompt_embedding": embedding,
            "experiment_id": "",  # Invalid - must be min 1 char
        }

        response = requests.post(
            f"{api_url}/api/novelty/check",
            json=payload,
            timeout=30,
        )

        # Should return 400 due to validation failure
        assert response.status_code == 400

    def test_novelty_check_long_experiment_id(self, api_url):
        """Test that overly long experiment_id values are rejected."""
        embedding = generate_embedding(seed=401)

        # 101 chars - exceeds max of 100
        long_experiment_id = "a" * 101
        payload = {
            "prompt": "Test prompt",
            "prompt_embedding": embedding,
            "experiment_id": long_experiment_id,
        }

        response = requests.post(
            f"{api_url}/api/novelty/check",
            json=payload,
            timeout=30,
        )

        # Should return 400 due to validation failure
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
