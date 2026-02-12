"""
Tests for validator-to-collector API data flow.

These tests verify that validators can submit execution data
to the collector API and that data persists correctly.
"""

import uuid

import pytest

from .helpers.api_client import (
    CollectorAPIClient,
    generate_deterministic_embedding,
)

pytestmark = [pytest.mark.e2e, pytest.mark.requires_docker]


class TestHealthEndpoints:
    """Tests for collector API health checks."""

    def test_health_endpoint(self, collector_api: CollectorAPIClient) -> None:
        """Verify health endpoint returns healthy status."""
        response = collector_api.health()
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "database" in data
        assert data["database"]["connected"] is True

    def test_detailed_health(self, collector_api: CollectorAPIClient) -> None:
        """Verify detailed health endpoint provides metrics."""
        response = collector_api.health_detailed()
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "system" in data
        assert "memory" in data["system"]
        assert "rateLimiter" in data


class TestExecutionSubmission:
    """Tests for execution data submission."""

    def test_submit_execution_basic(
        self,
        collector_api: CollectorAPIClient,
        validator_hotkey: str,
        validator_uid: int,
        sample_prompt: str,
        sample_response: str,
        sample_danger_score: float,
    ) -> None:
        """Submit basic execution data without wallet signing."""
        response = collector_api.submit_execution(
            prompt=sample_prompt,
            response=sample_response,
            danger_score=sample_danger_score,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            accepted=True,
        )

        # May succeed or fail based on signature requirements
        # During migration period, unsigned submissions might be allowed
        assert response.status_code in [201, 401], (
            f"Unexpected status: {response.status_code} - {response.text}"
        )

        if response.status_code == 201:
            data = response.json()
            assert data["success"] is True
            assert "execution_id" in data
            assert data["validator"]["hotkey"] == validator_hotkey

    def test_submit_execution_with_signature(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
        validator_uid: int,
        sample_prompt: str,
        sample_response: str,
        sample_danger_score: float,
    ) -> None:
        """Submit execution data with wallet signature."""
        response = collector_api.submit_execution(
            prompt=sample_prompt,
            response=sample_response,
            danger_score=sample_danger_score,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            wallet=validator_wallet,
            accepted=True,
        )

        assert response.status_code == 201, f"Failed: {response.text}"

        data = response.json()
        assert data["success"] is True
        assert "execution_id" in data
        assert data["validator"]["hotkey"] == validator_hotkey

    def test_submit_execution_with_miner_info(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
        validator_uid: int,
    ) -> None:
        """Submit execution with miner information."""
        # Generate fake miner hotkey for testing
        miner_hotkey = "5FakeTestMinerHotkey" + "A" * 28

        response = collector_api.submit_execution(
            prompt="Test prompt for miner submission",
            response="Test response from LLM",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            wallet=validator_wallet,
            miner_hotkey=miner_hotkey,
            miner_uid=42,
            accepted=True,
        )

        assert response.status_code == 201, f"Failed: {response.text}"

        data = response.json()
        assert data["success"] is True

    def test_submit_execution_with_embedding(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
        validator_uid: int,
        unique_prompt: str,
    ) -> None:
        """Submit execution with embedding for novelty scoring."""
        embedding = generate_deterministic_embedding(unique_prompt)

        response = collector_api.submit_execution(
            prompt=unique_prompt,
            response="A detailed explanation of quantum entanglement...",
            danger_score=0.05,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            wallet=validator_wallet,
            accepted=True,
            prompt_embedding=embedding,
        )

        assert response.status_code == 201, f"Failed: {response.text}"

        data = response.json()
        assert data["success"] is True
        # Should have novelty info if embedding was processed
        if data.get("novelty"):
            assert "score" in data["novelty"]
            # First submission of unique prompt should have high novelty
            assert data["novelty"]["score"] > 0.5

    def test_submit_consensus_verified(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
        validator_uid: int,
    ) -> None:
        """Submit consensus-verified execution."""
        response = collector_api.submit_execution(
            prompt="Consensus verification test prompt",
            response="Response verified by multiple validators",
            danger_score=0.02,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            wallet=validator_wallet,
            accepted=True,
            consensus_verified=True,
            validator_votes=[{"hotkey": "5ValidatorA" + "X" * 38, "vote": True}],
            mean_danger_score=0.025,
            std_dev_danger_score=0.01,
            validator_count=2,
        )

        assert response.status_code == 201, f"Failed: {response.text}"

        data = response.json()
        assert data["success"] is True

    def test_submit_rejected_prompt(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
        validator_uid: int,
    ) -> None:
        """Submit rejected (dangerous) prompt."""
        response = collector_api.submit_execution(
            prompt="This is a test of rejected content handling",
            response="I cannot help with that request.",
            danger_score=0.85,  # High danger score
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            wallet=validator_wallet,
            accepted=False,  # Rejected
        )

        assert response.status_code == 201, f"Failed: {response.text}"

        data = response.json()
        assert data["success"] is True


class TestExecutionRetrieval:
    """Tests for retrieving execution data."""

    def test_get_executions(self, collector_api: CollectorAPIClient) -> None:
        """Retrieve execution records."""
        response = collector_api.get_executions(limit=10)

        assert response.status_code == 200

        data = response.json()
        assert "executions" in data
        assert "count" in data

    def test_get_executions_by_validator(
        self,
        collector_api: CollectorAPIClient,
        validator_hotkey: str,
    ) -> None:
        """Retrieve executions filtered by validator."""
        response = collector_api.get_executions(
            validator_hotkey=validator_hotkey,
            limit=10,
        )

        assert response.status_code == 200

        data = response.json()
        # All returned executions should be from our validator
        for execution in data["executions"]:
            assert execution["validator_hotkey"] == validator_hotkey

    def test_get_execution_stats(self, collector_api: CollectorAPIClient) -> None:
        """Get execution statistics."""
        response = collector_api.get_execution_stats()

        assert response.status_code == 200

        data = response.json()
        assert "stats" in data
        assert "total_executions" in data["stats"]
        assert "unique_validators" in data["stats"]


class TestValidationErrors:
    """Tests for API validation error handling."""

    def test_missing_required_fields(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Submit with missing required fields."""
        import requests

        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={"prompt": "Only prompt, missing other fields"},
            headers={"Content-Type": "application/json"},
        )

        # Should fail with 400 (validation) or 401 (auth required)
        assert response.status_code in [400, 401]
        data = response.json()
        assert "error" in data

    def test_invalid_danger_score(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit with invalid danger score."""
        import requests

        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={
                "prompt": "Test prompt",
                "response": "Test response",
                "danger_score": 2.0,  # Invalid: should be 0-1
                "validator_hotkey": validator_hotkey,
                "accepted": True,
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400

    def test_invalid_embedding_dimensions(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit with wrong embedding dimensions."""
        import requests

        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={
                "prompt": "Test prompt",
                "response": "Test response",
                "danger_score": 0.1,
                "validator_hotkey": validator_hotkey,
                "accepted": True,
                "prompt_embedding": [0.1] * 100,  # Wrong: should be 384
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400


class TestDataPersistence:
    """Tests for data persistence in database."""

    def test_submitted_data_persists(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
        validator_uid: int,
    ) -> None:
        """Verify submitted execution persists and can be retrieved."""
        # Create unique prompt to identify our submission
        unique_id = str(uuid.uuid4())[:8]
        prompt = f"Persistence test prompt {unique_id}"

        # Submit
        submit_response = collector_api.submit_execution(
            prompt=prompt,
            response=f"Response for persistence test {unique_id}",
            danger_score=0.05,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            wallet=validator_wallet,
            accepted=True,
        )

        assert submit_response.status_code == 201
        execution_id = submit_response.json()["execution_id"]

        # Retrieve and verify
        get_response = collector_api.get_executions(
            validator_hotkey=validator_hotkey,
            limit=50,
        )

        assert get_response.status_code == 200
        executions = get_response.json()["executions"]

        # Find our submission
        found = False
        for execution in executions:
            if execution.get("id") == execution_id:
                found = True
                assert prompt in execution["prompt"]
                break

        assert found, f"Submitted execution {execution_id} not found in retrieval"
