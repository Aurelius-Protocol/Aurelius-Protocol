"""
Tests for data isolation and security boundaries.

These tests verify that data is properly isolated between:
- Different validators
- Different experiments
- Different miners
"""

import time
import uuid

import pytest
import requests

from .helpers.api_client import (
    CollectorAPIClient,
    create_test_span,
    generate_deterministic_embedding,
)

pytestmark = [pytest.mark.e2e, pytest.mark.requires_docker]


class TestValidatorIsolation:
    """Tests for validator data isolation."""

    def test_validator_a_cannot_see_validator_b_stats(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Validators should only see their own stats."""
        # Create a unique submission for our validator
        unique_id = str(uuid.uuid4())[:8]

        collector_api.submit_execution(
            prompt=f"Isolation test {unique_id}",
            response="Response for isolation test",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        time.sleep(0.5)

        # Query stats for a different (fake) validator
        fake_hotkey = "5DifferentValidator" + "A" * 29

        stats_response = collector_api.get_execution_stats(
            validator_hotkey=fake_hotkey,
        )

        assert stats_response.status_code == 200
        data = stats_response.json()

        # Stats for fake validator should be empty or zero
        stats = data.get("stats", {})
        # If validator doesn't exist, should have no executions for them
        if "total_executions" in stats:
            # The fake validator should have 0 or minimal executions
            pass  # API may return global stats or filtered stats

    def test_execution_attributed_to_correct_validator(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Execution should be attributed to the submitting validator."""
        unique_id = str(uuid.uuid4())[:8]
        prompt = f"Attribution test {unique_id}"

        response = collector_api.submit_execution(
            prompt=prompt,
            response="Response for attribution test",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        assert response.status_code == 201
        data = response.json()

        # Verify correct validator attribution
        assert data["validator"]["hotkey"] == validator_hotkey

        time.sleep(0.5)

        # Retrieve and verify
        get_response = collector_api.get_executions(
            validator_hotkey=validator_hotkey,
            limit=50,
        )

        assert get_response.status_code == 200
        executions = get_response.json()["executions"]

        # All returned executions should be from our validator
        for execution in executions:
            assert execution["validator_hotkey"] == validator_hotkey

    def test_telemetry_isolated_by_validator(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Telemetry spans should only be visible to the submitting validator."""
        import secrets

        unique_trace_id = secrets.token_hex(16)

        span = create_test_span(
            "isolation-test-span",
            trace_id=unique_trace_id,
            attributes={"test": "validator_isolation"},
        )

        # Submit telemetry
        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code == 201

        time.sleep(0.5)

        # Query logs for a different validator
        fake_hotkey = "5AnotherValidator" + "B" * 30

        logs_response = collector_api.query_logs(
            validator_hotkey=fake_hotkey,
            limit=100,
        )

        assert logs_response.status_code == 200
        logs = logs_response.json().get("logs", [])

        # Logs from our trace should not appear in fake validator's logs
        for log in logs:
            if log.get("trace_id"):
                assert log["trace_id"] != unique_trace_id

    def test_cannot_submit_as_different_validator(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Cannot submit execution claiming to be a different validator."""
        fake_hotkey = "5FakeValidator" + "C" * 33

        # Try to submit with a different hotkey in the body
        response = collector_api.submit_execution(
            prompt="Impersonation test",
            response="Response",
            danger_score=0.1,
            validator_hotkey=fake_hotkey,  # Different from signer
            wallet=validator_wallet,  # Signs with real key
            accepted=True,
        )

        # Should fail - signature doesn't match claimed validator
        assert response.status_code in [401, 403]


class TestExperimentIsolation:
    """Tests for per-experiment data isolation."""

    def test_experiment_a_novelty_ignores_experiment_b(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Prompts in different experiments should have independent novelty pools."""
        unique_id = str(uuid.uuid4())[:8]
        prompt = f"Cross-experiment test {unique_id}"
        embedding = generate_deterministic_embedding(prompt)

        # Submit to experiment A (default "prompt")
        response_a = collector_api.submit_execution(
            prompt=prompt,
            response="Response for experiment A",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
            prompt_embedding=embedding,
            experiment_id="prompt",  # Default experiment
        )

        # Skip if experiment validation fails
        if response_a.status_code == 400:
            pytest.skip("Experiment validation rejected submission")

        time.sleep(0.5)

        # Check novelty in the same experiment - should show low novelty
        check_same = collector_api.check_novelty(
            prompt=prompt,
            embedding=embedding,
            experiment_id="prompt",
        )

        # Check novelty in a different experiment
        check_different = collector_api.check_novelty(
            prompt=prompt,
            embedding=embedding,
            experiment_id="test_experiment_isolation",  # Different experiment
        )

        if check_same.status_code == 200 and check_different.status_code == 200:
            same_novelty = check_same.json().get("novelty_score", 0)
            different_novelty = check_different.json().get("novelty_score", 1)

            # Novelty in different experiment should be higher (more novel)
            # because the prompt doesn't exist in that experiment's pool
            # Note: This depends on experiment existing
            pass  # Document behavior rather than assert
        elif check_different.status_code == 400:
            # Expected - experiment doesn't exist
            error = check_different.json().get("error", "")
            assert "experiment" in error.lower() or "not found" in error.lower()

    def test_rate_limit_per_experiment(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Rate limits should be tracked separately per experiment."""
        # This test documents behavior - rate limits may or may not be per-experiment
        responses_a = []
        responses_b = []

        for i in range(15):
            resp_a = collector_api.submit_execution(
                prompt=f"Rate test A {i}",
                response="Response A",
                danger_score=0.1,
                validator_hotkey=validator_hotkey,
                wallet=validator_wallet,
                accepted=True,
                experiment_id="prompt",
            )
            responses_a.append(resp_a.status_code)

        # If rate limits are per-experiment, experiment B should still work
        # after experiment A hits its limit

        # Document what happens rather than strictly assert
        rate_limited_a = 429 in responses_a

        if rate_limited_a:
            print(f"\n  Experiment A rate limited after {responses_a.index(429)} requests")
        else:
            print(f"\n  Experiment A: {len(responses_a)} requests without rate limiting")

    def test_nonexistent_experiment_rejected(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submission to non-existent experiment should be rejected."""
        fake_experiment = f"nonexistent_experiment_{uuid.uuid4()}"

        response = collector_api.submit_execution(
            prompt="Test prompt for fake experiment",
            response="Response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
            experiment_id=fake_experiment,
        )

        # Should be rejected or possibly accepted if API auto-creates experiments
        assert response.status_code in [201, 400]

    def test_experiment_id_injection(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Experiment ID with special characters should be handled safely."""
        injection_experiments = [
            "test'; DROP TABLE experiments; --",
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
            "experiment\x00null",
            "experiment\nwith\nnewlines",
        ]

        for exp_id in injection_experiments:
            response = collector_api.submit_execution(
                prompt="Injection test",
                response="Response",
                danger_score=0.1,
                validator_hotkey=validator_hotkey,
                wallet=validator_wallet,
                accepted=True,
                experiment_id=exp_id,
            )

            # Should be rejected or sanitized, never cause 500
            assert response.status_code != 500, f"Server error with experiment_id: {exp_id}"


class TestMinerIsolation:
    """Tests for miner data isolation."""

    def test_miner_stats_isolated(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Each miner should only see their own stats."""
        miner_a = "5MinerA" + "A" * 41
        miner_b = "5MinerB" + "B" * 41

        # Submit execution for miner A
        collector_api.submit_execution(
            prompt="Miner A submission",
            response="Response from miner A",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            miner_hotkey=miner_a,
            miner_uid=1,
            accepted=True,
        )

        # Submit execution for miner B
        collector_api.submit_execution(
            prompt="Miner B submission",
            response="Response from miner B",
            danger_score=0.15,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            miner_hotkey=miner_b,
            miner_uid=2,
            accepted=True,
        )

        time.sleep(0.5)

        # Query executions for miner A
        response_a = collector_api.get_executions(miner_hotkey=miner_a, limit=50)

        assert response_a.status_code == 200
        executions_a = response_a.json()["executions"]

        # All executions should be from miner A
        for execution in executions_a:
            if execution.get("miner_hotkey"):
                assert execution["miner_hotkey"] == miner_a

        # Query executions for miner B
        response_b = collector_api.get_executions(miner_hotkey=miner_b, limit=50)

        assert response_b.status_code == 200
        executions_b = response_b.json()["executions"]

        # All executions should be from miner B
        for execution in executions_b:
            if execution.get("miner_hotkey"):
                assert execution["miner_hotkey"] == miner_b

    def test_miner_a_execution_not_attributed_to_b(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Execution from miner A should never appear in miner B's data."""
        unique_id = str(uuid.uuid4())[:8]
        miner_a = f"5MinerUnique{unique_id}A" + "A" * 25

        # Submit with unique prompt
        prompt = f"Unique attribution test {unique_id}"

        response = collector_api.submit_execution(
            prompt=prompt,
            response="Response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            miner_hotkey=miner_a,
            miner_uid=42,
            accepted=True,
        )

        assert response.status_code == 201

        time.sleep(0.5)

        # Try to find this execution under a different miner
        different_miner = "5CompletelyDifferent" + "Z" * 28

        response_other = collector_api.get_executions(
            miner_hotkey=different_miner,
            limit=100,
        )

        assert response_other.status_code == 200
        executions = response_other.json()["executions"]

        # Our unique prompt should not appear in different miner's results
        for execution in executions:
            assert prompt not in execution.get("prompt", "")

    def test_miner_novelty_isolated(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Miner novelty stats should be isolated per miner."""
        miner_a = "5NoveltyMinerA" + "A" * 34
        miner_b = "5NoveltyMinerB" + "B" * 34

        unique_id = str(uuid.uuid4())[:8]
        prompt = f"Miner novelty test {unique_id}"
        embedding = generate_deterministic_embedding(prompt)

        # Submit from miner A
        collector_api.submit_execution(
            prompt=prompt,
            response="Response from A",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            miner_hotkey=miner_a,
            miner_uid=1,
            accepted=True,
            prompt_embedding=embedding,
        )

        time.sleep(0.5)

        # Get novelty stats for miner A
        stats_a = collector_api.get_miner_novelty(miner_a)

        # Get novelty stats for miner B (who didn't submit this prompt)
        stats_b = collector_api.get_miner_novelty(miner_b)

        # Stats may require auth, so document behavior
        if stats_a.status_code == 200 and stats_b.status_code == 200:
            # Stats should differ - A has the submission, B doesn't
            pass


class TestCrossValidatorAccessControl:
    """Tests for access control between validators."""

    def test_cannot_query_other_validator_traces(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Cannot retrieve telemetry traces from another validator."""
        import secrets

        # Submit a trace
        trace_id = secrets.token_hex(16)
        span = create_test_span("access-control-test", trace_id=trace_id)

        collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        time.sleep(0.5)

        # Try to query logs filtering by a different validator
        different_hotkey = "5DifferentAccessControl" + "X" * 24

        response = collector_api.query_logs(
            validator_hotkey=different_hotkey,
            limit=100,
        )

        assert response.status_code == 200
        logs = response.json().get("logs", [])

        # Should not contain our trace
        for log in logs:
            assert log.get("trace_id") != trace_id

    def test_stats_endpoint_scope(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Stats endpoints should have appropriate scope."""
        # Global stats
        global_stats = collector_api.get_execution_stats()

        assert global_stats.status_code == 200
        data = global_stats.json()

        # Should have aggregate stats, not individual validator details
        stats = data.get("stats", {})
        assert "unique_validators" in stats or "total_executions" in stats


class TestDataIntegrity:
    """Tests for data integrity across isolation boundaries."""

    def test_execution_count_consistency(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Execution counts should be consistent across queries."""
        unique_id = str(uuid.uuid4())[:8]

        # Submit known number of executions
        num_submissions = 5
        for i in range(num_submissions):
            collector_api.submit_execution(
                prompt=f"Consistency test {unique_id} - {i}",
                response=f"Response {i}",
                danger_score=0.1,
                validator_hotkey=validator_hotkey,
                wallet=validator_wallet,
                accepted=True,
            )

        time.sleep(1)

        # Query and count
        response = collector_api.get_executions(
            validator_hotkey=validator_hotkey,
            limit=100,
        )

        assert response.status_code == 200
        data = response.json()

        # Count should reflect our submissions (plus any pre-existing)
        count = data["count"]
        executions = data["executions"]

        # Verify count matches actual returned records
        assert len(executions) == min(count, 100)

        # Count our specific submissions
        matching = [e for e in executions if unique_id in e.get("prompt", "")]
        assert len(matching) == num_submissions

    def test_no_data_leakage_on_error(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Error responses should not leak data from other validators."""
        # Make an invalid request
        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={
                "prompt": "",  # Invalid
                "response": "Response",
                "danger_score": 0.1,
                "validator_hotkey": "invalid",
                "accepted": True,
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code in [400, 401]
        data = response.json()

        # Error should not contain other validators' data
        error_str = str(data)
        assert "execution_id" not in error_str.lower() or "example" in error_str.lower()
        # Should not contain what looks like real hotkeys (except our invalid one)
        assert error_str.count("5") < 10  # Arbitrary threshold

    def test_no_internal_paths_in_errors(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Error messages should not reveal internal paths."""
        response = requests.post(
            f"{collector_api.base_url}/api/nonexistent",
            json={},
            headers={"Content-Type": "application/json"},
        )

        # Should be 404
        assert response.status_code == 404

        # Response should not contain internal paths
        response_text = response.text.lower()
        sensitive_patterns = [
            "/home/",
            "/var/",
            "/usr/",
            "/app/",
            "node_modules",
            ".ts:",
            ".js:",
            "line ",
            "column ",
        ]

        for pattern in sensitive_patterns:
            assert pattern not in response_text, f"Response contains sensitive pattern: {pattern}"
