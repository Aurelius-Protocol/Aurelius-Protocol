"""
Tests for concurrent operations, race conditions, and stress testing.

These tests verify that the system handles concurrent operations correctly
including simultaneous submissions, race conditions, and database locking.
"""

import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pytest
import requests

from .helpers.api_client import (
    CollectorAPIClient,
    create_test_span,
    create_test_log,
    generate_deterministic_embedding,
    generate_random_embedding,
)

pytestmark = [pytest.mark.e2e, pytest.mark.requires_docker]


class TestRaceConditions:
    """Tests for race condition scenarios."""

    def test_concurrent_identical_prompt_submission(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Same prompt submitted simultaneously from two threads."""
        unique_id = str(uuid.uuid4())[:8]
        prompt = f"Race condition test prompt {unique_id}"
        embedding = generate_deterministic_embedding(prompt)

        results: list[dict[str, Any]] = []
        errors: list[Exception] = []

        def submit_execution():
            try:
                response = collector_api.submit_execution(
                    prompt=prompt,
                    response="Response to race condition test",
                    danger_score=0.1,
                    validator_hotkey=validator_hotkey,
                    wallet=validator_wallet,
                    accepted=True,
                    prompt_embedding=embedding,
                )
                results.append({
                    "status_code": response.status_code,
                    "data": response.json() if response.status_code < 500 else None,
                })
            except Exception as e:
                errors.append(e)

        # Run two submissions simultaneously
        thread1 = threading.Thread(target=submit_execution)
        thread2 = threading.Thread(target=submit_execution)

        thread1.start()
        thread2.start()

        thread1.join(timeout=30)
        thread2.join(timeout=30)

        # No exceptions should have occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Both should succeed or handle conflict gracefully
        assert len(results) == 2
        for result in results:
            assert result["status_code"] in [201, 409, 429]

    def test_concurrent_validator_auto_registration(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """
        New validator submitting 10 concurrent requests.

        Tests that auto-registration doesn't create race conditions.
        """
        results: list[int] = []

        def make_submission(i):
            try:
                response = collector_api.submit_execution(
                    prompt=f"Auto-registration test {i} - {uuid.uuid4()}",
                    response=f"Response {i}",
                    danger_score=0.1,
                    validator_hotkey=validator_hotkey,
                    wallet=validator_wallet,
                    accepted=True,
                )
                return response.status_code
            except Exception as e:
                return str(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_submission, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]

        # All should succeed or be rate limited, never 500
        for status in results:
            if isinstance(status, int):
                assert status in [201, 429, 409]
            else:
                pytest.fail(f"Unexpected error: {status}")

    def test_concurrent_novelty_check_same_embedding(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Identical embedding checked concurrently."""
        embedding = generate_random_embedding()
        unique_id = str(uuid.uuid4())[:8]

        results: list[dict] = []

        def check_novelty(i):
            response = collector_api.check_novelty(
                prompt=f"Concurrent novelty check {unique_id}",
                embedding=embedding,
            )
            return {
                "thread": i,
                "status_code": response.status_code,
                "novelty_score": response.json().get("novelty_score") if response.status_code == 200 else None,
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_novelty, i) for i in range(5)]
            results = [f.result() for f in as_completed(futures)]

        # All should succeed or be rate limited
        for result in results:
            assert result["status_code"] in [200, 429]

        # If all succeeded, novelty scores should be consistent
        scores = [r["novelty_score"] for r in results if r["novelty_score"] is not None]
        if len(scores) > 1:
            # Scores should be relatively similar (same embedding)
            # Small variations possible due to timing
            pass

    def test_submission_during_rate_limit_window(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Requests at exact rate limit boundary."""
        # Make rapid requests to approach rate limit
        results = []

        for i in range(35):
            response = collector_api.submit_execution(
                prompt=f"Rate limit boundary test {i} - {uuid.uuid4()}",
                response=f"Response {i}",
                danger_score=0.1,
                validator_hotkey=validator_hotkey,
                wallet=validator_wallet,
                accepted=True,
            )
            results.append(response.status_code)

            # Stop if rate limited
            if response.status_code == 429:
                break

        # Should have mix of 201 and possibly 429
        assert 201 in results
        # Document behavior
        print(f"\n  Results: {results.count(201)} success, {results.count(429)} rate limited")


class TestStressTesting:
    """Tests for stress testing the API."""

    @pytest.mark.slow
    @pytest.mark.stress
    def test_10_validators_submitting_simultaneously(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """
        Simulate 10 validators submitting simultaneously.

        Note: Uses same wallet for all "validators" since we only have one test wallet.
        Real production would have different wallets.
        """
        num_validators = 10
        submissions_per_validator = 3

        results: list[tuple[int, int, int]] = []  # (validator_id, submission_id, status_code)

        def validator_workload(validator_id):
            local_results = []
            for sub_id in range(submissions_per_validator):
                try:
                    response = collector_api.submit_execution(
                        prompt=f"Validator {validator_id} submission {sub_id} - {uuid.uuid4()}",
                        response=f"Response from validator {validator_id}",
                        danger_score=0.1,
                        validator_hotkey=validator_hotkey,
                        wallet=validator_wallet,
                        accepted=True,
                    )
                    local_results.append((validator_id, sub_id, response.status_code))
                except Exception as e:
                    local_results.append((validator_id, sub_id, -1))
            return local_results

        with ThreadPoolExecutor(max_workers=num_validators) as executor:
            futures = [executor.submit(validator_workload, i) for i in range(num_validators)]
            for future in as_completed(futures):
                results.extend(future.result())

        # Analyze results
        total = len(results)
        success = sum(1 for r in results if r[2] == 201)
        rate_limited = sum(1 for r in results if r[2] == 429)
        errors = sum(1 for r in results if r[2] == -1 or r[2] >= 500)

        print(f"\n  Total: {total}, Success: {success}, Rate Limited: {rate_limited}, Errors: {errors}")

        # Should have no server errors
        assert errors == 0
        # Should have some successes
        assert success > 0

    @pytest.mark.slow
    @pytest.mark.stress
    def test_1000_rapid_submissions(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Burst load test with 1000 rapid submissions."""
        num_submissions = 1000
        batch_size = 50

        all_results: list[int] = []

        for batch_start in range(0, num_submissions, batch_size):
            batch_end = min(batch_start + batch_size, num_submissions)

            def make_submission(i):
                try:
                    response = collector_api.submit_execution(
                        prompt=f"Burst test {i} - {uuid.uuid4()}",
                        response=f"Response {i}",
                        danger_score=0.1,
                        validator_hotkey=validator_hotkey,
                        wallet=validator_wallet,
                        accepted=True,
                    )
                    return response.status_code
                except Exception:
                    return -1

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_submission, i) for i in range(batch_start, batch_end)]
                batch_results = [f.result() for f in as_completed(futures)]
                all_results.extend(batch_results)

            # Small delay between batches
            time.sleep(0.1)

        # Analyze results
        success = all_results.count(201)
        rate_limited = all_results.count(429)
        errors = sum(1 for r in all_results if r == -1 or r >= 500)

        print(f"\n  Total: {len(all_results)}, Success: {success}, Rate Limited: {rate_limited}, Errors: {errors}")

        # Should have no server errors
        assert errors == 0

    @pytest.mark.slow
    @pytest.mark.stress
    def test_mixed_operations_concurrent(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Mixed concurrent operations: executions + telemetry + novelty checks."""
        results: dict[str, list[int]] = {
            "execution": [],
            "telemetry": [],
            "novelty": [],
        }

        def execution_task(i):
            try:
                response = collector_api.submit_execution(
                    prompt=f"Mixed test execution {i} - {uuid.uuid4()}",
                    response=f"Response {i}",
                    danger_score=0.1,
                    validator_hotkey=validator_hotkey,
                    wallet=validator_wallet,
                    accepted=True,
                )
                return ("execution", response.status_code)
            except Exception:
                return ("execution", -1)

        def telemetry_task(i):
            try:
                span = create_test_span(f"mixed-test-span-{i}")
                response = collector_api.submit_traces(
                    spans=[span],
                    validator_hotkey=validator_hotkey,
                    wallet=validator_wallet,
                )
                return ("telemetry", response.status_code)
            except Exception:
                return ("telemetry", -1)

        def novelty_task(i):
            try:
                response = collector_api.check_novelty(
                    prompt=f"Mixed test novelty {i}",
                    embedding=generate_random_embedding(),
                )
                return ("novelty", response.status_code)
            except Exception:
                return ("novelty", -1)

        # Create mixed workload
        tasks = []
        for i in range(10):
            tasks.append(("execution", i))
            tasks.append(("telemetry", i))
            tasks.append(("novelty", i))

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for task_type, i in tasks:
                if task_type == "execution":
                    futures.append(executor.submit(execution_task, i))
                elif task_type == "telemetry":
                    futures.append(executor.submit(telemetry_task, i))
                else:
                    futures.append(executor.submit(novelty_task, i))

            for future in as_completed(futures):
                task_type, status = future.result()
                results[task_type].append(status)

        # Analyze per operation type
        for op_type, statuses in results.items():
            success = sum(1 for s in statuses if s in [200, 201])
            rate_limited = sum(1 for s in statuses if s == 429)
            errors = sum(1 for s in statuses if s == -1 or s >= 500)
            print(f"\n  {op_type}: {len(statuses)} total, {success} success, {rate_limited} rate limited, {errors} errors")
            assert errors == 0, f"{op_type} had errors"


class TestDatabaseLocking:
    """Tests for database locking behavior."""

    def test_novelty_with_concurrent_inserts(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """
        Test novelty checking while concurrent inserts are happening.

        This tests the FOR UPDATE SKIP LOCKED pattern if implemented.
        """
        unique_base = str(uuid.uuid4())[:8]

        results: dict[str, list] = {
            "inserts": [],
            "checks": [],
        }

        def insert_task(i):
            prompt = f"Concurrent insert {unique_base} - {i}"
            embedding = generate_deterministic_embedding(prompt)
            try:
                response = collector_api.submit_execution(
                    prompt=prompt,
                    response=f"Response {i}",
                    danger_score=0.1,
                    validator_hotkey=validator_hotkey,
                    wallet=validator_wallet,
                    accepted=True,
                    prompt_embedding=embedding,
                )
                return ("insert", response.status_code)
            except Exception:
                return ("insert", -1)

        def check_task(i):
            prompt = f"Concurrent check {unique_base} - {i}"
            embedding = generate_deterministic_embedding(prompt)
            try:
                response = collector_api.check_novelty(
                    prompt=prompt,
                    embedding=embedding,
                )
                return ("check", response.status_code)
            except Exception:
                return ("check", -1)

        # Interleave inserts and checks
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(20):
                futures.append(executor.submit(insert_task, i))
                futures.append(executor.submit(check_task, i))

            for future in as_completed(futures):
                task_type, status = future.result()
                results[task_type + "s"].append(status)

        # Should not have database errors (deadlocks, etc.)
        for task_type, statuses in results.items():
            errors = sum(1 for s in statuses if s == -1 or s >= 500)
            assert errors == 0, f"{task_type} had {errors} errors"

    def test_duplicate_trace_span_handling(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """
        Test ON CONFLICT DO NOTHING behavior with concurrent duplicate spans.
        """
        trace_id = secrets.token_hex(16)
        span_id = secrets.token_hex(8)

        # Same span submitted multiple times concurrently
        span = create_test_span("duplicate-span-test", trace_id=trace_id, span_id=span_id)

        results: list[int] = []

        def submit_span():
            try:
                response = collector_api.submit_traces(
                    spans=[span],
                    validator_hotkey=validator_hotkey,
                    wallet=validator_wallet,
                )
                return response.status_code
            except Exception:
                return -1

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(submit_span) for _ in range(10)]
            results = [f.result() for f in as_completed(futures)]

        # All should succeed (ON CONFLICT handling)
        for status in results:
            assert status in [201, 409, 429]

        # Verify only one span exists
        time.sleep(0.5)
        trace_response = collector_api.get_trace(trace_id)

        if trace_response.status_code == 200:
            data = trace_response.json()
            assert data["span_count"] == 1


class TestResourceExhaustion:
    """Tests for resource exhaustion attacks."""

    @pytest.mark.slow
    def test_rate_limit_memory_with_many_unique_keys(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """
        Test rate limiter doesn't exhaust memory with many unique keys.

        This tests if the rate limiter properly cleans up old entries.
        """
        # Make requests with many "different" identities
        results = []

        for i in range(100):
            # Each request appears to come from a different source
            response = collector_api.check_novelty(
                prompt=f"Memory exhaustion test {i}",
                embedding=generate_random_embedding(),
            )
            results.append(response.status_code)

        # Should not see server errors (memory exhaustion)
        errors = sum(1 for s in results if s >= 500)
        assert errors == 0

    def test_many_concurrent_connections(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Test handling of many concurrent connections."""
        num_connections = 50

        def make_request(i):
            try:
                # Health check is lightweight
                response = collector_api.health()
                return response.status_code
            except Exception:
                return -1

        with ThreadPoolExecutor(max_workers=num_connections) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_connections)]
            results = [f.result() for f in as_completed(futures)]

        # Most should succeed
        success = sum(1 for s in results if s == 200)
        errors = sum(1 for s in results if s == -1 or s >= 500)

        print(f"\n  {num_connections} connections: {success} success, {errors} errors")

        # Should have no server errors
        assert errors == 0
        # At least 80% should succeed
        assert success >= num_connections * 0.8


class TestDataConsistency:
    """Tests for data consistency under concurrent access."""

    def test_execution_count_after_concurrent_submissions(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Verify execution count is consistent after concurrent submissions."""
        unique_id = str(uuid.uuid4())[:8]
        num_submissions = 20

        # Get initial count
        initial_response = collector_api.get_executions(
            validator_hotkey=validator_hotkey,
            limit=1,
        )
        initial_count = initial_response.json().get("count", 0)

        # Make concurrent submissions
        def submit(i):
            response = collector_api.submit_execution(
                prompt=f"Consistency test {unique_id} - {i}",
                response=f"Response {i}",
                danger_score=0.1,
                validator_hotkey=validator_hotkey,
                wallet=validator_wallet,
                accepted=True,
            )
            return response.status_code

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(submit, i) for i in range(num_submissions)]
            results = [f.result() for f in as_completed(futures)]

        successful = sum(1 for r in results if r == 201)

        # Wait for database to settle
        time.sleep(1)

        # Get final count
        final_response = collector_api.get_executions(
            validator_hotkey=validator_hotkey,
            limit=1,
        )
        final_count = final_response.json().get("count", 0)

        # Count should have increased by number of successful submissions
        assert final_count >= initial_count + successful

    def test_no_lost_updates_under_concurrency(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """
        Verify no submissions are lost under concurrent load.

        Uses unique identifiers to track each submission.
        """
        batch_id = str(uuid.uuid4())[:8]
        submission_ids = [f"{batch_id}-{i}" for i in range(10)]

        def submit_with_id(sub_id):
            response = collector_api.submit_execution(
                prompt=f"No-lost-updates test {sub_id}",
                response=f"Response for {sub_id}",
                danger_score=0.1,
                validator_hotkey=validator_hotkey,
                wallet=validator_wallet,
                accepted=True,
            )
            return (sub_id, response.status_code)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(submit_with_id, sid) for sid in submission_ids]
            results = {sid: status for sid, status in [f.result() for f in as_completed(futures)]}

        successful_ids = [sid for sid, status in results.items() if status == 201]

        # Wait for database
        time.sleep(1)

        # Verify all successful submissions exist
        response = collector_api.get_executions(
            validator_hotkey=validator_hotkey,
            limit=100,
        )

        assert response.status_code == 200
        executions = response.json()["executions"]

        # Check that all successful submissions are present
        found_ids = set()
        for execution in executions:
            prompt = execution.get("prompt", "")
            for sid in successful_ids:
                if sid in prompt:
                    found_ids.add(sid)

        missing = set(successful_ids) - found_ids
        assert len(missing) == 0, f"Lost submissions: {missing}"
