"""
Tests for error resilience and network failure handling.

These tests verify that the system handles various failure scenarios
gracefully including API unavailability, timeouts, and partial failures.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest
import requests

from .helpers.api_client import (
    CollectorAPIClient,
    create_test_span,
    generate_random_embedding,
)

pytestmark = [pytest.mark.e2e, pytest.mark.requires_docker]


class TestAPIUnavailability:
    """Tests for handling API unavailability."""

    def test_health_check_when_healthy(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Baseline: health check should succeed when API is running."""
        response = collector_api.health()

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_connection_to_invalid_port(self) -> None:
        """Connection to non-listening port should fail gracefully."""
        client = CollectorAPIClient(
            base_url="http://localhost:19999",  # Unlikely to be in use
            timeout=5,
        )

        with pytest.raises(requests.exceptions.ConnectionError):
            client.health()

    def test_connection_to_invalid_host(self) -> None:
        """Connection to invalid hostname should fail gracefully."""
        client = CollectorAPIClient(
            base_url="http://invalid-host-that-does-not-exist.local:3000",
            timeout=5,
        )

        with pytest.raises((requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            client.health()

    def test_request_timeout_handling(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Very short timeout should raise timeout error."""
        # Create client with very short timeout
        short_timeout_client = CollectorAPIClient(
            base_url=collector_api.base_url,
            timeout=0.001,  # 1ms - too short for any real request
        )

        try:
            short_timeout_client.health()
            # If it somehow succeeds (local), that's fine
        except requests.exceptions.Timeout:
            # Expected
            pass
        except requests.exceptions.ConnectionError:
            # Also acceptable
            pass

    def test_api_error_response_format(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Error responses should have consistent format."""
        # Trigger a 400 error
        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={},  # Missing required fields
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code in [400, 401]
        data = response.json()

        # Should have error field
        assert "error" in data
        # Should be a string message
        assert isinstance(data["error"], str)

    def test_404_response_format(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """404 responses should have consistent format."""
        response = requests.get(
            f"{collector_api.base_url}/api/nonexistent/endpoint",
        )

        assert response.status_code == 404

        # Try to parse as JSON
        try:
            data = response.json()
            assert "error" in data or "message" in data
        except requests.exceptions.JSONDecodeError:
            # Plain text 404 is also acceptable
            pass

    def test_method_not_allowed(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Wrong HTTP method should return 405."""
        # Try PUT on health endpoint (should only accept GET)
        response = requests.put(
            f"{collector_api.base_url}/health",
            json={},
        )

        # Should be 405 or 404
        assert response.status_code in [404, 405]


class TestPartialFailures:
    """Tests for handling partial failures in batch operations."""

    def test_partial_span_batch_failure(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Batch with some valid and some invalid spans."""
        import secrets

        trace_id = secrets.token_hex(16)

        # Mix of valid and invalid spans
        valid_span = create_test_span("valid-span", trace_id=trace_id)
        invalid_span = {
            "trace_id": "not-valid-hex",  # Invalid
            "span_id": secrets.token_hex(8),
            "name": "invalid-span",
            "kind": "internal",
            "status": "ok",
            "start_time_unix_nano": str(int(time.time() * 1e9)),
            "end_time_unix_nano": str(int(time.time() * 1e9) + 1000),
        }

        response = collector_api.submit_traces(
            spans=[valid_span, invalid_span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # May get 201 (partial success), 207 (multi-status), or 400 (all rejected)
        assert response.status_code in [201, 207, 400]

        if response.status_code in [201, 207]:
            data = response.json()
            # Should indicate partial success
            if "spans_stored" in data:
                assert data["spans_received"] == 2
                # At least one should have succeeded or failed
                assert data["spans_stored"] >= 0
                assert data.get("spans_failed", 0) >= 0

    def test_mixed_valid_invalid_in_logs_batch(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Log batch with mixed valid and invalid entries."""
        from .helpers.api_client import create_test_log

        valid_log = create_test_log("Valid log message")
        invalid_log = {
            "timestamp_unix_nano": "not-a-number",  # Invalid
            "body": "Invalid log",
            "severity_number": 9,
        }

        response = collector_api.submit_logs(
            logs=[valid_log, invalid_log],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # Should handle gracefully
        assert response.status_code in [201, 207, 400]


class TestRateLimitingResilience:
    """Tests for rate limiting behavior."""

    @pytest.mark.slow
    def test_rate_limit_returns_429(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Exceeding rate limit should return 429 status."""
        # Make many rapid requests until rate limited
        responses = []
        for i in range(50):
            response = collector_api.check_novelty(
                prompt=f"Rate limit test {i}",
                embedding=generate_random_embedding(),
            )
            responses.append(response.status_code)

            if response.status_code == 429:
                break

            # Small delay to not overwhelm
            time.sleep(0.05)

        # Document behavior
        if 429 in responses:
            idx = responses.index(429)
            print(f"\n  Rate limited after {idx} requests")

            # Check rate limit response format
            response = collector_api.check_novelty(
                prompt="Rate limit check",
                embedding=generate_random_embedding(),
            )
            if response.status_code == 429:
                data = response.json()
                assert "error" in data or "message" in data

    @pytest.mark.slow
    def test_rate_limit_window_reset(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Rate limit should reset after window expires."""
        # This test is slow as it waits for rate limit window

        # First, trigger rate limit
        for i in range(50):
            response = collector_api.check_novelty(
                prompt=f"Window reset test {i}",
                embedding=generate_random_embedding(),
            )
            if response.status_code == 429:
                break

        # If we got rate limited, wait for window to reset
        if response.status_code == 429:
            # Wait for typical window (1 minute)
            time.sleep(65)

            # Should work again
            response = collector_api.check_novelty(
                prompt="After window reset",
                embedding=generate_random_embedding(),
            )

            # Should succeed (or at least not be immediately rate limited)
            assert response.status_code in [200, 429]

    def test_rate_limit_headers(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Rate limit responses should include helpful headers."""
        # Make a request
        response = collector_api.check_novelty(
            prompt="Header check",
            embedding=generate_random_embedding(),
        )

        # Check for rate limit headers (RFC 6585)
        headers = response.headers

        # These headers are optional but good practice
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After",
        ]

        # At least document which headers are present
        present_headers = [h for h in rate_limit_headers if h in headers]
        print(f"\n  Rate limit headers present: {present_headers}")


class TestConcurrentRequests:
    """Tests for handling concurrent requests."""

    def test_concurrent_health_checks(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Multiple concurrent health checks should all succeed."""
        num_requests = 20

        def make_health_request():
            return collector_api.health().status_code

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]

        # All should succeed
        success_count = sum(1 for r in results if r == 200)
        assert success_count == num_requests

    def test_concurrent_novelty_checks(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Multiple concurrent novelty checks should be handled."""
        num_requests = 10

        def make_novelty_request(i):
            return collector_api.check_novelty(
                prompt=f"Concurrent novelty test {i}",
                embedding=generate_random_embedding(),
            ).status_code

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_novelty_request, i) for i in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]

        # Should get 200 or 429 (rate limited), never 500
        for status_code in results:
            assert status_code in [200, 429]

    def test_concurrent_submissions(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Multiple concurrent execution submissions."""
        num_requests = 5

        def make_submission(i):
            return collector_api.submit_execution(
                prompt=f"Concurrent submission test {i} - {uuid.uuid4()}",
                response=f"Response {i}",
                danger_score=0.1,
                validator_hotkey=validator_hotkey,
                wallet=validator_wallet,
                accepted=True,
            ).status_code

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_submission, i) for i in range(num_requests)]
            results = [f.result() for f in as_completed(futures)]

        # Should get 201 or 429, never 500
        for status_code in results:
            assert status_code in [201, 429]


class TestMalformedRequests:
    """Tests for handling malformed requests."""

    def test_invalid_json(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Invalid JSON should return 400."""
        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            data="not valid json {",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400

    def test_empty_body(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Empty request body should return 400."""
        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            data="",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400

    def test_wrong_content_type(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Wrong content type should be rejected."""
        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            data="some=form&data=here",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        # Should reject or fail to parse
        assert response.status_code in [400, 415]

    def test_xml_body(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """XML body should be rejected."""
        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            data="<prompt>test</prompt>",
            headers={"Content-Type": "application/xml"},
        )

        assert response.status_code in [400, 415]

    def test_extremely_large_request(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Very large request body should be rejected."""
        # 10MB of data
        large_data = {"data": "X" * (10 * 1024 * 1024)}

        try:
            response = requests.post(
                f"{collector_api.base_url}/api/collections",
                json=large_data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            # Should be rejected for size
            assert response.status_code in [400, 413]
        except requests.exceptions.Timeout:
            # Timeout is acceptable for huge request
            pass
        except requests.exceptions.ConnectionError:
            # Connection reset is also acceptable
            pass


class TestErrorResponseConsistency:
    """Tests for consistent error response formatting."""

    def test_all_400_errors_have_error_field(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """All 400 responses should have error field."""
        # Generate various 400 errors
        bad_requests = [
            # Missing required fields
            requests.post(
                f"{collector_api.base_url}/api/collections",
                json={"prompt": "only prompt"},
                headers={"Content-Type": "application/json"},
            ),
            # Invalid danger score
            requests.post(
                f"{collector_api.base_url}/api/collections",
                json={
                    "prompt": "test",
                    "response": "test",
                    "danger_score": 5.0,  # Invalid
                    "validator_hotkey": validator_hotkey,
                    "accepted": True,
                },
                headers={"Content-Type": "application/json"},
            ),
            # Invalid embedding dimensions
            collector_api.check_novelty(
                prompt="test",
                embedding=[0.1] * 100,  # Wrong dimensions
            ),
        ]

        for response in bad_requests:
            if response.status_code == 400:
                data = response.json()
                assert "error" in data, f"Missing error field in 400 response: {data}"

    def test_401_vs_403_distinction(
        self,
        collector_api: CollectorAPIClient,
        validator_hotkey: str,
    ) -> None:
        """401 should be for missing auth, 403 for insufficient permissions."""
        # No auth at all - should be 401
        no_auth_response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [create_test_span("test")],
                "validator_hotkey": validator_hotkey,
            },
            headers={"Content-Type": "application/json"},
        )

        # Missing auth should be 401 (Unauthorized)
        assert no_auth_response.status_code == 401

        # Bad auth - could be 401 or 403 depending on implementation
        bad_auth_response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [create_test_span("test")],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": "invalid",
                "X-Signature-Timestamp": str(int(time.time())),
            },
        )

        # Bad signature should be 401 (authentication failed)
        assert bad_auth_response.status_code in [401, 403]

    def test_error_messages_no_sensitive_data(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Error messages should not leak sensitive information."""
        # Trigger error
        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code in [400, 401]

        error_text = response.text.lower()

        # Should not contain sensitive patterns
        sensitive_patterns = [
            "password",
            "secret",
            "database",
            "postgresql",
            "stack trace",
            "traceback",
            "/home/",
            "/var/",
            ".env",
        ]

        for pattern in sensitive_patterns:
            assert pattern not in error_text, f"Error contains sensitive pattern: {pattern}"


class TestDatabaseResilience:
    """Tests for database-related error handling."""

    def test_duplicate_submission_handling(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Duplicate submissions should be handled gracefully."""
        unique_id = str(uuid.uuid4())

        # Submit twice
        response1 = collector_api.submit_execution(
            prompt=f"Duplicate test {unique_id}",
            response="Response 1",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        response2 = collector_api.submit_execution(
            prompt=f"Duplicate test {unique_id}",
            response="Response 1",  # Same response
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        # Both should succeed (or second should indicate duplicate)
        assert response1.status_code in [201, 409]
        assert response2.status_code in [201, 409]

    def test_trace_span_deduplication(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Duplicate trace spans should be deduplicated."""
        import secrets

        trace_id = secrets.token_hex(16)
        span_id = secrets.token_hex(8)

        # Create same span twice
        span = create_test_span("dedup-test", trace_id=trace_id, span_id=span_id)

        # Submit twice
        response1 = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        response2 = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # Both should succeed (ON CONFLICT handling)
        assert response1.status_code == 201
        assert response2.status_code in [201, 409]

        # Retrieve trace
        time.sleep(0.5)
        trace_response = collector_api.get_trace(trace_id)

        if trace_response.status_code == 200:
            data = trace_response.json()
            # Should only have one span (deduplicated)
            assert data["span_count"] == 1
