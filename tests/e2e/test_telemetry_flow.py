"""
Tests for telemetry submission flow.

These tests verify that validators can submit trace spans and logs
to the telemetry endpoints using SR25519 signature authentication.
"""

import secrets
import time
import uuid

import pytest

from .helpers.api_client import (
    CollectorAPIClient,
    create_test_log,
    create_test_span,
)

pytestmark = [pytest.mark.e2e, pytest.mark.requires_docker]


class TestTelemetryAuthentication:
    """Tests for telemetry authentication."""

    def test_traces_require_signature(
        self,
        collector_api: CollectorAPIClient,
        validator_hotkey: str,
    ) -> None:
        """Submitting traces without signature should fail."""
        import requests

        span = create_test_span("test-span")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={"Content-Type": "application/json"},
        )

        # Should require authentication
        assert response.status_code == 401

    def test_logs_require_signature(
        self,
        collector_api: CollectorAPIClient,
        validator_hotkey: str,
    ) -> None:
        """Submitting logs without signature should fail."""
        import requests

        log = create_test_log("Test log message")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/logs",
            json={
                "logs": [log],
                "validator_hotkey": validator_hotkey,
            },
            headers={"Content-Type": "application/json"},
        )

        # Should require authentication
        assert response.status_code == 401

    def test_invalid_signature_rejected(
        self,
        collector_api: CollectorAPIClient,
        validator_hotkey: str,
    ) -> None:
        """Invalid signature should be rejected."""
        import requests

        span = create_test_span("test-span")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": "invalid_signature_hex",
                "X-Signature-Timestamp": str(int(time.time())),
            },
        )

        assert response.status_code == 401

    def test_mismatched_hotkey_rejected(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature from different hotkey should be rejected."""
        span = create_test_span("test-span")
        fake_hotkey = "5FakeDifferentHotkey" + "C" * 28

        # Sign with real wallet but claim different hotkey in body
        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=fake_hotkey,  # Different from signing wallet
            wallet=validator_wallet,
        )

        # Should fail - body hotkey doesn't match signature hotkey
        assert response.status_code in [401, 403]


class TestTraceSubmission:
    """Tests for trace span submission."""

    def test_submit_single_span(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit a single trace span."""
        trace_id = secrets.token_hex(16)
        span = create_test_span(
            "e2e-test-span",
            trace_id=trace_id,
            status="ok",
            attributes={"test": True, "environment": "e2e"},
        )

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            netuid=290,
            network="test",
        )

        assert response.status_code == 201, f"Failed: {response.text}"

        data = response.json()
        assert data["success"] is True
        assert data["spans_stored"] == 1
        assert data["spans_failed"] == 0
        assert data["validator_hotkey"] == validator_hotkey

    def test_submit_multiple_spans(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit multiple spans in a single batch."""
        trace_id = secrets.token_hex(16)

        # Create parent span
        parent_span_id = secrets.token_hex(8)
        parent = create_test_span(
            "parent-operation",
            trace_id=trace_id,
            span_id=parent_span_id,
            duration_ns=10_000_000,  # 10ms
        )

        # Create child spans
        child1 = create_test_span(
            "child-db-query",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            duration_ns=5_000_000,
        )

        child2 = create_test_span(
            "child-api-call",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            duration_ns=3_000_000,
        )

        response = collector_api.submit_traces(
            spans=[parent, child1, child2],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code == 201, f"Failed: {response.text}"

        data = response.json()
        assert data["spans_stored"] == 3

    def test_submit_error_span(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit span with error status."""
        span = create_test_span(
            "failed-operation",
            status="error",
            attributes={
                "error.type": "TestError",
                "error.message": "Simulated test error",
            },
        )

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code == 201

    def test_retrieve_trace(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit trace and retrieve it."""
        trace_id = secrets.token_hex(16)
        span = create_test_span("retrievable-span", trace_id=trace_id)

        # Submit
        submit_response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )
        assert submit_response.status_code == 201

        # Small delay for DB write
        time.sleep(0.5)

        # Retrieve
        get_response = collector_api.get_trace(trace_id)

        assert get_response.status_code == 200

        data = get_response.json()
        assert data["trace_id"] == trace_id
        assert data["span_count"] >= 1
        assert len(data["spans"]) >= 1


class TestLogSubmission:
    """Tests for log record submission."""

    def test_submit_single_log(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit a single log record."""
        log = create_test_log(
            "E2E test log message",
            severity="INFO",
            attributes={"test": True, "run_id": str(uuid.uuid4())},
        )

        response = collector_api.submit_logs(
            logs=[log],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            netuid=290,
            network="test",
        )

        assert response.status_code == 201, f"Failed: {response.text}"

        data = response.json()
        assert data["success"] is True
        assert data["logs_stored"] == 1

    def test_submit_multiple_logs(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit multiple logs in a batch."""
        logs = [
            create_test_log("Debug message", severity="DEBUG"),
            create_test_log("Info message", severity="INFO"),
            create_test_log("Warning message", severity="WARN"),
            create_test_log("Error message", severity="ERROR"),
        ]

        response = collector_api.submit_logs(
            logs=logs,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code == 201

        data = response.json()
        assert data["logs_stored"] == 4

    def test_submit_log_with_trace_correlation(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit log correlated with a trace."""
        trace_id = secrets.token_hex(16)
        span_id = secrets.token_hex(8)

        # Submit span first
        span = create_test_span("logged-operation", trace_id=trace_id, span_id=span_id)
        collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # Submit correlated log
        log = create_test_log(
            "Operation completed successfully",
            trace_id=trace_id,
            span_id=span_id,
        )

        response = collector_api.submit_logs(
            logs=[log],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code == 201

    def test_query_logs(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit and query logs."""
        unique_id = str(uuid.uuid4())
        log = create_test_log(
            f"Queryable log {unique_id}",
            severity="INFO",
            attributes={"unique_id": unique_id},
        )

        # Submit
        collector_api.submit_logs(
            logs=[log],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        time.sleep(0.5)

        # Query
        response = collector_api.query_logs(
            validator_hotkey=validator_hotkey,
            limit=10,
        )

        assert response.status_code == 200

        data = response.json()
        assert "logs" in data
        assert "count" in data


class TestTelemetryValidation:
    """Tests for telemetry validation."""

    def test_invalid_trace_id_format(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Invalid trace_id format should be rejected."""
        span = {
            "trace_id": "invalid",  # Should be 32 hex chars
            "span_id": secrets.token_hex(8),
            "name": "test-span",
            "kind": "internal",
            "status": "ok",
            "start_time_unix_nano": str(int(time.time() * 1e9)),
            "end_time_unix_nano": str(int(time.time() * 1e9) + 1000000),
        }

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code == 400

    def test_invalid_span_id_format(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Invalid span_id format should be rejected."""
        span = {
            "trace_id": secrets.token_hex(16),
            "span_id": "short",  # Should be 16 hex chars
            "name": "test-span",
            "kind": "internal",
            "status": "ok",
            "start_time_unix_nano": str(int(time.time() * 1e9)),
            "end_time_unix_nano": str(int(time.time() * 1e9) + 1000000),
        }

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code == 400

    def test_empty_span_batch_rejected(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Empty span batch should be rejected."""
        response = collector_api.submit_traces(
            spans=[],  # Empty
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code == 400

    def test_get_nonexistent_trace(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Getting nonexistent trace should return 404."""
        fake_trace_id = secrets.token_hex(16)

        response = collector_api.get_trace(fake_trace_id)

        assert response.status_code == 404


class TestTelemetryStats:
    """Tests for telemetry statistics."""

    def test_stats_requires_auth(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Telemetry stats should require authentication."""
        client_no_auth = CollectorAPIClient(
            base_url=collector_api.base_url,
            api_key=None,
        )

        response = client_no_auth.get_telemetry_stats()

        assert response.status_code in [401, 403]


class TestBulkTelemetry:
    """Tests for bulk telemetry operations."""

    @pytest.mark.slow
    def test_submit_large_span_batch(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit a large batch of spans."""
        trace_id = secrets.token_hex(16)

        # Create 100 spans
        spans = [
            create_test_span(
                f"bulk-span-{i}",
                trace_id=trace_id,
                attributes={"index": i},
            )
            for i in range(100)
        ]

        response = collector_api.submit_traces(
            spans=spans,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code in [201, 207], f"Failed: {response.text}"

        data = response.json()
        assert data["spans_received"] == 100
        assert data["spans_stored"] >= 90  # Allow some dedup

    @pytest.mark.slow
    def test_submit_large_log_batch(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submit a large batch of logs."""
        logs = [
            create_test_log(
                f"Bulk log message {i}",
                severity="INFO",
                attributes={"index": i},
            )
            for i in range(100)
        ]

        response = collector_api.submit_logs(
            logs=logs,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code in [201, 207], f"Failed: {response.text}"

        data = response.json()
        assert data["logs_received"] == 100
        assert data["logs_stored"] >= 90
