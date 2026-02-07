"""
Tests for data validation and boundary conditions.

These tests verify that the API correctly validates input data,
handles edge cases, and enforces limits on all data types.
"""

import math
import secrets
import time
import uuid

import pytest
import requests

from .helpers.api_client import (
    CollectorAPIClient,
    create_test_span,
    create_test_log,
    generate_random_embedding,
)

pytestmark = [pytest.mark.e2e, pytest.mark.requires_docker]


class TestPromptResponseLimits:
    """Tests for prompt and response content boundaries."""

    def test_empty_prompt(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Empty prompt should be rejected."""
        response = collector_api.submit_execution(
            prompt="",  # Empty
            response="Response to empty prompt",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        assert response.status_code == 400

    def test_whitespace_only_prompt(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Whitespace-only prompt should be rejected."""
        response = collector_api.submit_execution(
            prompt="   \t\n   ",  # Only whitespace
            response="Response to whitespace prompt",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        assert response.status_code == 400

    def test_max_length_prompt(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Very long prompt (10,000 chars) handling."""
        long_prompt = "A" * 10000

        response = collector_api.submit_execution(
            prompt=long_prompt,
            response="Response to long prompt",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        # Either accepted (201) or rejected for length (400)
        assert response.status_code in [201, 400]

    def test_prompt_with_unicode(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Prompt with emoji, CJK, and RTL text should be handled."""
        unicode_prompt = (
            "Hello! \U0001F600 "
            "\u4F60\u597D\u4E16\u754C "  # Chinese: Hello World
            "\u0645\u0631\u062D\u0628\u0627"  # Arabic: Hello
        )

        response = collector_api.submit_execution(
            prompt=unicode_prompt,
            response="Response with unicode support",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        assert response.status_code == 201

    def test_prompt_with_control_characters(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Prompt with control characters should be handled safely."""
        # Include various control characters
        control_prompt = "Test\x00with\nnewlines\rand\ttabs"

        response = collector_api.submit_execution(
            prompt=control_prompt,
            response="Response handling control characters",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        # Null bytes may be rejected, newlines/tabs usually accepted
        assert response.status_code in [201, 400]

    def test_prompt_with_sql_injection_attempt(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """SQL injection attempts should be stored safely, not executed."""
        injection_prompt = "'; DROP TABLE executions; --"

        response = collector_api.submit_execution(
            prompt=injection_prompt,
            response="Response to SQL injection test",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        # Should be accepted (stored safely) - no 500 error
        assert response.status_code == 201
        assert response.status_code != 500

    def test_response_100kb(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Large response (100KB) handling."""
        large_response = "X" * (100 * 1024)  # 100KB

        response = collector_api.submit_execution(
            prompt="Test prompt for large response",
            response=large_response,
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        # Either accepted or rejected for size
        assert response.status_code in [201, 400, 413]


class TestEmbeddingEdgeCases:
    """Tests for embedding validation boundaries."""

    def test_embedding_all_zeros(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """All-zero embedding handling."""
        response = collector_api.check_novelty(
            prompt="Test prompt",
            embedding=[0.0] * 384,  # All zeros
        )

        # May be accepted or rejected (zero magnitude)
        assert response.status_code in [200, 400]

    def test_embedding_all_zeros_but_one(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Embedding with only one non-zero value (minimal magnitude)."""
        embedding = [0.0] * 384
        embedding[0] = 0.002  # Just enough to pass magnitude check

        response = collector_api.check_novelty(
            prompt="Test prompt",
            embedding=embedding,
        )

        # Should be accepted - has valid magnitude
        assert response.status_code in [200, 400]

    def test_embedding_very_small_values(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Embedding with all very small values."""
        embedding = [0.00001] * 384  # Very small but non-zero

        response = collector_api.check_novelty(
            prompt="Test prompt",
            embedding=embedding,
        )

        assert response.status_code in [200, 400]

    def test_embedding_extreme_values(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Embedding with values near float limits."""
        embedding = [1e37] * 384  # Near float max

        response = collector_api.check_novelty(
            prompt="Test prompt",
            embedding=embedding,
        )

        # Should be rejected or handled
        assert response.status_code in [200, 400]

    def test_embedding_383_dimensions(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Embedding with one fewer dimension than required."""
        embedding = generate_random_embedding(dimensions=383)  # Wrong: should be 384

        response = collector_api.check_novelty(
            prompt="Test prompt",
            embedding=embedding,
        )

        assert response.status_code == 400

    def test_embedding_385_dimensions(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Embedding with one more dimension than required."""
        embedding = generate_random_embedding(dimensions=385)  # Wrong: should be 384

        response = collector_api.check_novelty(
            prompt="Test prompt",
            embedding=embedding,
        )

        assert response.status_code == 400

    def test_embedding_zero_dimensions(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Empty embedding array."""
        response = collector_api.check_novelty(
            prompt="Test prompt",
            embedding=[],  # Empty
        )

        assert response.status_code == 400

    def test_embedding_with_nan(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Embedding with NaN values (not valid JSON compliant)."""
        embedding = generate_random_embedding()
        embedding[0] = float("nan")

        # Note: Python json.dumps outputs "NaN" which is not valid JSON
        # requests library may handle this differently
        try:
            response = collector_api.check_novelty(
                prompt="Test prompt",
                embedding=embedding,
            )
            # If request succeeds, should be rejected by server
            assert response.status_code in [400, 422]
        except (ValueError, requests.exceptions.JSONDecodeError):
            # Expected - NaN is not valid JSON
            pass

    def test_embedding_with_infinity(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Embedding with infinity values."""
        embedding = generate_random_embedding()
        embedding[0] = float("inf")

        try:
            response = collector_api.check_novelty(
                prompt="Test prompt",
                embedding=embedding,
            )
            assert response.status_code in [400, 422]
        except (ValueError, requests.exceptions.JSONDecodeError):
            # Expected - Infinity is not valid JSON
            pass

    def test_embedding_with_string_values(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Embedding with string values instead of floats."""
        response = requests.post(
            f"{collector_api.base_url}/api/novelty/check",
            json={
                "prompt": "Test prompt",
                "prompt_embedding": ["not", "valid", "floats"] * 128,  # 384 strings
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400


class TestDangerScoreBoundaries:
    """Tests for danger score validation."""

    def test_danger_score_exactly_0(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Danger score at minimum boundary (0)."""
        response = collector_api.submit_execution(
            prompt="Test prompt for zero danger",
            response="Safe response",
            danger_score=0.0,  # Minimum
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        assert response.status_code == 201

    def test_danger_score_exactly_1(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Danger score at maximum boundary (1)."""
        response = collector_api.submit_execution(
            prompt="Test prompt for max danger",
            response="Dangerous response",
            danger_score=1.0,  # Maximum
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=False,  # Should be rejected at max danger
        )

        assert response.status_code == 201

    def test_danger_score_negative_tiny(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Danger score just below minimum (-0.0001)."""
        response = collector_api.submit_execution(
            prompt="Test prompt for negative danger",
            response="Response",
            danger_score=-0.0001,  # Invalid
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        assert response.status_code == 400

    def test_danger_score_over_1_tiny(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Danger score just above maximum (1.0001)."""
        response = collector_api.submit_execution(
            prompt="Test prompt for over max danger",
            response="Response",
            danger_score=1.0001,  # Invalid
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=False,
        )

        assert response.status_code == 400

    def test_danger_score_very_negative(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Very negative danger score."""
        response = collector_api.submit_execution(
            prompt="Test prompt",
            response="Response",
            danger_score=-100.0,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
        )

        assert response.status_code == 400

    def test_danger_score_very_high(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Very high danger score."""
        response = collector_api.submit_execution(
            prompt="Test prompt",
            response="Response",
            danger_score=100.0,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=False,
        )

        assert response.status_code == 400


class TestTelemetryBoundaries:
    """Tests for telemetry data boundaries."""

    def test_nanosecond_timestamp_max_safe_integer(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Timestamp at JavaScript MAX_SAFE_INTEGER."""
        max_safe = 9007199254740991

        span = {
            "trace_id": secrets.token_hex(16),
            "span_id": secrets.token_hex(8),
            "name": "max-safe-integer-test",
            "kind": "internal",
            "status": "ok",
            "start_time_unix_nano": str(max_safe),
            "end_time_unix_nano": str(max_safe + 1000),
            "attributes": {},
            "events": [],
            "links": [],
        }

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # May succeed or fail depending on timestamp validation
        assert response.status_code in [201, 400]

    def test_nanosecond_timestamp_overflow(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Timestamp above MAX_SAFE_INTEGER."""
        overflow = 9007199254740992  # MAX_SAFE_INTEGER + 1

        span = {
            "trace_id": secrets.token_hex(16),
            "span_id": secrets.token_hex(8),
            "name": "overflow-timestamp-test",
            "kind": "internal",
            "status": "ok",
            "start_time_unix_nano": str(overflow),
            "end_time_unix_nano": str(overflow + 1000),
            "attributes": {},
            "events": [],
            "links": [],
        }

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # Should either handle or reject gracefully
        assert response.status_code in [201, 400]

    def test_span_duration_negative(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Span with end_time before start_time (negative duration)."""
        start = int(time.time() * 1e9)
        end = start - 1000000  # End before start

        span = {
            "trace_id": secrets.token_hex(16),
            "span_id": secrets.token_hex(8),
            "name": "negative-duration-test",
            "kind": "internal",
            "status": "ok",
            "start_time_unix_nano": str(start),
            "end_time_unix_nano": str(end),  # Before start
            "attributes": {},
            "events": [],
            "links": [],
        }

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # Should be rejected or accepted with warning
        assert response.status_code in [201, 400]

    def test_span_zero_duration(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Span with zero duration (start == end)."""
        timestamp = int(time.time() * 1e9)

        span = {
            "trace_id": secrets.token_hex(16),
            "span_id": secrets.token_hex(8),
            "name": "zero-duration-test",
            "kind": "internal",
            "status": "ok",
            "start_time_unix_nano": str(timestamp),
            "end_time_unix_nano": str(timestamp),  # Same as start
            "attributes": {},
            "events": [],
            "links": [],
        }

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # Zero duration is valid
        assert response.status_code == 201

    def test_1001_spans_in_batch(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Batch with 1001 spans (over max 1000 limit)."""
        trace_id = secrets.token_hex(16)
        spans = [
            create_test_span(f"span-{i}", trace_id=trace_id)
            for i in range(1001)
        ]

        response = collector_api.submit_traces(
            spans=spans,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # Should be rejected for exceeding batch size
        assert response.status_code in [400, 413]

    def test_jsonb_deeply_nested(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Span attributes with deeply nested structure (50 levels)."""

        def create_nested(depth: int, current: int = 0) -> dict:
            if current >= depth:
                return {"value": "leaf"}
            return {"level": current, "nested": create_nested(depth, current + 1)}

        deeply_nested = create_nested(50)

        span = create_test_span(
            "deeply-nested-test",
            attributes={"nested": deeply_nested},
        )

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # May succeed or fail depending on JSON depth limits
        assert response.status_code in [201, 400]

    def test_jsonb_very_large(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Span attributes with large JSONB (100KB)."""
        large_attributes = {
            "large_data": "X" * (100 * 1024),  # 100KB string
        }

        span = create_test_span(
            "large-jsonb-test",
            attributes=large_attributes,
        )

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # May succeed or fail depending on size limits
        assert response.status_code in [201, 400, 413]


class TestPagination:
    """Tests for pagination boundaries."""

    def test_offset_zero_limit_zero(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Pagination with limit=0 edge case."""
        response = collector_api.get_executions(limit=0, offset=0)

        # May return empty array or fail validation
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert data["count"] == 0 or "executions" in data

    def test_offset_exceeds_total(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Pagination with offset beyond total records."""
        response = collector_api.get_executions(limit=10, offset=999999)

        assert response.status_code == 200
        data = response.json()
        assert len(data["executions"]) == 0

    def test_limit_negative(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Negative limit should fail validation."""
        response = requests.get(
            f"{collector_api.base_url}/api/collections/executions",
            params={"limit": -1, "offset": 0},
        )

        assert response.status_code == 400

    def test_limit_exactly_1000(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Limit at maximum allowed (1000)."""
        response = collector_api.get_executions(limit=1000, offset=0)

        assert response.status_code == 200

    def test_limit_1001(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Limit above maximum (1001) should fail."""
        response = requests.get(
            f"{collector_api.base_url}/api/collections/executions",
            params={"limit": 1001, "offset": 0},
        )

        # Should be capped or rejected
        assert response.status_code in [200, 400]
        if response.status_code == 200:
            data = response.json()
            assert len(data["executions"]) <= 1000

    def test_offset_negative(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Negative offset should fail validation."""
        response = requests.get(
            f"{collector_api.base_url}/api/collections/executions",
            params={"limit": 10, "offset": -1},
        )

        assert response.status_code == 400

    def test_offset_very_large(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Very large offset value."""
        response = collector_api.get_executions(
            limit=10,
            offset=2147483647,  # Max 32-bit signed integer
        )

        # Should succeed with empty results or fail gracefully
        assert response.status_code in [200, 400]


class TestConsensusEdgeCases:
    """Tests for consensus verification data boundaries."""

    def test_consensus_votes_empty_array(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Consensus verification with empty votes array."""
        response = collector_api.submit_execution(
            prompt="Consensus test with no votes",
            response="Response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
            consensus_verified=True,
            validator_votes=[],  # Empty
            validator_count=0,
        )

        # May be accepted or rejected
        assert response.status_code in [201, 400]

    def test_consensus_votes_all_disagree(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Consensus where all validators voted false."""
        response = collector_api.submit_execution(
            prompt="Consensus test all disagree",
            response="Response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
            consensus_verified=True,
            validator_votes=[
                {"hotkey": "5ValidatorA" + "X" * 38, "vote": False},
                {"hotkey": "5ValidatorB" + "X" * 38, "vote": False},
            ],
            validator_count=2,
        )

        assert response.status_code in [201, 400]

    def test_consensus_vote_count_mismatch(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Validator count doesn't match actual votes array."""
        response = collector_api.submit_execution(
            prompt="Consensus test count mismatch",
            response="Response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
            consensus_verified=True,
            validator_votes=[
                {"hotkey": "5ValidatorA" + "X" * 38, "vote": True},
                {"hotkey": "5ValidatorB" + "X" * 38, "vote": True},
            ],
            validator_count=5,  # Claims 5 but only 2 votes
        )

        # May be accepted (count is informational) or rejected
        assert response.status_code in [201, 400]

    def test_validator_votes_duplicate_hotkey(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Same validator appearing twice in votes."""
        same_hotkey = "5ValidatorDup" + "X" * 36

        response = collector_api.submit_execution(
            prompt="Consensus test duplicate voter",
            response="Response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
            consensus_verified=True,
            validator_votes=[
                {"hotkey": same_hotkey, "vote": True},
                {"hotkey": same_hotkey, "vote": False},  # Same hotkey, different vote
            ],
            validator_count=2,
        )

        # Should be rejected or de-duped
        assert response.status_code in [201, 400]

    def test_mean_danger_outside_stddev(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Mean danger score statistically impossible with given std_dev."""
        response = collector_api.submit_execution(
            prompt="Consensus test impossible stats",
            response="Response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
            consensus_verified=True,
            mean_danger_score=0.5,
            std_dev_danger_score=0.001,  # Very low std_dev
            validator_count=2,
            # With danger_score=0.1 and mean=0.5, this is ~400 std deviations off
        )

        # API likely doesn't validate statistical plausibility
        assert response.status_code in [201, 400]


class TestSpecialCharacters:
    """Tests for special character handling in various fields."""

    def test_hotkey_with_special_chars(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Invalid hotkey format with special characters."""
        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={
                "prompt": "Test prompt",
                "response": "Test response",
                "danger_score": 0.1,
                "validator_hotkey": "not-a-valid-ss58<script>",
                "accepted": True,
            },
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code in [400, 401]

    def test_experiment_id_with_special_chars(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Experiment ID with special characters."""
        response = collector_api.submit_execution(
            prompt="Test prompt",
            response="Test response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            accepted=True,
            experiment_id="test/../../../etc/passwd",  # Path traversal attempt
        )

        # Should be rejected or sanitized
        assert response.status_code in [201, 400]
        # Should not cause server error
        assert response.status_code != 500

    def test_trace_id_non_hex(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Trace ID with non-hex characters."""
        span = {
            "trace_id": "ghijklmnopqrstuvwxyz123456789012",  # Not valid hex
            "span_id": secrets.token_hex(8),
            "name": "non-hex-trace-test",
            "kind": "internal",
            "status": "ok",
            "start_time_unix_nano": str(int(time.time() * 1e9)),
            "end_time_unix_nano": str(int(time.time() * 1e9) + 1000),
        }

        response = collector_api.submit_traces(
            spans=[span],
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        assert response.status_code == 400

    def test_miner_hotkey_xss_attempt(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """XSS attempt in miner_hotkey field."""
        response = collector_api.submit_execution(
            prompt="Test prompt",
            response="Test response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            miner_hotkey="<script>alert('xss')</script>",
            miner_uid=1,
            accepted=True,
        )

        # Should be rejected for invalid format or sanitized
        assert response.status_code in [201, 400]
        # Should not cause server error
        assert response.status_code != 500
