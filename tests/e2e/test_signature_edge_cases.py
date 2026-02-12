"""
Tests for signature and authentication edge cases.

These tests verify the security boundaries of SR25519 signature authentication
including timestamp validation, replay attacks, and header manipulation.
"""

import secrets
import time

import pytest
import requests

from .helpers.api_client import (
    CollectorAPIClient,
    create_test_span,
    generate_random_embedding,
)

pytestmark = [pytest.mark.e2e, pytest.mark.requires_docker]


class TestTimestampBoundaries:
    """Tests for signature timestamp validation boundaries."""

    def test_signature_timestamp_exactly_at_boundary(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature with timestamp exactly at 30s boundary."""
        # Create a timestamp exactly 30 seconds ago
        timestamp = str(int(time.time()) - 30)
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("boundary-timestamp-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        # Should either succeed (exactly at boundary) or fail (past boundary)
        # Both 201 and 401 are valid behaviors depending on implementation
        assert response.status_code in [201, 401]

    def test_signature_timestamp_1ms_expired(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature with timestamp 31 seconds ago (just past boundary)."""
        # Create a timestamp 31 seconds ago
        timestamp = str(int(time.time()) - 31)
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("expired-timestamp-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        # Should be rejected - timestamp too old
        assert response.status_code == 401

    def test_signature_timestamp_future(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature with timestamp in the future should be rejected."""
        # Create a timestamp 60 seconds in the future
        timestamp = str(int(time.time()) + 60)
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("future-timestamp-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        # Should be rejected - timestamp in future
        assert response.status_code == 401

    def test_signature_timestamp_very_old(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature with timestamp 1 hour ago should be rejected."""
        timestamp = str(int(time.time()) - 3600)  # 1 hour ago
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("very-old-timestamp-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        assert response.status_code == 401

    def test_signature_timestamp_epoch_zero(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature with timestamp=0 (Unix epoch) should be rejected."""
        timestamp = "0"
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("epoch-zero-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        assert response.status_code == 401

    def test_signature_timestamp_negative(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature with negative timestamp should be rejected."""
        timestamp = "-1000"
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("negative-timestamp-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        assert response.status_code in [400, 401]

    def test_signature_timestamp_non_numeric(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature with non-numeric timestamp should be rejected."""
        timestamp = "not-a-number"
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("non-numeric-timestamp-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        assert response.status_code in [400, 401]


class TestSignatureAttacks:
    """Tests for signature manipulation and replay attacks."""

    def test_signature_replay_attack(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Replaying a valid signature with different body should fail."""
        # First, create a valid signature
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        # Submit first request (should succeed)
        span1 = create_test_span("original-request")
        response1 = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span1],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )
        assert response1.status_code == 201

        # Replay same signature with different body
        span2 = create_test_span("replay-attack-different-body")
        response2 = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span2],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        # If there's replay protection (nonce tracking), should fail
        # Otherwise might succeed if signature is still valid (within time window)
        # Both behaviors are documented - just verify no server error
        assert response2.status_code in [201, 401, 403, 429]

    def test_signature_wrong_hotkey(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature from validator A claiming to be validator B should fail."""
        # Sign with real wallet
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        # But claim different hotkey in headers
        fake_hotkey = "5FakeHotkeyThatDoesNotMatch" + "A" * 20

        span = create_test_span("wrong-hotkey-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": fake_hotkey,  # Wrong hotkey in body
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": fake_hotkey,  # Wrong hotkey in header
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        # Should fail - signature doesn't match claimed hotkey
        assert response.status_code in [401, 403]

    def test_signature_malformed_hex(
        self,
        collector_api: CollectorAPIClient,
        validator_hotkey: str,
    ) -> None:
        """Invalid hex string signature should be rejected."""
        span = create_test_span("malformed-signature-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": "not-valid-hex-GHIJKL",
                "X-Signature-Timestamp": str(int(time.time())),
            },
        )

        assert response.status_code in [400, 401]

    def test_signature_empty_string(
        self,
        collector_api: CollectorAPIClient,
        validator_hotkey: str,
    ) -> None:
        """Empty signature string should be rejected."""
        span = create_test_span("empty-signature-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": "",
                "X-Signature-Timestamp": str(int(time.time())),
            },
        )

        assert response.status_code in [400, 401]

    def test_signature_truncated(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Truncated signature should be rejected."""
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        full_signature = validator_wallet.hotkey.sign(message.encode()).hex()

        # Truncate signature
        truncated_signature = full_signature[:32]

        span = create_test_span("truncated-signature-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": truncated_signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        assert response.status_code in [400, 401]

    def test_signature_with_extra_bytes(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature with appended bytes should be rejected."""
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        full_signature = validator_wallet.hotkey.sign(message.encode()).hex()

        # Append extra bytes
        modified_signature = full_signature + "deadbeef"

        span = create_test_span("extra-bytes-signature-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": modified_signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        assert response.status_code in [400, 401]

    def test_signature_bit_flip(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Signature with single bit flip should be rejected."""
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature_bytes = validator_wallet.hotkey.sign(message.encode())

        # Flip a bit in the signature
        modified_bytes = bytearray(signature_bytes)
        modified_bytes[0] ^= 0x01  # Flip one bit
        modified_signature = bytes(modified_bytes).hex()

        span = create_test_span("bit-flip-signature-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": modified_signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        assert response.status_code in [400, 401]


class TestHeaderEdgeCases:
    """Tests for authentication header edge cases."""

    def test_missing_x_validator_hotkey(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Missing X-Validator-Hotkey header should fail."""
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("missing-hotkey-header-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                # X-Validator-Hotkey missing
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        assert response.status_code == 401

    def test_missing_x_signature_timestamp(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Missing X-Signature-Timestamp header should fail."""
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("missing-timestamp-header-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                # X-Signature-Timestamp missing
            },
        )

        assert response.status_code == 401

    def test_missing_x_validator_signature(
        self,
        collector_api: CollectorAPIClient,
        validator_hotkey: str,
    ) -> None:
        """Missing X-Validator-Signature header should fail."""
        span = create_test_span("missing-signature-header-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,
                "X-Signature-Timestamp": str(int(time.time())),
                # X-Validator-Signature missing
            },
        )

        assert response.status_code == 401

    def test_extra_whitespace_in_headers(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Headers with extra whitespace may be handled differently."""
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        span = create_test_span("whitespace-headers-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": validator_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": f"  {validator_hotkey}  ",  # Extra whitespace
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        # Whitespace might cause mismatch - expect failure
        # Some implementations might trim whitespace
        assert response.status_code in [201, 401]

    def test_case_sensitivity_in_hotkey(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """SS58 addresses are case-sensitive - wrong case should fail."""
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        # Change case of hotkey (SS58 is case-sensitive)
        wrong_case_hotkey = validator_hotkey.swapcase()

        span = create_test_span("case-sensitivity-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": wrong_case_hotkey,
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": wrong_case_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        # Should fail - case matters in SS58
        assert response.status_code in [400, 401]

    def test_header_body_hotkey_mismatch(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Hotkey in header should match hotkey in body."""
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        different_hotkey = "5Different" + "X" * 38

        span = create_test_span("header-body-mismatch-test")

        response = requests.post(
            f"{collector_api.base_url}/api/telemetry/traces",
            json={
                "spans": [span],
                "validator_hotkey": different_hotkey,  # Different in body
            },
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": validator_hotkey,  # Original in header
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": timestamp,
            },
        )

        # Should fail - hotkey mismatch
        assert response.status_code in [400, 401, 403]


class TestAPIKeyEdgeCases:
    """Tests for API key authentication edge cases."""

    def test_expired_api_key_format(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Malformed API key should be rejected."""
        # Create client with malformed API key
        client = CollectorAPIClient(
            base_url=collector_api.base_url,
            api_key="malformed-not-a-real-key",
        )

        response = client.get_telemetry_stats()

        # Should be rejected
        assert response.status_code in [401, 403]

    def test_empty_api_key(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """Empty API key should be rejected."""
        client = CollectorAPIClient(
            base_url=collector_api.base_url,
            api_key="",
        )

        response = client.get_telemetry_stats()

        # Should be rejected
        assert response.status_code in [401, 403]

    def test_api_key_with_bearer_prefix_doubled(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """API key with 'Bearer ' prefix when we already add it should fail."""
        response = requests.get(
            f"{collector_api.base_url}/api/telemetry/stats",
            headers={
                "Authorization": "Bearer Bearer some-api-key",  # Double Bearer
            },
        )

        assert response.status_code in [401, 403]

    def test_api_key_wrong_scheme(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """API key with wrong auth scheme should fail."""
        response = requests.get(
            f"{collector_api.base_url}/api/telemetry/stats",
            headers={
                "Authorization": "Basic c29tZS1hcGkta2V5",  # Basic instead of Bearer
            },
        )

        assert response.status_code in [401, 403]

    def test_api_key_injection_attempt(
        self,
        collector_api: CollectorAPIClient,
    ) -> None:
        """API key with injection characters should be handled safely."""
        response = requests.get(
            f"{collector_api.base_url}/api/telemetry/stats",
            headers={
                "Authorization": "Bearer '; DROP TABLE api_keys; --",
            },
        )

        # Should be rejected, not cause SQL error
        assert response.status_code in [401, 403]
        # Verify no server error
        assert response.status_code != 500


class TestSubmissionSignatureVariants:
    """Tests for submission endpoint signature format variations."""

    def test_submission_with_millisecond_timestamp(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submission uses millisecond timestamps (not seconds)."""
        # Submission format uses milliseconds
        timestamp = str(int(time.time() * 1000))
        message = f"aurelius-submission:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={
                "prompt": "Test prompt for millisecond timestamp",
                "response": "Test response",
                "danger_score": 0.1,
                "validator_hotkey": validator_hotkey,
                "accepted": True,
            },
            headers={
                "Content-Type": "application/json",
                "X-Signature": signature,
                "X-Timestamp": timestamp,
            },
        )

        assert response.status_code == 201

    def test_submission_with_seconds_timestamp_fails(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Submission with seconds timestamp (wrong format) should fail."""
        # Wrong format - seconds instead of milliseconds
        timestamp = str(int(time.time()))  # Seconds, not milliseconds
        message = f"aurelius-submission:{timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(message.encode()).hex()

        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={
                "prompt": "Test prompt for wrong timestamp format",
                "response": "Test response",
                "danger_score": 0.1,
                "validator_hotkey": validator_hotkey,
                "accepted": True,
            },
            headers={
                "Content-Type": "application/json",
                "X-Signature": signature,
                "X-Timestamp": timestamp,
            },
        )

        # Should fail because timestamp appears to be in the far past
        # (seconds value interpreted as milliseconds = year ~1970)
        assert response.status_code == 401

    def test_mixed_auth_headers_rejected(
        self,
        collector_api: CollectorAPIClient,
        validator_wallet,
        validator_hotkey: str,
    ) -> None:
        """Mixing telemetry and submission auth headers should fail."""
        # Use telemetry-style headers on submission endpoint
        telemetry_timestamp = str(int(time.time()))  # Seconds
        telemetry_message = f"aurelius-telemetry:{telemetry_timestamp}:{validator_hotkey}"
        signature = validator_wallet.hotkey.sign(telemetry_message.encode()).hex()

        response = requests.post(
            f"{collector_api.base_url}/api/collections",
            json={
                "prompt": "Test prompt",
                "response": "Test response",
                "danger_score": 0.1,
                "validator_hotkey": validator_hotkey,
                "accepted": True,
            },
            headers={
                "Content-Type": "application/json",
                # Telemetry-style headers (wrong for submissions)
                "X-Validator-Hotkey": validator_hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": telemetry_timestamp,
            },
        )

        # Should fail - wrong auth header format for this endpoint
        assert response.status_code == 401
