"""Tests for miner experiment registration module (T063-T065)."""

from unittest.mock import MagicMock, patch

import pytest


class TestRegistrationFunctions:
    """Tests for registration functions."""

    @pytest.fixture
    def mock_wallet(self):
        """Create a mock wallet."""
        wallet = MagicMock()
        wallet.hotkey.ss58_address = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        wallet.hotkey.sign.return_value = b"test_signature"
        return wallet

    def test_register_for_experiment_success(self, mock_wallet):
        """Test successful experiment registration."""
        from aurelius.miner.registration import register_for_experiment

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "registration": {
                    "status": "active",
                    "registered_at": "2024-01-01T00:00:00Z",
                },
                "message": "Successfully registered",
            }
            mock_post.return_value = mock_response

            result = register_for_experiment(
                wallet=mock_wallet,
                experiment_id="jailbreak-v1",
                api_endpoint="http://test-api.example.com",
            )

            assert result.success is True
            assert result.experiment_id == "jailbreak-v1"
            assert result.status == "active"
            assert result.registered_at == "2024-01-01T00:00:00Z"

    def test_register_for_prompt_experiment_auto_registered(self, mock_wallet):
        """Test that 'prompt' experiment returns auto-registered."""
        from aurelius.miner.registration import register_for_experiment

        result = register_for_experiment(
            wallet=mock_wallet,
            experiment_id="prompt",
            api_endpoint="http://test-api.example.com",
        )

        assert result.success is True
        assert result.experiment_id == "prompt"
        assert result.status == "active"
        assert "auto-registered" in result.message.lower()

    def test_register_no_endpoint_fails(self, mock_wallet):
        """Test registration fails when no endpoint configured."""
        from aurelius.miner.registration import register_for_experiment

        with patch("aurelius.miner.registration.Config") as mock_config:
            mock_config.CENTRAL_API_ENDPOINT = None

            result = register_for_experiment(
                wallet=mock_wallet,
                experiment_id="test-exp",
                api_endpoint=None,
            )

            assert result.success is False
            assert "endpoint" in result.error.lower()

    def test_withdraw_from_experiment_success(self, mock_wallet):
        """Test successful experiment withdrawal."""
        from aurelius.miner.registration import withdraw_from_experiment

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "registration": {
                    "status": "withdrawn",
                    "registered_at": "2024-01-01T00:00:00Z",
                    "withdrawn_at": "2024-01-02T00:00:00Z",
                },
                "message": "Successfully withdrawn",
            }
            mock_post.return_value = mock_response

            result = withdraw_from_experiment(
                wallet=mock_wallet,
                experiment_id="jailbreak-v1",
                api_endpoint="http://test-api.example.com",
            )

            assert result.success is True
            assert result.experiment_id == "jailbreak-v1"
            assert result.status == "withdrawn"
            assert result.withdrawn_at == "2024-01-02T00:00:00Z"

    def test_withdraw_from_prompt_experiment_fails(self, mock_wallet):
        """Test that withdrawing from 'prompt' experiment fails."""
        from aurelius.miner.registration import withdraw_from_experiment

        result = withdraw_from_experiment(
            wallet=mock_wallet,
            experiment_id="prompt",
            api_endpoint="http://test-api.example.com",
        )

        assert result.success is False
        assert "cannot withdraw" in result.error.lower()

    def test_list_registrations_success(self, mock_wallet):
        """Test listing experiment registrations."""
        from aurelius.miner.registration import list_registrations

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "registrations": [
                    {
                        "experiment_id": "jailbreak-v1",
                        "status": "active",
                        "registered_at": "2024-01-01T00:00:00Z",
                    },
                    {
                        "experiment_id": "prompt-v2",
                        "status": "withdrawn",
                        "registered_at": "2024-01-01T00:00:00Z",
                        "withdrawn_at": "2024-01-02T00:00:00Z",
                    },
                ]
            }
            mock_get.return_value = mock_response

            result = list_registrations(
                wallet=mock_wallet,
                api_endpoint="http://test-api.example.com",
            )

            assert result.error is None
            assert len(result.registrations) == 2
            assert result.registrations[0]["experiment_id"] == "jailbreak-v1"
            assert result.registrations[1]["status"] == "withdrawn"

    def test_list_registrations_no_endpoint(self, mock_wallet):
        """Test list registrations with no endpoint."""
        from aurelius.miner.registration import list_registrations

        with patch("aurelius.miner.registration.Config") as mock_config:
            mock_config.CENTRAL_API_ENDPOINT = None

            result = list_registrations(
                wallet=mock_wallet,
                api_endpoint=None,
            )

            assert result.error is not None
            assert "endpoint" in result.error.lower()
            assert result.registrations == []


class TestRegistrationResult:
    """Tests for RegistrationResult dataclass."""

    def test_registration_result_defaults(self):
        """Test RegistrationResult has correct defaults."""
        from aurelius.miner.registration import RegistrationResult

        result = RegistrationResult(success=True)

        assert result.success is True
        assert result.experiment_id is None
        assert result.status is None
        assert result.registered_at is None
        assert result.withdrawn_at is None
        assert result.message is None
        assert result.error is None


class TestMinerRegistrations:
    """Tests for MinerRegistrations dataclass."""

    def test_miner_registrations_defaults(self):
        """Test MinerRegistrations has correct defaults."""
        from aurelius.miner.registration import MinerRegistrations

        result = MinerRegistrations(
            miner_hotkey="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            registrations=[],
        )

        assert result.miner_hotkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        assert result.registrations == []
        assert result.error is None


class TestSignatureGeneration:
    """Tests for request signing."""

    @pytest.fixture
    def mock_wallet(self):
        """Create a mock wallet with signing capability."""
        wallet = MagicMock()
        wallet.hotkey.ss58_address = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
        wallet.hotkey.sign.return_value = b"\x01\x02\x03\x04"
        return wallet

    def test_register_sends_signature_header(self, mock_wallet):
        """Test that registration request includes signature."""
        from aurelius.miner.registration import register_for_experiment

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "registration": {"status": "active"},
            }
            mock_post.return_value = mock_response

            register_for_experiment(
                wallet=mock_wallet,
                experiment_id="test-exp",
                api_endpoint="http://test-api.example.com",
            )

            # Verify signature header was sent
            call_args = mock_post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert "X-Signature" in headers
            assert "X-Timestamp" in headers

    def test_withdraw_sends_signature_header(self, mock_wallet):
        """Test that withdrawal request includes signature."""
        from aurelius.miner.registration import withdraw_from_experiment

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "registration": {"status": "withdrawn"},
            }
            mock_post.return_value = mock_response

            withdraw_from_experiment(
                wallet=mock_wallet,
                experiment_id="test-exp",
                api_endpoint="http://test-api.example.com",
            )

            # Verify signature header was sent
            call_args = mock_post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert "X-Signature" in headers
            assert "X-Timestamp" in headers
