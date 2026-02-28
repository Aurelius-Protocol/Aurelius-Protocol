"""Tests for chat client fallback logic."""

from unittest.mock import MagicMock, patch

import pytest
from openai import APIStatusError

from aurelius.shared.chat_client import ModelUnavailableError, call_chat_api_with_fallback


class MockResponse:
    """Mock response from OpenAI API."""

    def __init__(self, content: str = "Test response"):
        self.choices = [MagicMock()]
        self.choices[0].message.content = content


class TestCallChatApiWithFallback:
    """Tests for call_chat_api_with_fallback function."""

    def test_primary_model_succeeds(self):
        """When primary model works, should return it immediately."""
        mock_client = MagicMock()
        mock_response = MockResponse("Hello world")
        mock_client.chat.completions.create.return_value = mock_response

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        response, model_used = call_chat_api_with_fallback(
            mock_client, api_params, fallback_models=["fallback-1", "fallback-2"]
        )

        assert response == mock_response
        assert model_used == "primary-model"
        assert mock_client.chat.completions.create.call_count == 1
        # Verify the model was set correctly
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "primary-model"

    def test_fallback_on_503_error(self):
        """When primary returns 503, should try fallback models."""
        mock_client = MagicMock()
        mock_response = MockResponse("Fallback response")

        # Create proper mock for APIStatusError
        mock_request = MagicMock()
        mock_request.url = "https://api.example.com/v1/chat"

        # First call fails with 503, second succeeds
        mock_client.chat.completions.create.side_effect = [
            APIStatusError(
                "Service unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Service unavailable"}},
            ),
            mock_response,
        ]

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            response, model_used = call_chat_api_with_fallback(
                mock_client, api_params, fallback_models=["fallback-1"]
            )

        assert response == mock_response
        assert model_used == "fallback-1"
        assert mock_client.chat.completions.create.call_count == 2

    def test_fallback_on_502_error(self):
        """When primary returns 502, should try fallback models."""
        mock_client = MagicMock()
        mock_response = MockResponse("Fallback response")

        mock_client.chat.completions.create.side_effect = [
            APIStatusError(
                "Bad gateway",
                response=MagicMock(status_code=502),
                body={"error": {"message": "Bad gateway"}},
            ),
            mock_response,
        ]

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            response, model_used = call_chat_api_with_fallback(
                mock_client, api_params, fallback_models=["fallback-1"]
            )

        assert response == mock_response
        assert model_used == "fallback-1"

    def test_fallback_on_504_error(self):
        """When primary returns 504, should try fallback models."""
        mock_client = MagicMock()
        mock_response = MockResponse("Fallback response")

        mock_client.chat.completions.create.side_effect = [
            APIStatusError(
                "Gateway timeout",
                response=MagicMock(status_code=504),
                body={"error": {"message": "Gateway timeout"}},
            ),
            mock_response,
        ]

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            response, model_used = call_chat_api_with_fallback(
                mock_client, api_params, fallback_models=["fallback-1"]
            )

        assert response == mock_response
        assert model_used == "fallback-1"

    def test_multiple_fallbacks_tried_in_order(self):
        """When multiple models fail, should try each fallback in order."""
        mock_client = MagicMock()
        mock_response = MockResponse("Third model response")

        mock_client.chat.completions.create.side_effect = [
            APIStatusError(
                "Primary unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Unavailable"}},
            ),
            APIStatusError(
                "Fallback 1 unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Unavailable"}},
            ),
            mock_response,
        ]

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            response, model_used = call_chat_api_with_fallback(
                mock_client, api_params, fallback_models=["fallback-1", "fallback-2"]
            )

        assert response == mock_response
        assert model_used == "fallback-2"
        assert mock_client.chat.completions.create.call_count == 3

    def test_all_models_fail_raises_error(self):
        """When all models fail with 503, should raise ModelUnavailableError."""
        mock_client = MagicMock()

        mock_client.chat.completions.create.side_effect = [
            APIStatusError(
                "Primary unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Unavailable"}},
            ),
            APIStatusError(
                "Fallback 1 unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Unavailable"}},
            ),
            APIStatusError(
                "Fallback 2 unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Unavailable"}},
            ),
        ]

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            with pytest.raises(ModelUnavailableError) as exc_info:
                call_chat_api_with_fallback(
                    mock_client, api_params, fallback_models=["fallback-1", "fallback-2"]
                )

        assert exc_info.value.primary_model == "primary-model"
        assert exc_info.value.fallback_models == ["fallback-1", "fallback-2"]
        assert len(exc_info.value.errors) == 3

    def test_non_503_error_not_retried(self):
        """Errors other than 502/503/504 should not trigger fallback."""
        mock_client = MagicMock()

        # 400 Bad Request - should NOT trigger fallback
        mock_client.chat.completions.create.side_effect = APIStatusError(
            "Bad request",
            response=MagicMock(status_code=400),
            body={"error": {"message": "Invalid request"}},
        )

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            with pytest.raises(APIStatusError) as exc_info:
                call_chat_api_with_fallback(
                    mock_client, api_params, fallback_models=["fallback-1"]
                )

        # Should only have tried once (no fallback)
        assert mock_client.chat.completions.create.call_count == 1
        assert exc_info.value.status_code == 400

    def test_rate_limit_triggers_fallback(self):
        """429 rate limit errors should trigger fallback to avoid losing submissions."""
        mock_client = MagicMock()
        mock_response = MagicMock()

        mock_client.chat.completions.create.side_effect = [
            APIStatusError(
                "Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={"error": {"message": "Rate limit exceeded"}},
            ),
            mock_response,
        ]

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            response, model = call_chat_api_with_fallback(
                mock_client, api_params, fallback_models=["fallback-1"]
            )

        assert mock_client.chat.completions.create.call_count == 2
        assert response == mock_response

    def test_auth_error_not_retried(self):
        """401 auth errors should not trigger fallback."""
        mock_client = MagicMock()

        mock_client.chat.completions.create.side_effect = APIStatusError(
            "Unauthorized",
            response=MagicMock(status_code=401),
            body={"error": {"message": "Invalid API key"}},
        )

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            with pytest.raises(APIStatusError) as exc_info:
                call_chat_api_with_fallback(
                    mock_client, api_params, fallback_models=["fallback-1"]
                )

        assert mock_client.chat.completions.create.call_count == 1
        assert exc_info.value.status_code == 401

    def test_non_api_error_not_retried(self):
        """Non-API errors (e.g., network errors) should not trigger fallback."""
        mock_client = MagicMock()

        mock_client.chat.completions.create.side_effect = ConnectionError("Network error")

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with pytest.raises(ConnectionError):
            call_chat_api_with_fallback(
                mock_client, api_params, fallback_models=["fallback-1"]
            )

        assert mock_client.chat.completions.create.call_count == 1

    def test_uses_config_fallback_models_by_default(self):
        """When fallback_models is None, should use Config.FALLBACK_MODELS."""
        mock_client = MagicMock()
        mock_response = MockResponse("Response")

        mock_client.chat.completions.create.side_effect = [
            APIStatusError(
                "Unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Unavailable"}},
            ),
            mock_response,
        ]

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            with patch(
                "aurelius.shared.chat_client.Config.FALLBACK_MODELS",
                ["config-fallback-model"],
            ):
                response, model_used = call_chat_api_with_fallback(
                    mock_client, api_params, fallback_models=None
                )

        assert response == mock_response
        assert model_used == "config-fallback-model"

    def test_api_params_preserved_except_model(self):
        """Original api_params should be preserved, only model changes."""
        mock_client = MagicMock()
        mock_response = MockResponse("Response")

        mock_client.chat.completions.create.side_effect = [
            APIStatusError(
                "Unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Unavailable"}},
            ),
            mock_response,
        ]

        api_params = {
            "model": "primary-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        with patch("bittensor.logging"):
            call_chat_api_with_fallback(
                mock_client, api_params, fallback_models=["fallback-1"]
            )

        # Check second call (fallback) preserved other params
        second_call = mock_client.chat.completions.create.call_args_list[1]
        assert second_call.kwargs["model"] == "fallback-1"
        assert second_call.kwargs["messages"] == [{"role": "user", "content": "Hi"}]
        assert second_call.kwargs["temperature"] == 0.7
        assert second_call.kwargs["max_tokens"] == 1000

    def test_timeout_passed_to_api(self):
        """Timeout parameter should be passed to API calls."""
        mock_client = MagicMock()
        mock_response = MockResponse("Response")
        mock_client.chat.completions.create.return_value = mock_response

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        call_chat_api_with_fallback(
            mock_client, api_params, fallback_models=[], timeout=30.0
        )

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["timeout"] == 30.0

    def test_timeout_none_not_added(self):
        """When timeout is None, it should not be added to params."""
        mock_client = MagicMock()
        mock_response = MockResponse("Response")
        mock_client.chat.completions.create.return_value = mock_response

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        call_chat_api_with_fallback(
            mock_client, api_params, fallback_models=[], timeout=None
        )

        call_args = mock_client.chat.completions.create.call_args
        assert "timeout" not in call_args.kwargs

    def test_timeout_passed_to_fallback_calls(self):
        """Timeout should be passed to fallback model calls too."""
        mock_client = MagicMock()
        mock_response = MockResponse("Fallback response")

        mock_client.chat.completions.create.side_effect = [
            APIStatusError(
                "Unavailable",
                response=MagicMock(status_code=503),
                body={"error": {"message": "Unavailable"}},
            ),
            mock_response,
        ]

        api_params = {"model": "primary-model", "messages": [{"role": "user", "content": "Hi"}]}

        with patch("bittensor.logging"):
            call_chat_api_with_fallback(
                mock_client, api_params, fallback_models=["fallback-1"], timeout=45.0
            )

        # Both calls should have the timeout
        first_call = mock_client.chat.completions.create.call_args_list[0]
        second_call = mock_client.chat.completions.create.call_args_list[1]
        assert first_call.kwargs["timeout"] == 45.0
        assert second_call.kwargs["timeout"] == 45.0


    def test_prefer_deepseek_tries_deepseek_first(self):
        """When prefer_deepseek=True, DeepSeek client should be called before primary client."""
        mock_client = MagicMock()
        mock_deepseek_client = MagicMock()
        mock_response = MockResponse("DeepSeek response")
        mock_deepseek_client.chat.completions.create.return_value = mock_response

        api_params = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        with patch("bittensor.logging"):
            response, model_used = call_chat_api_with_fallback(
                mock_client,
                api_params,
                fallback_models=["fallback-1"],
                deepseek_client=mock_deepseek_client,
                prefer_deepseek=True,
            )

        assert response == mock_response
        assert model_used == "deepseek-direct/deepseek-chat"
        # DeepSeek should have been called
        assert mock_deepseek_client.chat.completions.create.call_count == 1
        # Primary client should NOT have been called
        assert mock_client.chat.completions.create.call_count == 0

    def test_prefer_deepseek_skips_chutes_on_failure(self):
        """When prefer_deepseek=True and DeepSeek fails, should raise immediately without trying Chutes."""
        mock_client = MagicMock()
        mock_deepseek_client = MagicMock()

        mock_deepseek_client.chat.completions.create.side_effect = APIStatusError(
            "Rate limited",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limited"}},
        )

        api_params = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        with patch("bittensor.logging"):
            with pytest.raises(ModelUnavailableError) as exc_info:
                call_chat_api_with_fallback(
                    mock_client,
                    api_params,
                    fallback_models=["fallback-1"],
                    deepseek_client=mock_deepseek_client,
                    prefer_deepseek=True,
                )

        assert "deepseek-direct/deepseek-chat" in str(exc_info.value)
        # DeepSeek was tried once
        assert mock_deepseek_client.chat.completions.create.call_count == 1
        # Chutes primary and fallback clients should NOT have been called
        assert mock_client.chat.completions.create.call_count == 0

    def test_prefer_deepseek_skips_all_chutes_phases(self):
        """When prefer_deepseek=True and DeepSeek fails, neither Chutes primary nor fallback models are tried."""
        mock_client = MagicMock()
        mock_deepseek_client = MagicMock()

        # DeepSeek fails in Phase 0
        mock_deepseek_client.chat.completions.create.side_effect = APIStatusError(
            "Service unavailable",
            response=MagicMock(status_code=503),
            body={"error": {"message": "Unavailable"}},
        )

        api_params = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        with patch("bittensor.logging"):
            with pytest.raises(ModelUnavailableError) as exc_info:
                call_chat_api_with_fallback(
                    mock_client,
                    api_params,
                    fallback_models=["fallback-1", "fallback-2"],
                    deepseek_client=mock_deepseek_client,
                    prefer_deepseek=True,
                )

        # Error should list no fallback models since Chutes was skipped
        assert exc_info.value.fallback_models == []
        # DeepSeek should only have been tried once (Phase 0)
        assert mock_deepseek_client.chat.completions.create.call_count == 1
        # Chutes should not have been touched at all
        assert mock_client.chat.completions.create.call_count == 0


class TestModelUnavailableError:
    """Tests for ModelUnavailableError exception."""

    def test_error_message_contains_all_info(self):
        """Error message should contain primary model, fallbacks, and errors."""
        error = ModelUnavailableError(
            primary_model="primary",
            fallback_models=["fallback-1", "fallback-2"],
            errors=["error1", "error2", "error3"],
        )

        assert "primary" in str(error)
        assert "fallback-1" in str(error)
        assert "fallback-2" in str(error)
        assert "error1" in str(error)

    def test_error_attributes(self):
        """Error should have correct attributes."""
        error = ModelUnavailableError(
            primary_model="my-primary",
            fallback_models=["fb1", "fb2"],
            errors=["err1", "err2"],
        )

        assert error.primary_model == "my-primary"
        assert error.fallback_models == ["fb1", "fb2"]
        assert error.errors == ["err1", "err2"]
