"""Chat client helper with model fallback logic."""

from typing import Any

import bittensor as bt
from openai import APIStatusError, OpenAI

from aurelius.shared.circuit_breaker import CircuitBreakerConfig, CircuitBreakerOpen, get_circuit_breaker
from aurelius.shared.config import Config

# Circuit breaker for chat API
_chat_circuit_breaker = get_circuit_breaker(
    "chat-api",
    CircuitBreakerConfig(
        failure_threshold=3,  # Lower threshold since chat is critical
        recovery_timeout=30.0,  # Shorter recovery for faster service restoration
        half_open_max_calls=1,
        success_threshold=1,
    ),
)


class ModelUnavailableError(Exception):
    """Raised when all models (primary and fallbacks) are unavailable."""

    def __init__(self, primary_model: str, fallback_models: list[str], errors: list[str]):
        self.primary_model = primary_model
        self.fallback_models = fallback_models
        self.errors = errors
        models_tried = [primary_model] + fallback_models
        super().__init__(
            f"All models unavailable. Tried: {models_tried}. Errors: {errors}"
        )


def call_chat_api_with_fallback(
    client: OpenAI,
    api_params: dict[str, Any],
    fallback_models: list[str] | None = None,
    timeout: float | None = None,
) -> tuple[Any, str]:
    """
    Call chat API with automatic fallback on model unavailability.

    When the primary model returns 502/503/504 (service unavailable errors),
    this function automatically tries fallback models in order.

    Includes circuit breaker protection - if the API has failed repeatedly,
    requests will fail fast without attempting the API call.

    Args:
        client: OpenAI client instance
        api_params: API parameters including 'model' key
        fallback_models: List of fallback model names. If None, uses Config.FALLBACK_MODELS
        timeout: Request timeout in seconds. If None, uses client default.

    Returns:
        Tuple of (response, actual_model_used) where response is the API response
        and actual_model_used is the model that successfully responded.

    Raises:
        ModelUnavailableError: If all models (primary and fallbacks) fail with 502/503/504
        CircuitBreakerOpen: If circuit breaker is open due to repeated failures
        Other exceptions: Re-raised if the error is not a service unavailability error
    """
    # Check circuit breaker first - fail fast if API is known to be down
    if not _chat_circuit_breaker.can_execute():
        retry_time = _chat_circuit_breaker.get_time_until_retry()
        bt.logging.warning(
            f"Chat API circuit breaker OPEN - failing fast (retry in {retry_time:.1f}s)"
        )
        raise CircuitBreakerOpen("chat-api", retry_time)

    if fallback_models is None:
        fallback_models = Config.FALLBACK_MODELS

    primary_model = api_params.get("model", Config.DEFAULT_MODEL)
    models_to_try = [primary_model] + fallback_models
    errors: list[str] = []

    # HTTP status codes indicating temporary unavailability
    unavailable_status_codes = {502, 503, 504}

    for i, model in enumerate(models_to_try):
        try:
            # Update the model in params
            params = {**api_params, "model": model}
            if timeout is not None:
                params["timeout"] = timeout

            if i > 0:
                bt.logging.info(f"Trying fallback model: {model}")

            response = client.chat.completions.create(**params)

            if i > 0:
                bt.logging.success(f"Fallback model {model} succeeded")

            # Record success with circuit breaker
            _chat_circuit_breaker.record_success()

            return response, model

        except APIStatusError as e:
            if e.status_code in unavailable_status_codes:
                error_msg = f"{model}: {e.status_code} - {e.message}"
                errors.append(error_msg)
                bt.logging.warning(f"Model unavailable: {error_msg}")

                # Continue to next model if we have more to try
                if i < len(models_to_try) - 1:
                    continue
                else:
                    # No more models to try - record failure with circuit breaker
                    _chat_circuit_breaker.record_failure()
                    raise ModelUnavailableError(
                        primary_model=primary_model,
                        fallback_models=fallback_models,
                        errors=errors,
                    ) from e
            else:
                # Non-availability error (e.g., 400, 401, 429) - don't try fallbacks
                # Don't count these as circuit breaker failures (client errors)
                raise

        except Exception as e:
            # For non-API errors (network issues, timeouts), record failure
            _chat_circuit_breaker.record_failure()
            raise

    # Should not reach here, but just in case
    _chat_circuit_breaker.record_failure()
    raise ModelUnavailableError(
        primary_model=primary_model,
        fallback_models=fallback_models,
        errors=errors,
    )
