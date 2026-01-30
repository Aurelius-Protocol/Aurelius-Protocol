"""Chat client helper with model fallback logic."""

from typing import Any

import bittensor as bt
from openai import APIStatusError, OpenAI

from aurelius.shared.config import Config


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
        Other exceptions: Re-raised if the error is not a service unavailability error
    """
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
                    # No more models to try
                    raise ModelUnavailableError(
                        primary_model=primary_model,
                        fallback_models=fallback_models,
                        errors=errors,
                    ) from e
            else:
                # Non-availability error (e.g., 400, 401, 429) - don't try fallbacks
                raise

        except Exception:
            # For non-API errors, don't try fallbacks
            raise

    # Should not reach here, but just in case
    raise ModelUnavailableError(
        primary_model=primary_model,
        fallback_models=fallback_models,
        errors=errors,
    )
