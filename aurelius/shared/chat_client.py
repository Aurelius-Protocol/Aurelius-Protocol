"""Chat client helper with model fallback logic."""

from typing import Any

import bittensor as bt
from openai import APIStatusError, APITimeoutError, OpenAI

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
        super().__init__(f"All models unavailable. Tried: {models_tried}. Errors: {errors}")


# Mapping from Chutes model names to DeepSeek native API model names
DEEPSEEK_MODEL_MAP: dict[str, str] = {
    "deepseek-ai/DeepSeek-V3": "deepseek-chat",
    "deepseek-ai/DeepSeek-V3-0324": "deepseek-chat",
    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek-chat",
    "deepseek-ai/DeepSeek-R1": "deepseek-reasoner",
}


def _map_to_deepseek_model(chutes_model: str) -> str | None:
    """Map a Chutes model name to a DeepSeek native API model name.

    Returns None if the model is not a DeepSeek model.
    """
    if chutes_model in DEEPSEEK_MODEL_MAP:
        return DEEPSEEK_MODEL_MAP[chutes_model]
    if chutes_model.startswith("deepseek-ai/DeepSeek-V3"):
        return "deepseek-chat"
    if chutes_model.startswith("deepseek-ai/DeepSeek-R1"):
        return "deepseek-reasoner"
    return None


def call_chat_api_with_fallback(
    client: OpenAI,
    api_params: dict[str, Any],
    fallback_models: list[str] | None = None,
    timeout: float | None = None,
    deepseek_client: OpenAI | None = None,
) -> tuple[Any, str]:
    """
    Call chat API with automatic fallback on model unavailability.

    Fallback order:
    1. Primary model on primary client (Chutes)
    2. DeepSeek direct API (if deepseek_client provided and model is DeepSeek)
    3. Fallback models on primary client (Chutes)

    Includes circuit breaker protection - if the API has failed repeatedly,
    requests will fail fast without attempting the API call.

    Args:
        client: OpenAI client instance
        api_params: API parameters including 'model' key
        fallback_models: List of fallback model names. If None, uses Config.FALLBACK_MODELS
        timeout: Request timeout in seconds. If None, uses client default.
        deepseek_client: Optional OpenAI client configured for DeepSeek direct API.
            When provided, acts as a fallback between primary and fallback models.

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
        bt.logging.warning(f"Chat API circuit breaker OPEN - failing fast (retry in {retry_time:.1f}s)")
        raise CircuitBreakerOpen("chat-api", retry_time)

    if fallback_models is None:
        fallback_models = Config.FALLBACK_MODELS

    primary_model = api_params.get("model", Config.DEFAULT_MODEL)
    errors: list[str] = []

    # HTTP status codes indicating temporary unavailability
    unavailable_status_codes = {502, 503, 504}

    # Phase 1: Try primary model on primary client
    try:
        params = {**api_params, "model": primary_model}
        if timeout is not None:
            params["timeout"] = timeout

        response = client.chat.completions.create(**params)
        _chat_circuit_breaker.record_success()
        return response, primary_model

    except APIStatusError as e:
        if e.status_code in unavailable_status_codes:
            error_msg = f"{primary_model}: {e.status_code} - {e.message}"
            errors.append(error_msg)
            bt.logging.warning(f"Model unavailable: {error_msg}")
            # Continue to Phase 2
        else:
            # Non-availability error (e.g., 400, 401, 429) - don't try fallbacks
            raise

    except APITimeoutError as e:
        error_msg = f"{primary_model}: timeout - {e}"
        errors.append(error_msg)
        bt.logging.warning(f"Model timeout: {error_msg}")
        # Continue to Phase 2

    except Exception:
        _chat_circuit_breaker.record_failure()
        raise

    # Phase 2: Try DeepSeek direct API (if available and model is DeepSeek)
    if deepseek_client is not None:
        deepseek_model = _map_to_deepseek_model(primary_model)
        if deepseek_model is not None:
            try:
                bt.logging.info(f"Trying DeepSeek direct API: {deepseek_model} (mapped from {primary_model})")
                params = {**api_params, "model": deepseek_model}
                if timeout is not None:
                    params["timeout"] = timeout

                response = deepseek_client.chat.completions.create(**params)
                bt.logging.success(f"DeepSeek direct API succeeded: {deepseek_model}")
                _chat_circuit_breaker.record_success()
                return response, f"deepseek-direct/{deepseek_model}"

            except APIStatusError as e:
                if e.status_code in unavailable_status_codes:
                    error_msg = f"deepseek-direct/{deepseek_model}: {e.status_code} - {e.message}"
                    errors.append(error_msg)
                    bt.logging.warning(f"DeepSeek direct unavailable: {error_msg}")
                    # Continue to Phase 3
                else:
                    raise

            except Exception as e:
                error_msg = f"deepseek-direct/{deepseek_model}: {type(e).__name__} - {e}"
                errors.append(error_msg)
                bt.logging.warning(f"DeepSeek direct error: {error_msg}")
                # Intentionally not recording circuit breaker failure here:
                # DeepSeek direct is a separate API from the primary provider,
                # so its errors should not trip the primary circuit breaker.
                # Continue to Phase 3.

    # Phase 3: Try fallback models on primary client
    for model in fallback_models:
        try:
            params = {**api_params, "model": model}
            if timeout is not None:
                params["timeout"] = timeout

            bt.logging.info(f"Trying fallback model: {model}")
            response = client.chat.completions.create(**params)
            bt.logging.success(f"Fallback model {model} succeeded")
            _chat_circuit_breaker.record_success()
            return response, model

        except APIStatusError as e:
            if e.status_code in unavailable_status_codes:
                error_msg = f"{model}: {e.status_code} - {e.message}"
                errors.append(error_msg)
                bt.logging.warning(f"Model unavailable: {error_msg}")
                continue
            else:
                raise

        except APITimeoutError as e:
            error_msg = f"{model}: timeout - {e}"
            errors.append(error_msg)
            bt.logging.warning(f"Model timeout: {error_msg}")
            continue

        except Exception:
            _chat_circuit_breaker.record_failure()
            raise

    # All models exhausted - record failure
    _chat_circuit_breaker.record_failure()
    raise ModelUnavailableError(
        primary_model=primary_model,
        fallback_models=fallback_models,
        errors=errors,
    )
