"""Novelty detection client for calculating prompt uniqueness via central API."""

import time
from dataclasses import dataclass

import bittensor as bt
import requests
from opentelemetry.trace import SpanKind

from aurelius.shared.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker
from aurelius.shared.config import Config
from aurelius.shared.telemetry.otel_setup import get_tracer


@dataclass
class NoveltyResult:
    """Result from novelty detection check.

    Attributes:
        novelty_score: Score from 0-1 where 1 = completely novel, 0 = exact duplicate
        max_similarity: Maximum similarity found against existing prompts
        similar_count: Number of prompts above similarity threshold
        most_similar_id: ID of most similar prompt (if any)
    """

    novelty_score: float
    max_similarity: float
    similar_count: int
    most_similar_id: int | None = None


@dataclass
class MinerNoveltyStats:
    """Miner's novelty statistics.

    Attributes:
        avg_novelty: Average novelty score across all prompts
        total_prompts: Total prompts submitted
        novel_prompts: Prompts considered novel (above threshold)
        duplicate_prompts: Prompts considered duplicates
    """

    avg_novelty: float
    total_prompts: int
    novel_prompts: int
    duplicate_prompts: int


class NoveltyClient:
    """Client for novelty detection API calls."""

    def __init__(self, api_endpoint: str | None = None, timeout: int = 10, api_key: str | None = None):
        """
        Initialize novelty client.

        Args:
            api_endpoint: Base URL for novelty API (e.g., https://api.example.com/api/novelty)
            timeout: Request timeout in seconds
            api_key: API key for authenticated requests
        """
        self.api_endpoint = api_endpoint or Config.NOVELTY_API_ENDPOINT
        self.timeout = timeout
        self._api_key = api_key or getattr(Config, "CENTRAL_API_KEY", None)
        self._session = requests.Session()
        self._tracer = get_tracer("aurelius.novelty") if Config.TELEMETRY_ENABLED else None

        # Initialize circuit breaker for API resilience
        self._circuit_breaker = get_circuit_breaker(
            "novelty-api",
            CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                half_open_max_calls=1,
                success_threshold=2,
            ),
        )

        if self.api_endpoint:
            bt.logging.info(f"Novelty client: API endpoint at {self.api_endpoint}")
        else:
            bt.logging.warning("Novelty client: No API endpoint configured")

    def is_available(self) -> bool:
        """Check if novelty detection is available."""
        return bool(self.api_endpoint)

    def check_novelty(
        self,
        prompt: str,
        prompt_embedding: list[float] | None = None,
        include_similar_prompt: bool = False,
        experiment_id: str = "prompt",
    ) -> NoveltyResult | None:
        """
        Check novelty of a prompt against existing database (T084).

        Args:
            prompt: The prompt text to check
            prompt_embedding: Pre-computed embedding (384 dimensions). Required by API.
            include_similar_prompt: Whether to include similar prompt text in response
            experiment_id: Experiment ID for per-experiment novelty pools (default: "prompt")

        Returns:
            NoveltyResult with novelty_score and related metrics, or None on error
        """
        if not self.api_endpoint:
            bt.logging.debug("Novelty check skipped: no API endpoint configured")
            return None

        if not prompt_embedding:
            bt.logging.debug("Novelty check skipped: no embedding provided")
            return None

        start_time = time.time()

        # Wrap with tracing span if enabled
        if self._tracer:
            with self._tracer.start_as_current_span(
                "novelty.check",
                kind=SpanKind.CLIENT,
                attributes={
                    "novelty.endpoint": self.api_endpoint,
                    "novelty.prompt_length": len(prompt),
                    "novelty.has_embedding": True,
                    "novelty.experiment_id": experiment_id,
                },
            ) as span:
                result = self._do_check_novelty(prompt, prompt_embedding, include_similar_prompt, experiment_id)
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("duration_ms", round(duration_ms, 2))
                if result:
                    span.set_attribute("novelty.score", result.novelty_score)
                    span.set_attribute("novelty.max_similarity", result.max_similarity)
                    span.set_attribute("novelty.similar_count", result.similar_count)
                return result
        else:
            return self._do_check_novelty(prompt, prompt_embedding, include_similar_prompt, experiment_id)

    def _do_check_novelty(
        self,
        prompt: str,
        prompt_embedding: list[float],
        include_similar_prompt: bool,
        experiment_id: str = "prompt",
    ) -> NoveltyResult | None:
        """Internal method to perform the novelty check API call (T084)."""
        # Check circuit breaker first - fail fast if API is known to be down
        if not self._circuit_breaker.can_execute():
            bt.logging.debug(
                f"Novelty circuit breaker OPEN - skipping check "
                f"(retry in {self._circuit_breaker.get_time_until_retry():.1f}s)"
            )
            return None

        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            response = self._session.post(
                f"{self.api_endpoint}/check",
                json={
                    "prompt": prompt,
                    "prompt_embedding": prompt_embedding,
                    "include_similar_prompt": include_similar_prompt,
                    "experiment_id": experiment_id,  # T084: Per-experiment novelty pool
                },
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                # Validate required fields to avoid silently defaulting to 1.0
                if "novelty_score" not in data:
                    bt.logging.warning("Novelty API returned 200 but missing 'novelty_score' field")
                    return None
                # Record success with circuit breaker
                self._circuit_breaker.record_success()
                return NoveltyResult(
                    novelty_score=data["novelty_score"],
                    max_similarity=data.get("max_similarity", 0.0),
                    similar_count=data.get("similar_count", 0),
                    most_similar_id=data.get("most_similar_id"),
                )
            elif response.status_code == 400:
                # Bad request is a client error, don't count as circuit failure
                bt.logging.warning(f"Novelty check: bad request - {response.text}")
                return None
            elif response.status_code in {502, 503, 504}:
                # Service unavailable - record failure
                self._circuit_breaker.record_failure()
                bt.logging.debug("Novelty check: service not available")
                return None
            else:
                # Other server errors - record failure
                if response.status_code >= 500:
                    self._circuit_breaker.record_failure()
                bt.logging.warning(f"Novelty check failed: HTTP {response.status_code} - {response.text}")
                return None

        except requests.Timeout:
            self._circuit_breaker.record_failure()
            bt.logging.warning(f"Novelty check timed out after {self.timeout}s")
            return None
        except requests.RequestException as e:
            self._circuit_breaker.record_failure()
            bt.logging.warning(f"Novelty check request failed: {e}")
            return None
        except Exception as e:
            self._circuit_breaker.record_failure()
            bt.logging.error(f"Novelty check unexpected error: {e}")
            return None

    def get_miner_stats(self, miner_hotkey: str) -> MinerNoveltyStats | None:
        """
        Get novelty statistics for a specific miner.

        Args:
            miner_hotkey: Miner's hotkey

        Returns:
            MinerNoveltyStats with average novelty and counts, or None on error
        """
        if not self.api_endpoint:
            return None

        # Check circuit breaker first
        if not self._circuit_breaker.can_execute():
            bt.logging.debug("Novelty circuit breaker OPEN - skipping miner stats")
            return None

        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            response = self._session.get(
                f"{self.api_endpoint}/miner/{miner_hotkey}",
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                self._circuit_breaker.record_success()
                return MinerNoveltyStats(
                    avg_novelty=data.get("avg_novelty", 1.0),
                    total_prompts=data.get("total_prompts", 0),
                    novel_prompts=data.get("novel_prompts", 0),
                    duplicate_prompts=data.get("duplicate_prompts", 0),
                )
            else:
                if response.status_code >= 500:
                    self._circuit_breaker.record_failure()
                bt.logging.warning(f"Get miner novelty stats failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            self._circuit_breaker.record_failure()
            bt.logging.warning(f"Get miner novelty stats failed: {e}")
            return None

    def close(self):
        """Close the HTTP session to release connection pool resources."""
        self._session.close()

    def get_global_stats(self) -> dict | None:
        """
        Get global novelty statistics.

        Returns:
            Dictionary with global stats or None on error
        """
        if not self.api_endpoint:
            return None

        # Check circuit breaker first
        if not self._circuit_breaker.can_execute():
            bt.logging.debug("Novelty circuit breaker OPEN - skipping global stats")
            return None

        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            response = self._session.get(
                f"{self.api_endpoint}/stats",
                headers=headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                self._circuit_breaker.record_success()
                return response.json()
            else:
                if response.status_code >= 500:
                    self._circuit_breaker.record_failure()
                bt.logging.warning(f"Get novelty stats failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            self._circuit_breaker.record_failure()
            bt.logging.warning(f"Get novelty stats failed: {e}")
            return None


# Singleton instance
_novelty_client: NoveltyClient | None = None


def get_novelty_client() -> NoveltyClient:
    """Get singleton novelty client instance."""
    global _novelty_client
    if _novelty_client is None:
        _novelty_client = NoveltyClient()
    return _novelty_client
