"""Novelty detection client for calculating prompt uniqueness via central API."""

from dataclasses import dataclass

import bittensor as bt
import requests

from aurelius.shared.config import Config


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

    def __init__(self, api_endpoint: str | None = None, timeout: int = 10):
        """
        Initialize novelty client.

        Args:
            api_endpoint: Base URL for novelty API (e.g., https://api.example.com/api/novelty)
            timeout: Request timeout in seconds
        """
        self.api_endpoint = api_endpoint or Config.NOVELTY_API_ENDPOINT
        self.timeout = timeout

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
    ) -> NoveltyResult | None:
        """
        Check novelty of a prompt against existing database.

        Args:
            prompt: The prompt text to check
            prompt_embedding: Pre-computed embedding (384 dimensions). Required by API.
            include_similar_prompt: Whether to include similar prompt text in response

        Returns:
            NoveltyResult with novelty_score and related metrics, or None on error
        """
        if not self.api_endpoint:
            bt.logging.debug("Novelty check skipped: no API endpoint configured")
            return None

        if not prompt_embedding:
            bt.logging.debug("Novelty check skipped: no embedding provided")
            return None

        try:
            response = requests.post(
                f"{self.api_endpoint}/check",
                json={
                    "prompt": prompt,
                    "prompt_embedding": prompt_embedding,
                    "include_similar_prompt": include_similar_prompt,
                },
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                return NoveltyResult(
                    novelty_score=data.get("novelty_score", 1.0),
                    max_similarity=data.get("max_similarity", 0.0),
                    similar_count=data.get("similar_count", 0),
                    most_similar_id=data.get("most_similar_id"),
                )
            elif response.status_code == 400:
                bt.logging.warning(f"Novelty check: bad request - {response.text}")
                return None
            elif response.status_code == 503:
                # Service unavailable
                bt.logging.debug("Novelty check: service not available")
                return None
            else:
                bt.logging.warning(f"Novelty check failed: HTTP {response.status_code} - {response.text}")
                return None

        except requests.Timeout:
            bt.logging.warning(f"Novelty check timed out after {self.timeout}s")
            return None
        except requests.RequestException as e:
            bt.logging.warning(f"Novelty check request failed: {e}")
            return None
        except Exception as e:
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

        try:
            response = requests.get(
                f"{self.api_endpoint}/miner/{miner_hotkey}",
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                return MinerNoveltyStats(
                    avg_novelty=data.get("avg_novelty", 1.0),
                    total_prompts=data.get("total_prompts", 0),
                    novel_prompts=data.get("novel_prompts", 0),
                    duplicate_prompts=data.get("duplicate_prompts", 0),
                )
            else:
                bt.logging.warning(f"Get miner novelty stats failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            bt.logging.warning(f"Get miner novelty stats failed: {e}")
            return None

    def get_global_stats(self) -> dict | None:
        """
        Get global novelty statistics.

        Returns:
            Dictionary with global stats or None on error
        """
        if not self.api_endpoint:
            return None

        try:
            response = requests.get(
                f"{self.api_endpoint}/stats",
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return response.json()
            else:
                bt.logging.warning(f"Get novelty stats failed: HTTP {response.status_code}")
                return None

        except Exception as e:
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
