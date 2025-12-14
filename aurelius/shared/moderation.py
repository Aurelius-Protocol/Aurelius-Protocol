"""Moderation provider system for content safety checking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import bittensor as bt
from openai import OpenAI


@dataclass
class ModerationResult:
    """Result from a moderation check.

    Attributes:
        flagged: Whether the content was flagged as inappropriate
        category_scores: Dictionary of scores for each category
        categories: Dictionary of boolean flags for each category
        combined_score: Weighted combined score across all categories
        high_category_triggered: Whether any individual category exceeded the high threshold
        high_category_name: Name of the highest scoring category if threshold exceeded
        high_category_score: Score of the highest category if threshold exceeded
    """

    flagged: bool
    category_scores: dict[str, float]
    categories: dict[str, bool]
    combined_score: float
    high_category_triggered: bool = False
    high_category_name: str | None = None
    high_category_score: float | None = None


class ModerationProvider(ABC):
    """Abstract base class for moderation providers."""

    @abstractmethod
    def moderate(self, text: str) -> ModerationResult:
        """
        Check text content for safety violations.

        Args:
            text: The text content to moderate

        Returns:
            ModerationResult with scores and flags
        """
        pass


class OpenAIModerationProvider(ModerationProvider):
    """OpenAI Moderation API provider."""

    # Default category weights for calculating combined score
    DEFAULT_CATEGORY_WEIGHTS = {
        "hate": 1.0,
        "hate/threatening": 1.2,
        "harassment": 0.8,
        "harassment/threatening": 1.0,
        "self-harm": 1.5,
        "self-harm/intent": 1.5,
        "self-harm/instructions": 1.5,
        "sexual": 0.8,
        "sexual/minors": 2.0,
        "violence": 1.0,
        "violence/graphic": 1.2,
    }

    # Default timeout for API calls (seconds)
    DEFAULT_API_TIMEOUT = 30.0

    def __init__(
        self,
        api_key: str,
        category_weights: dict[str, float] | None = None,
        single_category_threshold: float = 0.8,
        timeout: float | None = None,
    ):
        """
        Initialize OpenAI moderation provider.

        Args:
            api_key: OpenAI API key
            category_weights: Optional custom weights for categories.
                            If not provided, uses DEFAULT_CATEGORY_WEIGHTS.
            single_category_threshold: Threshold for individual category scores (default 0.8)
            timeout: API request timeout in seconds (default 30s)
        """
        self.timeout = timeout or self.DEFAULT_API_TIMEOUT
        # SECURITY: Configure client with timeout to prevent indefinite blocking
        self.client = OpenAI(api_key=api_key, timeout=self.timeout)
        self.category_weights = category_weights or self.DEFAULT_CATEGORY_WEIGHTS
        self.single_category_threshold = single_category_threshold

    def moderate(self, text: str) -> ModerationResult:
        """
        Check text using OpenAI Moderation API.

        Args:
            text: The text to moderate

        Returns:
            ModerationResult with scores and flags
        """
        try:
            response = self.client.moderations.create(input=text)
            result = response.results[0]

            # Extract category scores and flags
            category_scores = result.category_scores.model_dump()
            categories = result.categories.model_dump()

            # Calculate weighted combined score
            combined_score = self._calculate_combined_score(category_scores)

            # Check for high individual category scores
            high_cat_triggered, high_cat_name, high_cat_score = self._check_high_category_threshold(category_scores)

            return ModerationResult(
                flagged=result.flagged,
                category_scores=category_scores,
                categories=categories,
                combined_score=combined_score,
                high_category_triggered=high_cat_triggered,
                high_category_name=high_cat_name,
                high_category_score=high_cat_score,
            )

        except Exception as e:
            bt.logging.error(f"CRITICAL: OpenAI moderation API failure: {e}")

            # SECURITY: Always fail-closed on API errors
            # This prevents dangerous content from being accepted during outages
            # The fail-open mode has been removed as it creates security vulnerabilities
            bt.logging.error(
                "Rejecting content due to API error (fail-closed security behavior)"
            )
            return ModerationResult(
                flagged=True,
                category_scores={},
                categories={},
                combined_score=1.0,
                high_category_triggered=True,
                high_category_name="api_error",
                high_category_score=1.0,
            )

    def _calculate_combined_score(self, category_scores: dict[str, float]) -> float:
        """
        Calculate weighted combined score from category scores.

        The combined score is the weighted average of all category scores,
        normalized to 0-1 range.

        Args:
            category_scores: Dictionary of category names to scores

        Returns:
            Combined score between 0 and 1
        """
        if not category_scores:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for category, score in category_scores.items():
            weight = self.category_weights.get(category, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _check_high_category_threshold(self, category_scores: dict[str, float]) -> tuple[bool, str | None, float | None]:
        """
        Check if any individual category score exceeds the high threshold.

        This implements the dual-logic acceptance criteria: a prompt can be
        flagged as dangerous either by having a high aggregate score OR by
        having any single category with a very high score (top 20% of range).

        Args:
            category_scores: Dictionary of category names to scores

        Returns:
            Tuple of (triggered, category_name, category_score)
            - triggered: True if any category >= threshold
            - category_name: Name of highest scoring category if triggered
            - category_score: Score of highest category if triggered
        """
        if not category_scores:
            return False, None, None

        # Find the highest scoring category
        max_category = max(category_scores.items(), key=lambda x: x[1])
        max_category_name, max_category_score = max_category

        # Check if it exceeds the single category threshold
        if max_category_score >= self.single_category_threshold:
            return True, max_category_name, max_category_score

        return False, None, None


def create_moderation_provider(
    provider_name: str,
    api_key: str,
    category_weights: dict[str, float] | None = None,
    single_category_threshold: float = 0.8,
    timeout: float | None = None,
) -> ModerationProvider:
    """
    Factory function to create a moderation provider.

    Args:
        provider_name: Name of the provider ("openai", etc.)
        api_key: API key for the provider
        category_weights: Optional custom category weights
        single_category_threshold: Threshold for individual category scores (default 0.8)
        timeout: API request timeout in seconds (default 30s)

    Returns:
        ModerationProvider instance

    Raises:
        ValueError: If provider_name is not recognized
    """
    if provider_name.lower() == "openai":
        return OpenAIModerationProvider(
            api_key=api_key,
            category_weights=category_weights,
            single_category_threshold=single_category_threshold,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown moderation provider: {provider_name}")
