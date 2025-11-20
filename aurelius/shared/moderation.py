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
    """

    flagged: bool
    category_scores: dict[str, float]
    categories: dict[str, bool]
    combined_score: float


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

    def __init__(self, api_key: str, category_weights: dict[str, float] | None = None):
        """
        Initialize OpenAI moderation provider.

        Args:
            api_key: OpenAI API key
            category_weights: Optional custom weights for categories.
                            If not provided, uses DEFAULT_CATEGORY_WEIGHTS.
        """
        self.client = OpenAI(api_key=api_key)
        self.category_weights = category_weights or self.DEFAULT_CATEGORY_WEIGHTS

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

            return ModerationResult(
                flagged=result.flagged,
                category_scores=category_scores,
                categories=categories,
                combined_score=combined_score,
            )

        except Exception as e:
            bt.logging.error(f"CRITICAL: OpenAI moderation API failure: {e}")

            # Handle error based on fail mode configuration
            from aurelius.shared.config import Config

            if Config.MODERATION_FAIL_MODE == "closed":
                # Fail closed: Reject content on error (conservative/secure)
                bt.logging.error(
                    "MODERATION_FAIL_MODE=closed: Rejecting content due to API error (SECURE default behavior)"
                )
                return ModerationResult(flagged=True, category_scores={}, categories={}, combined_score=1.0)
            else:
                # Fail open: Accept content on error (permissive) - ONLY FOR LOCAL_MODE
                bt.logging.error(
                    "SECURITY WARNING: MODERATION_FAIL_MODE=open: Accepting content despite API error. "
                    "This should ONLY be used in LOCAL_MODE for testing!"
                )
                return ModerationResult(flagged=False, category_scores={}, categories={}, combined_score=0.0)

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


def create_moderation_provider(
    provider_name: str, api_key: str, category_weights: dict[str, float] | None = None
) -> ModerationProvider:
    """
    Factory function to create a moderation provider.

    Args:
        provider_name: Name of the provider ("openai", etc.)
        api_key: API key for the provider
        category_weights: Optional custom category weights

    Returns:
        ModerationProvider instance

    Raises:
        ValueError: If provider_name is not recognized
    """
    if provider_name.lower() == "openai":
        return OpenAIModerationProvider(api_key=api_key, category_weights=category_weights)
    else:
        raise ValueError(f"Unknown moderation provider: {provider_name}")
