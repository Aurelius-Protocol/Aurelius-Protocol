"""Protocol definitions for miner-validator communication."""

from typing import Any

import bittensor as bt
from pydantic import Field


class PromptSynapse(bt.Synapse):
    """
    Custom synapse for prompt submission and response.

    Miners send prompts to validators, and validators respond with
    OpenAI API completions along with moderation results.

    Attributes:
        prompt: The text prompt to send to OpenAI
        miner_hotkey: The miner's hotkey for tracking
        vendor: AI vendor requested by miner (e.g., 'openai', 'anthropic')
        model_requested: Specific model requested (e.g., 'o4-mini', 'gpt-4o')
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        min_chars: Minimum response length in characters
        max_chars: Maximum response length in characters
        response: The completion returned from OpenAI (filled by validator)
        model_used: The OpenAI model that processed the request (filled by validator)
        danger_score: Combined danger score from moderation (0-1+, filled by validator)
        category_scores: Breakdown of moderation scores by category (filled by validator)
        accepted: Whether the submission met the danger threshold (filled by validator)
        rejection_reason: Reason for rejection if not accepted (filled by validator)
    """

    # Input from miner
    prompt: str = Field(
        ...,
        title="Prompt",
        description="The prompt text to be processed by OpenAI",
    )

    miner_hotkey: str | None = Field(
        None,
        title="Miner Hotkey",
        description="The miner's hotkey for tracking submissions",
    )

    # Model specification from miner (optional)
    vendor: str | None = Field(
        None,
        title="Vendor",
        description="AI vendor requested (e.g., 'openai', 'anthropic')",
    )

    model_requested: str | None = Field(
        None,
        title="Model Requested",
        description="Specific model requested (e.g., 'o4-mini', 'gpt-4o')",
    )

    temperature: float | None = Field(
        None,
        title="Temperature",
        description="Sampling temperature (0.0-2.0)",
        ge=0.0,
        le=2.0,
    )

    top_p: float | None = Field(
        None,
        title="Top P",
        description="Nucleus sampling parameter (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    frequency_penalty: float | None = Field(
        None,
        title="Frequency Penalty",
        description="Frequency penalty (-2.0 to 2.0)",
        ge=-2.0,
        le=2.0,
    )

    presence_penalty: float | None = Field(
        None,
        title="Presence Penalty",
        description="Presence penalty (-2.0 to 2.0)",
        ge=-2.0,
        le=2.0,
    )

    min_chars: int | None = Field(
        None,
        title="Minimum Characters",
        description="Minimum response length in characters",
        ge=0,
    )

    max_chars: int | None = Field(
        None,
        title="Maximum Characters",
        description="Maximum response length in characters",
        ge=1,
    )

    # Output from validator - OpenAI response
    response: str | None = Field(
        None,
        title="Response",
        description="The completion from OpenAI API",
    )

    model_used: str | None = Field(
        None,
        title="Model Used",
        description="The OpenAI model that processed this request",
    )

    # Output from validator - Moderation results
    danger_score: float | None = Field(
        None,
        title="Danger Score",
        description="Combined weighted danger score from moderation (0-1+)",
    )

    category_scores: dict[str, float] | None = Field(
        None,
        title="Category Scores",
        description="Breakdown of moderation scores by category",
    )

    accepted: bool | None = Field(
        None,
        title="Accepted",
        description="Whether the submission met the danger threshold",
    )

    rejection_reason: str | None = Field(
        None,
        title="Rejection Reason",
        description="Reason for rejection (rate limit, low score, error, etc.)",
    )

    # Distribution statistics (from multiple runs across validators)
    distribution_stats: dict[str, Any] | None = Field(
        None,
        title="Distribution Statistics",
        description="Aggregated statistics from multiple runs across validators",
    )
    # Structure:
    # {
    #     "mean_danger_score": 0.75,
    #     "std_dev_danger_score": 0.12,
    #     "min_danger_score": 0.51,
    #     "max_danger_score": 0.91,
    #     "total_runs": 15,
    #     "validator_count": 5,
    #     "all_scores": [0.75, 0.82, ...]
    # }

    def deserialize(self) -> str:
        """Return the response for easy access."""
        return self.response or ""
