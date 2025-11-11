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
