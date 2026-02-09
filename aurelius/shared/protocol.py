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
        miner_novelty_avg: Miner's average novelty score across all submissions (filled by validator)
        miner_hit_rate: Miner's acceptance rate (filled by validator)
        miner_submission_count: Total number of submissions from this miner (filled by validator)
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

    # Miner statistics (filled by validator)
    miner_novelty_avg: float | None = Field(
        None,
        title="Miner Novelty Average",
        description="Miner's average novelty score across all submissions (0-1)",
    )

    miner_hit_rate: float | None = Field(
        None,
        title="Miner Hit Rate",
        description="Miner's acceptance rate (accepted/total submissions)",
    )

    miner_submission_count: int | None = Field(
        None,
        title="Miner Submission Count",
        description="Total number of submissions from this miner",
    )

    # Distribution statistics (from multiple runs across validators)
    # Also used for experiment-specific feedback to miners (T088: dual-use documentation)
    distribution_stats: dict[str, Any] | None = Field(
        None,
        title="Distribution Statistics",
        description="Aggregated statistics from multiple runs across validators. "
        "Also provides experiment-specific feedback including thresholds and miner stats.",
    )
    # Dual-use field (T088):
    #
    # 1. CONSENSUS DISTRIBUTION: Statistics from multiple validator runs
    #    Used when validators collaborate on scoring dangerous prompts.
    #
    # 2. EXPERIMENT FEEDBACK: Per-experiment context for miners
    #    Miners can read this to understand experiment-specific thresholds
    #    and their performance within a specific experiment.
    #
    # Structure:
    # {
    #     # Consensus distribution stats:
    #     "mean_danger_score": 0.75,
    #     "std_dev_danger_score": 0.12,
    #     "min_danger_score": 0.51,
    #     "max_danger_score": 0.91,
    #     "total_runs": 15,
    #     "validator_count": 5,
    #     "all_scores": [0.75, 0.82, ...],
    #
    #     # Experiment feedback (optional, included for multi-experiment):
    #     "experiment_id": "prompt",           # Current experiment
    #     "thresholds": {                       # Experiment-specific thresholds
    #         "acceptance": 0.3,                # Danger score threshold
    #         "novelty": 0.02                   # Novelty score threshold
    #     },
    #     "miner_stats": {                      # Miner's stats in this experiment
    #         "total_submissions": 100,
    #         "hit_rate": 0.45,
    #         "novelty_avg": 0.85
    #     }
    # }

    # Experiment targeting (NEW: multi-experiment framework)
    experiment_id: str | None = Field(
        None,
        title="Experiment ID",
        description="Target experiment ID. Defaults to 'prompt' if None for backward compatibility.",
    )

    registration_required: bool | None = Field(
        None,
        title="Registration Required",
        description="True if miner needs to register for this experiment before submitting. "
        "Set by validator on rejection responses.",
    )

    available_experiments: list[str] | None = Field(
        None,
        title="Available Experiments",
        description="List of active experiment IDs. Returned on invalid experiment_id rejection.",
    )

    # Async submission token (returned by validator on submit)
    submission_token: str | None = Field(
        None,
        title="Submission Token",
        description="Unique token for this submission. Use with SubmissionStatusSynapse to poll for results.",
    )

    def deserialize(self) -> str:
        """Return the response for easy access."""
        return self.response or ""


class ConsensusVerificationSynapse(bt.Synapse):
    """
    Synapse for consensus verification between validators.

    When a primary validator receives a potentially dangerous prompt,
    it initiates consensus by sending this synapse to other validators.
    Each validator independently runs the prompt and votes on dangerousness.

    Attributes:
        prompt: The prompt to verify
        request_id: Unique identifier for this verification request
        primary_validator_hotkey: Hotkey of the validator initiating consensus
        verification_result: Results from the verifying validator (filled by receiver)
    """

    # Input from primary validator
    prompt: str = Field(
        ...,
        title="Prompt",
        description="The prompt to verify for dangerousness",
    )

    request_id: str = Field(
        ...,
        title="Request ID",
        description="Unique identifier for this consensus verification",
    )

    primary_validator_hotkey: str = Field(
        ...,
        title="Primary Validator Hotkey",
        description="Hotkey of the validator that initiated this consensus",
    )

    runs_required: int = Field(
        3,
        title="Runs Required",
        description="Number of times this validator should run the prompt (adaptive based on total validators)",
    )

    # Output from verifying validator
    verification_result: dict[str, Any] | None = Field(
        None,
        title="Verification Result",
        description="Results from running the prompt multiple times",
    )
    # Structure:
    # {
    #     "runs": [0.85, 0.72, 0.91],  # Danger scores from 3 runs
    #     "vote": True,                 # True if dangerous (2+ runs above threshold)
    #     "validator_hotkey": "...",   # This validator's hotkey
    #     "timestamp": "...",           # ISO timestamp
    # }

    def deserialize(self) -> dict | None:
        """Return the verification result for easy access."""
        return self.verification_result


class PullRequestSynapse(bt.Synapse):
    """
    Synapse for pull-based experiment queries from validator to miner (T057).

    In pull experiments, validators initiate queries to miners on a schedule.
    Miners respond with requested data, which the validator then scores.

    This is the inverse of push experiments where miners initiate contact.

    Attributes:
        experiment_id: The pull experiment ID this query is for
        request_id: Unique identifier for this pull request
        validator_hotkey: The validator's hotkey making the query
        query_type: Type of data being requested (e.g., 'data', 'benchmark')
        query_params: Experiment-specific query parameters
        response_data: The miner's response data (filled by miner)
        response_timestamp: When the miner responded (filled by miner)
        error_message: Error message if miner couldn't respond (filled by miner)
    """

    # Input from validator
    experiment_id: str = Field(
        ...,
        title="Experiment ID",
        description="The pull experiment ID this query is for",
    )

    request_id: str = Field(
        ...,
        title="Request ID",
        description="Unique identifier for this pull request",
    )

    validator_hotkey: str = Field(
        ...,
        title="Validator Hotkey",
        description="The validator's hotkey making this query",
    )

    query_type: str = Field(
        "data",
        title="Query Type",
        description="Type of data being requested (e.g., 'data', 'benchmark', 'health')",
    )

    query_params: dict[str, Any] | None = Field(
        None,
        title="Query Parameters",
        description="Experiment-specific parameters for the query",
    )

    # Output from miner
    response_data: dict[str, Any] | None = Field(
        None,
        title="Response Data",
        description="The miner's response data for this pull request",
    )

    response_timestamp: str | None = Field(
        None,
        title="Response Timestamp",
        description="ISO timestamp when the miner responded",
    )

    error_message: str | None = Field(
        None,
        title="Error Message",
        description="Error message if miner couldn't fulfill the request",
    )

    def deserialize(self) -> dict | None:
        """Return the response data for easy access."""
        return self.response_data


class SubmissionStatusSynapse(bt.Synapse):
    """
    Synapse for polling async submission results.

    After submitting a PromptSynapse and receiving a submission_token,
    miners use this synapse to poll for processing results.

    Attributes:
        submission_token: The token received from PromptSynapse submission
        status: Current submission status (PENDING, PROCESSING, COMPLETED, FAILED, TIMEOUT)
        result: Experiment-specific result blob (filled when COMPLETED)
        error_message: Error details (filled when FAILED/TIMEOUT)
        experiment_id: Which experiment processed this submission
        created_at: ISO timestamp when submission was created
        completed_at: ISO timestamp when processing finished
    """

    # Input from miner
    submission_token: str = Field(
        ...,
        title="Submission Token",
        description="The token received from PromptSynapse submission",
    )

    # Output from validator
    status: str | None = Field(
        None,
        title="Status",
        description="Current status: PENDING, PROCESSING, COMPLETED, FAILED, TIMEOUT",
    )

    result: dict[str, Any] | None = Field(
        None,
        title="Result",
        description="Experiment-specific result data (filled when COMPLETED)",
    )

    error_message: str | None = Field(
        None,
        title="Error Message",
        description="Error details if processing failed",
    )

    experiment_id: str | None = Field(
        None,
        title="Experiment ID",
        description="Which experiment processed this submission",
    )

    created_at: str | None = Field(
        None,
        title="Created At",
        description="ISO timestamp when submission was created",
    )

    completed_at: str | None = Field(
        None,
        title="Completed At",
        description="ISO timestamp when processing finished",
    )

    def deserialize(self) -> dict | None:
        """Return the result for easy access."""
        return self.result
