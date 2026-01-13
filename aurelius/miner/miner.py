"""Miner implementation - submits prompts to validators."""

import argparse
import sys
import time
from dataclasses import dataclass

import bittensor as bt

from aurelius.miner.errors import (
    DiagnosticsFormatter,
    ErrorCategory,
    classify_error,
    create_error,
)
from aurelius.miner.health import ValidatorHealthChecker
from aurelius.miner.retry import RetryConfig, RetryHandler
from aurelius.shared.config import Config, ConfigurationError
from aurelius.shared.protocol import PromptSynapse


@dataclass
class ValidatorQueryResult:
    """Result from a single validator query."""

    uid: int
    success: bool
    synapse: PromptSynapse | None = None
    error_message: str | None = None
    latency_ms: float = 0.0


class QueryProgress:
    """Displays progress during multi-validator queries."""

    def __init__(self, validator_uids: list[int], use_colors: bool = True):
        """
        Initialize progress tracker.

        Args:
            validator_uids: List of validator UIDs being queried
            use_colors: Whether to use ANSI color codes
        """
        self.validator_uids = validator_uids
        self.total = len(validator_uids)
        self.use_colors = use_colors
        self.results: dict[int, ValidatorQueryResult] = {}
        self.start_time = time.time()

        # ANSI color codes
        self.GREEN = "\033[92m" if use_colors else ""
        self.RED = "\033[91m" if use_colors else ""
        self.YELLOW = "\033[93m" if use_colors else ""
        self.CYAN = "\033[96m" if use_colors else ""
        self.BOLD = "\033[1m" if use_colors else ""
        self.RESET = "\033[0m" if use_colors else ""

    def show_start(self) -> None:
        """Display initial query message."""
        print(f"\n{self.CYAN}Querying {self.total} validator(s): {self.validator_uids}{self.RESET}")

    def record_result(self, result: ValidatorQueryResult) -> None:
        """Record a validator result."""
        self.results[result.uid] = result

    def show_health_check_progress(self, uid: int, healthy: bool, latency_ms: float | None = None) -> None:
        """Display health check result for a validator."""
        if healthy:
            latency_str = f" ({latency_ms:.0f}ms)" if latency_ms else ""
            print(f"  {self.GREEN}✓{self.RESET} UID {uid} reachable{latency_str}")
        else:
            print(f"  {self.RED}✗{self.RESET} UID {uid} unreachable")

    def show_query_complete(self) -> None:
        """Display summary after all queries complete."""
        elapsed = time.time() - self.start_time
        successful = sum(1 for r in self.results.values() if r.success)
        failed = len(self.results) - successful

        # Progress bar
        bar_width = 20
        filled = int(bar_width * successful / max(self.total, 1))
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"\n{self.BOLD}Query complete{self.RESET} [{bar}] {successful}/{self.total} ({elapsed:.1f}s)")

        # Show per-validator status
        for uid in self.validator_uids:
            if uid in self.results:
                result = self.results[uid]
                if result.success:
                    latency_str = f" ({result.latency_ms:.0f}ms)" if result.latency_ms > 0 else ""
                    print(f"  {self.GREEN}✓{self.RESET} UID {uid} responded{latency_str}")
                else:
                    error_str = f": {result.error_message}" if result.error_message else ""
                    print(f"  {self.RED}✗{self.RESET} UID {uid} failed{error_str}")
            else:
                print(f"  {self.YELLOW}?{self.RESET} UID {uid} not queried")


def discover_validators(
    metagraph,
    min_stake: float = 0.0,
    max_count: int | None = None,
) -> list[int]:
    """
    Discover eligible validators from metagraph, sorted by stake (highest first).

    Args:
        metagraph: Bittensor metagraph object
        min_stake: Minimum stake required (default: 0.0)
        max_count: Maximum number of validators to return (default: None = all)

    Returns:
        List of validator UIDs sorted by stake (highest first) that:
        - Have stake >= min_stake
        - Are currently serving (axon.is_serving)
    """
    # Collect eligible validators with their stakes
    eligible: list[tuple[int, float]] = []
    for uid in range(len(metagraph.hotkeys)):
        # Get stake - handle different metagraph implementations
        stake = metagraph.S[uid]
        if hasattr(stake, "item"):
            stake = stake.item()  # Handle numpy/torch tensors
        stake = float(stake)

        if stake < min_stake:
            continue

        # Check if validator is serving
        axon = metagraph.axons[uid]
        if not axon.is_serving:
            continue

        eligible.append((uid, stake))

    # Sort by stake descending (highest first)
    eligible.sort(key=lambda x: x[1], reverse=True)

    # Extract UIDs only
    sorted_uids = [uid for uid, _ in eligible]

    # Limit count if specified
    if max_count and len(sorted_uids) > max_count:
        return sorted_uids[:max_count]

    return sorted_uids


def send_prompt(
    prompt: str,
    validator_uid: int = 0,
    vendor: str | None = None,
    model_requested: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    skip_preflight: bool = False,
    use_colors: bool | None = None,
    verbose: bool = False,
) -> str:
    """
    Send a prompt to a validator and return the response.

    Args:
        prompt: The prompt text to send
        validator_uid: The UID of the validator to query (default: 0)
        vendor: AI vendor to use (e.g., 'openai', 'anthropic')
        model_requested: Specific model to use (e.g., 'o4-mini', 'gpt-4o')
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        min_chars: Minimum response length in characters
        max_chars: Maximum response length in characters
        timeout: Query timeout in seconds (default: from config)
        max_retries: Maximum retry attempts (default: from config)
        skip_preflight: Skip pre-flight health check
        use_colors: Use colored output for diagnostics (default: from config)
        verbose: Enable verbose output (default: False for minimal output)

    Returns:
        The response from the validator (OpenAI completion)
    """
    # Setup logging
    Config.setup_logging()

    # Apply network-aware defaults based on BT_NETUID
    Config.apply_network_defaults()

    # Get effective configuration
    effective_timeout = timeout if timeout is not None else Config.MINER_TIMEOUT
    effective_retries = max_retries if max_retries is not None else Config.MINER_MAX_RETRIES
    effective_colors = use_colors if use_colors is not None else Config.MINER_COLORED_OUTPUT
    do_preflight = not skip_preflight and Config.MINER_PREFLIGHT_CHECK

    # Initialize components
    formatter = DiagnosticsFormatter(use_colors=effective_colors)
    health_checker = ValidatorHealthChecker(timeout=Config.MINER_PREFLIGHT_TIMEOUT)
    retry_handler = RetryHandler(
        RetryConfig(
            max_attempts=effective_retries,
            initial_delay_seconds=Config.MINER_RETRY_DELAY,
            max_delay_seconds=Config.MINER_RETRY_MAX_DELAY,
        )
    )

    # Context for error messages
    context = {
        "netuid": Config.BT_NETUID,
        "network": Config.BT_NETWORK,
        "timeout": effective_timeout,
        "max_length": Config.MAX_PROMPT_LENGTH,
    }

    # Detect wallet if not explicitly configured
    try:
        Config.detect_and_set_wallet(role="miner")
    except ConfigurationError as e:
        error = create_error(
            ErrorCategory.INVALID_WALLET,
            {
                **context,
                "wallet": Config.MINER_WALLET_NAME,
                "hotkey": Config.MINER_HOTKEY,
            },
            e,
        )
        if verbose:
            print(formatter.format_error(error, context))
        else:
            print(f"[ERROR] Invalid wallet configuration: {e}")
        sys.exit(1)

    if verbose:
        bt.logging.info(f"Initializing miner with wallet: {Config.MINER_WALLET_NAME}")

    # Initialize wallet
    try:
        wallet = bt.Wallet(name=Config.MINER_WALLET_NAME, hotkey=Config.MINER_HOTKEY)
    except Exception as e:
        error = create_error(
            ErrorCategory.INVALID_WALLET,
            {
                **context,
                "wallet": Config.MINER_WALLET_NAME,
                "hotkey": Config.MINER_HOTKEY,
            },
            e,
        )
        if verbose:
            print(formatter.format_error(error, context))
        else:
            print(f"[ERROR] Wallet initialization failed: {e}")
        return ""

    # Initialize dendrite (for sending requests)
    try:
        dendrite = bt.Dendrite(wallet=wallet)
    except Exception as e:
        error = classify_error(e, context)
        if verbose:
            print(formatter.format_error(error, context))
        else:
            print(f"[ERROR] Dendrite initialization failed: {e}")
        return ""

    if verbose:
        bt.logging.info(f"Prompt: {prompt}")
        if vendor or model_requested:
            bt.logging.info(f"Model specs: vendor={vendor}, model={model_requested}")
        if temperature is not None:
            bt.logging.info(f"Temperature: {temperature}")

    # Create the synapse with the prompt and model specifications
    synapse = PromptSynapse(
        prompt=prompt,
        vendor=vendor,
        model_requested=model_requested,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        min_chars=min_chars,
        max_chars=max_chars,
    )

    # Determine target axon based on mode
    target_axon = None

    if Config.LOCAL_MODE:
        # Local mode: Connect directly to validator IP:PORT
        if verbose:
            bt.logging.info("=" * 60)
            bt.logging.info("LOCAL MODE ENABLED")
            bt.logging.info("=" * 60)
            bt.logging.info(
                f"Connecting directly to validator at {Config.VALIDATOR_HOST}:{Config.BT_PORT_VALIDATOR}"
            )

        context.update(
            {
                "host": Config.VALIDATOR_HOST,
                "port": Config.BT_PORT_VALIDATOR,
            }
        )

        # Create a custom axon info for the local validator
        target_axon = bt.AxonInfo(
            version=1,
            ip=Config.VALIDATOR_HOST,
            port=Config.BT_PORT_VALIDATOR,
            ip_type=4,
            hotkey="local_validator",  # Dummy hotkey for local mode
            coldkey="local_coldkey",
        )
    else:
        # Normal mode: Query metagraph for validator UID
        context["uid"] = validator_uid

        try:
            subtensor_config = Config.get_subtensor_config()
            subtensor = bt.Subtensor(**subtensor_config)
            metagraph = subtensor.metagraph(netuid=Config.BT_NETUID)

            # Validate UID exists
            if validator_uid < 0 or validator_uid >= len(metagraph.axons):
                error = create_error(ErrorCategory.INVALID_VALIDATOR_UID, context)
                if verbose:
                    print(formatter.format_error(error, context))
                else:
                    print(f"[ERROR] Invalid validator UID: {validator_uid}")
                return ""

            target_axon = metagraph.axons[validator_uid]
            context.update(
                {
                    "host": target_axon.ip,
                    "port": target_axon.port,
                }
            )

            if verbose:
                bt.logging.info(f"Sending prompt to validator UID {validator_uid}")

        except IndexError:
            error = create_error(ErrorCategory.INVALID_VALIDATOR_UID, context)
            if verbose:
                print(formatter.format_error(error, context))
            else:
                print(f"[ERROR] Invalid validator UID: {validator_uid}")
            return ""
        except Exception as e:
            error = classify_error(e, context)
            if verbose:
                print(formatter.format_error(error, context))
            else:
                print(f"[ERROR] Network error: {e}")
            return ""

    # Pre-flight health check
    if do_preflight:
        health_result = health_checker.check_validator_reachable(
            host=context["host"],
            port=context["port"],
        )
        if verbose:
            print(
                formatter.format_health_check(
                    is_healthy=health_result.is_healthy,
                    latency_ms=health_result.latency_ms,
                    error=health_result.error,
                    context=context,
                )
            )

        if not health_result.is_healthy:
            if not verbose:
                print(f"[ERROR] Validator unreachable at {context['host']}:{context['port']}")
            return ""

    # Define the query operation for retry
    def do_query():
        return dendrite.query(
            axons=target_axon,
            synapse=synapse,
            timeout=effective_timeout,
            deserialize=False,  # Get full synapse back, not just deserialized string
        )

    # Execute with retry
    result = retry_handler.execute_with_retry(
        operation=do_query,
        context=context,
        on_retry=lambda attempt, error, delay: (
            print(formatter.format_retry_progress(attempt, effective_retries, delay, error, context))
            if verbose
            else None
        ),
    )

    if not result.success:
        if verbose:
            print(formatter.format_error(result.last_error, context))
        else:
            error_msg = result.last_error.message if result.last_error else "Unknown error"
            print(f"[ERROR] Query failed: {error_msg}")
        return ""

    # Process response
    responses = result.result

    # dendrite.query can return different types:
    # - Sometimes the deserialized response (string from synapse.deserialize())
    # - Sometimes the synapse object itself
    # - Sometimes a list

    # Extract the synapse from response
    result_synapse = None

    # dendrite.query returns the synapse itself when querying a single axon
    # For lists, it returns a list of synapses
    if isinstance(responses, PromptSynapse):
        result_synapse = responses
    elif isinstance(responses, list) and len(responses) > 0:
        result_synapse = responses[0]
    elif isinstance(responses, str):
        # Shouldn't happen with single axon query, but handle it
        # Create a dummy synapse with the response
        result_synapse = PromptSynapse(prompt=prompt)
        result_synapse.response = responses
    else:
        result_synapse = responses if responses else synapse

    # Check for validator-side errors
    if result_synapse and hasattr(result_synapse, "rejection_reason") and result_synapse.rejection_reason:
        rejection = result_synapse.rejection_reason.lower()

        if "rate limit" in rejection:
            error = create_error(ErrorCategory.RATE_LIMITED, context)
            context["reason"] = result_synapse.rejection_reason
            if verbose:
                print(formatter.format_error(error, context))
                print(f"Rejection: {result_synapse.rejection_reason}")
            else:
                print(f"[ERROR] Rate limited: {result_synapse.rejection_reason}")
            return ""
        elif "error" in rejection:
            error = create_error(ErrorCategory.API_ERROR, context)
            context["reason"] = result_synapse.rejection_reason
            if verbose:
                print(formatter.format_error(error, context))
            else:
                print(f"[ERROR] {result_synapse.rejection_reason}")
            return ""

    # Check for response with Error prefix
    if (
        result_synapse
        and hasattr(result_synapse, "response")
        and result_synapse.response
        and result_synapse.response.startswith("Error:")
    ):
        error = create_error(ErrorCategory.API_ERROR, context)
        if verbose:
            print(formatter.format_error(error, context))
            print(f"Validator response: {result_synapse.response}")
        else:
            print(f"[ERROR] {result_synapse.response}")
        return ""

    if result_synapse and hasattr(result_synapse, "response") and result_synapse.response:
        if verbose:
            # Full verbose output
            print("\n" + "=" * 60)
            print("RESPONSE FROM VALIDATOR")
            print("=" * 60)
            print(f"Prompt:   {result_synapse.prompt}")
            print(f"Response: {result_synapse.response}")
            print(f"Model:    {result_synapse.model_used or 'unknown'}")

            # Display moderation results if available
            if result_synapse.danger_score is not None:
                print("\n--- Moderation Results ---")
                print(f"Danger Score:  {result_synapse.danger_score:.4f}")
                print(f"Accepted:      {'✓ YES' if result_synapse.accepted else '✗ NO'}")

                if result_synapse.rejection_reason:
                    print(f"Rejection:     {result_synapse.rejection_reason}")

                if result_synapse.category_scores:
                    print("\nTop Category Scores:")
                    sorted_cats = sorted(
                        result_synapse.category_scores.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    for category, score in sorted_cats:
                        if score > 0.01:
                            print(f"  {category:25s}: {score:.4f}")

            # Display miner statistics if available
            if result_synapse.miner_submission_count is not None:
                print("\n--- Your Statistics ---")
                print(f"Submissions:   {result_synapse.miner_submission_count}")
                if result_synapse.miner_hit_rate is not None:
                    print(f"Hit Rate:      {result_synapse.miner_hit_rate:.1%}")
                if result_synapse.miner_novelty_avg is not None:
                    print(f"Avg Novelty:   {result_synapse.miner_novelty_avg:.3f}")

            # Display reward mechanism info
            _display_reward_info(result_synapse, verbose=True)

            print("=" * 60 + "\n")
        else:
            # Minimal non-verbose output
            _display_minimal_result(result_synapse)

        return result_synapse.response
    else:
        # No response received - create appropriate error
        error = create_error(
            ErrorCategory.VALIDATOR_BUSY,
            context,
        )
        # Customize for no response case
        error.message = "No response received from validator"
        error.cause = (
            "The validator did not return a response. "
            "This could indicate the validator is overloaded, misconfigured, "
            "or the request was silently dropped."
        )
        error.suggestions = [
            "Check validator logs for errors",
            "Try a different validator: --validator-uid <other_uid>",
            "Verify the validator is running and accepting requests",
            "Increase timeout: --timeout 60",
        ]
        if verbose:
            print(formatter.format_error(error, context))
        else:
            print("[ERROR] No response received from validator")
        return ""


def query_validators_batched(
    dendrite,
    axons: list,
    synapse: PromptSynapse,
    timeout: int,
    batch_size: int = 3,
    batch_delay_ms: int = 100,
    verbose: bool = False,
) -> list:
    """
    Query validators in batches to avoid network congestion.

    When querying many validators simultaneously, network saturation can cause
    dropped packets and connection resets. This function splits the validators
    into smaller batches and queries them sequentially with a small delay.

    Args:
        dendrite: Bittensor dendrite instance
        axons: List of AxonInfo objects to query
        synapse: The synapse to send
        timeout: Query timeout in seconds (per batch)
        batch_size: Number of validators to query per batch
        batch_delay_ms: Delay between batches in milliseconds
        verbose: Enable verbose logging

    Returns:
        List of responses in the same order as axons
    """
    all_responses: list = [None] * len(axons)
    total_batches = (len(axons) + batch_size - 1) // batch_size

    # Process in batches
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(axons))
        batch_axons = axons[batch_start:batch_end]

        if verbose:
            bt.logging.info(
                f"Querying batch {batch_idx + 1}/{total_batches}: "
                f"validators {batch_start + 1}-{batch_end} of {len(axons)}"
            )

        # Query this batch
        batch_responses = dendrite.query(
            axons=batch_axons,
            synapse=synapse,
            timeout=timeout,
            deserialize=False,
        )

        # Handle single axon case (returns synapse, not list)
        if not isinstance(batch_responses, list):
            batch_responses = [batch_responses]

        # Store responses in correct positions
        for i, response in enumerate(batch_responses):
            all_responses[batch_start + i] = response

        # Inter-batch delay (skip after last batch)
        if batch_end < len(axons) and batch_delay_ms > 0:
            time.sleep(batch_delay_ms / 1000.0)

    return all_responses


def send_prompt_multi(
    prompt: str,
    validator_uids: list[int] | None = None,
    max_validators: int | None = None,
    min_stake: float | None = None,
    vendor: str | None = None,
    model_requested: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    skip_preflight: bool = False,
    use_colors: bool | None = None,
    verbose: bool = False,
    batch_size: int | None = None,
    batch_delay_ms: int | None = None,
    no_batching: bool = False,
) -> dict[int, ValidatorQueryResult]:
    """
    Send a prompt to multiple validators in parallel (or batched).

    Args:
        prompt: The prompt text to send
        validator_uids: Specific validator UIDs to query (None = auto-discover)
        max_validators: Max validators to query when auto-discovering
        min_stake: Minimum validator stake requirement for auto-discovery
        vendor: AI vendor to use (e.g., 'openai', 'anthropic')
        model_requested: Specific model to use
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        min_chars: Minimum response length in characters
        max_chars: Maximum response length in characters
        timeout: Query timeout in seconds
        max_retries: Maximum retry attempts for the query (default: from config)
        skip_preflight: Skip pre-flight health checks
        use_colors: Use colored output for diagnostics
        verbose: Enable verbose output (default: False for minimal output)
        batch_size: Number of validators to query per batch (default: from config)
        batch_delay_ms: Delay between batches in milliseconds (default: from config)
        no_batching: Disable batching and query all validators at once

    Returns:
        Dict mapping validator UID to ValidatorQueryResult.
        Includes both successful and failed results with error details.
    """
    # Setup logging
    Config.setup_logging()
    Config.apply_network_defaults()

    # Get effective configuration
    effective_timeout = timeout if timeout is not None else Config.MINER_TIMEOUT
    effective_retries = max_retries if max_retries is not None else Config.MINER_MAX_RETRIES
    effective_max = max_validators if max_validators is not None else Config.MINER_MAX_VALIDATORS
    effective_stake = min_stake if min_stake is not None else Config.MINER_MIN_VALIDATOR_STAKE
    effective_colors = use_colors if use_colors is not None else Config.MINER_COLORED_OUTPUT

    formatter = DiagnosticsFormatter(use_colors=effective_colors)
    retry_handler = RetryHandler(
        RetryConfig(
            max_attempts=effective_retries,
            initial_delay_seconds=Config.MINER_RETRY_DELAY,
            max_delay_seconds=Config.MINER_RETRY_MAX_DELAY,
        )
    )

    context = {
        "netuid": Config.BT_NETUID,
        "network": Config.BT_NETWORK,
        "timeout": effective_timeout,
    }

    # Detect wallet if not explicitly configured
    try:
        Config.detect_and_set_wallet(role="miner")
    except ConfigurationError as e:
        error = create_error(
            ErrorCategory.INVALID_WALLET,
            {**context, "wallet": Config.MINER_WALLET_NAME, "hotkey": Config.MINER_HOTKEY},
            e,
        )
        if verbose:
            print(formatter.format_error(error, context))
        else:
            print(f"[ERROR] Invalid wallet configuration: {e}")
        return {}

    if verbose:
        bt.logging.info(f"Initializing miner with wallet: {Config.MINER_WALLET_NAME}")

    # Initialize wallet
    try:
        wallet = bt.Wallet(name=Config.MINER_WALLET_NAME, hotkey=Config.MINER_HOTKEY)
    except Exception as e:
        error = create_error(
            ErrorCategory.INVALID_WALLET,
            {**context, "wallet": Config.MINER_WALLET_NAME, "hotkey": Config.MINER_HOTKEY},
            e,
        )
        if verbose:
            print(formatter.format_error(error, context))
        else:
            print(f"[ERROR] Wallet initialization failed: {e}")
        return {}

    # Initialize dendrite
    try:
        dendrite = bt.Dendrite(wallet=wallet)
    except Exception as e:
        error = classify_error(e, context)
        if verbose:
            print(formatter.format_error(error, context))
        else:
            print(f"[ERROR] Dendrite initialization failed: {e}")
        return {}

    # Get metagraph and discover validators
    try:
        subtensor_config = Config.get_subtensor_config()
        subtensor = bt.Subtensor(**subtensor_config)
        metagraph = subtensor.metagraph(netuid=Config.BT_NETUID)
    except Exception as e:
        error = classify_error(e, context)
        if verbose:
            print(formatter.format_error(error, context))
        else:
            print(f"[ERROR] Network error: {e}")
        return {}

    # Determine which validators to query
    if validator_uids is not None:
        # Use specified UIDs
        target_uids = [uid for uid in validator_uids if 0 <= uid < len(metagraph.axons)]
        if not target_uids:
            if verbose:
                bt.logging.error("No valid validator UIDs provided")
            else:
                print("[ERROR] No valid validator UIDs provided")
            return {}
    else:
        # Auto-discover validators
        target_uids = discover_validators(
            metagraph=metagraph,
            min_stake=effective_stake,
            max_count=effective_max,
        )

    if not target_uids:
        if verbose:
            bt.logging.error("No eligible validators found")
        print("[ERROR] No eligible validators found")
        return {}

    # Initialize progress tracker (only used in verbose mode)
    progress = QueryProgress(target_uids, use_colors=effective_colors)
    if verbose:
        progress.show_start()

    # Pre-flight health checks (optional)
    do_preflight = not skip_preflight and Config.MINER_PREFLIGHT_CHECK
    if do_preflight:
        if verbose:
            print("Running pre-flight health checks...")
        health_checker = ValidatorHealthChecker(timeout=Config.MINER_PREFLIGHT_TIMEOUT)
        healthy_uids = []
        unhealthy_results: dict[int, ValidatorQueryResult] = {}

        for uid in target_uids:
            axon = metagraph.axons[uid]
            health_result = health_checker.check_validator_reachable(axon.ip, axon.port)
            if verbose:
                progress.show_health_check_progress(uid, health_result.is_healthy, health_result.latency_ms)

            if health_result.is_healthy:
                healthy_uids.append(uid)
            else:
                # Record failed health check as a result
                error_msg = str(health_result.error.message) if health_result.error else "Health check failed"
                unhealthy_results[uid] = ValidatorQueryResult(
                    uid=uid,
                    success=False,
                    error_message=f"Unreachable: {error_msg}",
                )

        if not healthy_uids:
            print("[ERROR] No validators passed health check")
            return unhealthy_results

        # Update target_uids to only include healthy validators
        target_uids = healthy_uids
        if verbose:
            print(f"{len(healthy_uids)} validator(s) passed health check")
    else:
        unhealthy_results = {}

    if verbose:
        bt.logging.info(f"Querying {len(target_uids)} validator(s): {target_uids}")

    # Get axons for target validators
    target_axons = [metagraph.axons[uid] for uid in target_uids]

    # Create synapse
    synapse = PromptSynapse(
        prompt=prompt,
        vendor=vendor,
        model_requested=model_requested,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        min_chars=min_chars,
        max_chars=max_chars,
    )

    # Determine batching configuration
    effective_batch_size = batch_size if batch_size is not None else Config.MINER_QUERY_BATCH_SIZE
    effective_batch_delay = batch_delay_ms if batch_delay_ms is not None else Config.MINER_BATCH_DELAY_MS
    enable_batching = not no_batching and Config.MINER_ENABLE_BATCHING

    # Define the query operation for retry
    def do_query():
        if enable_batching and len(target_axons) > effective_batch_size:
            # Use batched queries to avoid network congestion
            if verbose:
                bt.logging.info(
                    f"Using batched queries: {effective_batch_size} validators per batch, "
                    f"{effective_batch_delay}ms delay"
                )
            return query_validators_batched(
                dendrite=dendrite,
                axons=target_axons,
                synapse=synapse,
                timeout=effective_timeout,
                batch_size=effective_batch_size,
                batch_delay_ms=effective_batch_delay,
                verbose=verbose,
            )
        else:
            # Original behavior: query all at once
            return dendrite.query(
                axons=target_axons,
                synapse=synapse,
                timeout=effective_timeout,
                deserialize=False,
            )

    # Query validators (batched or parallel) with retry
    if verbose:
        print("Sending queries...")
    query_start_time = time.time()
    result = retry_handler.execute_with_retry(
        operation=do_query,
        context=context,
        on_retry=lambda attempt, error, delay: (
            bt.logging.warning(
                f"Query failed (attempt {attempt}/{effective_retries}), retrying in {delay:.1f}s: {error}"
            )
            if verbose
            else None
        ),
    )

    query_end_time = time.time()
    query_duration_ms = (query_end_time - query_start_time) * 1000

    if not result.success:
        error = classify_error(result.last_error, context) if result.last_error else None
        if verbose and error:
            print(formatter.format_error(error, context))
        elif not verbose:
            error_msg = result.last_error.message if result.last_error else "Unknown error"
            print(f"[ERROR] Query failed: {error_msg}")
        # Return unhealthy results from health check even if query failed
        return unhealthy_results

    responses = result.result

    # Process responses - track both successful and failed
    results: dict[int, ValidatorQueryResult] = {}

    # Handle case where responses might not be a list
    if not isinstance(responses, list):
        responses = [responses]

    for uid, response in zip(target_uids, responses):
        if response is None:
            query_result = ValidatorQueryResult(
                uid=uid,
                success=False,
                error_message="No response (timeout or error)",
            )
            results[uid] = query_result
            progress.record_result(query_result)
            continue

        # Handle different response types
        if isinstance(response, PromptSynapse):
            result_synapse = response
        elif isinstance(response, str):
            result_synapse = PromptSynapse(prompt=prompt)
            result_synapse.response = response
        else:
            query_result = ValidatorQueryResult(
                uid=uid,
                success=False,
                error_message=f"Unexpected response type: {type(response).__name__}",
            )
            results[uid] = query_result
            progress.record_result(query_result)
            continue

        # Check for valid response
        if result_synapse.response and not result_synapse.response.startswith("Error:"):
            # Check for rejection reasons
            if result_synapse.rejection_reason:
                rejection = result_synapse.rejection_reason.lower()
                if "rate limit" in rejection or "error" in rejection:
                    bt.logging.debug(f"Validator {uid} rejected: {result_synapse.rejection_reason}")
                    query_result = ValidatorQueryResult(
                        uid=uid,
                        success=False,
                        synapse=result_synapse,
                        error_message=result_synapse.rejection_reason,
                    )
                    results[uid] = query_result
                    progress.record_result(query_result)
                    continue

            # Success
            query_result = ValidatorQueryResult(
                uid=uid,
                success=True,
                synapse=result_synapse,
                latency_ms=query_duration_ms / len(target_uids),  # Approximate per-validator
            )
            results[uid] = query_result
            progress.record_result(query_result)
        else:
            error_msg = "No response content"
            if result_synapse.response and result_synapse.response.startswith("Error:"):
                error_msg = result_synapse.response
            query_result = ValidatorQueryResult(
                uid=uid,
                success=False,
                synapse=result_synapse,
                error_message=error_msg,
            )
            results[uid] = query_result
            progress.record_result(query_result)

    # Merge with unhealthy results from health check
    all_results = {**unhealthy_results, **results}

    # Show progress summary (verbose only)
    if verbose:
        progress.show_query_complete()
        successful_count = sum(1 for r in all_results.values() if r.success)
        bt.logging.info(f"Received {successful_count} successful response(s) from {len(all_results)} validator(s)")

    return all_results


def display_multi_results(
    results: dict[int, ValidatorQueryResult],
    use_colors: bool = True,
    verbose: bool = False,
) -> None:
    """
    Display results from multiple validators.

    Args:
        results: Dict mapping validator UID to ValidatorQueryResult
        use_colors: Use colored output
        verbose: Enable verbose output (default: False for minimal output)
    """
    # ANSI color codes
    GREEN = "\033[92m" if use_colors else ""
    RED = "\033[91m" if use_colors else ""
    YELLOW = "\033[93m" if use_colors else ""
    CYAN = "\033[96m" if use_colors else ""
    BOLD = "\033[1m" if use_colors else ""
    RESET = "\033[0m" if use_colors else ""

    if not results:
        print("[ERROR] No results from validators")
        return

    # Separate successful and failed results
    successful = {uid: r for uid, r in results.items() if r.success}
    failed = {uid: r for uid, r in results.items() if not r.success}

    if not verbose:
        # Minimal output: just summary of accepted/rejected counts
        if failed:
            print(f"Failed: {len(failed)} validator(s)")

        if successful:
            # Show aggregate result
            accepted_count = sum(
                1 for r in successful.values() if r.synapse and r.synapse.accepted
            )
            print(f"Results: {accepted_count}/{len(successful)} accepted across {len(successful)} validator(s)")

            # Show reward info from first successful response
            for r in successful.values():
                if r.synapse:
                    _display_reward_info(r.synapse, verbose=False)
                    break
        return

    # Verbose output - show all details
    # Summary header
    print("\n" + "=" * 60)
    print(f"{BOLD}RESULTS: {len(successful)}/{len(results)} VALIDATOR(S) RESPONDED{RESET}")
    print("=" * 60)

    # Show failed validators first (brief)
    if failed:
        print(f"\n{RED}Failed validators:{RESET}")
        for uid, result in failed.items():
            print(f"  UID {uid}: {result.error_message or 'Unknown error'}")

    # Show successful responses
    if successful:
        print(f"\n{GREEN}Successful responses:{RESET}")
        for uid, result in successful.items():
            synapse = result.synapse
            if not synapse:
                continue

            print(f"\n{CYAN}--- Validator UID {uid} ---{RESET}")

            if synapse.response:
                # Truncate long responses for display
                response_display = synapse.response
                if len(response_display) > 500:
                    response_display = response_display[:500] + "... [truncated]"
                print(f"Response: {response_display}")

            print(f"Model:    {synapse.model_used or 'unknown'}")

            if synapse.danger_score is not None:
                if synapse.accepted:
                    accepted_str = f"{GREEN}YES{RESET}"
                else:
                    accepted_str = f"{RED}NO{RESET}"
                print(f"Danger:   {synapse.danger_score:.4f} (Accepted: {accepted_str})")

            if synapse.miner_submission_count is not None:
                print(f"Stats:    {synapse.miner_submission_count} submissions", end="")
                if synapse.miner_hit_rate is not None:
                    print(f", {synapse.miner_hit_rate:.1%} hit rate", end="")
                print()

            # Show reward info for each validator in verbose mode
            _display_reward_info(synapse, verbose=True)

    print("\n" + "=" * 60)

    # Summary statistics (only from successful responses)
    if len(successful) > 1:
        danger_scores = [
            r.synapse.danger_score
            for r in successful.values()
            if r.synapse and r.synapse.danger_score is not None
        ]
        if danger_scores:
            avg_danger = sum(danger_scores) / len(danger_scores)
            print(f"Average Danger Score: {avg_danger:.4f}")
            print(f"Min: {min(danger_scores):.4f}, Max: {max(danger_scores):.4f}")
        print("=" * 60 + "\n")


def _display_reward_info(synapse: PromptSynapse, verbose: bool = False) -> None:
    """
    Display reward mechanism info from validator.

    Args:
        synapse: The response synapse containing reward info
        verbose: If True, show detailed breakdown; if False, show one-line summary
    """
    # Check if reward info is available
    if synapse.reward_top_miners_count is None:
        return

    if verbose:
        print("\n--- Reward Mechanism ---")
        print(f"Top Miners:    {synapse.reward_top_miners_count}")
        if synapse.reward_burn_enabled:
            burn_pct = (synapse.reward_burn_percentage or 0) * 100
            print(f"Burn Rate:     {burn_pct:.0f}%")
        else:
            print("Burn Rate:     Disabled")
        if synapse.reward_min_hit_rate is not None:
            print(f"Min Hit Rate:  {synapse.reward_min_hit_rate:.0%}")
        if synapse.reward_min_novelty is not None:
            print(f"Min Novelty:   {synapse.reward_min_novelty:.0%}")
        if synapse.reward_danger_threshold is not None:
            print(f"Danger Thresh: {synapse.reward_danger_threshold}")
    else:
        # One-line summary
        parts = [f"Top {synapse.reward_top_miners_count} miners"]
        if synapse.reward_burn_enabled and synapse.reward_burn_percentage:
            parts.append(f"{synapse.reward_burn_percentage:.0%} burn")
        if synapse.reward_min_hit_rate is not None:
            parts.append(f"Min hit rate: {synapse.reward_min_hit_rate:.0%}")
        if synapse.reward_min_novelty is not None:
            parts.append(f"Min novelty: {synapse.reward_min_novelty:.0%}")

        print(f"Rewards: {' | '.join(parts)}")


def _display_minimal_result(synapse: PromptSynapse) -> None:
    """
    Display minimal result for non-verbose mode.

    Args:
        synapse: The response synapse
    """
    danger = synapse.danger_score or 0.0
    novelty = synapse.miner_novelty_avg

    if synapse.accepted:
        novelty_str = f", novelty: {novelty:.2f}" if novelty is not None else ""
        print(f"Prompt accepted (danger: {danger:.2f}{novelty_str})")
    else:
        threshold = synapse.reward_danger_threshold or 0.3
        if synapse.rejection_reason and "rate limit" in synapse.rejection_reason.lower():
            print(f"Prompt rejected: {synapse.rejection_reason}")
        else:
            print(f"Prompt rejected (danger: {danger:.2f} < threshold {threshold:.2f})")

    _display_reward_info(synapse, verbose=False)


def main():
    """Main entry point for the miner."""
    parser = argparse.ArgumentParser(
        description="Bittensor Miner - Submit prompts to validators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-validator mode (default): query all eligible validators
  python miner.py --prompt "test prompt"

  # Limit to 5 validators
  python miner.py --prompt "test prompt" --max-validators 5

  # Single validator mode (backwards compatible)
  python miner.py --prompt "test prompt" --validator-uid 3

  # Force single validator mode with default UID 1
  python miner.py --prompt "test prompt" --single
""",
    )
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to send to the validator")

    # Validator selection arguments
    parser.add_argument(
        "--validator-uid",
        type=int,
        default=None,
        help="Query single validator by UID (disables multi-validator mode)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Force single-validator mode (uses UID 1 if --validator-uid not set)",
    )
    parser.add_argument(
        "--max-validators",
        type=int,
        default=None,
        help=f"Max validators to query in multi-validator mode (default: {Config.MINER_MAX_VALIDATORS})",
    )
    parser.add_argument(
        "--min-stake",
        type=float,
        default=None,
        help=f"Minimum validator stake requirement (default: {Config.MINER_MIN_VALIDATOR_STAKE})",
    )
    parser.add_argument("--netuid", type=int, default=None, help=f"Override the netuid (default: {Config.BT_NETUID})")

    # Model specification arguments
    parser.add_argument("--vendor", type=str, default=None, help="AI vendor to use (e.g., 'openai', 'anthropic')")
    parser.add_argument("--model", type=str, default=None, help="Specific model to use (e.g., 'o4-mini', 'gpt-4o')")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (0.0-2.0)")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling parameter (0.0-1.0)")
    parser.add_argument("--frequency-penalty", type=float, default=None, help="Frequency penalty (-2.0 to 2.0)")
    parser.add_argument("--presence-penalty", type=float, default=None, help="Presence penalty (-2.0 to 2.0)")
    parser.add_argument("--min-chars", type=int, default=None, help="Minimum response length in characters")
    parser.add_argument("--max-chars", type=int, default=None, help="Maximum response length in characters")

    # Connection configuration arguments
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=f"Query timeout in seconds (default: {Config.MINER_TIMEOUT})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help=f"Maximum retry attempts (default: {Config.MINER_MAX_RETRIES})",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip pre-flight health checks",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    # Batching configuration (to prevent network congestion with many validators)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Number of validators to query per batch (default: {Config.MINER_QUERY_BATCH_SIZE})",
    )
    parser.add_argument(
        "--batch-delay",
        type=int,
        default=None,
        help=f"Delay in milliseconds between batches (default: {Config.MINER_BATCH_DELAY_MS})",
    )
    parser.add_argument(
        "--no-batching",
        action="store_true",
        help="Disable batching and query all validators at once (original behavior)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (show all details, preflight checks, category scores)",
    )

    args = parser.parse_args()

    # Override netuid if provided
    if args.netuid is not None:
        Config.BT_NETUID = args.netuid

    use_colors = not args.no_color
    verbose = args.verbose

    # Determine mode: single validator or multi-validator
    # Use single mode if: --validator-uid specified, --single flag, or MINER_MULTI_VALIDATOR=false
    use_single_mode = (
        args.validator_uid is not None
        or args.single
        or not Config.MINER_MULTI_VALIDATOR
    )

    if use_single_mode:
        # Single validator mode (backwards compatible)
        uid = args.validator_uid if args.validator_uid is not None else 1
        send_prompt(
            prompt=args.prompt,
            validator_uid=uid,
            vendor=args.vendor,
            model_requested=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            timeout=args.timeout,
            max_retries=args.retries,
            skip_preflight=args.no_preflight,
            use_colors=use_colors,
            verbose=verbose,
        )
    else:
        # Multi-validator mode (default when MINER_MULTI_VALIDATOR=true)
        results = send_prompt_multi(
            prompt=args.prompt,
            max_validators=args.max_validators,
            min_stake=args.min_stake,
            vendor=args.vendor,
            model_requested=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            max_retries=args.retries,
            timeout=args.timeout,
            skip_preflight=args.no_preflight,
            use_colors=use_colors,
            verbose=verbose,
            batch_size=args.batch_size,
            batch_delay_ms=args.batch_delay,
            no_batching=args.no_batching,
        )
        display_multi_results(results, use_colors=use_colors, verbose=verbose)


if __name__ == "__main__":
    main()
