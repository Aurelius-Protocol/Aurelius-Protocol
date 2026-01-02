"""Miner implementation - submits prompts to validators."""

import argparse
import sys

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
        print(formatter.format_error(error, context))
        sys.exit(1)

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
        print(formatter.format_error(error, context))
        return ""

    # Initialize dendrite (for sending requests)
    try:
        dendrite = bt.Dendrite(wallet=wallet)
    except Exception as e:
        error = classify_error(e, context)
        print(formatter.format_error(error, context))
        return ""

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
                print(formatter.format_error(error, context))
                return ""

            target_axon = metagraph.axons[validator_uid]
            context.update(
                {
                    "host": target_axon.ip,
                    "port": target_axon.port,
                }
            )

            bt.logging.info(f"Sending prompt to validator UID {validator_uid}")

        except IndexError:
            error = create_error(ErrorCategory.INVALID_VALIDATOR_UID, context)
            print(formatter.format_error(error, context))
            return ""
        except Exception as e:
            error = classify_error(e, context)
            print(formatter.format_error(error, context))
            return ""

    # Pre-flight health check
    if do_preflight:
        health_result = health_checker.check_validator_reachable(
            host=context["host"],
            port=context["port"],
        )
        print(
            formatter.format_health_check(
                is_healthy=health_result.is_healthy,
                latency_ms=health_result.latency_ms,
                error=health_result.error,
                context=context,
            )
        )

        if not health_result.is_healthy:
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
        on_retry=lambda attempt, error, delay: print(
            formatter.format_retry_progress(attempt, effective_retries, delay, error, context)
        ),
    )

    if not result.success:
        print(formatter.format_error(result.last_error, context))
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
            print(formatter.format_error(error, context))
            # Still show the rejection reason
            print(f"Rejection: {result_synapse.rejection_reason}")
            return ""
        elif "error" in rejection:
            error = create_error(ErrorCategory.API_ERROR, context)
            context["reason"] = result_synapse.rejection_reason
            print(formatter.format_error(error, context))
            return ""

    # Check for response with Error prefix
    if (
        result_synapse
        and hasattr(result_synapse, "response")
        and result_synapse.response
        and result_synapse.response.startswith("Error:")
    ):
        error = create_error(ErrorCategory.API_ERROR, context)
        print(formatter.format_error(error, context))
        print(f"Validator response: {result_synapse.response}")
        return ""

    if result_synapse and hasattr(result_synapse, "response") and result_synapse.response:
        # Display results
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

        print("=" * 60 + "\n")
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
        print(formatter.format_error(error, context))
        return ""


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
    use_colors: bool | None = None,
) -> dict[int, PromptSynapse]:
    """
    Send a prompt to multiple validators in parallel.

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
        use_colors: Use colored output for diagnostics

    Returns:
        Dict mapping validator UID to response synapse.
        Only successful responses are included.
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
        print(formatter.format_error(error, context))
        return {}

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
        print(formatter.format_error(error, context))
        return {}

    # Initialize dendrite
    try:
        dendrite = bt.Dendrite(wallet=wallet)
    except Exception as e:
        error = classify_error(e, context)
        print(formatter.format_error(error, context))
        return {}

    # Get metagraph and discover validators
    try:
        subtensor_config = Config.get_subtensor_config()
        subtensor = bt.Subtensor(**subtensor_config)
        metagraph = subtensor.metagraph(netuid=Config.BT_NETUID)
    except Exception as e:
        error = classify_error(e, context)
        print(formatter.format_error(error, context))
        return {}

    # Determine which validators to query
    if validator_uids is not None:
        # Use specified UIDs
        target_uids = [uid for uid in validator_uids if 0 <= uid < len(metagraph.axons)]
        if not target_uids:
            bt.logging.error("No valid validator UIDs provided")
            return {}
    else:
        # Auto-discover validators
        target_uids = discover_validators(
            metagraph=metagraph,
            min_stake=effective_stake,
            max_count=effective_max,
        )

    if not target_uids:
        bt.logging.error("No eligible validators found")
        print("No eligible validators found. Check network connectivity and stake thresholds.")
        return {}

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

    # Define the query operation for retry
    def do_query():
        return dendrite.query(
            axons=target_axons,
            synapse=synapse,
            timeout=effective_timeout,
            deserialize=False,
        )

    # Query all validators in parallel with retry
    result = retry_handler.execute_with_retry(
        operation=do_query,
        context=context,
        on_retry=lambda attempt, error, delay: bt.logging.warning(
            f"Query failed (attempt {attempt}/{effective_retries}), retrying in {delay:.1f}s: {error}"
        ),
    )

    if not result.success:
        error = classify_error(result.last_error, context) if result.last_error else None
        if error:
            print(formatter.format_error(error, context))
        return {}

    responses = result.result

    # Process responses - keep only successful ones
    results: dict[int, PromptSynapse] = {}

    # Handle case where responses might not be a list
    if not isinstance(responses, list):
        responses = [responses]

    for uid, response in zip(target_uids, responses):
        if response is None:
            continue

        # Handle different response types
        if isinstance(response, PromptSynapse):
            result_synapse = response
        elif isinstance(response, str):
            result_synapse = PromptSynapse(prompt=prompt)
            result_synapse.response = response
        else:
            continue

        # Check for valid response
        if result_synapse.response and not result_synapse.response.startswith("Error:"):
            # Skip if rate limited or has rejection reason with error
            if result_synapse.rejection_reason:
                rejection = result_synapse.rejection_reason.lower()
                if "rate limit" in rejection or "error" in rejection:
                    bt.logging.debug(f"Validator {uid} rejected: {result_synapse.rejection_reason}")
                    continue

            results[uid] = result_synapse

    bt.logging.info(f"Received {len(results)} successful response(s) from {len(target_uids)} validator(s)")
    return results


def display_multi_results(
    responses: dict[int, PromptSynapse],
    use_colors: bool = True,
) -> None:
    """
    Display results from multiple validators.

    Args:
        responses: Dict mapping validator UID to response synapse
        use_colors: Use colored output
    """
    # ANSI color codes
    GREEN = "\033[92m" if use_colors else ""
    RED = "\033[91m" if use_colors else ""
    YELLOW = "\033[93m" if use_colors else ""
    CYAN = "\033[96m" if use_colors else ""
    BOLD = "\033[1m" if use_colors else ""
    RESET = "\033[0m" if use_colors else ""

    if not responses:
        print(f"\n{YELLOW}No responses received from validators.{RESET}")
        return

    print("\n" + "=" * 60)
    print(f"{BOLD}RESPONSES FROM {len(responses)} VALIDATOR(S){RESET}")
    print("=" * 60)

    for uid, synapse in responses.items():
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

    print("\n" + "=" * 60)

    # Summary statistics
    if len(responses) > 1:
        danger_scores = [s.danger_score for s in responses.values() if s.danger_score is not None]
        if danger_scores:
            avg_danger = sum(danger_scores) / len(danger_scores)
            print(f"Average Danger Score: {avg_danger:.4f}")
            print(f"Min: {min(danger_scores):.4f}, Max: {max(danger_scores):.4f}")
        print("=" * 60 + "\n")


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
        help="Skip pre-flight health check (only applies to single-validator mode)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    # Override netuid if provided
    if args.netuid is not None:
        Config.BT_NETUID = args.netuid

    use_colors = not args.no_color

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
        )
    else:
        # Multi-validator mode (default when MINER_MULTI_VALIDATOR=true)
        responses = send_prompt_multi(
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
            use_colors=use_colors,
        )
        display_multi_results(responses, use_colors=use_colors)


if __name__ == "__main__":
    main()
