"""Miner implementation - submits prompts to validators."""

import argparse
import os
import signal
import sys
import threading
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
from aurelius.miner.registration import (
    list_registrations,
    register_for_experiment,
    withdraw_from_experiment,
)
from aurelius.miner.retry import RetryConfig, RetryHandler
from aurelius.shared.config import Config, ConfigurationError
from aurelius.shared.protocol import PromptSynapse, SubmissionStatusSynapse


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
    experiment_id: str | None = None,
    poll_interval: int = 5,
    max_poll_time: int = 300,
    submit_only: bool = False,
) -> str:
    """
    Send a prompt to a validator using the async token-based flow.

    Phase 1: Submit prompt and receive a submission token instantly.
    Phase 2: Poll the validator for results using the token.

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
        poll_interval: Seconds between status polls (default: 5)
        max_poll_time: Max seconds to poll before giving up (default: 300)
        submit_only: Submit and print token, don't poll for result

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

    bt.logging.info(f"Prompt: {Config.truncate_sensitive_data(prompt)}")
    if vendor or model_requested:
        bt.logging.info(f"Model specs: vendor={vendor}, model={model_requested}")
    if temperature is not None:
        bt.logging.info(f"Temperature: {temperature}")

    # Validate and clamp parameters before synapse construction
    import math

    def _validate_param(
        name: str, value: float | None, lo: float, hi: float
    ) -> float | None:
        if value is None:
            return None
        if math.isnan(value) or math.isinf(value):
            bt.logging.warning(f"Parameter {name}={value} is NaN/Infinity, ignoring")
            return None
        return max(lo, min(hi, value))

    temperature = _validate_param("temperature", temperature, 0.0, 2.0)
    top_p = _validate_param("top_p", top_p, 0.0, 1.0)
    frequency_penalty = _validate_param("frequency_penalty", frequency_penalty, -2.0, 2.0)
    presence_penalty = _validate_param("presence_penalty", presence_penalty, -2.0, 2.0)

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
        experiment_id=experiment_id,
    )

    if experiment_id:
        bt.logging.info(f"Experiment: {experiment_id}")

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
        # Use real validator hotkey if available (env VALIDATOR_HOTKEY_SS58)
        # so that Bittensor's signature verification passes
        local_hotkey = os.environ.get("VALIDATOR_HOTKEY_SS58", "local_validator")
        local_coldkey = os.environ.get("VALIDATOR_COLDKEY_SS58", "local_coldkey")
        target_axon = bt.AxonInfo(
            version=1,
            ip=Config.VALIDATOR_HOST,
            port=Config.BT_PORT_VALIDATOR,
            ip_type=4,
            hotkey=local_hotkey,
            coldkey=local_coldkey,
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

    # --- Phase 1: Submit prompt and get token ---
    def do_submit():
        return dendrite.query(
            axons=target_axon,
            synapse=synapse,
            timeout=10,  # Submit should be fast
            deserialize=False,
        )

    result = retry_handler.execute_with_retry(
        operation=do_submit,
        context=context,
        on_retry=lambda attempt, error, delay: print(
            formatter.format_retry_progress(attempt, effective_retries, delay, error, context)
        ),
    )

    if not result.success:
        print(formatter.format_error(result.last_error, context))
        return ""

    responses = result.result
    result_synapse = None
    if isinstance(responses, PromptSynapse):
        result_synapse = responses
    elif isinstance(responses, list) and len(responses) > 0:
        result_synapse = responses[0]
    else:
        result_synapse = responses if responses else synapse

    # Check for rejection (rate limit, validation errors)
    if result_synapse and hasattr(result_synapse, "rejection_reason") and result_synapse.rejection_reason:
        code = getattr(result_synapse, "rejection_code", None) or ""
        rejection = result_synapse.rejection_reason.lower()
        if code == "RATE_LIMITED" or (not code and "rate limit" in rejection):
            error = create_error(ErrorCategory.RATE_LIMITED, context)
            context["reason"] = result_synapse.rejection_reason
            print(formatter.format_error(error, context))
            return ""
        elif code.startswith("INVALID") or (not code and "error" in rejection):
            error = create_error(ErrorCategory.API_ERROR, context)
            context["reason"] = result_synapse.rejection_reason
            print(formatter.format_error(error, context))
            return ""

    # Extract submission token
    token = getattr(result_synapse, "submission_token", None) if result_synapse else None

    if not token:
        # Fallback: validator may still use legacy sync flow — check for response
        if result_synapse and hasattr(result_synapse, "response") and result_synapse.response:
            _display_legacy_response(result_synapse, effective_colors)
            return result_synapse.response

        error = create_error(ErrorCategory.VALIDATOR_BUSY, context)
        error.message = "No submission token or response received"
        error.suggestions = [
            "Validator may be running an older version without async support",
            "Try a different validator: --validator-uid <other_uid>",
        ]
        print(formatter.format_error(error, context))
        return ""

    GREEN = "\033[92m" if effective_colors else ""
    CYAN = "\033[96m" if effective_colors else ""
    YELLOW = "\033[93m" if effective_colors else ""
    BOLD = "\033[1m" if effective_colors else ""
    RESET = "\033[0m" if effective_colors else ""

    print(f"\n{GREEN}Submission accepted{RESET}")
    print(f"  Token: {BOLD}{token}{RESET}")

    if submit_only:
        print(f"\n{CYAN}Submit-only mode. Use --poll-token {token} to check results later.{RESET}")
        return token

    # --- Phase 2: Poll for results ---
    MAX_POLL_COUNT = 60
    print(f"\n{CYAN}Polling for results (interval={poll_interval}s, max={max_poll_time}s)...{RESET}")

    poll_start = time.time()
    last_status = None
    poll_count = 0
    current_interval = poll_interval

    while (time.time() - poll_start) < max_poll_time and poll_count < MAX_POLL_COUNT:
        poll_count += 1
        elapsed = int(time.time() - poll_start)

        status_synapse = SubmissionStatusSynapse(submission_token=token)
        try:
            status_response = dendrite.query(
                axons=target_axon,
                synapse=status_synapse,
                timeout=10,
                deserialize=False,
            )
        except Exception as e:
            bt.logging.warning(f"Poll failed: {e}")
            time.sleep(current_interval)
            current_interval = min(current_interval * 1.5, 30.0)
            continue

        # Extract status synapse
        status_result = None
        if isinstance(status_response, SubmissionStatusSynapse):
            status_result = status_response
        elif isinstance(status_response, list) and len(status_response) > 0:
            status_result = status_response[0]

        status = getattr(status_result, "status", None) if status_result else None

        if status and status != last_status:
            print(f"  Status: {BOLD}{status}{RESET} ({elapsed}s elapsed)")
            last_status = status

        if status in ("COMPLETED", "FAILED", "TIMEOUT"):
            break

        time.sleep(current_interval)
        current_interval = min(current_interval * 1.5, 30.0)

    if poll_count >= MAX_POLL_COUNT:
        bt.logging.warning(f"Max poll count ({MAX_POLL_COUNT}) reached for token {token}")

    if not status_result or not status:
        print(f"\n{YELLOW}Polling timed out after {max_poll_time}s. Token: {token}{RESET}")
        return ""

    # Display results
    if status == "COMPLETED" and status_result.result:
        _display_async_result(status_result, effective_colors)
        return status_result.result.get("response", "")
    elif status == "FAILED":
        print(f"\n{YELLOW}Processing failed: {status_result.error_message or 'Unknown error'}{RESET}")
        return ""
    elif status == "TIMEOUT":
        print(f"\n{YELLOW}Processing timed out on validator side{RESET}")
        return ""
    else:
        print(f"\n{YELLOW}Final status: {status}. Token: {token}{RESET}")
        return ""


def _display_legacy_response(synapse: PromptSynapse, use_colors: bool) -> None:
    """Display results from a legacy synchronous response."""
    print("\n" + "=" * 60)
    print("RESPONSE FROM VALIDATOR (sync)")
    print("=" * 60)
    print(f"Prompt:   {Config.truncate_sensitive_data(synapse.prompt)}")
    print(f"Response: {Config.truncate_sensitive_data(synapse.response)}")
    print(f"Model:    {synapse.model_used or 'unknown'}")

    if synapse.danger_score is not None:
        print(f"\nDanger Score:  {synapse.danger_score:.4f}")
        print(f"Accepted:      {'YES' if synapse.accepted else 'NO'}")

    print("=" * 60 + "\n")


def _display_async_result(status_synapse: SubmissionStatusSynapse, use_colors: bool) -> None:
    """Display results from an async submission."""
    GREEN = "\033[92m" if use_colors else ""
    RED = "\033[91m" if use_colors else ""
    BOLD = "\033[1m" if use_colors else ""
    RESET = "\033[0m" if use_colors else ""

    result = status_synapse.result or {}
    experiment_id = status_synapse.experiment_id or "unknown"

    print("\n" + "=" * 60)
    print(f"{BOLD}RESULT FROM VALIDATOR{RESET} (experiment: {experiment_id})")
    print("=" * 60)

    if experiment_id == "moral-reasoning":
        quality = result.get("quality_score", 0)
        screening = result.get("screening", "?")
        final = result.get("final_score", 0)
        response_text = result.get("response", "")
        model = result.get("model_used", "unknown")
        timing = result.get("timing_ms", {})

        print(f"Quality Score:  {quality:.3f}")
        screening_color = GREEN if screening == "PASS" else RED
        print(f"Screening:      {screening_color}{screening}{RESET}")
        print(f"Final Score:    {final:.3f}")
        print(f"Model:          {model}")
        if isinstance(timing, dict) and timing.get("total_ms"):
            print(f"Processing:     {timing['total_ms']:.0f}ms")

        signals = result.get("signals", {})
        if signals:
            true_count = sum(1 for v in signals.values() if v)
            print(f"Signals:        {true_count}/{len(signals)} true")

        if response_text:
            print("\nResponse preview:")
            print(f"  {Config.truncate_sensitive_data(response_text)}")

    else:
        # Prompt experiment
        danger = result.get("danger_score", 0)
        accepted = result.get("accepted", False)
        response_text = result.get("response", "")
        model = result.get("model_used", "unknown")

        print(f"Danger Score:   {danger:.4f}")
        accepted_str = f"{GREEN}YES{RESET}" if accepted else f"{RED}NO{RESET}"
        print(f"Accepted:       {accepted_str}")
        print(f"Model:          {model}")

        categories = result.get("category_scores", {})
        if categories:
            print("\nTop Categories:")
            sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
            for cat, score in sorted_cats:
                if score > 0.01:
                    print(f"  {cat:25s}: {score:.4f}")

        miner_stats = result.get("miner_stats", {})
        if miner_stats:
            print(f"\nSubmissions:    {miner_stats.get('submission_count', '?')}")
            hr = miner_stats.get("hit_rate")
            if hr is not None:
                print(f"Hit Rate:       {hr:.1%}")

        if response_text:
            print(f"\nResponse: {Config.truncate_sensitive_data(response_text)}")

    print("=" * 60 + "\n")


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
    experiment_id: str | None = None,
) -> dict[int, ValidatorQueryResult]:
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
        skip_preflight: Skip pre-flight health checks
        use_colors: Use colored output for diagnostics

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

    # Initialize progress tracker
    progress = QueryProgress(target_uids, use_colors=effective_colors)
    progress.show_start()

    # Pre-flight health checks (optional)
    do_preflight = not skip_preflight and Config.MINER_PREFLIGHT_CHECK
    if do_preflight:
        print("Running pre-flight health checks...")
        health_checker = ValidatorHealthChecker(timeout=Config.MINER_PREFLIGHT_TIMEOUT)
        healthy_uids = []
        unhealthy_results: dict[int, ValidatorQueryResult] = {}

        for uid in target_uids:
            axon = metagraph.axons[uid]
            health_result = health_checker.check_validator_reachable(axon.ip, axon.port)
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
            print("No validators passed health check. Aborting query.")
            return unhealthy_results

        # Update target_uids to only include healthy validators
        target_uids = healthy_uids
        print(f"{len(healthy_uids)} validator(s) passed health check")
    else:
        unhealthy_results = {}

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
        experiment_id=experiment_id,
    )

    if experiment_id:
        bt.logging.info(f"Experiment: {experiment_id}")

    # Define the query operation for retry
    def do_query():
        return dendrite.query(
            axons=target_axons,
            synapse=synapse,
            timeout=effective_timeout,
            deserialize=False,
        )

    # Query all validators in parallel with retry
    print("Sending queries...")
    query_start_time = time.time()
    result = retry_handler.execute_with_retry(
        operation=do_query,
        context=context,
        on_retry=lambda attempt, error, delay: bt.logging.warning(
            f"Query failed (attempt {attempt}/{effective_retries}), retrying in {delay:.1f}s: {error}"
        ),
    )

    query_end_time = time.time()
    query_duration_ms = (query_end_time - query_start_time) * 1000

    if not result.success:
        error = classify_error(result.last_error, context) if result.last_error else None
        if error:
            print(formatter.format_error(error, context))
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

    # Show progress summary
    progress.show_query_complete()

    successful_count = sum(1 for r in all_results.values() if r.success)
    bt.logging.info(f"Received {successful_count} successful response(s) from {len(all_results)} validator(s)")
    return all_results


def display_multi_results(
    results: dict[int, ValidatorQueryResult],
    use_colors: bool = True,
) -> None:
    """
    Display results from multiple validators.

    Args:
        results: Dict mapping validator UID to ValidatorQueryResult
        use_colors: Use colored output
    """
    # ANSI color codes
    GREEN = "\033[92m" if use_colors else ""
    RED = "\033[91m" if use_colors else ""
    YELLOW = "\033[93m" if use_colors else ""
    CYAN = "\033[96m" if use_colors else ""
    BOLD = "\033[1m" if use_colors else ""
    RESET = "\033[0m" if use_colors else ""

    if not results:
        print(f"\n{YELLOW}No results from validators.{RESET}")
        return

    # Separate successful and failed results
    successful = {uid: r for uid, r in results.items() if r.success}
    failed = {uid: r for uid, r in results.items() if not r.success}

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
                print(f"Response: {Config.truncate_sensitive_data(synapse.response)}")

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


def _handle_experiment_commands(args, use_colors: bool) -> None:
    """Handle experiment registration CLI commands (T063).

    Args:
        args: Parsed command-line arguments
        use_colors: Whether to use colored output
    """
    # ANSI color codes
    GREEN = "\033[92m" if use_colors else ""
    RED = "\033[91m" if use_colors else ""
    CYAN = "\033[96m" if use_colors else ""
    BOLD = "\033[1m" if use_colors else ""
    RESET = "\033[0m" if use_colors else ""

    # Setup logging and config
    Config.setup_logging()
    Config.apply_network_defaults()

    # Detect wallet
    try:
        Config.detect_and_set_wallet(role="miner")
    except ConfigurationError as e:
        print(f"{RED}Error: Could not detect wallet: {e}{RESET}")
        sys.exit(1)

    # Initialize wallet
    try:
        wallet = bt.Wallet(name=Config.MINER_WALLET_NAME, hotkey=Config.MINER_HOTKEY)
    except Exception as e:
        print(f"{RED}Error: Could not initialize wallet: {e}{RESET}")
        sys.exit(1)

    print(f"{CYAN}Using wallet: {Config.MINER_WALLET_NAME}/{Config.MINER_HOTKEY}{RESET}")
    print(f"{CYAN}Hotkey: {wallet.hotkey.ss58_address}{RESET}\n")

    # Handle registration
    if args.register_experiment:
        experiment_id = args.register_experiment
        print(f"Registering for experiment: {BOLD}{experiment_id}{RESET}...")

        result = register_for_experiment(wallet, experiment_id)

        if result.success:
            print(f"\n{GREEN}✓ Successfully registered for '{experiment_id}'{RESET}")
            if result.status:
                print(f"  Status: {result.status}")
            if result.registered_at:
                print(f"  Registered at: {result.registered_at}")
            if result.message:
                print(f"  {result.message}")
        else:
            print(f"\n{RED}✗ Registration failed{RESET}")
            if result.error:
                print(f"  Error: {result.error}")

    # Handle withdrawal
    elif args.withdraw_experiment:
        experiment_id = args.withdraw_experiment
        print(f"Withdrawing from experiment: {BOLD}{experiment_id}{RESET}...")

        result = withdraw_from_experiment(wallet, experiment_id)

        if result.success:
            print(f"\n{GREEN}✓ Successfully withdrawn from '{experiment_id}'{RESET}")
            if result.status:
                print(f"  Status: {result.status}")
            if result.withdrawn_at:
                print(f"  Withdrawn at: {result.withdrawn_at}")
            if result.message:
                print(f"  {result.message}")
        else:
            print(f"\n{RED}✗ Withdrawal failed{RESET}")
            if result.error:
                print(f"  Error: {result.error}")

    # Handle list registrations
    elif args.list_registrations:
        print(f"Fetching experiment registrations...")

        result = list_registrations(wallet)

        if result.error:
            print(f"\n{RED}✗ Failed to fetch registrations{RESET}")
            print(f"  Error: {result.error}")
        else:
            registrations = result.registrations

            if not registrations:
                print(f"\n{CYAN}No experiment registrations found.{RESET}")
                print("  All miners are automatically registered for the 'prompt' experiment.")
            else:
                print(f"\n{BOLD}Experiment Registrations:{RESET}")
                print("-" * 50)

                for reg in registrations:
                    exp_id = reg.get("experiment_id", "unknown")
                    status = reg.get("status", "unknown")
                    registered_at = reg.get("registered_at", "")

                    # Color status
                    if status == "active":
                        status_str = f"{GREEN}{status}{RESET}"
                    elif status == "withdrawn":
                        status_str = f"{RED}{status}{RESET}"
                    else:
                        status_str = status

                    print(f"  {BOLD}{exp_id}{RESET}")
                    print(f"    Status: {status_str}")
                    if registered_at:
                        print(f"    Registered: {registered_at}")
                    if reg.get("withdrawn_at"):
                        print(f"    Withdrawn: {reg.get('withdrawn_at')}")
                    print()

                print("-" * 50)
                print(f"Total: {len(registrations)} registration(s)")


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

Experiment Registration:
  # Register for an experiment
  python miner.py --register-experiment jailbreak-v1

  # Withdraw from an experiment
  python miner.py --withdraw-experiment jailbreak-v1

  # List all your experiment registrations
  python miner.py --list-registrations
""",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="The prompt to send to the validator (required unless using experiment commands)",
    )

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
    parser.add_argument("--experiment", type=str, default=None, help="Target experiment ID (e.g., 'moral-reasoning')")

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

    # Async submission arguments
    async_group = parser.add_argument_group("async submission")
    async_group.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between status polls (default: 5)",
    )
    async_group.add_argument(
        "--max-poll-time",
        type=int,
        default=300,
        help="Max seconds to poll before giving up (default: 300)",
    )
    async_group.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit and print token, don't poll for result",
    )

    # Experiment registration arguments (T063)
    experiment_group = parser.add_argument_group("experiment registration")
    experiment_group.add_argument(
        "--register-experiment",
        type=str,
        metavar="EXPERIMENT_ID",
        help="Register for an experiment (e.g., 'jailbreak-v1')",
    )
    experiment_group.add_argument(
        "--withdraw-experiment",
        type=str,
        metavar="EXPERIMENT_ID",
        help="Withdraw from an experiment",
    )
    experiment_group.add_argument(
        "--list-registrations",
        action="store_true",
        help="List all experiment registrations for this miner",
    )

    args = parser.parse_args()

    # Override netuid if provided
    if args.netuid is not None:
        Config.BT_NETUID = args.netuid

    use_colors = not args.no_color

    # Setup graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        bt.logging.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Handle experiment registration commands (T063)
    if args.register_experiment or args.withdraw_experiment or args.list_registrations:
        _handle_experiment_commands(args, use_colors)
        return

    # Validate prompt is provided for sending prompts
    if not args.prompt:
        parser.error("--prompt is required when sending prompts to validators")

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
            experiment_id=args.experiment,
            poll_interval=args.poll_interval,
            max_poll_time=args.max_poll_time,
            submit_only=args.submit_only,
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
            experiment_id=args.experiment,
        )
        display_multi_results(results, use_colors=use_colors)


if __name__ == "__main__":
    main()
