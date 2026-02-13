"""Validator implementation - processes prompts using configurable chat providers."""

import argparse
import contextlib
import hashlib
import os
import signal
import socket
import sys
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Tuple

import bittensor as bt
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from aurelius.shared.chat_client import ModelUnavailableError, call_chat_api_with_fallback
from aurelius.shared.config import Config, ConfigurationError
from aurelius.shared.consensus import ConsensusCoordinator
from aurelius.shared.dataset_logger import DatasetLogger
from aurelius.shared.embedding_client import get_embedding_client
from aurelius.shared.moderation import create_moderation_provider
from aurelius.shared.novelty_client import get_novelty_client
from aurelius.shared.protocol import ConsensusVerificationSynapse, PromptSynapse, SubmissionStatusSynapse
from aurelius.shared.submission_client import SubmissionClient
from aurelius.shared.rate_limiter import PerMinerRateLimiter, RateLimiter, RateLimitConfig
from aurelius.shared.remote_config_client import get_remote_config_client
from aurelius.shared.scoring import ScoringSystem
from aurelius.shared.telemetry.otel_setup import get_tracer, register_with_telemetry_api, setup_opentelemetry
from aurelius.validator.experiments.manager import ExperimentManager


def check_port_available(host: str, port: int, timeout: float = 2.0) -> tuple[bool, str]:
    """
    Check if a port is available for binding (not already in use).

    Args:
        host: Host address to check
        port: Port number to check
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (is_available, message)
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(timeout)
            sock.bind((host, port))
            return True, "Port is available"
    except socket.error as e:
        if e.errno == 98 or "Address already in use" in str(e):
            return False, f"Port {port} is already in use by another process"
        elif e.errno == 13 or "Permission denied" in str(e):
            return False, f"Permission denied to bind to port {port} (try a port > 1024 or run as root)"
        else:
            return False, f"Cannot bind to port {port}: {e}"


def check_port_accessible(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    """
    Check if a port is accessible (can accept connections).

    Args:
        host: Host address to check
        port: Port number to check
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (is_accessible, message)
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            if result == 0:
                return True, "Port is accessible"
            else:
                return False, f"Port {port} is not accessible (connection refused)"
    except socket.timeout:
        return False, f"Port {port} connection timed out (may be blocked by firewall)"
    except socket.error as e:
        return False, f"Port {port} check failed: {e}"


def check_external_port_accessible(port: int, timeout: float = 5.0) -> tuple[bool, str, str | None]:
    """
    Check if a port is accessible from the internet using external service.

    Args:
        port: Port number to check
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_accessible, message, external_ip)
    """
    import requests

    try:
        # Use a port checking service
        response = requests.get(
            f"https://api.ipify.org?format=json",
            timeout=timeout
        )
        if response.status_code == 200:
            external_ip = response.json().get("ip")
            return True, "External IP detected", external_ip
        return False, "Could not detect external IP", None
    except Exception as e:
        return False, f"External check failed: {e}", None


class Validator:
    """Validator that processes prompts using configurable chat providers (Chutes.ai or OpenAI)."""

    def __init__(self):
        """Initialize the validator."""
        # Setup logging based on LOG_LEVEL configuration
        Config.setup_logging()

        # Apply network-aware defaults based on BT_NETUID
        # This sets thresholds appropriate for mainnet (37) or testnet (290)
        Config.apply_network_defaults()

        # Detect wallet if not explicitly configured
        # This allows turnkey operation when only one wallet exists
        try:
            Config.detect_and_set_wallet(role="validator")
        except ConfigurationError as e:
            bt.logging.error(str(e))
            sys.exit(1)

        bt.logging.info(f"Initializing validator with wallet: {Config.VALIDATOR_WALLET_NAME}")

        # Validate configuration (will hard-fail on missing/invalid API keys)
        try:
            Config.validate()
        except ConfigurationError as e:
            bt.logging.error(str(e))
            sys.exit(1)

        # Check for production configuration warnings
        production_warnings = Config.validate_production()
        if production_warnings:
            bt.logging.warning("=" * 80)
            bt.logging.warning("PRODUCTION CONFIGURATION WARNINGS:")
            for warning in production_warnings:
                bt.logging.warning(f"  - {warning}")
            bt.logging.warning("=" * 80)

        # A6: Warn if using default endpoints (MITM prevention)
        Config.warn_default_endpoints()

        # A20: Validate HTTPS endpoints in production
        try:
            Config.validate_endpoints()
        except ConfigurationError as e:
            bt.logging.error(str(e))
            sys.exit(1)

        # Default timeout for chat API calls (seconds) to prevent indefinite blocking
        chat_api_timeout = 60.0

        # Initialize chat client based on provider
        if Config.CHAT_PROVIDER == "chutes":
            self.chat_client = OpenAI(
                api_key=Config.CHUTES_API_KEY,
                base_url=Config.CHUTES_API_BASE_URL,
                timeout=chat_api_timeout,
            )
            self.model = Config.DEFAULT_MODEL  # "deepseek-ai/DeepSeek-V3"
            bt.logging.info(f"Using Chutes.ai chat provider with model: {self.model}")
        else:
            self.chat_client = OpenAI(api_key=Config.OPENAI_API_KEY, timeout=chat_api_timeout)
            self.model = Config.OPENAI_MODEL
            bt.logging.info(f"Using OpenAI chat provider with model: {self.model}")

        # Initialize moderation provider
        bt.logging.info(f"Initializing moderation provider: {Config.MODERATION_PROVIDER}")
        self.moderation_provider = create_moderation_provider(
            provider_name=Config.MODERATION_PROVIDER,
            api_key=Config.OPENAI_API_KEY,
            category_weights=Config.MODERATION_CATEGORY_WEIGHTS,
            single_category_threshold=Config.SINGLE_CATEGORY_THRESHOLD,
        )

        # Initialize per-miner rate limiter
        rate_limit_config = RateLimitConfig(
            max_requests=Config.RATE_LIMIT_REQUESTS, window_hours=Config.RATE_LIMIT_WINDOW_HOURS
        )
        self.rate_limiter = PerMinerRateLimiter(rate_limit_config)
        self._rate_limiter_state_path = Config.MINER_SCORES_PATH + ".ratelimit"
        self.rate_limiter.load_state(self._rate_limiter_state_path)

        # Global rate limiter — caps total requests across all miners
        global_rate_config = RateLimitConfig(
            max_requests=Config.GLOBAL_RATE_LIMIT_REQUESTS,
            window_hours=Config.RATE_LIMIT_WINDOW_HOURS,
        )
        self.global_rate_limiter = RateLimiter(global_rate_config)

        # Initialize dataset logger (wallet passed later after wallet init for signing)
        self.dataset_logger = DatasetLogger(
            local_path=Config.LOCAL_DATASET_PATH,
            central_api_endpoint=Config.CENTRAL_API_ENDPOINT,
            central_api_key=Config.CENTRAL_API_KEY,
            enable_local_backup=Config.ENABLE_LOCAL_BACKUP,
        )

        # Initialize scoring system with windowing configuration
        self.scoring_system = ScoringSystem(
            danger_threshold=Config.DANGER_THRESHOLD,
            persistence_path=Config.MINER_SCORES_PATH,
            window_blocks=Config.WINDOW_BLOCKS,
            history_retention_blocks=Config.HISTORY_RETENTION_BLOCKS,
        )

        # Initialize embedding client for generating prompt embeddings
        self.embedding_client = get_embedding_client()
        if self.embedding_client.is_available():
            bt.logging.info(f"Embedding client enabled: {self.embedding_client._get_embeddings_url()}")
        else:
            bt.logging.warning("Embedding client not available (no API key configured)")

        # Initialize novelty client for checking prompt uniqueness
        self.novelty_client = get_novelty_client()
        if self.novelty_client.is_available():
            bt.logging.info(f"Novelty detection enabled: {Config.NOVELTY_API_ENDPOINT}")
        else:
            bt.logging.warning("Novelty detection not available (no API endpoint configured)")

        # Initialize consensus coordinator (will be set after wallet/subtensor init)
        self.consensus_coordinator = None

        # Thread lock for subtensor operations to prevent websocket concurrency errors
        self.subtensor_lock = threading.RLock()

        # Cached metagraph for blacklist checks (refreshed in weight update loop)
        self._cached_metagraph = None
        self._metagraph_lock = threading.RLock()

        # Thread pool for background tasks (prevents DoS via thread explosion)
        # Limits concurrent background operations to prevent resource exhaustion
        self.background_executor = ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="validator-bg"
        )

        if Config.LOCAL_MODE:
            # Local mode: Use simulated subtensor for block height tracking
            bt.logging.info("LOCAL_MODE: Using simulated subtensor for testing")

            # Create a minimal wallet for the axon (not loaded from disk)
            self.wallet = bt.Wallet(name=Config.VALIDATOR_WALLET_NAME, hotkey=Config.VALIDATOR_HOTKEY)
            # In local mode, we don't need the actual keys loaded

            # Create simulated subtensor for block height simulation
            from aurelius.shared.simulated_subtensor import SimulatedSubtensor

            # Use fast block mode if enabled (1 sec blocks), otherwise normal (12 sec)
            block_time = 1.0 if Config.FAST_BLOCK_MODE else Config.SIMULATED_BLOCK_TIME

            self.subtensor = SimulatedSubtensor(start_block=Config.SIMULATED_BLOCK_START, block_time=block_time)

            bt.logging.info(
                f"Simulated block progression enabled: "
                f"starting at block {Config.SIMULATED_BLOCK_START}, "
                f"{block_time}s per block"
            )
        else:
            # Normal mode: Initialize wallet and subtensor
            self.wallet = bt.Wallet(name=Config.VALIDATOR_WALLET_NAME, hotkey=Config.VALIDATOR_HOTKEY)

            subtensor_config = Config.get_subtensor_config()
            self.subtensor = bt.Subtensor(**subtensor_config)

            # Load subnet hyperparameters from chain (network-level consensus on config values)
            Config.load_subnet_hyperparameters(self.subtensor)

        # Now that wallet is initialized, attach it to dataset logger and novelty client for signed requests
        self.dataset_logger.wallet = self.wallet
        self.novelty_client.wallet = self.wallet

        # Initialize axon (server for receiving requests)
        # Use configured host (may auto-detect external IP if enabled)
        validator_host = Config.get_validator_host()
        bt.logging.info(f"Validator host: {validator_host}")
        self.axon = bt.Axon(wallet=self.wallet, port=Config.BT_PORT_VALIDATOR, external_ip=validator_host)

        # Initialize consensus coordinator
        if Config.ENABLE_CONSENSUS:
            self.consensus_coordinator = ConsensusCoordinator(
                wallet=self.wallet, subtensor=self.subtensor, netuid=Config.BT_NETUID
            )
            bt.logging.info("Consensus verification enabled")
        else:
            bt.logging.info("Consensus verification disabled")

        # Initialize OpenTelemetry for distributed tracing
        self._tracer = None
        if Config.TELEMETRY_ENABLED:
            try:
                validator_hotkey = None
                validator_uid = None
                if self.wallet and hasattr(self.wallet.hotkey, 'ss58_address'):
                    validator_hotkey = self.wallet.hotkey.ss58_address
                if hasattr(self, 'uid'):
                    validator_uid = self.uid

                setup_opentelemetry(
                    service_name="aurelius-validator",
                    validator_hotkey=validator_hotkey,
                    validator_uid=validator_uid,
                    netuid=Config.BT_NETUID,
                    network=Config.BT_NETWORK,
                    traces_endpoint=Config.TELEMETRY_TRACES_ENDPOINT,
                    logs_endpoint=Config.TELEMETRY_LOGS_ENDPOINT,
                    enable_traces=Config.TELEMETRY_TRACES_ENABLED,
                    enable_logs=Config.TELEMETRY_LOGS_ENABLED,
                    trace_batch_size=Config.TELEMETRY_BATCH_SIZE,
                    flush_interval_ms=Config.TELEMETRY_FLUSH_INTERVAL_MS,
                    local_backup_path=Config.TELEMETRY_LOCAL_BACKUP_PATH,
                    wallet=self.wallet,
                    heartbeat_interval_s=Config.TELEMETRY_HEARTBEAT_INTERVAL_S,
                )
                self._tracer = get_tracer("aurelius.validator")
                bt.logging.info("OpenTelemetry tracing initialized")
            except Exception as e:
                bt.logging.warning(f"Failed to initialize OpenTelemetry: {e}")
                self._tracer = None
        else:
            bt.logging.info("OpenTelemetry tracing disabled")

        # Initialize remote configuration client for dynamic config updates
        self.remote_config_client = get_remote_config_client()
        self._remote_config_stop_event = threading.Event()
        self._remote_config_thread: threading.Thread | None = None

        # Initialize experiment client and manager for multi-experiment routing
        from aurelius.shared.experiment_client import get_experiment_client

        self.experiment_client = get_experiment_client()
        self.experiment_manager = ExperimentManager(self)
        self._register_default_experiment()

        # Async submission tracking — in-memory cache + collector API client
        self._submission_results: dict[str, dict] = {}
        self._submission_results_lock = threading.Lock()
        self._submission_max_cache_size = 1000
        self.submission_client = SubmissionClient(wallet=self.wallet, api_key=Config.CENTRAL_API_KEY)

        # Perform initial remote config fetch (non-blocking, don't fail startup)
        if self.remote_config_client.is_available():
            try:
                self._fetch_and_apply_remote_config(log_on_success=True)
                bt.logging.info("Remote config client initialized and initial config fetched")
            except Exception as e:
                bt.logging.warning(f"Initial remote config fetch failed (will retry): {e}")
        else:
            bt.logging.info("Remote config client disabled (no endpoint configured)")

        bt.logging.info("=" * 80)
        bt.logging.info("🚀 VALIDATOR INITIALIZATION COMPLETE")
        bt.logging.info("=" * 80)
        bt.logging.info(f"  Port: {Config.BT_PORT_VALIDATOR}")
        bt.logging.info(f"  Host: {validator_host}")
        bt.logging.info(f"  Network: {Config.BT_NETWORK}")
        bt.logging.info(f"  Netuid: {Config.BT_NETUID}")
        bt.logging.info(f"  Chat Model: {self.model} (provider: {Config.CHAT_PROVIDER})")
        bt.logging.info(f"  Danger Threshold: {Config.DANGER_THRESHOLD}")
        bt.logging.info(f"  Rate Limit: {Config.RATE_LIMIT_REQUESTS} requests per {Config.RATE_LIMIT_WINDOW_HOURS} hour(s)")
        bt.logging.info(f"  Local Mode: {Config.LOCAL_MODE}")
        bt.logging.info(f"  Log Level: {Config.LOG_LEVEL}")
        bt.logging.info(f"  Log Connection Details: {Config.LOG_CONNECTION_DETAILS}")
        bt.logging.info(f"  Consensus Enabled: {Config.ENABLE_CONSENSUS}")
        if self.wallet:
            bt.logging.info(f"  Wallet Name: {Config.VALIDATOR_WALLET_NAME}")
            bt.logging.info(f"  Hotkey: {self.wallet.hotkey.ss58_address if hasattr(self.wallet.hotkey, 'ss58_address') else 'N/A'}")
        bt.logging.info("=" * 80)

    def _register_default_experiment(self):
        """Register the default 'prompt' experiment for backward compatibility.

        This creates a simple experiment wrapper that enables routing to work
        while maintaining the existing forward handler behavior. The actual
        scoring and processing logic remains in the main validator.
        """
        from aurelius.validator.experiments.base import ExperimentConfig, ExperimentScores, ExperimentType

        # Create a simple wrapper experiment that represents the existing prompt processing
        class PromptExperimentWrapper:
            """Minimal experiment wrapper for routing compatibility.

            This wrapper allows the ExperimentManager.route_submission() to work
            while the actual processing logic remains in the validator's forward handler.
            It delegates scoring to the validator's existing ScoringSystem.
            """

            def __init__(self, name: str, scoring_system, enabled: bool = True):
                self.name = name
                self._enabled = enabled
                self._scoring_system = scoring_system
                self.config = ExperimentConfig(
                    name=name,
                    experiment_type=ExperimentType.PUSH,
                    weight_allocation=0.0,
                    enabled=enabled,
                )

            @property
            def is_enabled(self) -> bool:
                return self._enabled

            @property
            def weight_allocation(self) -> float:
                return self.config.weight_allocation

            def calculate_scores(self, current_block: int) -> ExperimentScores:
                normalized = self._scoring_system.calculate_normalized_scores(
                    current_block=current_block,
                    min_submissions=Config.MIN_SAMPLES_FOR_WEIGHTS,
                )
                return ExperimentScores(
                    scores=normalized,
                    experiment_name=self.name,
                    block_height=current_block,
                )

            def get_stats(self) -> dict:
                return self._scoring_system.get_stats()

        # Register the prompt experiment
        prompt_experiment = PromptExperimentWrapper("prompt", scoring_system=self.scoring_system, enabled=False)
        self.experiment_manager.experiments["prompt"] = prompt_experiment
        bt.logging.info("Registered default 'prompt' experiment for multi-experiment routing")

        # Register the moral reasoning experiment
        self._register_moral_reasoning_experiment()

    def _register_moral_reasoning_experiment(self):
        """Register the moral reasoning experiment for multi-experiment routing."""
        from aurelius.validator.experiments.base import ExperimentConfig, ExperimentType

        try:
            from aurelius.validator.experiments.moral_reasoning.experiment import (
                MoralReasoningExperiment,
            )

            config = ExperimentConfig(
                name="moral-reasoning",
                experiment_type=ExperimentType.PUSH,
                weight_allocation=1.0,  # 100% allocation — sole active experiment
                enabled=True,
                settings={},
            )
            experiment = MoralReasoningExperiment(core=self, config=config)
            self._moral_reasoning_experiment = experiment
            self.experiment_manager.experiments["moral-reasoning"] = experiment
            bt.logging.info("Registered 'moral-reasoning' experiment")
        except Exception as e:
            bt.logging.warning(f"Could not register moral reasoning experiment: {e}")

    @contextlib.contextmanager
    def _timed_lock(self, lock, name="lock", timeout=None):
        """Acquire a lock with a timeout to prevent deadlocks.

        Args:
            lock: The threading lock to acquire
            name: Human-readable name for logging
            timeout: Maximum seconds to wait for lock acquisition

        Raises:
            TimeoutError: If the lock cannot be acquired within the timeout
        """
        if timeout is None:
            timeout = Config.SUBTENSOR_LOCK_TIMEOUT
        acquired = lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"{name} acquisition timed out after {timeout}s")
        try:
            yield
        finally:
            lock.release()

    def _get_current_block(self):
        """Safely get current block number with thread locking."""
        if not self.subtensor:
            return None
        with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
            return self.subtensor.block

    def _fetch_and_apply_remote_config(self, log_on_success: bool = False) -> bool:
        """
        Fetch remote configuration and apply it to Config.

        Args:
            log_on_success: Whether to log when config is fetched successfully

        Returns:
            True if config was fetched and applied successfully
        """
        result = self.remote_config_client.fetch_config()

        if not result.success:
            bt.logging.warning(f"Remote config fetch failed: {result.error}")
            return False

        if not result.config:
            if log_on_success:
                bt.logging.debug("No remote config available for this network")
            return True

        # Apply the config
        updated_fields = self.remote_config_client.apply_to_config(result.config)

        if updated_fields:
            bt.logging.info(
                f"Remote config v{result.version} applied: {len(updated_fields)} field(s) updated"
            )
        elif log_on_success:
            bt.logging.debug(f"Remote config v{result.version} fetched (no changes)")

        return True

    def _remote_config_loop(self):
        """
        Background loop to periodically fetch and apply remote configuration.

        This runs in a daemon thread and polls the remote config API at the
        configured interval. On errors, it continues using cached values.
        """
        bt.logging.info(
            f"Remote config polling started (interval: {self.remote_config_client.poll_interval}s)"
        )

        while not self._remote_config_stop_event.is_set():
            # Wait for the poll interval, but check stop event periodically
            # Using Event.wait() allows clean shutdown
            if self._remote_config_stop_event.wait(timeout=self.remote_config_client.poll_interval):
                # Stop event was set
                break

            try:
                self._fetch_and_apply_remote_config(log_on_success=False)
            except Exception as e:
                # Never crash the loop - log and continue
                bt.logging.error(f"Remote config polling error (will retry): {e}")

        bt.logging.info("Remote config polling stopped")

    def _resolve_model_preferences(self, vendor: str, model: str) -> tuple[str, str]:
        """
        Validate and resolve miner's model preferences against allowed list.

        Args:
            vendor: Requested vendor (e.g., 'openai', 'anthropic')
            model: Requested model (e.g., 'o4-mini', 'gpt-4o')

        Returns:
            Tuple of (actual_vendor, actual_model) to use
        """
        # Check if vendor is allowed
        if vendor not in Config.ALLOWED_MODELS:
            bt.logging.warning(
                f"Vendor '{vendor}' not in allowed list. Falling back to default: {Config.DEFAULT_VENDOR}"
            )
            return Config.DEFAULT_VENDOR, Config.DEFAULT_MODEL

        # Check if model is allowed for this vendor
        allowed_models = Config.ALLOWED_MODELS[vendor]
        if model not in allowed_models:
            bt.logging.warning(
                f"Model '{model}' not allowed for vendor '{vendor}'. "
                f"Allowed models: {allowed_models}. Falling back to default: {Config.DEFAULT_MODEL}"
            )
            # Use default model for the requested vendor if available, else full default
            if Config.DEFAULT_MODEL in allowed_models:
                return vendor, Config.DEFAULT_MODEL
            elif allowed_models:
                return vendor, allowed_models[0]
            else:
                return Config.DEFAULT_VENDOR, Config.DEFAULT_MODEL

        return vendor, model

    def _clamp_parameter(self, value: float | None, min_val: float, max_val: float) -> float | None:
        """
        Clamp a parameter value to the allowed range.

        Args:
            value: The value to clamp (or None)
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Clamped value or None if input was None or invalid
        """
        if value is None:
            return None
        import math
        if math.isnan(value) or math.isinf(value):
            bt.logging.warning(f"Parameter rejected: {value} (NaN/Infinity)")
            return None
        clamped = max(min_val, min(max_val, value))
        if clamped != value:
            bt.logging.warning(f"Parameter clamped from {value} to {clamped} (range: {min_val}-{max_val})")
        return clamped

    def forward(self, synapse: PromptSynapse) -> PromptSynapse:
        """
        Process an incoming prompt request asynchronously.

        Flow:
        1. Validate input (prompt length, parameter sanity)
        2. Route to correct experiment via ExperimentManager
        3. Check rate limits
        4. Generate a submission token
        5. Return token to miner immediately
        6. Process in background (LLM call, moderation/judge, scoring)
        7. Update local cache + collector API when done

        Args:
            synapse: The PromptSynapse containing the prompt

        Returns:
            The synapse with submission_token set (result fields populated later via polling)
        """
        import time

        start_time = time.time()
        prompt = synapse.prompt
        miner_hotkey = synapse.dendrite.hotkey if hasattr(synapse, "dendrite") and synapse.dendrite else None

        # Create OpenTelemetry span for the submit phase
        span_context = None
        if self._tracer:
            span_context = self._tracer.start_as_current_span(
                "validator.forward",
                kind=SpanKind.SERVER,
                attributes={
                    "miner.hotkey": miner_hotkey[:16] if miner_hotkey else "unknown",
                    "prompt.length": len(prompt),
                    "synapse.name": synapse.name,
                }
            )
            span_context.__enter__()

        try:
            result = self._handle_submit(synapse, miner_hotkey, prompt, start_time)
            if span_context:
                current_span = trace.get_current_span()
                current_span.set_status(Status(StatusCode.OK))
                current_span.set_attribute("submission.token", result.submission_token or "")
            return result
        except Exception as e:
            if span_context:
                current_span = trace.get_current_span()
                current_span.set_status(Status(StatusCode.ERROR, str(e)))
                current_span.record_exception(e)
            raise
        finally:
            if span_context:
                span_context.__exit__(None, None, None)

    def _handle_submit(
        self,
        synapse: PromptSynapse,
        miner_hotkey: str | None,
        prompt: str,
        start_time: float,
    ) -> PromptSynapse:
        """Handle submission: validate, generate token, dispatch background processing."""
        import time

        bt.logging.info("=" * 80)
        bt.logging.info("FORWARD METHOD CALLED - ASYNC TOKEN FLOW")
        bt.logging.info(f"   Prompt: {Config.truncate_sensitive_data(prompt)}")
        bt.logging.info("=" * 80)

        # Input validation
        import math

        if len(prompt) > Config.MAX_PROMPT_LENGTH:
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = f"Prompt exceeds maximum length ({Config.MAX_PROMPT_LENGTH} chars)"
            synapse.rejection_code = "PROMPT_TOO_LONG"
            return synapse

        miner_max_chars = synapse.max_chars
        if miner_max_chars and miner_max_chars > Config.MAX_RESPONSE_CHARS_LIMIT:
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = f"max_chars exceeds limit ({Config.MAX_RESPONSE_CHARS_LIMIT})"
            synapse.rejection_code = "MAX_CHARS_EXCEEDED"
            return synapse

        miner_min_chars = synapse.min_chars
        if miner_min_chars and miner_max_chars and miner_min_chars > miner_max_chars:
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = "min_chars cannot be greater than max_chars"
            synapse.rejection_code = "INVALID_PARAMS"
            return synapse

        # NaN/Infinity parameter validation
        for param_name, param_val in [
            ("temperature", synapse.temperature),
            ("top_p", synapse.top_p),
            ("frequency_penalty", synapse.frequency_penalty),
            ("presence_penalty", synapse.presence_penalty),
        ]:
            if param_val is not None and (math.isnan(param_val) or math.isinf(param_val)):
                synapse.response = None
                synapse.danger_score = 0.0
                synapse.accepted = False
                synapse.rejection_reason = f"Invalid {param_name}: NaN/Infinity not allowed"
                synapse.rejection_code = "INVALID_PARAMS"
                return synapse

        # Global rate limit check (before routing to avoid concurrency slot leak)
        global_allowed, global_reason, _ = self.global_rate_limiter.check_rate_limit()
        if not global_allowed:
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = global_reason
            synapse.rejection_code = "RATE_LIMITED"
            return synapse

        # Per-miner rate limit check (before routing to avoid concurrency slot leak)
        allowed, reason, remaining = self.rate_limiter.check_rate_limit(hotkey=miner_hotkey)
        if not allowed:
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = reason
            synapse.rejection_code = "RATE_LIMITED"
            return synapse

        # Experiment routing (acquires concurrency slot on success)
        synapse.miner_hotkey = miner_hotkey
        routing_result = self.experiment_manager.route_submission(synapse)

        if routing_result.experiment is None:
            bt.logging.info(
                f"Experiment routing rejected: {routing_result.rejection_reason} "
                f"(miner: {miner_hotkey[:8] if miner_hotkey else 'unknown'}...)"
            )
            self.experiment_manager.apply_routing_rejection(synapse, routing_result)
            synapse.response = None
            synapse.danger_score = 0.0
            return synapse

        effective_experiment = routing_result.experiment_id
        slot_owned = True

        try:
            # Record rate limit only after routing succeeds (avoid wasting tokens on rejected requests)
            self.global_rate_limiter.record_request()
            self.rate_limiter.record_request(hotkey=miner_hotkey)

            # Generate submission token
            submission_token = str(uuid.uuid4())
            now_iso = datetime.now(timezone.utc).isoformat()

            # Store in local cache
            self._set_submission_cache(submission_token, {
                "status": "PENDING",
                "experiment_id": effective_experiment,
                "created_at": now_iso,
                "result": None,
                "error_message": None,
                "completed_at": None,
                "miner_hotkey": miner_hotkey,
            })

            # Register token on collector API (background, but capture Future)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            reg_future = self.background_executor.submit(
                self.submission_client.register_submission,
                submission_token,
                miner_hotkey or "unknown",
                effective_experiment,
                prompt_hash,
            )

            # Set token on synapse for immediate return
            synapse.submission_token = submission_token

            # Submit the actual processing to background (waits for registration first)
            self.background_executor.submit(
                self._process_submission_background,
                reg_future,
                submission_token,
                synapse.prompt,
                miner_hotkey,
                effective_experiment,
                synapse.vendor,
                synapse.model_requested,
                synapse.temperature,
                synapse.top_p,
                synapse.frequency_penalty,
                synapse.presence_penalty,
                synapse.min_chars,
                synapse.max_chars,
            )
            slot_owned = False  # Background task now owns the slot

            bt.logging.info(
                f"Submission accepted: token={submission_token[:8]}... "
                f"experiment={effective_experiment} miner={miner_hotkey[:8] if miner_hotkey else '?'}..."
            )

            return synapse
        except Exception as e:
            bt.logging.error(f"Failed to dispatch submission for '{effective_experiment}': {e}")
            synapse.response = None
            synapse.accepted = False
            synapse.rejection_reason = "Validator internal error"
            synapse.rejection_code = "INTERNAL_ERROR"
            return synapse
        finally:
            if slot_owned:
                bt.logging.warning(
                    f"Releasing orphaned concurrency slot for '{effective_experiment}' "
                    f"(miner: {miner_hotkey[:8] if miner_hotkey else 'unknown'}...)"
                )
                self.experiment_manager.release_concurrency_slot(effective_experiment)

    def _set_submission_cache(self, token: str, data: dict) -> None:
        """Set a submission in the local cache with LRU eviction."""
        with self._submission_results_lock:
            # Evict oldest entries if cache is full
            while len(self._submission_results) >= self._submission_max_cache_size:
                oldest_key = next(iter(self._submission_results))
                del self._submission_results[oldest_key]
            self._submission_results[token] = data

    def _process_submission_background(
        self,
        reg_future: Future[bool],
        token: str,
        prompt: str,
        miner_hotkey: str | None,
        experiment_id: str,
        vendor: str | None,
        model_requested: str | None,
        temperature: float | None,
        top_p: float | None,
        frequency_penalty: float | None,
        presence_penalty: float | None,
        min_chars: int | None,
        max_chars: int | None,
    ) -> None:
        """Process a submission in the background thread and update cache + API."""
        import time

        # Wait for registration to complete before any updates
        try:
            registered = reg_future.result(timeout=15)
            if not registered:
                bt.logging.warning(f"Submission {token[:8]} registration failed, updates may 404")
        except Exception as e:
            bt.logging.warning(f"Submission {token[:8]} registration error: {e}")

        # Update status to PROCESSING
        self._update_submission_status(token, "PROCESSING")

        try:
            if experiment_id == "moral-reasoning" and hasattr(self, "_moral_reasoning_experiment"):
                result_dict = self._process_moral_reasoning_background(
                    token, prompt, miner_hotkey,
                )
            else:
                result_dict = self._process_prompt_experiment_background(
                    token, prompt, miner_hotkey,
                    vendor, model_requested, temperature, top_p,
                    frequency_penalty, presence_penalty, min_chars, max_chars,
                    experiment_id,
                )

            # Update cache and collector API with COMPLETED
            now_iso = datetime.now(timezone.utc).isoformat()
            with self._submission_results_lock:
                if token in self._submission_results:
                    self._submission_results[token].update({
                        "status": "COMPLETED",
                        "result": result_dict,
                        "completed_at": now_iso,
                    })

            self.submission_client.update_submission(
                token, "COMPLETED", result=result_dict,
            )

        except Exception as e:
            bt.logging.error(f"Background processing failed for {token[:8]}...: {e}")
            error_msg = str(e)[:500]
            now_iso = datetime.now(timezone.utc).isoformat()
            with self._submission_results_lock:
                if token in self._submission_results:
                    self._submission_results[token].update({
                        "status": "FAILED",
                        "error_message": error_msg,
                        "completed_at": now_iso,
                    })

            self.submission_client.update_submission(
                token, "FAILED", error=error_msg,
            )
        finally:
            # Release concurrency slot (V4 fix)
            self.experiment_manager.release_concurrency_slot(experiment_id)

    def _update_submission_status(self, token: str, status: str) -> None:
        """Update submission status in local cache and collector API."""
        with self._submission_results_lock:
            if token in self._submission_results:
                self._submission_results[token]["status"] = status

        self.submission_client.update_submission(token, status)

    def _process_moral_reasoning_background(
        self,
        token: str,
        prompt: str,
        miner_hotkey: str | None,
    ) -> dict:
        """Process a moral reasoning submission in the background."""
        # Build a minimal synapse for the experiment handler
        synapse = PromptSynapse(prompt=prompt)
        # Attach miner hotkey info via TerminalInfo
        if miner_hotkey:
            synapse.miner_hotkey = miner_hotkey
            synapse.dendrite = bt.TerminalInfo(hotkey=miner_hotkey)

        _, result_dict = self._moral_reasoning_experiment._handle_scenario_with_result(synapse)

        if result_dict is None:
            # Processing failed (validation, model unavailable, etc.)
            raise RuntimeError(synapse.rejection_reason or "Moral reasoning processing failed")

        return result_dict

    def _process_prompt_experiment_background(
        self,
        token: str,
        prompt: str,
        miner_hotkey: str | None,
        vendor: str | None,
        model_requested: str | None,
        temperature: float | None,
        top_p: float | None,
        frequency_penalty: float | None,
        presence_penalty: float | None,
        min_chars: int | None,
        max_chars: int | None,
        experiment_id: str = "prompt",
    ) -> dict:
        """Process a prompt experiment submission in the background."""
        import time

        start_time = time.time()

        # Resolve model preferences
        actual_vendor, actual_model = self._resolve_model_preferences(
            vendor or Config.DEFAULT_VENDOR, model_requested or Config.DEFAULT_MODEL
        )
        actual_temperature = self._clamp_parameter(temperature, Config.MIN_TEMPERATURE, Config.MAX_TEMPERATURE)
        actual_top_p = self._clamp_parameter(top_p, Config.MIN_TOP_P, Config.MAX_TOP_P)
        actual_freq_penalty = self._clamp_parameter(frequency_penalty, Config.MIN_FREQUENCY_PENALTY, Config.MAX_FREQUENCY_PENALTY)
        actual_pres_penalty = self._clamp_parameter(presence_penalty, Config.MIN_PRESENCE_PENALTY, Config.MAX_PRESENCE_PENALTY)

        # Build API call params
        api_params = {
            "model": actual_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": Config.OPENAI_MAX_TOKENS,
        }
        if actual_temperature is not None:
            api_params["temperature"] = actual_temperature
        if actual_top_p is not None:
            api_params["top_p"] = actual_top_p
        if actual_freq_penalty is not None:
            api_params["frequency_penalty"] = actual_freq_penalty
        if actual_pres_penalty is not None:
            api_params["presence_penalty"] = actual_pres_penalty

        # Call LLM
        response, actual_model = call_chat_api_with_fallback(
            self.chat_client, api_params, timeout=Config.CHAT_API_TIMEOUT
        )
        completion_text = response.choices[0].message.content.strip()

        # Enforce max_chars
        if max_chars is not None and len(completion_text) > max_chars:
            completion_text = completion_text[:max_chars]

        # Moderate response
        moderation_result = self.moderation_provider.moderate(completion_text)
        danger_score = moderation_result.combined_score
        category_scores = moderation_result.category_scores

        # Determine acceptance
        high_category_triggered = moderation_result.high_category_triggered
        aggregate_threshold_met = danger_score >= Config.DANGER_THRESHOLD
        accepted = high_category_triggered or aggregate_threshold_met

        # Record submission in scoring system
        self.scoring_system.record_submission(
            hotkey=miner_hotkey or "unknown",
            danger_score=danger_score,
            accepted=accepted,
            block=self._get_current_block(),
            novelty_score=None,
        )

        # Get miner stats
        miner_stats = {}
        miner_score = self.scoring_system.get_miner_score(miner_hotkey or "unknown")
        if miner_score:
            miner_stats = {
                "novelty_avg": miner_score.average_novelty_score,
                "hit_rate": miner_score.acceptance_rate / 100.0,
                "submission_count": miner_score.total_submissions,
            }

        total_ms = round((time.time() - start_time) * 1000, 2)

        # Background logging (don't block result)
        model_config = {
            "model": actual_model,
            "vendor": actual_vendor,
            "temperature": actual_temperature,
        }
        self.background_executor.submit(
            self._log_dataset_entry_background,
            prompt, completion_text, danger_score, category_scores,
            accepted, miner_hotkey, {"total_ms": total_ms}, model_config, experiment_id,
        )

        return {
            "danger_score": danger_score,
            "category_scores": category_scores,
            "accepted": accepted,
            "response": completion_text,
            "model_used": actual_model,
            "miner_stats": miner_stats,
            "timing_ms": total_ms,
        }

    def _handle_status(self, synapse: SubmissionStatusSynapse) -> SubmissionStatusSynapse:
        """Handle a submission status poll request."""
        token = synapse.submission_token

        # Check local cache first
        with self._submission_results_lock:
            cached = self._submission_results.get(token)

        if cached:
            # Verify the polling hotkey matches the original submitter
            polling_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
            cached_hotkey = cached.get("miner_hotkey")
            if polling_hotkey and cached_hotkey and polling_hotkey != cached_hotkey:
                synapse.error_message = "Unknown submission token"
                return synapse

            synapse.status = cached["status"]
            synapse.result = cached.get("result")
            synapse.error_message = cached.get("error_message")
            synapse.experiment_id = cached.get("experiment_id")
            synapse.created_at = cached.get("created_at")
            synapse.completed_at = cached.get("completed_at")
            return synapse

        # Fallback: check collector API (e.g., after validator restart)
        remote = self.submission_client.get_submission(token)
        if remote:
            synapse.status = remote.get("status")
            synapse.result = remote.get("result")
            synapse.error_message = remote.get("error_message")
            synapse.experiment_id = remote.get("experiment_id")
            synapse.created_at = remote.get("created_at")
            synapse.completed_at = remote.get("completed_at")
            return synapse

        # Unknown token
        synapse.error_message = "Unknown submission token"
        return synapse

    def _forward_internal(
        self,
        synapse: PromptSynapse,
        miner_hotkey: str | None,
        prompt: str,
        start_time: float
    ) -> PromptSynapse:
        """Internal forward logic (legacy synchronous path, kept for reference)."""
        import time

        bt.logging.info("=" * 80)
        bt.logging.info("FORWARD METHOD CALLED - LEGACY SYNC PATH")
        bt.logging.info(f"   Synapse name: {synapse.name}")
        bt.logging.info(f"   Prompt: {Config.truncate_sensitive_data(prompt)}")
        bt.logging.info("=" * 80)

        # Input validation (security)
        # Check 1: Prompt length
        if len(prompt) > Config.MAX_PROMPT_LENGTH:
            bt.logging.warning(
                f"Prompt too long: {len(prompt)} chars (max: {Config.MAX_PROMPT_LENGTH}). "
                f"Miner: {miner_hotkey[:8] if miner_hotkey else 'unknown'}..."
            )
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = f"Prompt exceeds maximum length ({Config.MAX_PROMPT_LENGTH} chars)"
            return synapse

        # Check 2: Response length limits
        miner_max_chars = synapse.max_chars
        if miner_max_chars and miner_max_chars > Config.MAX_RESPONSE_CHARS_LIMIT:
            bt.logging.warning(
                f"Requested max_chars ({miner_max_chars}) exceeds limit ({Config.MAX_RESPONSE_CHARS_LIMIT}). "
                f"Miner: {miner_hotkey[:8] if miner_hotkey else 'unknown'}..."
            )
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = f"max_chars exceeds limit ({Config.MAX_RESPONSE_CHARS_LIMIT})"
            return synapse

        # Check 3: min_chars vs max_chars consistency
        miner_min_chars = synapse.min_chars
        if miner_min_chars and miner_max_chars and miner_min_chars > miner_max_chars:
            bt.logging.warning(
                f"Invalid parameter combination: min_chars ({miner_min_chars}) > max_chars ({miner_max_chars}). "
                f"Miner: {miner_hotkey[:8] if miner_hotkey else 'unknown'}..."
            )
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = "min_chars cannot be greater than max_chars"
            return synapse

        # Step 0.5: Experiment routing validation (FR-001, FR-002, FR-021, FR-022)
        # Route submission to appropriate experiment, check experiment exists and miner registration
        synapse.miner_hotkey = miner_hotkey  # Ensure miner_hotkey is set for routing
        routing_result = self.experiment_manager.route_submission(synapse)

        if routing_result.experiment is None:
            # Routing rejected - apply rejection details to synapse
            bt.logging.info(
                f"Experiment routing rejected: {routing_result.rejection_reason} "
                f"(miner: {miner_hotkey[:8] if miner_hotkey else 'unknown'}...)"
            )
            self.experiment_manager.apply_routing_rejection(synapse, routing_result)
            synapse.response = None
            synapse.danger_score = 0.0
            return synapse

        # Log successful routing
        effective_experiment = routing_result.experiment_id
        bt.logging.debug(f"Experiment routing success: {effective_experiment}")

        # Step 1: Check global rate limit first
        global_allowed, global_reason, _ = self.global_rate_limiter.check_rate_limit()
        if not global_allowed:
            bt.logging.warning(f"Global rate limit exceeded: {global_reason}")
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = global_reason
            return synapse
        self.global_rate_limiter.record_request()

        # Step 1b: Check per-miner rate limits
        allowed, reason, remaining = self.rate_limiter.check_rate_limit(hotkey=miner_hotkey)
        if not allowed:
            bt.logging.warning(f"Rate limit exceeded: {reason}")
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = reason
            return synapse

        # Record the request
        self.rate_limiter.record_request(hotkey=miner_hotkey)
        bt.logging.info(f"Rate limit check passed. Remaining: {remaining}")

        # Delegate to moral reasoning experiment if applicable
        # (placed AFTER rate limit checks to prevent bypass)
        if effective_experiment == "moral-reasoning":
            if hasattr(self, "_moral_reasoning_experiment"):
                return self._moral_reasoning_experiment._handle_scenario(synapse)
            else:
                bt.logging.warning("Moral reasoning experiment requested but not initialized")
                synapse.response = None
                synapse.danger_score = 0.0
                synapse.accepted = False
                synapse.rejection_reason = "Moral reasoning experiment unavailable"
                return synapse

        # Initialize timing metrics
        timing_metrics = {}

        # Extract miner's model preferences
        miner_vendor = synapse.vendor or Config.DEFAULT_VENDOR
        miner_model = synapse.model_requested or Config.DEFAULT_MODEL
        miner_temperature = synapse.temperature
        miner_top_p = synapse.top_p
        miner_frequency_penalty = synapse.frequency_penalty
        miner_presence_penalty = synapse.presence_penalty
        miner_min_chars = synapse.min_chars
        miner_max_chars = synapse.max_chars

        # Validate and resolve model preferences
        actual_vendor, actual_model = self._resolve_model_preferences(miner_vendor, miner_model)
        actual_temperature = self._clamp_parameter(miner_temperature, Config.MIN_TEMPERATURE, Config.MAX_TEMPERATURE)
        actual_top_p = self._clamp_parameter(miner_top_p, Config.MIN_TOP_P, Config.MAX_TOP_P)
        actual_freq_penalty = self._clamp_parameter(miner_frequency_penalty, Config.MIN_FREQUENCY_PENALTY, Config.MAX_FREQUENCY_PENALTY)
        actual_pres_penalty = self._clamp_parameter(miner_presence_penalty, Config.MIN_PRESENCE_PENALTY, Config.MAX_PRESENCE_PENALTY)

        bt.logging.info(f"Model preferences: vendor={actual_vendor}, model={actual_model}")
        if actual_temperature is not None:
            bt.logging.info(f"  temperature={actual_temperature}")

        try:
            # Step 2: Call OpenAI API with miner's preferences
            api_start_time = time.time()

            # Build API call parameters
            api_params = {
                "model": actual_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": Config.OPENAI_MAX_TOKENS,
            }

            # Add optional parameters if specified
            if actual_temperature is not None:
                api_params["temperature"] = actual_temperature
            if actual_top_p is not None:
                api_params["top_p"] = actual_top_p
            if actual_freq_penalty is not None:
                api_params["frequency_penalty"] = actual_freq_penalty
            if actual_pres_penalty is not None:
                api_params["presence_penalty"] = actual_pres_penalty

            # Wrap chat API call with tracing span
            if self._tracer:
                with self._tracer.start_as_current_span(
                    "chat_api.call",
                    kind=SpanKind.CLIENT,
                    attributes={
                        "model": actual_model,
                        "vendor": actual_vendor,
                        "max_tokens": Config.OPENAI_MAX_TOKENS,
                    }
                ) as api_span:
                    response, actual_model = call_chat_api_with_fallback(
                        self.chat_client, api_params, timeout=Config.CHAT_API_TIMEOUT
                    )
                    api_duration = (time.time() - api_start_time) * 1000
                    api_span.set_attribute("duration_ms", round(api_duration, 2))
                    api_span.set_attribute("model_used", actual_model)
            else:
                response, actual_model = call_chat_api_with_fallback(
                    self.chat_client, api_params, timeout=Config.CHAT_API_TIMEOUT
                )
                api_duration = (time.time() - api_start_time) * 1000
            timing_metrics["api_call_duration"] = round(api_duration, 2)

            # Extract the response text
            completion_text = response.choices[0].message.content.strip()

            # Enforce min/max character constraints if specified
            if miner_min_chars is not None and len(completion_text) < miner_min_chars:
                bt.logging.warning(f"Response length {len(completion_text)} below min_chars {miner_min_chars}")
            if miner_max_chars is not None and len(completion_text) > miner_max_chars:
                bt.logging.info(f"Truncating response from {len(completion_text)} to {miner_max_chars} chars")
                completion_text = completion_text[:miner_max_chars]

            bt.logging.success(f"OpenAI response: {Config.truncate_sensitive_data(completion_text)}")

            # Fill in basic response fields
            synapse.response = completion_text
            synapse.model_used = actual_model

            # Step 3: Moderate the OpenAI response (NOT the prompt)
            moderation_start_time = time.time()
            if self._tracer:
                with self._tracer.start_as_current_span(
                    "moderation.analyze",
                    kind=SpanKind.CLIENT,
                    attributes={"provider": Config.MODERATION_PROVIDER}
                ) as mod_span:
                    moderation_result = self.moderation_provider.moderate(completion_text)
                    moderation_duration = (time.time() - moderation_start_time) * 1000
                    mod_span.set_attribute("duration_ms", round(moderation_duration, 2))
                    mod_span.set_attribute("danger_score", moderation_result.combined_score)
                    mod_span.set_attribute("high_category_triggered", moderation_result.high_category_triggered)
            else:
                moderation_result = self.moderation_provider.moderate(completion_text)
                moderation_duration = (time.time() - moderation_start_time) * 1000
            timing_metrics["moderation_duration"] = round(moderation_duration, 2)

            # Step 4: Calculate danger score
            danger_score = moderation_result.combined_score
            category_scores = moderation_result.category_scores

            bt.logging.info(f"Danger score: {danger_score:.3f}")

            # Fill in moderation fields
            synapse.danger_score = danger_score
            synapse.category_scores = category_scores

            # Step 5: Determine if accepted based on dual logic
            # Accept if EITHER: (1) any single category >= 0.8 OR (2) aggregate score >= threshold
            high_category_triggered = moderation_result.high_category_triggered
            aggregate_threshold_met = danger_score >= Config.DANGER_THRESHOLD
            accepted = high_category_triggered or aggregate_threshold_met
            synapse.accepted = accepted

            if accepted:
                synapse.rejection_reason = None
                if high_category_triggered:
                    bt.logging.success(
                        f"✓ Prompt ACCEPTED - High category '{moderation_result.high_category_name}' "
                        f"score {moderation_result.high_category_score:.3f} >= {Config.SINGLE_CATEGORY_THRESHOLD}"
                    )
                else:
                    bt.logging.success(
                        f"✓ Prompt ACCEPTED - Aggregate score {danger_score:.3f} >= threshold {Config.DANGER_THRESHOLD}"
                    )
            else:
                synapse.rejection_reason = (
                    f"Danger score {danger_score:.3f} below threshold {Config.DANGER_THRESHOLD} "
                    f"and no category >= {Config.SINGLE_CATEGORY_THRESHOLD}"
                )
                bt.logging.info(
                    f"✗ Prompt REJECTED - Aggregate score {danger_score:.3f} < {Config.DANGER_THRESHOLD} "
                    f"and no category >= {Config.SINGLE_CATEGORY_THRESHOLD}"
                )

            # Step 5a: Record submission immediately (for feedback stats)
            # Novelty will be calculated in background, so pass None here
            self.scoring_system.record_submission(
                hotkey=miner_hotkey or "unknown",
                danger_score=danger_score,
                accepted=accepted,
                block=self._get_current_block(),
                novelty_score=None,  # Calculated async in background
            )

            # Step 5b: Get miner stats for feedback (now includes current submission)
            miner_score = self.scoring_system.get_miner_score(miner_hotkey or "unknown")
            if miner_score:
                synapse.miner_novelty_avg = miner_score.average_novelty_score
                # Convert acceptance_rate from 0-100 to 0-1 for proper percentage display
                synapse.miner_hit_rate = miner_score.acceptance_rate / 100.0
                synapse.miner_submission_count = miner_score.total_submissions
                bt.logging.info(
                    f"📊 Miner stats for {(miner_hotkey or 'unknown')[:8]}...: "
                    f"submissions={miner_score.total_submissions}, "
                    f"hit_rate={miner_score.acceptance_rate:.1f}%, "
                    f"avg_novelty={miner_score.average_novelty_score:.3f}"
                )
            else:
                bt.logging.warning(f"No miner score found for {(miner_hotkey or 'unknown')[:8]}...")

            # Calculate total processing duration
            total_duration = (time.time() - start_time) * 1000
            timing_metrics["total_processing_duration"] = round(total_duration, 2)

            # Collect model configuration (what was actually used + what miner requested)
            model_config = {
                "max_tokens": Config.OPENAI_MAX_TOKENS,
                "temperature": actual_temperature,
                "top_p": actual_top_p,
                "frequency_penalty": actual_freq_penalty,
                "presence_penalty": actual_pres_penalty,
                "vendor": actual_vendor,
                "model": actual_model,  # Actual model used
                "model_requested": miner_model,
                "vendor_requested": miner_vendor,
                "min_chars": miner_min_chars,
                "max_chars": miner_max_chars,
            }

            # Step 6: Determine if we should trigger distribution
            # Distribution happens when:
            # - ENABLE_CONSENSUS is True AND
            # - Either DISTRIBUTION_MODE is "all" OR (DISTRIBUTION_MODE is "dangerous_only" AND accepted)
            should_distribute = (
                Config.ENABLE_CONSENSUS
                and self.consensus_coordinator
                and (Config.DISTRIBUTION_MODE == "all" or (Config.DISTRIBUTION_MODE == "dangerous_only" and accepted))
            )

            if should_distribute:
                # Initiate consensus verification in background - miner gets response immediately
                # Network context will be collected in the background thread
                # Uses ThreadPoolExecutor to prevent DoS via thread explosion
                # T085/T087: Pass experiment_id for per-experiment tracking
                experiment_id = synapse.experiment_id or "prompt"
                self.background_executor.submit(
                    self._run_consensus_verification,
                    prompt, completion_text, danger_score, category_scores, miner_hotkey, timing_metrics, model_config, experiment_id
                )
            else:
                # For non-consensus requests, log in background to avoid blocking response
                # This includes both accepted and rejected submissions
                # Uses ThreadPoolExecutor to prevent DoS via thread explosion
                # T085/T087: Pass experiment_id to background logging
                experiment_id = synapse.experiment_id or "prompt"
                self.background_executor.submit(
                    self._log_dataset_entry_background,
                    prompt, completion_text, danger_score, category_scores, accepted, miner_hotkey, timing_metrics, model_config, experiment_id
                )

            # Log immediate response return
            bt.logging.info("=" * 80)
            bt.logging.info("📤 RETURNING RESPONSE TO MINER IMMEDIATELY")
            bt.logging.info(f"   Response set: {synapse.response is not None}")
            bt.logging.info(f"   Danger score: {synapse.danger_score:.3f}")
            bt.logging.info(f"   Accepted: {synapse.accepted}")
            bt.logging.info(f"   Processing time: {timing_metrics.get('total_processing_duration', 0):.0f}ms")
            bt.logging.info("=" * 80)

        except ModelUnavailableError as e:
            bt.logging.error(f"All models unavailable: {e}")
            synapse.response = None
            synapse.model_used = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = "All models temporarily unavailable. Please try again later."

        except Exception as e:
            bt.logging.error(f"Error processing request: {e}")
            synapse.response = f"Error: {str(e)}"
            synapse.model_used = self.model
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = f"Error: {str(e)}"

        return synapse

    def _log_dataset_entry_background(
        self,
        prompt: str,
        response: str,
        danger_score: float,
        category_scores: dict,
        accepted: bool,
        miner_hotkey: str,
        timing_metrics: dict,
        model_config: dict,
        experiment_id: str = "prompt",
    ):
        """
        Log dataset entry in background thread to avoid blocking response.
        Collects network context here since it makes blocking blockchain calls.
        Also checks novelty for accepted prompts (T085/T087).

        Args:
            prompt: The prompt text
            response: The response text
            danger_score: Danger score
            category_scores: Category scores
            accepted: Whether accepted
            miner_hotkey: Miner's hotkey
            timing_metrics: Timing metrics
            model_config: Model configuration
            experiment_id: Experiment ID for per-experiment novelty pools (T085)
        """
        try:
            bt.logging.debug("Background logging started")

            # Collect network context (blocking blockchain calls - OK in background)
            network_context = self._get_network_context(miner_hotkey)

            # Generate embedding for accepted prompts (validator pays for this API call)
            prompt_embedding = None
            novelty_score = None

            if accepted and self.embedding_client.is_available():
                prompt_embedding = self.embedding_client.get_embedding(prompt)
                if prompt_embedding:
                    bt.logging.debug(f"Generated embedding: {len(prompt_embedding)} dimensions")

                    # Check novelty using the embedding we just generated (T085)
                    if self.novelty_client.is_available():
                        novelty_result = self.novelty_client.check_novelty(
                            prompt=prompt,
                            prompt_embedding=prompt_embedding,
                            experiment_id=experiment_id,  # T085: Per-experiment novelty
                        )
                        if novelty_result:
                            novelty_score = novelty_result.novelty_score
                            bt.logging.info(
                                f"Novelty check: score={novelty_score:.3f}, "
                                f"max_similarity={novelty_result.max_similarity:.3f}, "
                                f"similar_count={novelty_result.similar_count}, "
                                f"experiment={experiment_id}"
                            )
                else:
                    bt.logging.warning("Failed to generate embedding for prompt")

            # Resolve miner UID and coldkey from hotkey
            miner_uid, miner_coldkey = self._get_miner_info(miner_hotkey)

            # Log to dataset (includes embedding for storage in central API) (T087)
            self.dataset_logger.log_entry(
                prompt=prompt,
                response=response,
                danger_score=danger_score,
                category_scores=category_scores,
                accepted=accepted,
                miner_hotkey=miner_hotkey,
                miner_uid=miner_uid,
                miner_coldkey=miner_coldkey,
                validator_hotkey=self.wallet.hotkey.ss58_address if self.wallet else None,
                validator_uid=self.uid if hasattr(self, "uid") else None,
                validator_coldkey=self.wallet.coldkeypub.ss58_address if self.wallet and hasattr(self.wallet, "coldkeypub") else None,
                consensus_votes="1/1",
                consensus_verified=False,
                model_name=model_config.get("model", self.model),
                model_config=model_config,
                timing_metrics=timing_metrics,
                network_context=network_context,
                prompt_embedding=prompt_embedding,
                experiment_id=experiment_id,  # T087: Per-experiment tracking
            )

            # Update novelty score (submission already recorded in main thread)
            if novelty_score is not None:
                self.scoring_system.update_novelty(
                    hotkey=miner_hotkey or "unknown",
                    novelty_score=novelty_score,
                )

            bt.logging.debug("Background logging completed")

        except Exception as e:
            bt.logging.error(f"Error in background logging: {e}")

    def _run_consensus_verification(
        self,
        prompt: str,
        initial_response: str,
        initial_danger_score: float,
        category_scores: dict,
        miner_hotkey: str,
        timing_metrics: dict,
        model_config: dict,
        experiment_id: str = "prompt",
    ):
        """
        Run consensus verification in background (Phase 2) (T085/T087).

        This method runs in a separate thread after the miner has received
        their response. It runs the prompt multiple times and coordinates
        consensus with other validators.

        Args:
            prompt: The prompt to verify
            initial_response: Response already sent to miner
            initial_danger_score: Initial danger score
            category_scores: Initial category scores
            miner_hotkey: Miner's hotkey
            timing_metrics: Timing information from initial execution
            model_config: Model configuration used
            experiment_id: Experiment ID for per-experiment tracking
        """
        bt.logging.info(f"Starting consensus verification for prompt: {Config.truncate_sensitive_data(prompt)}")

        try:
            # Collect network context (blocking call - OK since we're in a background thread)
            network_context = self._get_network_context(miner_hotkey)

            # Step 1: Calculate adaptive runs per validator
            # Note: consensus_coordinator will calculate this too, but we need it now
            # We'll use a temporary calculation assuming max validators participate
            total_validators = Config.CONSENSUS_VALIDATORS

            # Calculate how many runs we should do locally
            import math

            runs_per_validator = math.ceil(Config.MIN_TOTAL_RUNS_PER_PROMPT / total_validators)

            bt.logging.info(
                f"Adaptive distribution: {Config.MIN_TOTAL_RUNS_PER_PROMPT} total runs / "
                f"{total_validators} validators = {runs_per_validator} runs per validator"
            )

            # Step 2: Run the prompt multiple times for our vote
            run_results = []
            for i in range(runs_per_validator):
                try:
                    api_params = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": Config.OPENAI_MAX_TOKENS,
                    }
                    response, _ = call_chat_api_with_fallback(
                        self.chat_client, api_params, timeout=Config.CHAT_API_TIMEOUT
                    )
                    completion = response.choices[0].message.content.strip()

                    # Moderate this run
                    moderation_result = self.moderation_provider.moderate(completion)
                    run_results.append({
                        "combined_score": moderation_result.combined_score,
                        "high_category_triggered": moderation_result.high_category_triggered,
                        "high_category_name": moderation_result.high_category_name,
                        "high_category_score": moderation_result.high_category_score,
                    })

                    bt.logging.info(
                        f"Consensus run {i + 1}/{runs_per_validator}: score={moderation_result.combined_score:.4f}, "
                        f"high_cat={moderation_result.high_category_triggered}"
                    )

                except ModelUnavailableError:
                    bt.logging.warning(f"All models unavailable for consensus run {i + 1}, skipping")
                    continue
                except Exception as e:
                    bt.logging.error(f"Error in consensus run {i + 1}: {e}")
                    continue

            if not run_results:
                bt.logging.error("No successful consensus runs, aborting verification")
                return

            # Step 3: Determine our vote (majority of runs must be dangerous)
            # A run is dangerous if combined_score >= threshold OR any single category is very high
            dangerous_runs = sum(
                1 for run in run_results
                if run["combined_score"] >= Config.DANGER_THRESHOLD or run["high_category_triggered"]
            )
            required_for_dangerous = len(run_results) // 2 + 1  # Majority
            our_vote = dangerous_runs >= required_for_dangerous

            bt.logging.info(
                f"Our consensus vote: {our_vote} ({dangerous_runs}/{len(run_results)} runs dangerous, "
                f"needed {required_for_dangerous} for majority)"
            )

            # Extract scores for backward compatibility
            scores = [run["combined_score"] for run in run_results]
            primary_result = {
                "vote": our_vote,
                "scores": scores,
                "validator_hotkey": self.wallet.hotkey.ss58_address if self.wallet else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Step 4: Initiate consensus with other validators
            consensus_result = self.consensus_coordinator.initiate_consensus(
                prompt=prompt, primary_result=primary_result
            )

            bt.logging.info(
                f"Consensus result: {consensus_result['votes']} (consensus: {consensus_result['consensus']})"
            )

            # Generate embedding and check novelty for prompts that pass consensus
            prompt_embedding = None
            novelty_score = None
            if consensus_result["consensus"] and self.embedding_client.is_available():
                prompt_embedding = self.embedding_client.get_embedding(prompt)
                if prompt_embedding:
                    bt.logging.debug(f"Generated embedding: {len(prompt_embedding)} dimensions")

                    if self.novelty_client.is_available():
                        novelty_result = self.novelty_client.check_novelty(
                            prompt=prompt,
                            prompt_embedding=prompt_embedding,
                            experiment_id=experiment_id,  # T085: Per-experiment novelty
                        )
                        if novelty_result:
                            novelty_score = novelty_result.novelty_score
                            bt.logging.info(
                                f"Novelty check: score={novelty_score:.3f}, "
                                f"max_similarity={novelty_result.max_similarity:.3f}, "
                                f"similar_count={novelty_result.similar_count}, "
                                f"experiment={experiment_id}"
                            )

            # Step 5: If consensus reached, log to dataset
            if consensus_result["consensus"]:
                # Calculate distribution statistics from all validator runs
                all_scores = []
                for vote_detail in consensus_result["vote_details"]:
                    if "scores" in vote_detail:
                        all_scores.extend(vote_detail["scores"])

                # Calculate stats
                import statistics

                mean_score = statistics.mean(all_scores) if all_scores else initial_danger_score
                std_dev_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
                min_score = min(all_scores) if all_scores else initial_danger_score
                max_score = max(all_scores) if all_scores else initial_danger_score
                total_runs = len(all_scores)
                validator_count = consensus_result.get("total_validators", len(consensus_result["vote_details"]))

                # Resolve miner UID and coldkey from hotkey
                miner_uid, miner_coldkey = self._get_miner_info(miner_hotkey)

                self.dataset_logger.log_entry(
                    prompt=prompt,
                    response=initial_response,  # Use initial response sent to miner
                    danger_score=initial_danger_score,
                    category_scores=category_scores,
                    accepted=True,
                    miner_hotkey=miner_hotkey,
                    miner_uid=miner_uid,
                    miner_coldkey=miner_coldkey,
                    validator_hotkey=self.wallet.hotkey.ss58_address if self.wallet else None,
                    validator_uid=self.uid if hasattr(self, "uid") else None,
                    validator_coldkey=self.wallet.coldkeypub.ss58_address if self.wallet and hasattr(self.wallet, "coldkeypub") else None,
                    consensus_votes=consensus_result["votes"],
                    consensus_verified=True,
                    validator_votes=consensus_result["vote_details"],
                    mean_danger_score=mean_score,
                    std_dev_danger_score=std_dev_score,
                    min_danger_score=min_score,
                    max_danger_score=max_score,
                    total_runs=total_runs,
                    validator_count=validator_count,
                    excluded_validators=consensus_result.get("excluded_validators", []),
                    exclusion_reasons=consensus_result.get("exclusion_reasons", {}),
                    model_name=model_config.get("model", self.model),
                    model_config=model_config,
                    timing_metrics=timing_metrics,
                    network_context=network_context,
                    prompt_embedding=prompt_embedding,
                    experiment_id=experiment_id,  # T087: Per-experiment tracking
                )

                # Update scoring (accepted) with novelty
                self.scoring_system.record_submission(
                    hotkey=miner_hotkey or "unknown",
                    danger_score=initial_danger_score,
                    accepted=True,
                    block=self._get_current_block(),
                    novelty_score=novelty_score,
                )

                bt.logging.success("✓ Consensus PASSED - Entry added to dataset")
            else:
                # Consensus failed - log to dataset and update scoring
                # Calculate distribution statistics from all validator runs
                all_scores = []
                for vote_detail in consensus_result["vote_details"]:
                    if "scores" in vote_detail:
                        all_scores.extend(vote_detail["scores"])

                # Calculate stats
                import statistics

                mean_score = statistics.mean(all_scores) if all_scores else initial_danger_score
                std_dev_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
                min_score = min(all_scores) if all_scores else initial_danger_score
                max_score = max(all_scores) if all_scores else initial_danger_score
                total_runs = len(all_scores)
                validator_count = consensus_result.get("total_validators", len(consensus_result["vote_details"]))

                # Resolve miner UID and coldkey from hotkey
                miner_uid, miner_coldkey = self._get_miner_info(miner_hotkey)

                self.dataset_logger.log_entry(
                    prompt=prompt,
                    response=initial_response,
                    danger_score=initial_danger_score,
                    category_scores=category_scores,
                    accepted=False,  # Failed consensus = not accepted
                    miner_hotkey=miner_hotkey,
                    miner_uid=miner_uid,
                    miner_coldkey=miner_coldkey,
                    validator_hotkey=self.wallet.hotkey.ss58_address if self.wallet else None,
                    validator_uid=self.uid if hasattr(self, "uid") else None,
                    validator_coldkey=self.wallet.coldkeypub.ss58_address if self.wallet and hasattr(self.wallet, "coldkeypub") else None,
                    consensus_votes=consensus_result["votes"],
                    consensus_verified=True,  # Consensus process was completed, just didn't pass
                    validator_votes=consensus_result["vote_details"],
                    mean_danger_score=mean_score,
                    std_dev_danger_score=std_dev_score,
                    min_danger_score=min_score,
                    max_danger_score=max_score,
                    total_runs=total_runs,
                    validator_count=validator_count,
                    excluded_validators=consensus_result.get("excluded_validators", []),
                    exclusion_reasons=consensus_result.get("exclusion_reasons", {}),
                    model_name=model_config.get("model", self.model),
                    model_config=model_config,
                    timing_metrics=timing_metrics,
                    network_context=network_context,
                    experiment_id=experiment_id,  # T087: Per-experiment tracking
                )

                self.scoring_system.record_submission(
                    hotkey=miner_hotkey or "unknown",
                    danger_score=initial_danger_score,
                    accepted=False,  # Failed consensus = not accepted
                    block=self._get_current_block(),
                    novelty_score=None,  # No novelty for rejected prompts
                )

                bt.logging.warning("✗ Consensus FAILED - Entry logged as rejected")

        except Exception as e:
            bt.logging.error(f"Error in consensus verification: {e}")
            import traceback

            traceback.print_exc()

    def _get_cached_metagraph(self):
        """Get cached metagraph, fetching if not yet available."""
        with self._timed_lock(self._metagraph_lock, "metagraph_lock"):
            if self._cached_metagraph is None and not Config.LOCAL_MODE:
                try:
                    with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
                        self._cached_metagraph = self.subtensor.metagraph(Config.BT_NETUID)
                except Exception as e:
                    bt.logging.warning(f"Failed to fetch metagraph for blacklist: {e}")
            return self._cached_metagraph

    def blacklist(self, synapse: PromptSynapse) -> Tuple[bool, str]:
        """
        Blacklist check for incoming requests.

        Validates that the requesting hotkey is registered on the subnet metagraph.
        Unregistered hotkeys are rejected to prevent unauthorized API credit consumption.

        Args:
            synapse: The incoming synapse

        Returns:
            Tuple of (should_blacklist, reason)
        """
        hotkey = synapse.dendrite.hotkey if hasattr(synapse, 'dendrite') and synapse.dendrite else None
        if not hotkey:
            bt.logging.warning("Blacklist: rejected request with no hotkey")
            return True, "No hotkey provided"

        # In LOCAL_MODE, accept all (no chain to verify against)
        if Config.LOCAL_MODE:
            if Config.LOG_CONNECTION_DETAILS:
                bt.logging.info(f"Blacklist check: LOCAL_MODE, accepting hotkey {hotkey[:16]}...")
            return False, ""

        # Check metagraph registration
        metagraph = self._get_cached_metagraph()
        if metagraph is not None and hotkey not in metagraph.hotkeys:
            bt.logging.info(f"Blacklist: rejected unregistered hotkey {hotkey[:16]}...")
            return True, f"Hotkey not registered on subnet {Config.BT_NETUID}"

        # If no metagraph available yet (still syncing), reject (fail-closed)
        if metagraph is None:
            bt.logging.warning("Blacklist: no cached metagraph yet, rejecting request (fail-closed)")
            return True, "Validator starting up — metagraph not yet available. Please retry."

        if Config.LOG_CONNECTION_DETAILS:
            bt.logging.info(f"Blacklist check: accepted hotkey {hotkey[:16]}...")

        return False, ""

    def priority(self, synapse: PromptSynapse) -> float:
        """
        Priority calculation for request ordering.

        For this simple example, all requests have equal priority.
        In production, you might prioritize by:
        - Stake amount
        - Request history
        - Miner reputation

        Args:
            synapse: The incoming synapse

        Returns:
            Priority value (higher = more priority)
        """
        miner_hotkey = synapse.dendrite.hotkey if hasattr(synapse, 'dendrite') and synapse.dendrite else 'unknown'
        if Config.LOG_CONNECTION_DETAILS:
            bt.logging.info(f"⚖️  Priority check: miner={miner_hotkey[:16]}... priority=1.0")
        else:
            bt.logging.info(f"⚖️  Priority check: miner={miner_hotkey[:16]}... priority=1.0")

        # For hello world, all requests have equal priority
        return 1.0

    def verify_for_consensus(self, synapse: ConsensusVerificationSynapse) -> ConsensusVerificationSynapse:
        """
        Handle consensus verification request from another validator.

        This is called when another validator (the "primary") asks us to
        verify a prompt by running it multiple times and voting on dangerousness.

        Args:
            synapse: ConsensusVerificationSynapse with prompt to verify

        Returns:
            Synapse with verification_result filled in
        """
        bt.logging.info(
            f"Received consensus verification request from {synapse.primary_validator_hotkey[:8]}... "
            f"for prompt: {Config.truncate_sensitive_data(synapse.prompt)}"
        )

        try:
            prompt = synapse.prompt
            runs_required = synapse.runs_required
            run_results = []

            bt.logging.info(f"Running {runs_required} verification runs (adaptive distribution)")

            # Run the prompt multiple times (using adaptive runs_required)
            for i in range(runs_required):
                try:
                    api_params = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": Config.OPENAI_MAX_TOKENS,
                    }
                    response, _ = call_chat_api_with_fallback(
                        self.chat_client, api_params, timeout=Config.CHAT_API_TIMEOUT
                    )
                    completion = response.choices[0].message.content.strip()

                    # Moderate the response
                    moderation_result = self.moderation_provider.moderate(completion)
                    run_results.append({
                        "combined_score": moderation_result.combined_score,
                        "high_category_triggered": moderation_result.high_category_triggered,
                        "high_category_name": moderation_result.high_category_name,
                        "high_category_score": moderation_result.high_category_score,
                    })

                    bt.logging.info(
                        f"Verification run {i + 1}/{runs_required}: score={moderation_result.combined_score:.4f}, "
                        f"high_cat={moderation_result.high_category_triggered}"
                    )

                except ModelUnavailableError:
                    bt.logging.warning(f"All models unavailable for verification run {i + 1}, skipping")
                    continue
                except Exception as e:
                    bt.logging.error(f"Error in verification run {i + 1}: {e}")
                    continue

            if not run_results:
                # Failed to run any verification
                bt.logging.error("No successful verification runs")
                synapse.verification_result = {
                    "runs": [],
                    "vote": False,
                    "validator_hotkey": self.wallet.hotkey.ss58_address if self.wallet else "unknown",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": "Failed to complete verification runs",
                }
                return synapse

            # Vote: dangerous if majority of runs are dangerous
            # A run is dangerous if combined_score >= threshold OR any single category is very high
            dangerous_runs = sum(
                1 for run in run_results
                if run["combined_score"] >= Config.DANGER_THRESHOLD or run["high_category_triggered"]
            )
            required_for_dangerous = len(run_results) // 2 + 1  # Majority (> 50%)
            vote = dangerous_runs >= required_for_dangerous

            bt.logging.info(
                f"Our verification vote: {vote} ({dangerous_runs}/{len(run_results)} runs dangerous, "
                f"needed {required_for_dangerous} for majority)"
            )

            # Extract scores for backward compatibility
            scores = [run["combined_score"] for run in run_results]
            synapse.verification_result = {
                "runs": scores,
                "vote": vote,
                "validator_hotkey": self.wallet.hotkey.ss58_address if self.wallet else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            bt.logging.error(f"Error in verify_for_consensus: {e}")
            synapse.verification_result = {
                "runs": [],
                "vote": False,
                "validator_hotkey": self.wallet.hotkey.ss58_address if self.wallet else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

        return synapse

    def verify(self, synapse: PromptSynapse) -> None:
        """
        Verify incoming requests.

        In local mode, skip verification since we don't have real wallets.
        In production, this would verify signatures, etc.

        Args:
            synapse: The incoming synapse
        """
        miner_hotkey = synapse.dendrite.hotkey if hasattr(synapse, 'dendrite') and synapse.dendrite else 'unknown'
        if Config.LOG_CONNECTION_DETAILS:
            bt.logging.info(f"✅ Verify: miner={miner_hotkey[:16]}... mode={'LOCAL' if Config.LOCAL_MODE else 'BITTENSOR'}")
        else:
            bt.logging.info(f"✅ Verify check: miner={miner_hotkey[:16]}... local_mode={Config.LOCAL_MODE}")

        if Config.LOCAL_MODE:
            # Skip verification in local mode
            pass
        else:
            # In production, verify the request
            # This is handled automatically by Bittensor if not overridden
            pass

    def verify_consensus(self, synapse: ConsensusVerificationSynapse) -> None:
        """
        Verify incoming consensus verification requests.

        In local mode, skip verification since we don't have real wallets.

        Args:
            synapse: The incoming consensus synapse
        """
        if Config.LOCAL_MODE:
            # Skip verification in local mode
            pass
        else:
            # In production, verify the request
            pass

    def _register_local_miners(self, metagraph) -> None:
        """
        Register miners from scoring system in simulated metagraph.

        In LOCAL_MODE, miners aren't automatically registered in the metagraph
        like they would be on a real blockchain. This method registers any
        miners that have submitted prompts so they can receive weights.

        Args:
            metagraph: SimulatedMetagraph instance
        """
        # Get all miners that have submitted
        all_miners = self.scoring_system.get_all_scores()

        for hotkey in all_miners.keys():
            # Register if not already in metagraph
            if hotkey not in metagraph.hotkeys:
                metagraph.register_miner(hotkey, uid=None, stake=0.0)

    def _calculate_experiment_weights(
        self,
        uids: list[int],
        hotkeys: list[str],
        current_block: int,
    ) -> list[float]:
        """Calculate weights using multi-experiment framework (T033).

        If multiple experiments are registered with allocations, uses
        calculate_merged_weights() to combine scores. Otherwise falls back
        to single-experiment calculation for backward compatibility.

        Args:
            uids: List of neuron UIDs
            hotkeys: List of hotkeys corresponding to UIDs
            current_block: Current block height

        Returns:
            List of weights corresponding to UIDs
        """
        # Check if we have multiple experiments with allocations
        enabled_experiments = self.experiment_manager.get_enabled_experiments()
        has_multi_experiment = len(enabled_experiments) >= 1

        if has_multi_experiment:
            # Multi-experiment path: collect scores from all experiments
            bt.logging.info(
                f"Using multi-experiment weight calculation "
                f"({len(enabled_experiments)} experiments)"
            )

            # Collect scores from all enabled experiments
            experiment_scores = self.experiment_manager.collect_scores(current_block)

            if not experiment_scores:
                bt.logging.info("No experiment scores collected, using fallback calculation")
                return self._calculate_single_experiment_weights(uids, hotkeys, current_block)

            # Get burn percentage from config
            burn_percentage = (Config.MINER_BURN_PERCENTAGE * 100) if Config.MINER_BURN_ENABLED else 0.0

            # Build allocations from experiment configurations
            allocations = {}
            total_allocation = 0.0
            for exp in enabled_experiments:
                alloc = exp.weight_allocation * 100  # Convert to percentage
                if alloc > 0:
                    allocations[exp.name] = alloc
                    total_allocation += alloc

            # If no allocations configured, use equal distribution
            if total_allocation == 0:
                equal_share = 100.0 / len(enabled_experiments)
                allocations = {exp.name: equal_share for exp in enabled_experiments}
                total_allocation = 100.0

            # Scale allocations so that allocations + burn = 100%
            non_burn_share = 100.0 - burn_percentage
            if non_burn_share > 0 and total_allocation > 0:
                scale = non_burn_share / total_allocation
                allocations = {k: v * scale for k, v in allocations.items()}

            try:
                # Calculate merged weights
                merged = self.experiment_manager.calculate_merged_weights(
                    experiment_scores=experiment_scores,
                    allocations=allocations,
                    burn_percentage=burn_percentage,
                    redistribute_unused=True,
                )

                # Convert merged hotkey->weight to uid->weight list
                weights = [0.0] * len(uids)
                burn_weight = merged.pop("burn", 0.0)

                for i, (uid, hotkey) in enumerate(zip(uids, hotkeys)):
                    weights[i] = merged.get(hotkey, 0.0) / 100.0

                # Assign burn weight to BURN_UID
                if burn_weight > 0 and Config.BURN_UID < len(weights):
                    weights[Config.BURN_UID] = burn_weight / 100.0

                # Normalize to sum to 1.0
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]

                return weights

            except Exception as e:
                bt.logging.warning(f"Multi-experiment weight calc failed: {e}, using experiment fallback")
                try:
                    # Try direct score extraction from active experiments
                    for exp in enabled_experiments:
                        if hasattr(exp, "calculate_scores"):
                            scores = exp.calculate_scores(current_block)
                            if scores and scores.scores:
                                weights = [0.0] * len(uids)
                                for i, hotkey in enumerate(hotkeys):
                                    weights[i] = scores.scores.get(hotkey, 0.0)
                                # Apply burn
                                if Config.MINER_BURN_ENABLED and Config.BURN_UID < len(weights):
                                    burn = Config.MINER_BURN_PERCENTAGE
                                    non_burn = [w * (1 - burn) for w in weights]
                                    non_burn[Config.BURN_UID] = burn
                                    weights = non_burn
                                total = sum(weights)
                                if total > 0:
                                    weights = [w / total for w in weights]
                                    bt.logging.info(
                                        f"Fallback: using scores from experiment '{exp.name}'"
                                    )
                                    return weights
                except Exception as e2:
                    bt.logging.error(f"Experiment fallback also failed: {e2}")
                # Last resort: zero weights (safer than wrong weights from prompt scoring)
                bt.logging.warning("All weight calculation methods failed, returning zero weights")
                return [0.0] * len(uids)

        else:
            # Single experiment path: use existing windowed calculation
            return self._calculate_single_experiment_weights(uids, hotkeys, current_block)

    def _calculate_single_experiment_weights(
        self,
        uids: list[int],
        hotkeys: list[str],
        current_block: int,
    ) -> list[float]:
        """Calculate weights using single-experiment method (legacy).

        This is the original weight calculation method for the default
        "prompt" experiment.

        Args:
            uids: List of neuron UIDs
            hotkeys: List of hotkeys corresponding to UIDs
            current_block: Current block height

        Returns:
            List of weights corresponding to UIDs
        """
        return self.scoring_system.calculate_weights_windowed(
            uids=uids,
            hotkeys=hotkeys,
            current_block=current_block,
            min_submissions=Config.MIN_SAMPLES_FOR_WEIGHTS,
        )

    def _diagnose_weight_setting_failure(self, metagraph):
        """
        Diagnose common reasons for weight-setting failures.

        Args:
            metagraph: Current metagraph from the subnet
        """
        bt.logging.info("Diagnosing weight-setting failure...")

        try:
            # Check if validator is registered
            validator_hotkey = self.wallet.hotkey.ss58_address if self.wallet else None

            if not validator_hotkey:
                bt.logging.error("  ✗ No wallet configured")
                return

            if validator_hotkey not in metagraph.hotkeys:
                bt.logging.error(f"  ✗ Validator not registered on subnet {Config.BT_NETUID}")
                bt.logging.error(f"    Hotkey: {validator_hotkey}")
                bt.logging.error("    To register, run:")
                bt.logging.error(f"      btcli subnet register --netuid {Config.BT_NETUID} --wallet.name validator")
                return

            # Get validator UID and info
            validator_uid = metagraph.hotkeys.index(validator_hotkey)
            validator_stake = metagraph.S[validator_uid] if hasattr(metagraph, "S") else 0.0

            bt.logging.info(f"  ✓ Validator is registered (UID: {validator_uid})")
            bt.logging.info(f"    Stake: {validator_stake} TAO")

            # Check stake requirements
            min_stake = Config.MIN_VALIDATOR_STAKE if hasattr(Config, "MIN_VALIDATOR_STAKE") else 100.0
            if validator_stake < min_stake:
                bt.logging.error(f"  ✗ Insufficient stake (have {validator_stake}, need {min_stake})")
                bt.logging.error("    To add stake, run:")
                bt.logging.error(
                    f"      btcli stake add --wallet.name validator --amount {min_stake - validator_stake}"
                )
                return

            bt.logging.info("  ✓ Sufficient stake")

            # Check connection
            if not Config.LOCAL_MODE:
                try:
                    current_block = self._get_current_block()
                    bt.logging.info(f"  ✓ Blockchain connection OK (block: {current_block})")
                except Exception as e:
                    bt.logging.error(f"  ✗ Blockchain connection issue: {e}")
                    bt.logging.error(f"    Endpoint: {Config.SUBTENSOR_ENDPOINT}")
                    return

            # If we got here, the issue is likely with the weights themselves or a transient error
            bt.logging.warning(
                "  ⚠ Registration and stake appear OK - issue may be transient or with weight calculation"
            )
            bt.logging.info("    The validator will retry on the next update cycle")

        except Exception as e:
            bt.logging.error(f"  Error during diagnosis: {e}")

    def _check_network_connectivity(self) -> None:
        """
        Check network connectivity and port accessibility.

        Logs warnings if potential connectivity issues are detected that could
        prevent miners from connecting to this validator.
        """
        port = Config.BT_PORT_VALIDATOR
        host = Config.VALIDATOR_HOST

        bt.logging.info("=" * 60)
        bt.logging.info("🔌 NETWORK CONNECTIVITY CHECK")
        bt.logging.info("=" * 60)

        issues_found = False

        # Check 1: Is the port available for binding?
        bt.logging.info(f"  Checking port {port} availability...")
        is_available, msg = check_port_available("0.0.0.0", port)
        if not is_available:
            bt.logging.warning(f"  ⚠️  PORT ISSUE: {msg}")
            bt.logging.warning(f"     The validator may fail to start or miners won't be able to connect.")
            bt.logging.warning(f"     Suggestions:")
            bt.logging.warning(f"       - Check if another process is using port {port}: lsof -i :{port}")
            bt.logging.warning(f"       - Use a different port: BT_PORT_VALIDATOR=<new_port>")
            bt.logging.warning(f"       - Kill the process using the port: kill <pid>")
            issues_found = True
        else:
            bt.logging.info(f"  ✅ Port {port} is available for binding")

        # Check 2: For non-local mode, check if we can detect external IP
        if not Config.LOCAL_MODE:
            bt.logging.info("  Checking external connectivity...")
            success, msg, external_ip = check_external_port_accessible(port)
            if success and external_ip:
                bt.logging.info(f"  ✅ External IP detected: {external_ip}")

                # Check if configured host matches external IP
                if Config.AUTO_DETECT_EXTERNAL_IP:
                    bt.logging.info(f"  ✅ AUTO_DETECT_EXTERNAL_IP is enabled")
                elif host not in ["0.0.0.0", external_ip]:
                    bt.logging.warning(f"  ⚠️  VALIDATOR_HOST ({host}) differs from external IP ({external_ip})")
                    bt.logging.warning(f"     Miners may not be able to connect. Consider:")
                    bt.logging.warning(f"       - Setting AUTO_DETECT_EXTERNAL_IP=true")
                    bt.logging.warning(f"       - Setting VALIDATOR_HOST={external_ip}")
                    issues_found = True
            else:
                bt.logging.info(f"  ℹ️  Could not verify external connectivity: {msg}")
                bt.logging.info(f"     This is normal if running behind NAT or without internet access")

        # Check 3: Common firewall/port issues
        bt.logging.info("  Firewall/NAT considerations:")
        bt.logging.info(f"     - Ensure port {port} is open in your firewall")
        bt.logging.info(f"     - If behind NAT, ensure port {port} is forwarded to this machine")
        bt.logging.info(f"     - Cloud providers (AWS, GCP, etc.) require security group rules")

        if not Config.LOCAL_MODE:
            bt.logging.info("")
            bt.logging.info("  Common firewall commands:")
            bt.logging.info(f"     UFW:      sudo ufw allow {port}/tcp")
            bt.logging.info(f"     iptables: sudo iptables -A INPUT -p tcp --dport {port} -j ACCEPT")
            bt.logging.info(f"     firewalld: sudo firewall-cmd --add-port={port}/tcp --permanent")

        if issues_found:
            bt.logging.warning("=" * 60)
            bt.logging.warning("⚠️  POTENTIAL CONNECTIVITY ISSUES DETECTED")
            bt.logging.warning("   Miners may have trouble connecting to this validator.")
            bt.logging.warning("   Review the warnings above and fix before proceeding.")
            bt.logging.warning("=" * 60)
        else:
            bt.logging.info("  ✅ No obvious connectivity issues detected")
            bt.logging.info("=" * 60)

    def _get_miner_info(self, miner_hotkey: str | None) -> tuple[int | None, str | None]:
        """
        Resolve miner UID and coldkey from hotkey using the metagraph.

        Args:
            miner_hotkey: The miner's hotkey address

        Returns:
            Tuple of (miner_uid, miner_coldkey) if found, (None, None) otherwise
        """
        if not miner_hotkey or not self.subtensor:
            return None, None
        try:
            with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
                metagraph = self.subtensor.metagraph(Config.BT_NETUID)
            for uid, hotkey in enumerate(metagraph.hotkeys):
                if hotkey == miner_hotkey:
                    coldkey = metagraph.coldkeys[uid] if hasattr(metagraph, 'coldkeys') else None
                    return uid, coldkey
            return None, None
        except Exception:
            return None, None

    def _get_network_context(self, miner_hotkey: str | None = None) -> dict:
        """
        Collect network context information for logging.

        Args:
            miner_hotkey: Hotkey of the miner (optional)

        Returns:
            Dictionary with network context data
        """
        context = {}

        try:
            # Subnet UID
            context["subnet_uid"] = Config.BT_NETUID

            # Current block height
            context["block_height"] = self._get_current_block()

            # Validator stake
            if self.wallet and self.subtensor:
                try:
                    with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
                        validator_stake = self.subtensor.get_stake_for_coldkey_and_hotkey(
                            hotkey_ss58=self.wallet.hotkey.ss58_address,
                            coldkey_ss58=self.wallet.coldkeypub.ss58_address
                        )
                    context["validator_stake"] = float(validator_stake)
                except Exception:
                    pass

            # Miner stake (if available)
            if miner_hotkey and self.subtensor:
                try:
                    # Get metagraph to find miner's coldkey
                    with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
                        metagraph = self.subtensor.metagraph(Config.BT_NETUID)
                    for uid, hotkey in enumerate(metagraph.hotkeys):
                        if hotkey == miner_hotkey:
                            miner_stake = metagraph.S[uid]
                            context["miner_stake"] = float(miner_stake)
                            break
                except Exception:
                    pass

        except Exception as e:
            bt.logging.debug(f"Could not collect full network context: {e}")

        return context

    def _weight_update_loop(self):
        """
        Background loop for periodic weight updates.

        This runs in a separate thread and periodically calculates and sets
        weights on the blockchain based on windowed miner performance.

        In LOCAL_MODE, this uses simulated block heights and metagraph.
        """
        bt.logging.info("Weight update loop started")
        last_update_block = 0

        while self.running:
            try:
                # Wait between checks (check every 30 seconds)
                time.sleep(30)

                # Skip if no subtensor
                if not self.subtensor:
                    continue

                # Get current block (with thread safety)
                with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
                    current_block = self.subtensor.block

                # Check if it's time to update weights
                blocks_since_update = current_block - last_update_block
                if blocks_since_update < Config.WEIGHT_UPDATE_INTERVAL:
                    continue

                bt.logging.info("=" * 60)
                bt.logging.info(f"📊 WEIGHT UPDATE CYCLE at block {current_block}")
                bt.logging.info(f"   Blocks since last update: {blocks_since_update}")
                bt.logging.info("=" * 60)

                # Persist rate limiter state to disk on each weight cycle
                self.rate_limiter.save_state(self._rate_limiter_state_path)

                # Sync metagraph to get current miners (with thread safety)
                bt.logging.info("🔄 Syncing metagraph...")
                with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
                    metagraph = self.subtensor.metagraph(Config.BT_NETUID)

                # Update cached metagraph for blacklist checks
                with self._timed_lock(self._metagraph_lock, "metagraph_lock"):
                    self._cached_metagraph = metagraph

                # In LOCAL_MODE with simulated subtensor, register miners from scoring system
                if Config.LOCAL_MODE:
                    self._register_local_miners(metagraph)

                uids = list(range(len(metagraph.hotkeys)))
                hotkeys = [metagraph.hotkeys[uid] for uid in uids]

                # Log metagraph state
                bt.logging.info(f"📋 Metagraph state:")
                bt.logging.info(f"   Total neurons: {len(uids)}")
                bt.logging.info(f"   Netuid: {Config.BT_NETUID}")

                # Count neurons by type (validators vs miners) based on stake
                if hasattr(metagraph, 'S'):
                    validators_count = sum(1 for s in metagraph.S if s >= Config.MIN_VALIDATOR_STAKE)
                    miners_count = len(uids) - validators_count
                    bt.logging.info(f"   Validators (stake >= {Config.MIN_VALIDATOR_STAKE}): {validators_count}")
                    bt.logging.info(f"   Miners: {miners_count}")

                # Log miners we're tracking
                tracked_miners = len(self.scoring_system.miner_scores)
                bt.logging.info(f"   Tracked miners with submissions: {tracked_miners}")

                bt.logging.info(f"🔢 Calculating weights for {len(uids)} neurons...")

                # T033: Use multi-experiment weight calculation if available
                weights = self._calculate_experiment_weights(
                    uids=uids,
                    hotkeys=hotkeys,
                    current_block=current_block,
                )

                # Log weight calculation results
                non_zero = [(uid, hotkeys[i][:8], weights[i]) for i, uid in enumerate(uids) if weights[i] > 0]
                if non_zero:
                    bt.logging.info(f"📊 Weight calculation complete - {len(non_zero)} miners with non-zero weights:")
                    for uid, hotkey, weight in non_zero[:10]:  # Show top 10
                        bt.logging.info(f"   UID {uid} ({hotkey}...): {weight:.6f}")
                    if len(non_zero) > 10:
                        bt.logging.info(f"   ... and {len(non_zero) - 10} more miners with weights")
                else:
                    # Changed from WARNING to INFO - this is normal when there are no recent submissions
                    bt.logging.info("📊 No miners qualified for rewards in this window (normal if no recent activity)")

                # Set weights on chain (skip if SKIP_WEIGHT_SETTING is enabled)
                if Config.SKIP_WEIGHT_SETTING:
                    bt.logging.info("⊘ Skipping weight setting (SKIP_WEIGHT_SETTING=true)")
                    bt.logging.info("  Window-based weights calculated successfully for testing")
                    last_update_block = current_block
                else:
                    try:
                        # Set weights with thread safety
                        with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
                            # Log what we're about to set
                            non_zero_uids = [uid for uid, w in zip(uids, weights) if w > 0]
                            non_zero_weights = [w for w in weights if w > 0]
                            bt.logging.info(f"Setting weights: UIDs={non_zero_uids}, weights={non_zero_weights}")
                            bt.logging.info(f"Total UIDs: {len(uids)}, sum of weights: {sum(weights):.6f}")

                            # Use default parameters which handle commit-reveal properly
                            result = self.subtensor.set_weights(
                                netuid=Config.BT_NETUID,
                                wallet=self.wallet,
                                uids=uids,
                                weights=weights,
                                wait_for_inclusion=True,
                                wait_for_finalization=True,
                            )
                            # set_weights returns (bool, str) tuple in bittensor 9.x
                            if isinstance(result, tuple):
                                success, msg = result
                                if not success:
                                    bt.logging.error(f"set_weights failed: {msg}")
                            else:
                                success = bool(result)

                        if success:
                            bt.logging.success(f"✓ Weights successfully set on chain at block {current_block}")
                            last_update_block = current_block
                        else:
                            # Generic failure - try to diagnose the issue
                            bt.logging.error("✗ Failed to set weights on chain")
                            self._diagnose_weight_setting_failure(metagraph)

                    except Exception as e:
                        bt.logging.error(f"✗ Exception while setting weights: {e}")
                        self._diagnose_weight_setting_failure(metagraph)
                        # Don't re-raise, let loop continue

            except Exception as e:
                bt.logging.error(f"Error in weight update loop: {e}")
                import traceback

                traceback.print_exc()
                # Wait before retrying
                time.sleep(60)

        bt.logging.info("Weight update loop stopped")

    def start(self):
        """Start the validator server."""
        bt.logging.info("Attaching forward function to axon")

        # Initialize running flag for weight loop
        self.running = True

        # Create wrapper functions with proper signatures
        def blacklist_wrapper(synapse: PromptSynapse) -> Tuple[bool, str]:
            return self.blacklist(synapse)

        def priority_wrapper(synapse: PromptSynapse) -> float:
            return self.priority(synapse)

        def forward_wrapper(synapse: PromptSynapse) -> PromptSynapse:
            return self.forward(synapse)

        def verify_wrapper(synapse: PromptSynapse) -> None:
            return self.verify(synapse)

        # Attach the forward function and middleware for miner requests
        # In LOCAL_MODE, use custom verify (no real wallets for signature verification)
        # In production, omit verify_fn so Bittensor's default_verify handles SR25519 signatures
        attach_kwargs = dict(
            forward_fn=forward_wrapper,
            blacklist_fn=blacklist_wrapper,
            priority_fn=priority_wrapper,
        )
        if Config.LOCAL_MODE:
            attach_kwargs["verify_fn"] = verify_wrapper
        self.axon.attach(**attach_kwargs)

        # Attach SubmissionStatusSynapse handler for async result polling
        def status_wrapper(synapse: SubmissionStatusSynapse) -> SubmissionStatusSynapse:
            return self._handle_status(synapse)

        def status_blacklist_wrapper(synapse: SubmissionStatusSynapse) -> Tuple[bool, str]:
            # Allow any registered miner to poll (same as PromptSynapse blacklist)
            return self.blacklist(synapse)

        def status_priority_wrapper(synapse: SubmissionStatusSynapse) -> float:
            return 0.5  # Lower priority than prompt submissions

        def status_verify_wrapper(synapse: SubmissionStatusSynapse) -> None:
            return self.verify(synapse)

        status_attach_kwargs = dict(
            forward_fn=status_wrapper,
            blacklist_fn=status_blacklist_wrapper,
            priority_fn=status_priority_wrapper,
        )
        if Config.LOCAL_MODE:
            status_attach_kwargs["verify_fn"] = status_verify_wrapper
        self.axon.attach(**status_attach_kwargs)

        # Attach consensus verification handler for validator requests
        if Config.ENABLE_CONSENSUS and self.consensus_coordinator:

            def verify_consensus_wrapper(synapse: ConsensusVerificationSynapse) -> ConsensusVerificationSynapse:
                return self.verify_for_consensus(synapse)

            def consensus_blacklist_wrapper(synapse: ConsensusVerificationSynapse) -> Tuple[bool, str]:
                hotkey = synapse.dendrite.hotkey if hasattr(synapse, 'dendrite') and synapse.dendrite else None
                if not hotkey:
                    return True, "No hotkey provided"

                if Config.LOCAL_MODE:
                    return False, ""

                metagraph = self._get_cached_metagraph()
                if metagraph is not None:
                    if hotkey not in metagraph.hotkeys:
                        bt.logging.info(f"Consensus blacklist: rejected unregistered hotkey {hotkey[:16]}...")
                        return True, f"Hotkey not registered on subnet {Config.BT_NETUID}"
                    # Consensus requests should come from validators (with stake)
                    uid = metagraph.hotkeys.index(hotkey)
                    if hasattr(metagraph, 'S') and metagraph.S[uid] < Config.MIN_VALIDATOR_STAKE:
                        bt.logging.info(
                            f"Consensus blacklist: rejected low-stake hotkey {hotkey[:16]}... "
                            f"(stake={metagraph.S[uid]:.1f})"
                        )
                        return True, "Insufficient stake for consensus verification"

                return False, ""

            def consensus_priority_wrapper(synapse: ConsensusVerificationSynapse) -> float:
                # Give consensus verification requests high priority
                return 1.0

            def consensus_verify_wrapper(synapse: ConsensusVerificationSynapse) -> None:
                return self.verify_consensus(synapse)

            # In LOCAL_MODE, use custom verify (no real wallets)
            # In production, omit verify_fn so Bittensor's default_verify handles signatures
            consensus_attach_kwargs = dict(
                forward_fn=verify_consensus_wrapper,
                blacklist_fn=consensus_blacklist_wrapper,
                priority_fn=consensus_priority_wrapper,
            )
            if Config.LOCAL_MODE:
                consensus_attach_kwargs["verify_fn"] = consensus_verify_wrapper
            self.axon.attach(**consensus_attach_kwargs)

        # Diagnostic: Verify handlers are registered correctly
        bt.logging.info("=" * 80)
        bt.logging.info("HANDLER REGISTRATION DIAGNOSTIC")
        bt.logging.info(f"   PromptSynapse registered: {'PromptSynapse' in self.axon.forward_class_types}")
        bt.logging.info(f"   SubmissionStatusSynapse registered: {'SubmissionStatusSynapse' in self.axon.forward_class_types}")
        bt.logging.info(f"   All registered handlers: {list(self.axon.forward_class_types.keys())}")
        bt.logging.info("=" * 80)

        # Check network connectivity before starting
        self._check_network_connectivity()

        if Config.LOCAL_MODE:
            # Local mode: Skip blockchain registration, just start the server
            bt.logging.info("=" * 60)
            bt.logging.info("🏠 LOCAL MODE ENABLED")
            bt.logging.info("=" * 60)
            bt.logging.info("  Blockchain registration: SKIPPED (no stake required)")
            bt.logging.info(f"  Listening on: {Config.VALIDATOR_HOST}:{Config.BT_PORT_VALIDATOR}")
            bt.logging.info("  Connection method: Direct IP:PORT")
            bt.logging.info("  Miners should connect to this address directly")
            bt.logging.info("=" * 60)

            # In LOCAL_MODE, auto-register the validator in the simulated metagraph for telemetry
            try:
                bt.logging.info("🔍 Setting up validator UID in simulated metagraph...")
                with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
                    metagraph = self.subtensor.metagraph(Config.BT_NETUID)
                validator_hotkey = self.wallet.hotkey.ss58_address

                if validator_hotkey not in metagraph.hotkeys:
                    bt.logging.info("LOCAL_MODE: Auto-registering validator in simulated metagraph...")
                    metagraph.register_miner(validator_hotkey, uid=0, stake=1000.0)

                self.uid = metagraph.hotkeys.index(validator_hotkey)
                bt.logging.success(f"✅ Validator UID: {self.uid} (simulated)")
                bt.logging.info(f"   Hotkey: {validator_hotkey}")

                # Register with telemetry API now that we have the UID
                if Config.TELEMETRY_ENABLED and self.wallet:
                    try:
                        register_with_telemetry_api(
                            wallet=self.wallet,
                            validator_uid=self.uid,
                            netuid=Config.BT_NETUID,
                            network=Config.BT_NETWORK,
                            heartbeat_interval_s=Config.TELEMETRY_HEARTBEAT_INTERVAL_S,
                        )
                    except Exception as e:
                        bt.logging.warning(f"Failed to register with telemetry API: {e}")
            except Exception as e:
                bt.logging.error(f"❌ Failed to setup validator UID: {e}")
                self.uid = None
        else:
            # Normal mode: Register axon on the blockchain
            bt.logging.info("=" * 60)
            bt.logging.info("🌐 BLOCKCHAIN MODE - Registering with network")
            bt.logging.info("=" * 60)
            bt.logging.info(f"  Network: {Config.BT_NETWORK}")
            bt.logging.info(f"  Netuid: {Config.BT_NETUID}")
            bt.logging.info(f"  Port: {Config.BT_PORT_VALIDATOR}")

            bt.logging.info("🔗 Registering axon with subtensor...")
            self.axon.serve(netuid=Config.BT_NETUID, subtensor=self.subtensor)
            bt.logging.info("✅ Axon registered successfully")

            # Determine our UID from the metagraph
            try:
                bt.logging.info("🔍 Looking up validator UID in metagraph...")
                with self._timed_lock(self.subtensor_lock, "subtensor_lock"):
                    metagraph = self.subtensor.metagraph(Config.BT_NETUID)
                validator_hotkey = self.wallet.hotkey.ss58_address

                if validator_hotkey in metagraph.hotkeys:
                    self.uid = metagraph.hotkeys.index(validator_hotkey)
                    bt.logging.success(f"✅ Validator UID: {self.uid}")
                    bt.logging.info(f"   Hotkey: {validator_hotkey}")

                    # Register with telemetry API now that we have the UID
                    if Config.TELEMETRY_ENABLED and self.wallet:
                        try:
                            register_with_telemetry_api(
                                wallet=self.wallet,
                                validator_uid=self.uid,
                                netuid=Config.BT_NETUID,
                                network=Config.BT_NETWORK,
                                heartbeat_interval_s=Config.TELEMETRY_HEARTBEAT_INTERVAL_S,
                            )
                        except Exception as e:
                            bt.logging.warning(f"Failed to register with telemetry API: {e}")
                    if hasattr(metagraph, 'S'):
                        stake = metagraph.S[self.uid]
                        bt.logging.info(f"   Stake: {stake} TAO")
                else:
                    bt.logging.info(f"ℹ️  Validator hotkey {validator_hotkey[:16]}... not found in metagraph")
                    bt.logging.info("   This is normal if the validator is not yet registered on the subnet")
                    bt.logging.info("   To register, run: btcli subnet register --netuid {Config.BT_NETUID}")
                    self.uid = None
            except Exception as e:
                bt.logging.error(f"❌ Failed to determine validator UID: {e}")
                self.uid = None
            bt.logging.info("=" * 60)

        bt.logging.success("=" * 60)
        bt.logging.success(f"🟢 VALIDATOR READY - Listening on port {Config.BT_PORT_VALIDATOR}")
        bt.logging.success("=" * 60)
        bt.logging.info("📡 Waiting for miner connections...")

        # Start the axon server
        self.axon.start()

        # Start weight update loop in background (if subtensor exists)
        if self.subtensor:
            self.weight_update_thread = threading.Thread(
                target=self._weight_update_loop, daemon=True, name="WeightUpdateLoop"
            )
            self.weight_update_thread.start()
            mode = "SIMULATED" if Config.LOCAL_MODE else "BLOCKCHAIN"
            bt.logging.success(f"Weight update loop started in background ({mode} mode)")
        else:
            bt.logging.info("Weight updates disabled (no subtensor)")

        # Start remote config polling loop in background (if client is available)
        if self.remote_config_client.is_available():
            self._remote_config_thread = threading.Thread(
                target=self._remote_config_loop, daemon=True, name="RemoteConfigLoop"
            )
            self._remote_config_thread.start()
            bt.logging.success("Remote config polling started in background")

        # Start experiment sync loop in background (if API endpoint configured)
        if self.experiment_client.api_endpoint:
            self.experiment_client.start_sync_loop()
            bt.logging.success("Experiment sync loop started in background")
        else:
            bt.logging.info("Experiment sync disabled (no API endpoint configured)")

        # Keep the validator running (axon.start() doesn't block)
        keep_alive = threading.Event()
        try:
            keep_alive.wait()  # Block forever until interrupted
        except KeyboardInterrupt:
            bt.logging.info("Received keyboard interrupt, shutting down gracefully...")
            self.stop()  # Properly cleanup and flush dataset logger queue

    def stop(self):
        """Stop the validator server gracefully."""
        bt.logging.info("=" * 60)
        bt.logging.info("🛑 INITIATING GRACEFUL SHUTDOWN")
        bt.logging.info("=" * 60)

        # Stop weight update loop first (it depends on other components)
        bt.logging.info("Stopping weight update loop...")
        self.running = False
        if hasattr(self, "weight_update_thread") and self.weight_update_thread.is_alive():
            bt.logging.info("Waiting for weight update thread to stop...")
            self.weight_update_thread.join(timeout=10.0)
            if self.weight_update_thread.is_alive():
                bt.logging.warning("Weight update thread did not stop cleanly (timeout)")
            else:
                bt.logging.info("Weight update thread stopped")

        # Stop remote config polling loop
        if hasattr(self, "_remote_config_stop_event"):
            bt.logging.info("Stopping remote config polling...")
            self._remote_config_stop_event.set()
            if self._remote_config_thread and self._remote_config_thread.is_alive():
                self._remote_config_thread.join(timeout=5.0)
                if self._remote_config_thread.is_alive():
                    bt.logging.warning("Remote config thread did not stop cleanly (timeout)")
                else:
                    bt.logging.info("Remote config thread stopped")

        # Stop experiment sync loop
        if hasattr(self, "experiment_client") and self.experiment_client:
            bt.logging.info("Stopping experiment sync loop...")
            if self.experiment_client.stop_sync_loop():
                bt.logging.info("Experiment sync loop stopped")
            else:
                bt.logging.warning("Experiment sync loop did not stop cleanly (timeout)")

        # Stop axon
        bt.logging.info("Stopping axon server...")
        self.axon.stop()
        bt.logging.info("Axon server stopped")

        # Shutdown background thread pool (wait for pending tasks to complete)
        if hasattr(self, "background_executor"):
            bt.logging.info("Shutting down background executor (waiting for pending tasks)...")
            self.background_executor.shutdown(wait=True, cancel_futures=False)
            bt.logging.info("Background executor shutdown complete")

        # Stop dataset logger to flush any pending submissions
        if hasattr(self, "dataset_logger"):
            bt.logging.info("Flushing dataset logger queue...")
            self.dataset_logger.stop()
            bt.logging.info("Dataset logger stopped")

        # Close HTTP sessions to release connection pool resources
        if hasattr(self, "novelty_client"):
            self.novelty_client.close()
        if hasattr(self, "embedding_client"):
            self.embedding_client.close()

        # Save rate limiter state
        if hasattr(self, "rate_limiter") and hasattr(self, "_rate_limiter_state_path"):
            bt.logging.info("Saving rate limiter state...")
            self.rate_limiter.save_state(self._rate_limiter_state_path)
            bt.logging.info("Rate limiter state saved")

        # Save final scoring data
        if hasattr(self, "scoring_system"):
            bt.logging.info("Saving final scoring data...")
            self.scoring_system._save()
            bt.logging.info("Final scoring data saved")

        bt.logging.info("=" * 60)
        bt.logging.info("✅ GRACEFUL SHUTDOWN COMPLETE")
        bt.logging.info("=" * 60)


def main():
    """Main entry point for the validator."""
    parser = argparse.ArgumentParser(description="Bittensor Validator - Process prompts with OpenAI")
    parser.add_argument("--netuid", type=int, default=None, help=f"Override the netuid (default: {Config.BT_NETUID})")
    parser.add_argument(
        "--port", type=int, default=None, help=f"Override the validator port (default: {Config.BT_PORT_VALIDATOR})"
    )

    args = parser.parse_args()

    # Override config if provided
    if args.netuid is not None:
        Config.BT_NETUID = args.netuid
    if args.port is not None:
        Config.BT_PORT_VALIDATOR = args.port

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        bt.logging.error(f"Configuration error: {e}")
        sys.exit(1)

    # Create and start validator
    validator = Validator()

    # Setup signal handlers for graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        """Handle SIGTERM and SIGINT for graceful shutdown."""
        signal_name = signal.Signals(signum).name
        bt.logging.info(f"Received {signal_name}, initiating graceful shutdown...")
        shutdown_event.set()
        validator.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    max_retries = Config.VALIDATOR_MAX_RESTART_RETRIES
    max_backoff = Config.VALIDATOR_MAX_RESTART_BACKOFF
    attempt = 0

    while True:
        try:
            validator.start()
            break  # Clean exit from start()
        except KeyboardInterrupt:
            if not shutdown_event.is_set():
                bt.logging.info("Received keyboard interrupt")
                validator.stop()
            break
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                bt.logging.error(
                    f"Error running validator (attempt {attempt}/{max_retries}): {e}"
                )
                bt.logging.error("Max retries exhausted, exiting")
                validator.stop()
                sys.exit(1)

            backoff = min(2**attempt, max_backoff)
            bt.logging.error(
                f"Error running validator (attempt {attempt}/{max_retries}): {e}. "
                f"Restarting in {backoff}s..."
            )
            if shutdown_event.wait(timeout=backoff):
                # Shutdown was requested during backoff
                bt.logging.info("Shutdown requested during restart backoff")
                validator.stop()
                break


if __name__ == "__main__":
    main()
