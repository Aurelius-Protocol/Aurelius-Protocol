"""Validator implementation - processes prompts using configurable chat providers."""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Tuple

import bittensor as bt
from openai import OpenAI

from aurelius.shared.config import Config
from aurelius.shared.consensus import ConsensusCoordinator
from aurelius.shared.dataset_logger import DatasetLogger
from aurelius.shared.moderation import create_moderation_provider
from aurelius.shared.embedding_client import get_embedding_client
from aurelius.shared.novelty_client import get_novelty_client
from aurelius.shared.protocol import ConsensusVerificationSynapse, PromptSynapse
from aurelius.shared.rate_limiter import PerMinerRateLimiter, RateLimitConfig
from aurelius.shared.scoring import ScoringSystem


class Validator:
    """Validator that processes prompts using configurable chat providers (Chutes.ai or OpenAI)."""

    def __init__(self):
        """Initialize the validator."""
        # Setup logging based on LOG_LEVEL configuration
        Config.setup_logging()

        bt.logging.info(f"Initializing validator with wallet: {Config.VALIDATOR_WALLET_NAME}")

        # Validate configuration
        Config.validate()

        # Check for production configuration warnings
        production_warnings = Config.validate_production()
        if production_warnings:
            bt.logging.warning("=" * 80)
            bt.logging.warning("PRODUCTION CONFIGURATION WARNINGS:")
            for warning in production_warnings:
                bt.logging.warning(f"  - {warning}")
            bt.logging.warning("=" * 80)

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

        # Initialize dataset logger
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

    def _get_current_block(self):
        """Safely get current block number with thread locking."""
        if not self.subtensor:
            return None
        with self.subtensor_lock:
            return self.subtensor.block

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
            Clamped value or None if input was None
        """
        if value is None:
            return None
        clamped = max(min_val, min(max_val, value))
        if clamped != value:
            bt.logging.warning(f"Parameter clamped from {value} to {clamped} (range: {min_val}-{max_val})")
        return clamped

    def forward(self, synapse: PromptSynapse) -> PromptSynapse:
        """
        Process an incoming prompt request with moderation and scoring.

        Flow:
        1. Check rate limits
        2. Get OpenAI response to prompt
        3. Moderate the OpenAI response (not the prompt)
        4. Calculate danger score
        5. Determine if accepted based on threshold
        6. Log to dataset and update scores
        7. Return enriched synapse

        Args:
            synapse: The PromptSynapse containing the prompt

        Returns:
            The synapse with response and moderation data filled in
        """
        import time

        # Start timing the entire processing
        start_time = time.time()

        prompt = synapse.prompt

        bt.logging.info("=" * 80)
        bt.logging.info("🚀 FORWARD METHOD CALLED - REQUEST REACHED HANDLER!")
        bt.logging.info(f"   Synapse name: {synapse.name}")
        bt.logging.info(f"   Prompt: {Config.truncate_sensitive_data(prompt)}")
        bt.logging.info("=" * 80)

        # Get miner hotkey for tracking
        miner_hotkey = synapse.dendrite.hotkey if hasattr(synapse, "dendrite") and synapse.dendrite else None

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

        # Step 1: Check rate limits
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

            response = self.chat_client.chat.completions.create(**api_params)
            api_duration = (time.time() - api_start_time) * 1000  # Convert to milliseconds
            timing_metrics["api_call_duration"] = round(api_duration, 2)

            # Extract the response text
            completion_text = response.choices[0].message.content.strip()

            # Enforce min/max character constraints if specified
            if miner_min_chars is not None and len(completion_text) < miner_min_chars:
                bt.logging.warning(f"Response length {len(completion_text)} below min_chars {miner_min_chars}")
            if miner_max_chars is not None and len(completion_text) > miner_max_chars:
                bt.logging.info(f"Truncating response from {len(completion_text)} to {miner_max_chars} chars")
                completion_text = completion_text[:miner_max_chars]

            bt.logging.success(f"OpenAI response: {completion_text}")

            # Fill in basic response fields
            synapse.response = completion_text
            synapse.model_used = actual_model

            # Step 3: Moderate the OpenAI response (NOT the prompt)
            moderation_start_time = time.time()
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
                self.background_executor.submit(
                    self._run_consensus_verification,
                    prompt, completion_text, danger_score, category_scores, miner_hotkey, timing_metrics, model_config
                )
            else:
                # For non-consensus requests, log in background to avoid blocking response
                # This includes both accepted and rejected submissions
                # Uses ThreadPoolExecutor to prevent DoS via thread explosion
                self.background_executor.submit(
                    self._log_dataset_entry_background,
                    prompt, completion_text, danger_score, category_scores, accepted, miner_hotkey, timing_metrics, model_config
                )

            # Log immediate response return
            bt.logging.info("=" * 80)
            bt.logging.info("📤 RETURNING RESPONSE TO MINER IMMEDIATELY")
            bt.logging.info(f"   Response set: {synapse.response is not None}")
            bt.logging.info(f"   Danger score: {synapse.danger_score:.3f}")
            bt.logging.info(f"   Accepted: {synapse.accepted}")
            bt.logging.info(f"   Processing time: {timing_metrics.get('total_processing_duration', 0):.0f}ms")
            bt.logging.info("=" * 80)

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
    ):
        """
        Log dataset entry in background thread to avoid blocking response.
        Collects network context here since it makes blocking blockchain calls.
        Also checks novelty for accepted prompts.

        Args:
            prompt: The prompt text
            response: The response text
            danger_score: Danger score
            category_scores: Category scores
            accepted: Whether accepted
            miner_hotkey: Miner's hotkey
            timing_metrics: Timing metrics
            model_config: Model configuration
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

                    # Check novelty using the embedding we just generated
                    if self.novelty_client.is_available():
                        novelty_result = self.novelty_client.check_novelty(
                            prompt=prompt,
                            prompt_embedding=prompt_embedding,
                        )
                        if novelty_result:
                            novelty_score = novelty_result.novelty_score
                            bt.logging.info(
                                f"Novelty check: score={novelty_score:.3f}, "
                                f"max_similarity={novelty_result.max_similarity:.3f}, "
                                f"similar_count={novelty_result.similar_count}"
                            )
                else:
                    bt.logging.warning("Failed to generate embedding for prompt")

            # Resolve miner UID and coldkey from hotkey
            miner_uid, miner_coldkey = self._get_miner_info(miner_hotkey)

            # Log to dataset (includes embedding for storage in central API)
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
            )

            # Update scoring system with novelty
            self.scoring_system.record_submission(
                hotkey=miner_hotkey or "unknown",
                danger_score=danger_score,
                accepted=accepted,
                block=self._get_current_block(),
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
    ):
        """
        Run consensus verification in background (Phase 2).

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
            scores = []
            for i in range(runs_per_validator):
                try:
                    response = self.chat_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=Config.OPENAI_MAX_TOKENS,
                    )
                    completion = response.choices[0].message.content.strip()

                    # Moderate this run
                    moderation_result = self.moderation_provider.moderate(completion)
                    scores.append(moderation_result.combined_score)

                    bt.logging.info(
                        f"Consensus run {i + 1}/{runs_per_validator}: score={moderation_result.combined_score:.4f}"
                    )

                except Exception as e:
                    bt.logging.error(f"Error in consensus run {i + 1}: {e}")
                    continue

            if not scores:
                bt.logging.error("No successful consensus runs, aborting verification")
                return

            # Step 3: Determine our vote (majority of runs must exceed threshold)
            dangerous_runs = sum(1 for s in scores if s >= Config.DANGER_THRESHOLD)
            required_for_dangerous = len(scores) // 2 + 1  # Majority
            our_vote = dangerous_runs >= required_for_dangerous

            bt.logging.info(
                f"Our consensus vote: {our_vote} ({dangerous_runs}/{len(scores)} runs dangerous, "
                f"needed {required_for_dangerous} for majority)"
            )

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
                        )
                        if novelty_result:
                            novelty_score = novelty_result.novelty_score
                            bt.logging.info(
                                f"Novelty check: score={novelty_score:.3f}, "
                                f"max_similarity={novelty_result.max_similarity:.3f}, "
                                f"similar_count={novelty_result.similar_count}"
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

    def blacklist(self, synapse: PromptSynapse) -> Tuple[bool, str]:
        """
        Blacklist check for incoming requests.

        For this simple example, we accept all requests.
        In production, you might check:
        - Stake amounts
        - Rate limiting
        - Request validity

        Args:
            synapse: The incoming synapse

        Returns:
            Tuple of (should_blacklist, reason)
        """
        if Config.LOG_CONNECTION_DETAILS:
            bt.logging.info("=" * 80)
            bt.logging.info("🔍 BLACKLIST CHECK - INCOMING CONNECTION")
            bt.logging.info(f"   Synapse name: {synapse.name}")
            bt.logging.info(f"   Synapse type: {type(synapse).__name__}")
            if hasattr(synapse, 'dendrite') and synapse.dendrite:
                bt.logging.info(f"   Miner hotkey: {synapse.dendrite.hotkey}")
                bt.logging.info(f"   Miner IP: {synapse.dendrite.ip}")
                bt.logging.info(f"   Miner port: {getattr(synapse.dendrite, 'port', 'N/A')}")
                bt.logging.info(f"   Miner version: {getattr(synapse.dendrite, 'version', 'N/A')}")
            else:
                bt.logging.info("   Dendrite: None (direct connection or missing info)")
            if hasattr(synapse, 'axon') and synapse.axon:
                bt.logging.info(f"   Axon IP: {synapse.axon.ip}")
                bt.logging.info(f"   Axon port: {synapse.axon.port}")
            bt.logging.info("   Result: ACCEPTED (not blacklisted)")
            bt.logging.info("=" * 80)
        else:
            bt.logging.debug(f"Blacklist check: synapse={synapse.name}, accepted=True")

        # For hello world, accept all requests
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
            bt.logging.debug(f"Priority check: miner={miner_hotkey[:16]}... priority=1.0")

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
            f"for prompt: {synapse.prompt[:50]}..."
        )

        try:
            prompt = synapse.prompt
            runs_required = synapse.runs_required
            scores = []

            bt.logging.info(f"Running {runs_required} verification runs (adaptive distribution)")

            # Run the prompt multiple times (using adaptive runs_required)
            for i in range(runs_required):
                try:
                    response = self.chat_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=Config.OPENAI_MAX_TOKENS,
                    )
                    completion = response.choices[0].message.content.strip()

                    # Moderate the response
                    moderation_result = self.moderation_provider.moderate(completion)
                    scores.append(moderation_result.combined_score)

                    bt.logging.info(
                        f"Verification run {i + 1}/{runs_required}: score={moderation_result.combined_score:.4f}"
                    )

                except Exception as e:
                    bt.logging.error(f"Error in verification run {i + 1}: {e}")
                    continue

            if not scores:
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

            # Vote: dangerous if majority of runs exceed threshold
            dangerous_runs = sum(1 for s in scores if s >= Config.DANGER_THRESHOLD)
            required_for_dangerous = len(scores) // 2 + 1  # Majority (> 50%)
            vote = dangerous_runs >= required_for_dangerous

            bt.logging.info(
                f"Our verification vote: {vote} ({dangerous_runs}/{len(scores)} runs dangerous, "
                f"needed {required_for_dangerous} for majority)"
            )

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
            bt.logging.debug(f"Verify check: miner={miner_hotkey[:16]}... local_mode={Config.LOCAL_MODE}")

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
            with self.subtensor_lock:
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
                    with self.subtensor_lock:
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
                    with self.subtensor_lock:
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
                with self.subtensor_lock:
                    current_block = self.subtensor.block

                # Check if it's time to update weights
                blocks_since_update = current_block - last_update_block
                if blocks_since_update < Config.WEIGHT_UPDATE_INTERVAL:
                    continue

                bt.logging.info("=" * 60)
                bt.logging.info(f"📊 WEIGHT UPDATE CYCLE at block {current_block}")
                bt.logging.info(f"   Blocks since last update: {blocks_since_update}")
                bt.logging.info("=" * 60)

                # Sync metagraph to get current miners (with thread safety)
                bt.logging.info("🔄 Syncing metagraph...")
                with self.subtensor_lock:
                    metagraph = self.subtensor.metagraph(Config.BT_NETUID)

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

                # Calculate weights using windowed method
                weights = self.scoring_system.calculate_weights_windowed(
                    uids=uids,
                    hotkeys=hotkeys,
                    current_block=current_block,
                    min_submissions=Config.MIN_SAMPLES_FOR_WEIGHTS,
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
                        with self.subtensor_lock:
                            success = self.subtensor.set_weights(
                                netuid=Config.BT_NETUID,
                                wallet=self.wallet,
                                uids=uids,
                                weights=weights,
                                wait_for_inclusion=False,
                                wait_for_finalization=False,
                                version_key=0,
                            )

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
        self.axon.attach(
            forward_fn=forward_wrapper,
            blacklist_fn=blacklist_wrapper,
            priority_fn=priority_wrapper,
            verify_fn=verify_wrapper,
        )

        # Attach consensus verification handler for validator requests
        if Config.ENABLE_CONSENSUS and self.consensus_coordinator:

            def verify_consensus_wrapper(synapse: ConsensusVerificationSynapse) -> ConsensusVerificationSynapse:
                return self.verify_for_consensus(synapse)

            def consensus_blacklist_wrapper(synapse: ConsensusVerificationSynapse) -> Tuple[bool, str]:
                # For consensus verification from other validators, be permissive
                # Only blacklist if obvious spam/attack
                return False, ""

            def consensus_priority_wrapper(synapse: ConsensusVerificationSynapse) -> float:
                # Give consensus verification requests high priority
                return 1.0

            def consensus_verify_wrapper(synapse: ConsensusVerificationSynapse) -> None:
                return self.verify_consensus(synapse)

            self.axon.attach(
                forward_fn=verify_consensus_wrapper,
                blacklist_fn=consensus_blacklist_wrapper,
                priority_fn=consensus_priority_wrapper,
                verify_fn=consensus_verify_wrapper,
            )

        # Diagnostic: Verify handlers are registered correctly
        bt.logging.info("=" * 80)
        bt.logging.info("📋 HANDLER REGISTRATION DIAGNOSTIC")
        bt.logging.info(f"   PromptSynapse in forward_class_types: {'PromptSynapse' in self.axon.forward_class_types}")
        bt.logging.info(f"   PromptSynapse in blacklist_fns: {'PromptSynapse' in self.axon.blacklist_fns}")
        bt.logging.info(f"   PromptSynapse in priority_fns: {'PromptSynapse' in self.axon.priority_fns}")
        bt.logging.info(f"   PromptSynapse in verify_fns: {'PromptSynapse' in self.axon.verify_fns}")
        bt.logging.info(f"   PromptSynapse in forward_fns: {'PromptSynapse' in self.axon.forward_fns}")
        bt.logging.info(f"   All registered handlers: {list(self.axon.forward_class_types.keys())}")
        bt.logging.info("=" * 80)

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
                with self.subtensor_lock:
                    metagraph = self.subtensor.metagraph(Config.BT_NETUID)
                validator_hotkey = self.wallet.hotkey.ss58_address
                if validator_hotkey in metagraph.hotkeys:
                    self.uid = metagraph.hotkeys.index(validator_hotkey)
                    bt.logging.success(f"✅ Validator UID: {self.uid}")
                    bt.logging.info(f"   Hotkey: {validator_hotkey}")
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

        # Keep the validator running (axon.start() doesn't block)
        keep_alive = threading.Event()
        try:
            keep_alive.wait()  # Block forever until interrupted
        except KeyboardInterrupt:
            bt.logging.info("Received keyboard interrupt, shutting down gracefully...")
            self.stop()  # Properly cleanup and flush dataset logger queue

    def stop(self):
        """Stop the validator server."""
        bt.logging.info("Stopping validator")

        # Stop weight update loop
        self.running = False

        # Stop axon
        self.axon.stop()

        # Shutdown background thread pool (wait for pending tasks to complete)
        if hasattr(self, "background_executor"):
            bt.logging.info("Shutting down background executor...")
            self.background_executor.shutdown(wait=True, cancel_futures=False)
            bt.logging.info("Background executor shutdown complete")

        # Stop dataset logger to flush any pending submissions
        if hasattr(self, "dataset_logger"):
            self.dataset_logger.stop()

        # Save final scoring data
        if hasattr(self, "scoring_system"):
            self.scoring_system._save()
            bt.logging.info("Final scoring data saved")


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

    try:
        validator.start()
    except KeyboardInterrupt:
        bt.logging.info("Received keyboard interrupt")
        validator.stop()
    except Exception as e:
        bt.logging.error(f"Error running validator: {e}")
        validator.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
