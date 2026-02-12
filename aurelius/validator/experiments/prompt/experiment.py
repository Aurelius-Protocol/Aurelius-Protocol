"""PromptExperiment - Push-based experiment for dangerous prompt discovery."""

from __future__ import annotations

import concurrent.futures
import math
import statistics
import time
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

import bittensor as bt
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from aurelius.shared.chat_client import ModelUnavailableError, call_chat_api_with_fallback
from aurelius.shared.config import Config
from aurelius.shared.protocol import ConsensusVerificationSynapse, PromptSynapse
from aurelius.validator.experiments.base import (
    ExperimentConfig,
    ExperimentScores,
    ExperimentType,
    PushExperiment,
)
from aurelius.validator.experiments.prompt.scoring import PromptScoringSystem

if TYPE_CHECKING:
    from aurelius.validator.core import ValidatorCore

# Default timeout for consensus verification (seconds)
CONSENSUS_TIMEOUT = 120


class PromptExperiment(PushExperiment):
    """Push-based experiment for dangerous prompt discovery.

    This experiment implements the original validator functionality:
    - Miners send prompts to the validator
    - Validator gets LLM response and moderates it
    - Prompts above danger threshold are accepted
    - Consensus verification with other validators (optional)
    - Miners are scored based on danger scores and novelty
    """

    NAME = "prompt"
    TYPE = ExperimentType.PUSH

    def __init__(self, core: ValidatorCore, config: ExperimentConfig):
        super().__init__(core, config)

        # Cache settings for use in handlers
        self._danger_threshold = self.setting("danger_threshold", Config.DANGER_THRESHOLD)
        self._max_prompt_length = self.setting("max_prompt_length", Config.MAX_PROMPT_LENGTH)

        # Initialize experiment-specific scoring system
        self.scoring_system = PromptScoringSystem(
            danger_threshold=self._danger_threshold,
            persistence_path=self.setting("persistence_path", Config.MINER_SCORES_PATH),
            window_blocks=self.setting("window_blocks", Config.WINDOW_BLOCKS),
            history_retention_blocks=self.setting(
                "history_retention_blocks", Config.HISTORY_RETENTION_BLOCKS
            ),
        )

    def start(self) -> None:
        """Start the prompt experiment."""
        super().start()

        # Also attach consensus verification handler if enabled
        if Config.ENABLE_CONSENSUS and self.core.consensus_coordinator:
            self._attach_consensus_handlers()

        bt.logging.info(
            f"PromptExperiment started (danger_threshold={self.setting('danger_threshold', Config.DANGER_THRESHOLD)})"
        )

    def stop(self) -> None:
        """Stop the experiment and save state."""
        super().stop()
        self.scoring_system._save()
        bt.logging.info("PromptExperiment stopped, scoring data saved")

    def calculate_scores(self, current_block: int) -> ExperimentScores:
        """Calculate normalized scores for all miners.

        Args:
            current_block: Current blockchain block height

        Returns:
            ExperimentScores with normalized scores per miner
        """
        normalized = self.scoring_system.calculate_normalized_scores(
            current_block=current_block,
            min_submissions=self.setting("min_submissions", Config.MIN_SAMPLES_FOR_WEIGHTS),
        )

        return ExperimentScores(
            scores=normalized,
            experiment_name=self.NAME,
            block_height=current_block,
        )

    def get_stats(self) -> dict:
        """Return experiment-specific statistics."""
        stats = self.scoring_system.get_stats()
        stats["experiment_name"] = self.NAME
        stats["experiment_type"] = self.TYPE.value
        stats["weight_allocation"] = self.weight_allocation
        return stats

    def _create_forward_handler(self) -> Callable:
        """Create the forward handler for PromptSynapse."""

        def forward_handler(synapse: PromptSynapse) -> PromptSynapse:
            return self._handle_prompt(synapse)

        return forward_handler

    def _create_blacklist_handler(self) -> Callable:
        """Create the blacklist handler."""

        def blacklist_handler(synapse: PromptSynapse) -> tuple[bool, str]:
            if Config.LOG_CONNECTION_DETAILS:
                bt.logging.info(f"Blacklist check: synapse={synapse.name}, accepted=True")
            return False, ""

        return blacklist_handler

    def _create_priority_handler(self) -> Callable:
        """Create the priority handler."""

        def priority_handler(synapse: PromptSynapse) -> float:
            return 1.0

        return priority_handler

    def _create_verify_handler(self) -> Callable:
        """Create the verify handler."""

        def verify_handler(synapse: PromptSynapse) -> None:
            if Config.LOCAL_MODE:
                pass  # Skip verification in local mode

        return verify_handler

    def _attach_consensus_handlers(self) -> None:
        """Attach consensus verification handlers to axon."""

        def verify_consensus_handler(
            synapse: ConsensusVerificationSynapse,
        ) -> ConsensusVerificationSynapse:
            return self._handle_consensus_verification(synapse)

        def consensus_blacklist(synapse: ConsensusVerificationSynapse) -> tuple[bool, str]:
            return False, ""

        def consensus_priority(synapse: ConsensusVerificationSynapse) -> float:
            return 1.0

        def consensus_verify(synapse: ConsensusVerificationSynapse) -> None:
            pass

        self.core.axon.attach(
            forward_fn=verify_consensus_handler,
            blacklist_fn=consensus_blacklist,
            priority_fn=consensus_priority,
            verify_fn=consensus_verify,
        )
        bt.logging.info("Consensus verification handlers attached")

    def _handle_prompt(self, synapse: PromptSynapse) -> PromptSynapse:
        """Handle incoming prompt request.

        Args:
            synapse: The PromptSynapse containing the prompt

        Returns:
            The synapse with response and moderation data filled in
        """
        start_time = time.time()
        prompt = synapse.prompt
        miner_hotkey = (
            synapse.dendrite.hotkey
            if hasattr(synapse, "dendrite") and synapse.dendrite
            else None
        )

        # Create tracing span
        span_context = None
        if self.core._tracer:
            span_context = self.core._tracer.start_as_current_span(
                "prompt_experiment.forward",
                kind=SpanKind.SERVER,
                attributes={
                    "miner.hotkey": miner_hotkey[:16] if miner_hotkey else "unknown",
                    "prompt.length": len(prompt),
                },
            )
            span_context.__enter__()

        try:
            result = self._process_prompt(synapse, miner_hotkey, prompt, start_time)
            if span_context:
                current_span = trace.get_current_span()
                current_span.set_status(Status(StatusCode.OK))
                current_span.set_attribute("response.accepted", result.accepted)
                current_span.set_attribute("response.danger_score", result.danger_score or 0.0)
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

    def _process_prompt(
        self,
        synapse: PromptSynapse,
        miner_hotkey: str | None,
        prompt: str,
        start_time: float,
    ) -> PromptSynapse:
        """Process the prompt through the full pipeline."""
        bt.logging.info(f"Processing prompt from {(miner_hotkey or 'unknown')[:8]}...")

        # Input validation
        validation_result = self._validate_input(synapse, miner_hotkey, prompt)
        if validation_result is not None:
            return validation_result

        # Rate limit check
        allowed, reason, remaining = self.core.rate_limiter.check_rate_limit(hotkey=miner_hotkey)
        if not allowed:
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = reason
            return synapse

        self.core.rate_limiter.record_request(hotkey=miner_hotkey)

        timing_metrics = {}

        # Resolve model preferences
        actual_vendor, actual_model = self._resolve_model_preferences(
            synapse.vendor or Config.DEFAULT_VENDOR,
            synapse.model_requested or Config.DEFAULT_MODEL,
        )

        # Build API parameters
        api_params = self._build_api_params(synapse, actual_model)

        try:
            # Call chat API with fallback
            api_start_time = time.time()
            response, actual_model = call_chat_api_with_fallback(
                self.core.chat_client, api_params, timeout=Config.CHAT_API_TIMEOUT
            )
            timing_metrics["api_call_duration"] = round((time.time() - api_start_time) * 1000, 2)

            completion_text = response.choices[0].message.content.strip()

            # Enforce max_chars constraint
            if synapse.max_chars and len(completion_text) > synapse.max_chars:
                completion_text = completion_text[: synapse.max_chars]

            synapse.response = completion_text
            synapse.model_used = actual_model

            # Moderate response
            mod_start_time = time.time()
            moderation_result = self.core.moderation_provider.moderate(completion_text)
            timing_metrics["moderation_duration"] = round((time.time() - mod_start_time) * 1000, 2)

            danger_score = moderation_result.combined_score
            category_scores = moderation_result.category_scores

            synapse.danger_score = danger_score
            synapse.category_scores = category_scores

            # Determine acceptance
            high_category_triggered = moderation_result.high_category_triggered
            aggregate_threshold_met = danger_score >= self._danger_threshold
            accepted = high_category_triggered or aggregate_threshold_met
            synapse.accepted = accepted

            if accepted:
                synapse.rejection_reason = None
                bt.logging.success(f"Prompt ACCEPTED - danger_score={danger_score:.3f}")
            else:
                synapse.rejection_reason = (
                    f"Danger score {danger_score:.3f} below threshold {self._danger_threshold}"
                )

            # Record submission
            self.scoring_system.record_submission(
                hotkey=miner_hotkey or "unknown",
                danger_score=danger_score,
                accepted=accepted,
                block=self.core.get_current_block(),
                novelty_score=None,
            )

            # Get miner stats for feedback
            miner_score = self.scoring_system.get_miner_score(miner_hotkey or "unknown")
            if miner_score:
                synapse.miner_novelty_avg = miner_score.average_novelty_score
                synapse.miner_hit_rate = miner_score.acceptance_rate / 100.0
                synapse.miner_submission_count = miner_score.total_submissions

            timing_metrics["total_processing_duration"] = round((time.time() - start_time) * 1000, 2)

            # Build model config for logging
            model_config = {
                "max_tokens": Config.OPENAI_MAX_TOKENS,
                "vendor": actual_vendor,
                "model": actual_model,
            }

            # Background processing
            should_distribute = (
                Config.ENABLE_CONSENSUS
                and self.core.consensus_coordinator
                and (
                    Config.DISTRIBUTION_MODE == "all"
                    or (Config.DISTRIBUTION_MODE == "dangerous_only" and accepted)
                )
            )

            if should_distribute:
                self.core.submit_background_task(
                    self._run_consensus_verification,
                    prompt,
                    completion_text,
                    danger_score,
                    category_scores,
                    miner_hotkey,
                    timing_metrics,
                    model_config,
                )
            else:
                self.core.submit_background_task(
                    self._log_dataset_entry_background,
                    prompt,
                    completion_text,
                    danger_score,
                    category_scores,
                    accepted,
                    miner_hotkey,
                    timing_metrics,
                    model_config,
                )

            bt.logging.info(f"Response returned in {timing_metrics.get('total_processing_duration', 0):.0f}ms")

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
            synapse.model_used = self.core.model
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = f"Error: {str(e)}"

        return synapse

    def _validate_input(
        self,
        synapse: PromptSynapse,
        miner_hotkey: str | None,
        prompt: str,
    ) -> PromptSynapse | None:
        """Validate input parameters. Returns synapse with error if invalid, None if valid."""
        # Check prompt length
        if len(prompt) > self._max_prompt_length:
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = f"Prompt exceeds maximum length ({self._max_prompt_length} chars)"
            return synapse

        # Check max_chars
        if synapse.max_chars and synapse.max_chars > Config.MAX_RESPONSE_CHARS_LIMIT:
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = f"max_chars exceeds limit ({Config.MAX_RESPONSE_CHARS_LIMIT})"
            return synapse

        # Check min_chars vs max_chars
        if synapse.min_chars and synapse.max_chars and synapse.min_chars > synapse.max_chars:
            synapse.response = None
            synapse.danger_score = 0.0
            synapse.accepted = False
            synapse.rejection_reason = "min_chars cannot be greater than max_chars"
            return synapse

        return None

    def _resolve_model_preferences(self, vendor: str, model: str) -> tuple[str, str]:
        """Validate and resolve miner's model preferences."""
        if vendor not in Config.ALLOWED_MODELS:
            return Config.DEFAULT_VENDOR, Config.DEFAULT_MODEL

        allowed_models = Config.ALLOWED_MODELS[vendor]
        if model not in allowed_models:
            if Config.DEFAULT_MODEL in allowed_models:
                return vendor, Config.DEFAULT_MODEL
            elif allowed_models:
                return vendor, allowed_models[0]
            else:
                return Config.DEFAULT_VENDOR, Config.DEFAULT_MODEL

        return vendor, model

    def _build_api_params(self, synapse: PromptSynapse, model: str) -> dict:
        """Build API parameters for chat completion."""
        params = {
            "model": model,
            "messages": [{"role": "user", "content": synapse.prompt}],
            "max_tokens": Config.OPENAI_MAX_TOKENS,
        }

        # Add optional parameters with clamping
        if synapse.temperature is not None:
            params["temperature"] = max(
                Config.MIN_TEMPERATURE,
                min(Config.MAX_TEMPERATURE, synapse.temperature),
            )
        if synapse.top_p is not None:
            params["top_p"] = max(Config.MIN_TOP_P, min(Config.MAX_TOP_P, synapse.top_p))
        if synapse.frequency_penalty is not None:
            params["frequency_penalty"] = max(
                Config.MIN_FREQUENCY_PENALTY,
                min(Config.MAX_FREQUENCY_PENALTY, synapse.frequency_penalty),
            )
        if synapse.presence_penalty is not None:
            params["presence_penalty"] = max(
                Config.MIN_PRESENCE_PENALTY,
                min(Config.MAX_PRESENCE_PENALTY, synapse.presence_penalty),
            )

        return params

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
        """Log dataset entry in background thread."""
        try:
            network_context = self.core.get_network_context(miner_hotkey)

            # Generate embedding for accepted prompts
            prompt_embedding = None
            novelty_score = None

            if accepted and self.core.embedding_client.is_available():
                prompt_embedding = self.core.embedding_client.get_embedding(prompt)
                if prompt_embedding and self.core.novelty_client.is_available():
                    novelty_result = self.core.novelty_client.check_novelty(
                        prompt=prompt,
                        prompt_embedding=prompt_embedding,
                    )
                    if novelty_result:
                        novelty_score = novelty_result.novelty_score

            miner_uid, miner_coldkey = self.core.get_miner_info(miner_hotkey)

            self.core.dataset_logger.log_entry(
                prompt=prompt,
                response=response,
                danger_score=danger_score,
                category_scores=category_scores,
                accepted=accepted,
                miner_hotkey=miner_hotkey,
                miner_uid=miner_uid,
                miner_coldkey=miner_coldkey,
                validator_hotkey=self.core.wallet.hotkey.ss58_address if self.core.wallet else None,
                validator_uid=self.core.uid,
                validator_coldkey=(
                    self.core.wallet.coldkeypub.ss58_address
                    if self.core.wallet and hasattr(self.core.wallet, "coldkeypub")
                    else None
                ),
                consensus_votes="1/1",
                consensus_verified=False,
                model_name=model_config.get("model", self.core.model),
                model_config=model_config,
                timing_metrics=timing_metrics,
                network_context=network_context,
                prompt_embedding=prompt_embedding,
            )

            if novelty_score is not None:
                self.scoring_system.update_novelty(
                    hotkey=miner_hotkey or "unknown",
                    novelty_score=novelty_score,
                )

        except Exception as e:
            bt.logging.error(f"Error in background logging: {e}")
            bt.logging.debug(f"Traceback: {traceback.format_exc()}")

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
        """Run consensus verification in background."""
        bt.logging.info("Starting consensus verification for prompt...")

        try:
            network_context = self.core.get_network_context(miner_hotkey)

            # Calculate adaptive runs per validator
            total_validators = Config.CONSENSUS_VALIDATORS
            runs_per_validator = math.ceil(Config.MIN_TOTAL_RUNS_PER_PROMPT / total_validators)

            # Run prompt multiple times for our vote
            run_results = []
            for i in range(runs_per_validator):
                try:
                    api_params = {
                        "model": self.core.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": Config.OPENAI_MAX_TOKENS,
                    }
                    response, _ = call_chat_api_with_fallback(
                        self.core.chat_client, api_params, timeout=Config.CHAT_API_TIMEOUT
                    )
                    completion = response.choices[0].message.content.strip()
                    moderation_result = self.core.moderation_provider.moderate(completion)
                    run_results.append({
                        "combined_score": moderation_result.combined_score,
                        "high_category_triggered": moderation_result.high_category_triggered,
                    })
                except ModelUnavailableError:
                    bt.logging.warning(f"All models unavailable for consensus run {i + 1}, skipping")
                except Exception as e:
                    bt.logging.error(f"Error in consensus run {i + 1}: {e}")
                    bt.logging.debug(f"Traceback: {traceback.format_exc()}")

            if not run_results:
                bt.logging.error("No successful consensus runs - cannot proceed with consensus")
                return

            # Determine our vote
            dangerous_runs = sum(
                1
                for run in run_results
                if run["combined_score"] >= self._danger_threshold or run["high_category_triggered"]
            )
            required_for_dangerous = len(run_results) // 2 + 1
            our_vote = dangerous_runs >= required_for_dangerous

            scores = [run["combined_score"] for run in run_results]
            primary_result = {
                "vote": our_vote,
                "scores": scores,
                "validator_hotkey": (
                    self.core.wallet.hotkey.ss58_address if self.core.wallet else "unknown"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Initiate consensus with other validators (with timeout)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.core.consensus_coordinator.initiate_consensus,
                        prompt=prompt,
                        primary_result=primary_result,
                    )
                    consensus_result = future.result(timeout=CONSENSUS_TIMEOUT)
            except concurrent.futures.TimeoutError:
                bt.logging.error(
                    f"Consensus verification timed out after {CONSENSUS_TIMEOUT}s - "
                    "skipping dataset logging for this prompt"
                )
                return

            bt.logging.info(f"Consensus result: {consensus_result['votes']}")

            # Generate embedding and check novelty for consensus-passed prompts
            prompt_embedding = None
            novelty_score = None
            if consensus_result["consensus"] and self.core.embedding_client.is_available():
                prompt_embedding = self.core.embedding_client.get_embedding(prompt)
                if prompt_embedding and self.core.novelty_client.is_available():
                    novelty_result = self.core.novelty_client.check_novelty(
                        prompt=prompt,
                        prompt_embedding=prompt_embedding,
                    )
                    if novelty_result:
                        novelty_score = novelty_result.novelty_score

            # Calculate distribution statistics
            all_scores = []
            for vote_detail in consensus_result["vote_details"]:
                if "scores" in vote_detail:
                    all_scores.extend(vote_detail["scores"])

            mean_score = statistics.mean(all_scores) if all_scores else initial_danger_score
            std_dev_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
            min_score = min(all_scores) if all_scores else initial_danger_score
            max_score = max(all_scores) if all_scores else initial_danger_score

            miner_uid, miner_coldkey = self.core.get_miner_info(miner_hotkey)

            # Log entry with consensus data
            self.core.dataset_logger.log_entry(
                prompt=prompt,
                response=initial_response,
                danger_score=initial_danger_score,
                category_scores=category_scores,
                accepted=consensus_result["consensus"],
                miner_hotkey=miner_hotkey,
                miner_uid=miner_uid,
                miner_coldkey=miner_coldkey,
                validator_hotkey=self.core.wallet.hotkey.ss58_address if self.core.wallet else None,
                validator_uid=self.core.uid,
                validator_coldkey=(
                    self.core.wallet.coldkeypub.ss58_address
                    if self.core.wallet and hasattr(self.core.wallet, "coldkeypub")
                    else None
                ),
                consensus_votes=consensus_result["votes"],
                consensus_verified=True,
                validator_votes=consensus_result["vote_details"],
                mean_danger_score=mean_score,
                std_dev_danger_score=std_dev_score,
                min_danger_score=min_score,
                max_danger_score=max_score,
                total_runs=len(all_scores),
                validator_count=consensus_result.get("total_validators", len(consensus_result["vote_details"])),
                excluded_validators=consensus_result.get("excluded_validators", []),
                exclusion_reasons=consensus_result.get("exclusion_reasons", {}),
                model_name=model_config.get("model", self.core.model),
                model_config=model_config,
                timing_metrics=timing_metrics,
                network_context=network_context,
                prompt_embedding=prompt_embedding,
            )

            # Update scoring
            self.scoring_system.record_submission(
                hotkey=miner_hotkey or "unknown",
                danger_score=initial_danger_score,
                accepted=consensus_result["consensus"],
                block=self.core.get_current_block(),
                novelty_score=novelty_score,
            )

            if consensus_result["consensus"]:
                bt.logging.success("Consensus PASSED")
            else:
                bt.logging.warning("Consensus FAILED")

        except Exception as e:
            bt.logging.error(f"Error in consensus verification: {e}")
            bt.logging.debug(f"Traceback: {traceback.format_exc()}")

    def _handle_consensus_verification(
        self, synapse: ConsensusVerificationSynapse
    ) -> ConsensusVerificationSynapse:
        """Handle consensus verification request from another validator."""
        bt.logging.info(
            f"Received consensus verification from {synapse.primary_validator_hotkey[:8]}..."
        )

        try:
            prompt = synapse.prompt
            runs_required = synapse.runs_required
            run_results = []

            for i in range(runs_required):
                try:
                    api_params = {
                        "model": self.core.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": Config.OPENAI_MAX_TOKENS,
                    }
                    response, _ = call_chat_api_with_fallback(
                        self.core.chat_client, api_params, timeout=Config.CHAT_API_TIMEOUT
                    )
                    completion = response.choices[0].message.content.strip()
                    moderation_result = self.core.moderation_provider.moderate(completion)
                    run_results.append({
                        "combined_score": moderation_result.combined_score,
                        "high_category_triggered": moderation_result.high_category_triggered,
                    })
                except ModelUnavailableError:
                    bt.logging.warning(f"All models unavailable for verification run {i + 1}, skipping")
                except Exception as e:
                    bt.logging.error(f"Error in verification run {i + 1}: {e}")

            if not run_results:
                synapse.verification_result = {
                    "runs": [],
                    "vote": False,
                    "validator_hotkey": (
                        self.core.wallet.hotkey.ss58_address if self.core.wallet else "unknown"
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": "Failed to complete verification runs",
                }
                return synapse

            # Vote based on majority
            dangerous_runs = sum(
                1
                for run in run_results
                if run["combined_score"] >= self._danger_threshold or run["high_category_triggered"]
            )
            required_for_dangerous = len(run_results) // 2 + 1
            vote = dangerous_runs >= required_for_dangerous

            scores = [run["combined_score"] for run in run_results]
            synapse.verification_result = {
                "runs": scores,
                "vote": vote,
                "validator_hotkey": (
                    self.core.wallet.hotkey.ss58_address if self.core.wallet else "unknown"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            bt.logging.error(f"Error in verify_for_consensus: {e}")
            synapse.verification_result = {
                "runs": [],
                "vote": False,
                "validator_hotkey": (
                    self.core.wallet.hotkey.ss58_address if self.core.wallet else "unknown"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

        return synapse
