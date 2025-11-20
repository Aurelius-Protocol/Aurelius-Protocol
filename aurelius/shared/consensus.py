"""Consensus coordination for multi-validator verification."""

import secrets
import statistics
import uuid
from datetime import datetime, timezone

import bittensor as bt

from aurelius.shared.config import Config
from aurelius.shared.protocol import ConsensusVerificationSynapse
from aurelius.shared.validator_trust import ValidatorTrustTracker


class ConsensusCoordinator:
    """Coordinates consensus verification across multiple validators."""

    def __init__(self, wallet: bt.wallet, subtensor: bt.subtensor | None, netuid: int):
        """
        Initialize consensus coordinator.

        Args:
            wallet: Validator's wallet
            subtensor: Subtensor connection (None in LOCAL_MODE)
            netuid: Network UID
        """
        self.wallet = wallet
        self.subtensor = subtensor
        self.netuid = netuid
        self.dendrite = bt.dendrite(wallet=wallet)

        # Cache metagraph
        self.metagraph = None
        if subtensor and not Config.LOCAL_MODE:
            self.metagraph = subtensor.metagraph(netuid)

        # Parse local validator endpoints for multi-validator local testing
        self.local_validator_axons = []
        if Config.LOCAL_MODE and Config.LOCAL_VALIDATOR_ENDPOINTS:
            endpoints = [e.strip() for e in Config.LOCAL_VALIDATOR_ENDPOINTS.split(",") if e.strip()]
            for endpoint in endpoints:
                if ":" in endpoint:
                    host, port = endpoint.rsplit(":", 1)
                    # Create a mock axon info for local validators
                    axon = bt.AxonInfo(
                        version=4,
                        ip=host,
                        port=int(port),
                        ip_type=4,
                        hotkey="local_validator_" + endpoint,  # Mock hotkey
                        coldkey="local_validator_" + endpoint,
                    )
                    self.local_validator_axons.append(axon)
            bt.logging.info(f"Local testing mode: configured {len(self.local_validator_axons)} validator endpoints")

        # Initialize trust tracker if enabled
        self.trust_tracker = None
        if Config.ENABLE_VALIDATOR_TRUST_TRACKING:
            self.trust_tracker = ValidatorTrustTracker(
                persistence_path=Config.VALIDATOR_TRUST_PERSISTENCE_PATH,
                min_trust_score=Config.MIN_VALIDATOR_TRUST_SCORE,
                trust_decay_rate=Config.VALIDATOR_TRUST_DECAY_RATE,
            )
            bt.logging.info("Validator trust tracking enabled")

        bt.logging.info("Consensus coordinator initialized")

    def validate_verification_result(
        self, result: dict, runs_required: int, validator_hotkey: str, response_time: float
    ) -> tuple[bool, str | None]:
        """
        Validate a verification result from another validator.

        Checks for:
        1. Correct number of runs
        2. Score variance (detect fabricated identical scores)
        3. Response time (detect instant/fake responses)

        Args:
            result: Verification result dict
            runs_required: Expected number of runs
            validator_hotkey: Validator's hotkey
            response_time: Time taken to respond (seconds)

        Returns:
            Tuple of (is_valid, failure_reason)
        """
        if not result:
            return False, "empty_result"

        # Check 1: Did they return the requested number of runs?
        runs = result.get("runs", [])
        if len(runs) < runs_required:
            bt.logging.warning(
                f"Validator {validator_hotkey[:8]}... returned {len(runs)} runs, expected {runs_required}"
            )
            return False, "insufficient_runs"

        # Check 2: Are all scores identical? (suspicious - LLM responses vary)
        # BUT: Skip this check if all scores are very low (< 0.01) - legitimately safe responses
        mean_score = sum(runs) / len(runs) if runs else 0

        if len(runs) >= 2 and mean_score >= 0.01:  # Only check variance for non-trivial scores
            if len(set(runs)) == 1:
                # All identical scores
                bt.logging.warning(
                    f"Validator {validator_hotkey[:8]}... returned identical scores: {runs} (suspicious)"
                )
                return False, "no_variance"

            # Check variance is above threshold
            if len(runs) >= 3:
                try:
                    stdev = statistics.stdev(runs)
                    if stdev < Config.MAX_SCORE_VARIANCE_THRESHOLD:
                        bt.logging.warning(
                            f"Validator {validator_hotkey[:8]}... returned suspiciously "
                            f"low variance: stdev={stdev:.4f}, scores={runs}"
                        )
                        return False, "low_variance"
                except statistics.StatisticsError:
                    pass  # Not enough data for stdev
        elif mean_score < 0.01:
            # Very low scores (safe responses) - skip variance check
            bt.logging.debug(
                f"Validator {validator_hotkey[:8]}... returned safe scores "
                f"(mean={mean_score:.4f}), skipping variance check"
            )

        # Check 3: Did they respond too fast? (running multiple API calls takes time)
        # Each run takes ~2-5 seconds (API call + moderation)
        # Skip this check in LOCAL_MODE - localhost responses are legitimately faster
        if not Config.LOCAL_MODE:
            min_expected_time = runs_required * Config.MIN_RESPONSE_TIME_SECONDS * 0.5  # 50% margin
            if response_time < min_expected_time:
                bt.logging.warning(
                    f"Validator {validator_hotkey[:8]}... responded suspiciously fast: "
                    f"{response_time:.1f}s (expected >{min_expected_time:.1f}s for {runs_required} runs)"
                )
                return False, "response_too_fast"
        else:
            bt.logging.debug("LOCAL_MODE: Skipping response time check (validators on localhost)")

        # All checks passed
        return True, None

    def calculate_runs_per_validator(self, available_validators: int) -> int:
        """
        Calculate how many runs each validator should perform to meet MIN_TOTAL_RUNS_PER_PROMPT.

        Uses ceiling division to ensure we meet or exceed the minimum.

        Args:
            available_validators: Number of validators that will participate

        Returns:
            Number of runs each validator should perform

        Example:
            MIN_TOTAL_RUNS_PER_PROMPT = 15
            available_validators = 5
            → 15 / 5 = 3 runs per validator

            available_validators = 4
            → ceil(15 / 4) = 4 runs per validator (total 16)
        """
        import math

        if available_validators < 1:
            # Fallback if no validators available
            return Config.CONSENSUS_RUNS_PER_VALIDATOR

        runs_per_validator = math.ceil(Config.MIN_TOTAL_RUNS_PER_PROMPT / available_validators)

        bt.logging.info(
            f"Adaptive runs calculation: {Config.MIN_TOTAL_RUNS_PER_PROMPT} total runs / "
            f"{available_validators} validators = {runs_per_validator} runs per validator"
        )

        return runs_per_validator

    def select_validators(self, exclude_hotkey: str, count: int = 4, min_stake: float = 0.0) -> list:
        """
        Select random validators for consensus verification.

        Args:
            exclude_hotkey: Don't select this validator (usually self)
            count: Number of validators to select
            min_stake: Minimum stake requirement (TAO)

        Returns:
            List of AxonInfo objects for local mode, or UIDs for network mode
        """
        # Use cryptographically secure random number generator for validator selection
        secure_random = secrets.SystemRandom()

        if Config.LOCAL_MODE:
            # In local mode with configured endpoints, use those
            if self.local_validator_axons:
                # Return up to 'count' local validator axons
                selected_count = min(count, len(self.local_validator_axons))
                selected = secure_random.sample(self.local_validator_axons, selected_count)
                bt.logging.info(f"LOCAL_MODE: Selected {len(selected)} local validators for consensus")
                return selected
            else:
                # No local endpoints configured
                bt.logging.warning("LOCAL_MODE: Cannot select other validators for consensus")
                return []

        if not self.metagraph:
            bt.logging.error("No metagraph available for validator selection")
            return []

        # Find validators with sufficient stake
        eligible_validators = []
        for uid in range(len(self.metagraph.hotkeys)):
            hotkey = self.metagraph.hotkeys[uid]

            # Skip self
            if hotkey == exclude_hotkey:
                continue

            # Check stake requirement
            stake = self.metagraph.S[uid]
            if stake < min_stake:
                continue

            # Check if axon is available
            if not self.metagraph.axons[uid].is_serving:
                continue

            # Check trust score if trust tracking enabled
            if self.trust_tracker and not self.trust_tracker.is_trusted(hotkey):
                bt.logging.debug(
                    f"Excluding validator UID {uid} ({hotkey[:8]}...) "
                    f"due to low trust score: {self.trust_tracker.get_trust_score(hotkey):.3f}"
                )
                continue

            eligible_validators.append(uid)

        if len(eligible_validators) < count:
            bt.logging.warning(f"Only {len(eligible_validators)} eligible validators found, requested {count}")

        # Randomly select from eligible using cryptographically secure randomness
        selected_count = min(count, len(eligible_validators))
        selected_uids = secure_random.sample(eligible_validators, selected_count)

        bt.logging.debug(
            f"Selected {len(selected_uids)} validators from {len(eligible_validators)} eligible "
            f"(min_stake={min_stake}, excluded self)"
        )

        return selected_uids

    def initiate_consensus(self, prompt: str, primary_result: dict) -> dict:
        """
        Initiate consensus verification with other validators.

        Args:
            prompt: The prompt to verify
            primary_result: This validator's result
                {
                    'vote': bool,  # Is dangerous?
                    'scores': List[float],  # Danger scores from multiple runs
                }

        Returns:
            Consensus result:
            {
                'consensus': bool,  # True if 4+/5 validators agree
                'votes': str,  # e.g., "4/5"
                'vote_details': List[Dict],  # All validator votes
            }
        """
        bt.logging.info(f"Initiating consensus verification for prompt: {prompt[:50]}...")

        # Generate unique request ID
        request_id = str(uuid.uuid4())

        # Select other validators
        validators = self.select_validators(
            exclude_hotkey=self.wallet.hotkey.ss58_address,
            count=Config.CONSENSUS_VALIDATORS - 1,  # -1 because we count ourselves
            min_stake=Config.MIN_VALIDATOR_STAKE,
        )

        if not validators:
            # Can't do consensus if no other validators
            bt.logging.warning("Cannot perform consensus: no other validators available")
            return {
                "consensus": primary_result["vote"],  # Trust our own vote
                "votes": "1/1",
                "vote_details": [primary_result],
            }

        # Calculate adaptive runs per validator
        # Total validators = len(validators) + 1 (ourselves)
        total_validators = len(validators) + 1
        runs_per_validator = self.calculate_runs_per_validator(total_validators)

        # Create consensus verification synapse
        synapse = ConsensusVerificationSynapse(
            prompt=prompt,
            request_id=request_id,
            primary_validator_hotkey=self.wallet.hotkey.ss58_address,
            runs_required=runs_per_validator,
        )

        # Query other validators
        bt.logging.info(f"Querying {len(validators)} validators for consensus")

        try:
            # In LOCAL_MODE, validators is a list of AxonInfo
            # In network mode, validators is a list of UIDs
            if Config.LOCAL_MODE:
                axons = validators  # Already AxonInfo objects
                validator_hotkeys = [axon.hotkey for axon in axons]
            else:
                axons = [self.metagraph.axons[uid] for uid in validators]
                validator_hotkeys = [self.metagraph.hotkeys[uid] for uid in validators]

            # Track request timestamps for response time validation
            request_start = datetime.now(timezone.utc)

            # Record verification requests if trust tracking enabled
            if self.trust_tracker:
                for hotkey in validator_hotkeys:
                    self.trust_tracker.record_verification_request(hotkey)

            responses = self.dendrite.query(
                axons=axons,
                synapse=synapse,
                timeout=Config.CONSENSUS_TIMEOUT,
                deserialize=False,
            )

            response_time = (datetime.now(timezone.utc) - request_start).total_seconds()

            # Collect and validate votes
            all_votes = [primary_result]
            excluded_validators = {}  # {hotkey: reason}

            for i, response in enumerate(responses):
                hotkey = validator_hotkeys[i]

                # Verify response authenticity
                if not response:
                    # No response received
                    if self.trust_tracker:
                        self.trust_tracker.record_response(hotkey, success=False)
                    excluded_validators[hotkey] = "no_response"
                    bt.logging.warning(f"No response from validator {hotkey[:8]}...")
                    continue

                # Bittensor's dendrite.query already verifies signatures, but check for tampering
                if not hasattr(response, "verification_result"):
                    # Invalid response format
                    if self.trust_tracker:
                        self.trust_tracker.record_response(hotkey, success=False)
                    excluded_validators[hotkey] = "invalid_format"
                    bt.logging.warning(
                        f"Invalid response format from validator {hotkey[:8]}... "
                        f"(missing verification_result field)"
                    )
                    continue

                # Additional security: Check if response was properly signed by Bittensor
                # dendrite.query uses Bittensor's built-in signature verification
                # If response reaches here, signature was already validated by dendrite
                # Log verification success for audit trail
                bt.logging.debug(f"Verified authentic response from validator {hotkey[:8]}...")

                result = response.verification_result
                if not result:
                    # Empty result
                    if self.trust_tracker:
                        self.trust_tracker.record_response(hotkey, success=False)
                    excluded_validators[hotkey] = "empty_result"
                    continue

                # Validate the result
                if self.trust_tracker:
                    is_valid, failure_reason = self.validate_verification_result(
                        result,
                        runs_per_validator,
                        hotkey,
                        response_time / len(validators),  # Approximate per-validator time
                    )

                    if is_valid:
                        # Valid result, add vote
                        all_votes.append(result)
                        self.trust_tracker.record_response(hotkey, success=True)
                    else:
                        # Invalid result, exclude and flag
                        excluded_validators[hotkey] = failure_reason
                        self.trust_tracker.record_validation_failure(hotkey, failure_reason)
                        bt.logging.warning(
                            f"Excluding validator {hotkey[:8]}... from consensus due to: {failure_reason}"
                        )
                else:
                    # No trust tracking, accept all results
                    all_votes.append(result)

            # Tally dangerous votes
            dangerous_votes = sum(1 for v in all_votes if v.get("vote", False))
            total_votes = len(all_votes)

            # Consensus: need CONSENSUS_REQUIRED_VOTES out of total
            consensus_reached = dangerous_votes >= Config.CONSENSUS_REQUIRED_VOTES

            bt.logging.info(
                f"Consensus result: {dangerous_votes}/{total_votes} votes for dangerous "
                f"(consensus: {consensus_reached})"
            )

            if excluded_validators:
                bt.logging.info(
                    f"Excluded {len(excluded_validators)} validators: "
                    f"{list(excluded_validators.keys())[:3]}"  # Show first 3
                )

            # Update trust scores based on consensus agreement (post-hoc)
            if self.trust_tracker and consensus_reached:
                for vote_detail in all_votes:
                    validator_hotkey = vote_detail.get("validator_hotkey")
                    if validator_hotkey and validator_hotkey != self.wallet.hotkey.ss58_address:
                        voted_dangerous = vote_detail.get("vote", False)
                        agreed = voted_dangerous == consensus_reached
                        self.trust_tracker.record_response(validator_hotkey, success=True, agreed_with_consensus=agreed)

            return {
                "consensus": consensus_reached,
                "votes": f"{dangerous_votes}/{total_votes}",
                "vote_details": all_votes,
                "runs_per_validator": runs_per_validator,
                "total_validators": total_validators,
                "excluded_validators": list(excluded_validators.keys()),
                "exclusion_reasons": excluded_validators,
            }

        except Exception as e:
            bt.logging.error(f"Error during consensus: {e}")
            # On error, fall back to just our vote
            return {
                "consensus": primary_result["vote"],
                "votes": "1/1 (error)",
                "vote_details": [primary_result],
            }

    def get_consensus_stats(self) -> dict:
        """
        Get statistics about consensus coordinator.

        Returns:
            Dictionary with stats
        """
        if Config.LOCAL_MODE:
            return {
                "mode": "local",
                "consensus_enabled": False,
            }

        eligible_count = 0
        if self.metagraph:
            for uid in range(len(self.metagraph.hotkeys)):
                if self.metagraph.S[uid] >= Config.MIN_VALIDATOR_STAKE and self.metagraph.axons[uid].is_serving:
                    eligible_count += 1

        return {
            "mode": "network",
            "consensus_enabled": True,
            "eligible_validators": eligible_count,
            "required_votes": Config.CONSENSUS_REQUIRED_VOTES,
            "total_validators": Config.CONSENSUS_VALIDATORS,
        }
