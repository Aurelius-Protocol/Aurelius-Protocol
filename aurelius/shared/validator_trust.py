"""Validator trust and reputation tracking system."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import bittensor as bt


@dataclass
class ValidatorStats:
    """Statistics for a single validator.

    Attributes:
        hotkey: Validator's hotkey
        total_verifications: Total number of verification requests sent
        agreements_with_consensus: Number of times validator agreed with consensus
        total_responses: Number of successful responses received
        failed_responses: Number of failed/timeout responses
        trust_score: Calculated trust score (0-1)
        last_seen: ISO timestamp of last verification
        flags: List of validation failures (e.g., "insufficient_runs", "no_variance")
    """

    hotkey: str
    total_verifications: int = 0
    agreements_with_consensus: int = 0
    total_responses: int = 0
    failed_responses: int = 0
    trust_score: float = 0.3  # Start with limited trust (prevents Sybil attacks)
    last_seen: str = ""
    flags: list[str] = None

    def __post_init__(self):
        if self.flags is None:
            self.flags = []


class ValidatorTrustTracker:
    """Tracks validator reputation and trust scores."""

    # Minimum verifications required before a validator can achieve full trust
    # This prevents Sybil attacks where new validators immediately get full voting rights
    # ~50 verifications at ~1 per block = ~10 minutes of activity
    MIN_VERIFICATIONS_FOR_FULL_TRUST = 50

    # Initial trust score for brand new validators (before warm-up)
    INITIAL_TRUST_SCORE = 0.3

    def __init__(
        self,
        persistence_path: str = "./validator_trust.json",
        min_trust_score: float = 0.7,
        trust_decay_rate: float = 0.95,
    ):
        """
        Initialize trust tracker.

        Args:
            persistence_path: Path to save trust data
            min_trust_score: Minimum trust score to participate (0-1)
            trust_decay_rate: Rate at which trust decays per failure (0-1)
        """
        self.persistence_path = persistence_path
        self.min_trust_score = min_trust_score
        self.trust_decay_rate = trust_decay_rate

        # In-memory store: {hotkey: ValidatorStats}
        self.validators: dict[str, ValidatorStats] = {}

        # Load existing data if available
        self._load()

        bt.logging.info(f"ValidatorTrustTracker initialized: min_trust={min_trust_score}, decay={trust_decay_rate}")

    def _load(self) -> None:
        """Load trust data from disk."""
        if not os.path.exists(self.persistence_path):
            bt.logging.info("No existing trust data found, starting fresh")
            return

        try:
            with open(self.persistence_path) as f:
                data = json.load(f)

            # Convert dict entries back to ValidatorStats objects
            for hotkey, stats_dict in data.items():
                self.validators[hotkey] = ValidatorStats(**stats_dict)

            bt.logging.info(f"Loaded trust data for {len(self.validators)} validators")

        except Exception as e:
            bt.logging.error(f"Failed to load trust data: {e}")

    def _save(self) -> None:
        """Save trust data to disk."""
        try:
            # Create parent directory if needed
            Path(self.persistence_path).parent.mkdir(parents=True, exist_ok=True)

            # Convert ValidatorStats objects to dicts
            data = {hotkey: asdict(stats) for hotkey, stats in self.validators.items()}

            with open(self.persistence_path, "w") as f:
                json.dump(data, f, indent=2)

            bt.logging.debug("Trust data saved")

        except Exception as e:
            bt.logging.error(f"Failed to save trust data: {e}")

    def get_or_create_stats(self, hotkey: str) -> ValidatorStats:
        """Get stats for a validator, creating new entry if doesn't exist."""
        if hotkey not in self.validators:
            self.validators[hotkey] = ValidatorStats(hotkey=hotkey, last_seen=datetime.now(timezone.utc).isoformat())
        return self.validators[hotkey]

    def record_verification_request(self, hotkey: str) -> None:
        """
        Record that we sent a verification request to this validator.

        Args:
            hotkey: Validator's hotkey
        """
        stats = self.get_or_create_stats(hotkey)
        stats.total_verifications += 1
        stats.last_seen = datetime.now(timezone.utc).isoformat()
        self._save()

    def record_response(self, hotkey: str, success: bool, agreed_with_consensus: bool | None = None) -> None:
        """
        Record a response from validator.

        Args:
            hotkey: Validator's hotkey
            success: Whether we received a valid response
            agreed_with_consensus: Whether their vote matched consensus (if known)
        """
        stats = self.get_or_create_stats(hotkey)

        if success:
            stats.total_responses += 1

            if agreed_with_consensus is not None:
                if agreed_with_consensus:
                    stats.agreements_with_consensus += 1
        else:
            stats.failed_responses += 1

        # Recalculate trust score
        self._update_trust_score(stats)
        self._save()

    def record_validation_failure(self, hotkey: str, failure_type: str) -> None:
        """
        Record a validation failure (e.g., insufficient runs, no variance).

        Args:
            hotkey: Validator's hotkey
            failure_type: Type of failure (e.g., "insufficient_runs")
        """
        stats = self.get_or_create_stats(hotkey)
        stats.flags.append(f"{failure_type}:{datetime.now(timezone.utc).isoformat()}")

        # Keep only last 10 flags to prevent unbounded growth
        if len(stats.flags) > 10:
            stats.flags = stats.flags[-10:]

        # Reduce trust score
        stats.trust_score *= self.trust_decay_rate

        bt.logging.warning(
            f"Validator {hotkey[:8]}... flagged for {failure_type}. Trust score: {stats.trust_score:.3f}"
        )

        self._save()

    def _update_trust_score(self, stats: ValidatorStats) -> None:
        """
        Calculate trust score based on validator performance.

        Trust score formula:
        - Base: agreement_rate (agreements / total_verifications)
        - Penalty: failed_responses reduce score
        - Warm-up: New validators must earn trust over MIN_VERIFICATIONS_FOR_FULL_TRUST
        - Floor: Never goes below 0.0
        - Ceiling: Capped at 1.0
        """
        if stats.total_verifications == 0:
            # Brand new validators start with limited trust (prevents Sybil attacks)
            stats.trust_score = self.INITIAL_TRUST_SCORE
            return

        # Agreement rate (0-1)
        if stats.total_verifications > 0:
            agreement_rate = stats.agreements_with_consensus / stats.total_verifications
        else:
            agreement_rate = 0.0

        # Response rate (0-1)
        response_rate = stats.total_responses / stats.total_verifications

        # Combine: 70% weight on agreement, 30% on response rate
        trust_score = (0.7 * agreement_rate) + (0.3 * response_rate)

        # Apply penalty for failures (each failure reduces by decay_rate)
        if stats.failed_responses > 0:
            penalty = self.trust_decay_rate**stats.failed_responses
            trust_score *= penalty

        # Apply warm-up factor: new validators must earn full trust over time
        # This prevents Sybil attacks where attackers create many new validators
        if stats.total_verifications < self.MIN_VERIFICATIONS_FOR_FULL_TRUST:
            warmup_factor = stats.total_verifications / self.MIN_VERIFICATIONS_FOR_FULL_TRUST
            # Interpolate between initial trust and calculated trust
            trust_score = self.INITIAL_TRUST_SCORE + (trust_score - self.INITIAL_TRUST_SCORE) * warmup_factor
            bt.logging.debug(
                f"Validator warm-up: {stats.total_verifications}/{self.MIN_VERIFICATIONS_FOR_FULL_TRUST} "
                f"verifications, warmup_factor={warmup_factor:.2f}, trust={trust_score:.3f}"
            )

        # Clamp to [0, 1]
        stats.trust_score = max(0.0, min(1.0, trust_score))

    def get_trust_score(self, hotkey: str) -> float:
        """
        Get current trust score for a validator.

        Args:
            hotkey: Validator's hotkey

        Returns:
            Trust score (0-1), or INITIAL_TRUST_SCORE if validator is new
        """
        if hotkey not in self.validators:
            # New validators start with limited trust (prevents Sybil attacks)
            return self.INITIAL_TRUST_SCORE

        return self.validators[hotkey].trust_score

    def is_trusted(self, hotkey: str) -> bool:
        """
        Check if validator meets minimum trust threshold.

        Args:
            hotkey: Validator's hotkey

        Returns:
            True if trust_score >= min_trust_score
        """
        trust_score = self.get_trust_score(hotkey)
        return trust_score >= self.min_trust_score

    def filter_trusted_validators(self, validator_hotkeys: list[str]) -> list[str]:
        """
        Filter list of validators to only include trusted ones.

        Args:
            validator_hotkeys: List of validator hotkeys to filter

        Returns:
            Filtered list of trusted validator hotkeys
        """
        trusted = [hotkey for hotkey in validator_hotkeys if self.is_trusted(hotkey)]

        excluded_count = len(validator_hotkeys) - len(trusted)
        if excluded_count > 0:
            bt.logging.info(f"Filtered {excluded_count} validators below trust threshold (min={self.min_trust_score})")

        return trusted

    def get_stats_summary(self) -> dict:
        """
        Get summary statistics for all tracked validators.

        Returns:
            Dictionary with summary stats
        """
        if not self.validators:
            return {
                "total_validators": 0,
                "trusted_validators": 0,
                "untrusted_validators": 0,
            }

        trusted_count = sum(1 for stats in self.validators.values() if stats.trust_score >= self.min_trust_score)

        return {
            "total_validators": len(self.validators),
            "trusted_validators": trusted_count,
            "untrusted_validators": len(self.validators) - trusted_count,
            "min_trust_score": self.min_trust_score,
            "average_trust_score": sum(s.trust_score for s in self.validators.values()) / len(self.validators),
        }

    def get_validator_stats(self, hotkey: str) -> dict | None:
        """
        Get detailed stats for a specific validator.

        Args:
            hotkey: Validator's hotkey

        Returns:
            Dict with stats, or None if validator not tracked
        """
        if hotkey not in self.validators:
            return None

        return asdict(self.validators[hotkey])
