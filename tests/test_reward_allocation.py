"""Unit tests for calculate_merged_weights() function in ExperimentManager.

Tests reward distribution across experiments according to configured allocation percentages.
"""

from collections import defaultdict
from unittest.mock import MagicMock

import pytest

from aurelius.shared.config import ConfigurationError
from aurelius.validator.experiments.base import ExperimentScores


class TestCalculateMergedWeights:
    """Tests for the calculate_merged_weights() function."""

    @pytest.fixture
    def experiment_manager(self):
        """Create an ExperimentManager with mock core."""
        from aurelius.validator.experiments.manager import ExperimentManager

        mock_core = MagicMock()
        mock_core.experiment_client = MagicMock()
        return ExperimentManager(mock_core)

    def test_single_experiment_full_allocation(self, experiment_manager):
        """Test single experiment gets full allocation."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 0.5, "miner2": 0.3, "miner3": 0.2},
                block_height=1000,
            )
        ]
        allocations = {"prompt": 100.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=0.0,
        )

        # Weights should sum to 100%
        total = sum(weights.values())
        assert abs(total - 100.0) < 0.01

        # Each miner's weight should be proportional to their score
        assert "miner1" in weights
        assert "miner2" in weights
        assert "miner3" in weights
        # miner1 has 50% of scores, should get 50% of allocation
        assert abs(weights["miner1"] - 50.0) < 0.01
        assert abs(weights["miner2"] - 30.0) < 0.01
        assert abs(weights["miner3"] - 20.0) < 0.01

    def test_multiple_experiments_proportional_allocation(self, experiment_manager):
        """Test multiple experiments distribute weights proportionally."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 0.6, "miner2": 0.4},
                block_height=1000,
            ),
            ExperimentScores(
                experiment_name="jailbreak-v1",
                scores={"miner2": 0.7, "miner3": 0.3},
                block_height=1000,
            ),
        ]
        allocations = {"prompt": 85.0, "jailbreak-v1": 10.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=5.0,
        )

        # Total weight should be 95% (100 - 5% burn)
        total = sum(w for k, w in weights.items() if k != "burn")
        assert abs(total - 95.0) < 0.01

        # miner1 only participates in prompt: 60% of 85% = 51%
        assert abs(weights["miner1"] - 51.0) < 0.01

        # miner2 participates in both:
        # prompt: 40% of 85% = 34%
        # jailbreak: 70% of 10% = 7%
        # total: 41%
        assert abs(weights["miner2"] - 41.0) < 0.01

        # miner3 only participates in jailbreak: 30% of 10% = 3%
        assert abs(weights["miner3"] - 3.0) < 0.01

    def test_burn_percentage_excluded_from_miner_weights(self, experiment_manager):
        """Test that burn percentage is tracked separately."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 1.0},
                block_height=1000,
            )
        ]
        allocations = {"prompt": 95.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=5.0,
        )

        # Miner gets 95%
        assert abs(weights["miner1"] - 95.0) < 0.01

        # Burn should be tracked
        assert "burn" in weights
        assert abs(weights["burn"] - 5.0) < 0.01

    def test_zero_activity_redistribution(self, experiment_manager):
        """Test that unused allocation is redistributed to active experiments."""
        # jailbreak has no activity (empty scores)
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 0.6, "miner2": 0.4},
                block_height=1000,
            ),
            ExperimentScores(
                experiment_name="jailbreak-v1",
                scores={},  # No activity
                block_height=1000,
            ),
        ]
        allocations = {"prompt": 85.0, "jailbreak-v1": 10.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=5.0,
            redistribute_unused=True,
        )

        # jailbreak's 10% should be redistributed to prompt
        # Effective prompt allocation: 85% + 10% = 95%
        # miner1: 60% of 95% = 57%
        # miner2: 40% of 95% = 38%
        total_miner = weights.get("miner1", 0) + weights.get("miner2", 0)
        assert abs(total_miner - 95.0) < 0.01

    def test_zero_activity_no_redistribution(self, experiment_manager):
        """Test that unused allocation goes to burn when redistribution disabled."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 1.0},
                block_height=1000,
            ),
            ExperimentScores(
                experiment_name="jailbreak-v1",
                scores={},  # No activity
                block_height=1000,
            ),
        ]
        allocations = {"prompt": 85.0, "jailbreak-v1": 10.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=5.0,
            redistribute_unused=False,
        )

        # miner1 only gets prompt's 85%
        assert abs(weights["miner1"] - 85.0) < 0.01

        # Burn gets original 5% + unused 10% = 15%
        assert abs(weights["burn"] - 15.0) < 0.01

    def test_allocation_validation_must_sum_to_100(self, experiment_manager):
        """Test that allocations must sum to 100%."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 1.0},
                block_height=1000,
            )
        ]
        # Allocations sum to 90% (missing 10%)
        allocations = {"prompt": 85.0}

        with pytest.raises(ConfigurationError) as exc_info:
            experiment_manager.calculate_merged_weights(
                experiment_scores=scores,
                allocations=allocations,
                burn_percentage=5.0,
            )

        assert "100" in str(exc_info.value) or "sum" in str(exc_info.value).lower()

    def test_allocation_validation_over_100(self, experiment_manager):
        """Test that allocations over 100% are rejected."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 1.0},
                block_height=1000,
            )
        ]
        # Allocations sum to 110%
        allocations = {"prompt": 100.0}

        with pytest.raises(ConfigurationError) as exc_info:
            experiment_manager.calculate_merged_weights(
                experiment_scores=scores,
                allocations=allocations,
                burn_percentage=10.0,
            )

        assert "100" in str(exc_info.value) or "sum" in str(exc_info.value).lower()

    def test_empty_scores_returns_burn_only(self, experiment_manager):
        """Test that empty scores only returns burn allocation."""
        scores = []
        allocations = {}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=100.0,
        )

        # Only burn should be in weights
        assert weights.get("burn", 0) == 100.0
        # No miner weights
        assert len([k for k in weights if k != "burn"]) == 0

    def test_normalized_scores_within_experiment(self, experiment_manager):
        """Test that scores are normalized within each experiment."""
        # Scores don't sum to 1.0 - should be normalized
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 100.0, "miner2": 100.0},  # Sum to 200
                block_height=1000,
            )
        ]
        allocations = {"prompt": 100.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=0.0,
        )

        # Each miner should get 50% regardless of raw score magnitude
        assert abs(weights["miner1"] - 50.0) < 0.01
        assert abs(weights["miner2"] - 50.0) < 0.01

    def test_negative_scores_treated_as_zero(self, experiment_manager):
        """Test that negative scores are treated as zero."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 1.0, "miner2": -0.5},  # Negative score
                block_height=1000,
            )
        ]
        allocations = {"prompt": 100.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=0.0,
        )

        # miner1 should get all allocation (miner2's negative is zeroed)
        assert abs(weights["miner1"] - 100.0) < 0.01
        assert weights.get("miner2", 0.0) == 0.0


class TestEdgeCases:
    """Edge case tests for reward allocation."""

    @pytest.fixture
    def experiment_manager(self):
        """Create an ExperimentManager with mock core."""
        from aurelius.validator.experiments.manager import ExperimentManager

        mock_core = MagicMock()
        mock_core.experiment_client = MagicMock()
        return ExperimentManager(mock_core)

    def test_single_miner_single_experiment(self, experiment_manager):
        """Test simplest case: one miner, one experiment."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 1.0},
                block_height=1000,
            )
        ]
        allocations = {"prompt": 100.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=0.0,
        )

        assert abs(weights["miner1"] - 100.0) < 0.01

    def test_all_zero_scores(self, experiment_manager):
        """Test when all miners have zero scores."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 0.0, "miner2": 0.0},
                block_height=1000,
            )
        ]
        allocations = {"prompt": 95.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=5.0,
            redistribute_unused=True,
        )

        # All allocation should go to burn when no activity
        assert weights.get("burn", 0) >= 5.0

    def test_very_small_allocations(self, experiment_manager):
        """Test with very small allocation percentages."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 1.0},
                block_height=1000,
            ),
            ExperimentScores(
                experiment_name="tiny-exp",
                scores={"miner2": 1.0},
                block_height=1000,
            ),
        ]
        allocations = {"prompt": 99.9, "tiny-exp": 0.1}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=0.0,
        )

        assert abs(weights["miner1"] - 99.9) < 0.01
        assert abs(weights["miner2"] - 0.1) < 0.01

    def test_many_experiments_many_miners(self, experiment_manager):
        """Test with many experiments and many miners."""
        scores = [
            ExperimentScores(
                experiment_name=f"exp{i}",
                scores={f"miner{j}": float(j + 1) for j in range(10)},
                block_height=1000,
            )
            for i in range(5)
        ]
        # Equal allocation across 5 experiments
        allocations = {f"exp{i}": 19.0 for i in range(5)}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=5.0,
        )

        # Total should be 95% (100 - 5% burn)
        total = sum(w for k, w in weights.items() if k != "burn")
        assert abs(total - 95.0) < 0.1

    def test_float_precision(self, experiment_manager):
        """Test that float precision doesn't cause issues."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 0.333333333, "miner2": 0.333333333, "miner3": 0.333333334},
                block_height=1000,
            )
        ]
        allocations = {"prompt": 99.999}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=0.001,
        )

        # Should not raise and total should be approximately 100%
        total = sum(weights.values())
        assert abs(total - 100.0) < 0.1


class TestMoralReasoningWeightIntegration:
    """Tests for moral reasoning experiment weight merging (T017, T018)."""

    @pytest.fixture
    def experiment_manager(self):
        from aurelius.validator.experiments.manager import ExperimentManager

        mock_core = MagicMock()
        mock_core.experiment_client = MagicMock()
        return ExperimentManager(mock_core)

    def test_moral_reasoning_merges_with_prompt(self, experiment_manager):
        """Moral reasoning scores merge with prompt scores at configured ratios."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 0.8, "miner2": 0.5},
                block_height=1000,
            ),
            ExperimentScores(
                experiment_name="moral-reasoning",
                scores={"miner2": 0.9, "miner3": 0.6},
                block_height=1000,
            ),
        ]
        allocations = {"prompt": 80.0, "moral-reasoning": 15.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=5.0,
        )

        # miner1: only prompt → 80% * (0.8 / 1.3) ≈ 49.23
        # miner2: prompt + moral → 80% * (0.5/1.3) + 15% * (0.9/1.5) = 30.77 + 9.0 = 39.77
        # miner3: only moral → 15% * (0.6/1.5) = 6.0
        # burn: 5%
        total = sum(weights.values())
        assert abs(total - 100.0) < 0.01

        # miner2 should have contributions from both experiments
        assert weights.get("miner2", 0) > 0
        # miner3 only in moral reasoning
        assert weights.get("miner3", 0) > 0
        assert weights.get("burn", 0) > 0

    def test_miner_in_both_gets_combined_weight(self, experiment_manager):
        """A miner in both experiments receives combined weight."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner_both": 1.0},
                block_height=1000,
            ),
            ExperimentScores(
                experiment_name="moral-reasoning",
                scores={"miner_both": 1.0},
                block_height=1000,
            ),
        ]
        allocations = {"prompt": 70.0, "moral-reasoning": 25.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=5.0,
        )

        # miner_both is sole miner in each experiment → gets full allocation from each
        # prompt: 70%, moral: 25%, burn: 5%
        assert abs(weights["miner_both"] - 95.0) < 0.01

    def test_no_moral_submissions_redistributes(self, experiment_manager):
        """Unused moral reasoning allocation redistributes when no miners submit."""
        scores = [
            ExperimentScores(
                experiment_name="prompt",
                scores={"miner1": 1.0},
                block_height=1000,
            ),
            ExperimentScores(
                experiment_name="moral-reasoning",
                scores={},  # No submissions
                block_height=1000,
            ),
        ]
        allocations = {"prompt": 80.0, "moral-reasoning": 15.0}

        weights = experiment_manager.calculate_merged_weights(
            experiment_scores=scores,
            allocations=allocations,
            burn_percentage=5.0,
            redistribute_unused=True,
        )

        # moral-reasoning 15% should be redistributed to prompt
        # miner1 gets 80% + 15% = 95%
        assert abs(weights["miner1"] - 95.0) < 0.01
