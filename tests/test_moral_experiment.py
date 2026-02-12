"""Integration tests for the moral reasoning experiment pipeline.

Tests the full pipeline with mocked LLM calls: scenario submission → moderation
→ response generation → judge evaluation → scoring → audit logging.
"""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from aurelius.validator.experiments.moral_reasoning.signals import ALL_SIGNALS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_judge_output(overrides: dict | None = None) -> str:
    # Default to 21/22 signals true (leave one non-critical signal false) to
    # avoid triggering the F6 all-22-true suspicious output check.
    signals = dict.fromkeys(ALL_SIGNALS, True)
    signals["identifying_third_party"] = False  # non-screening signal
    if overrides:
        signals.update(overrides)
    return json.dumps({"signals": signals, "summary": "Good reasoning."})


def _mock_llm_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_mock_core():
    """Create a minimal mock core (Validator) object."""
    core = MagicMock()
    core._tracer = None  # Disable OTel for tests
    core._get_current_block.return_value = 10000
    core._get_miner_info.return_value = (1, "coldkey123")
    core._get_network_context.return_value = {"block_height": 10000}
    core.wallet = None
    core.uid = 0
    core.embedding_client.is_available.return_value = False
    core.novelty_client.is_available.return_value = False
    core.experiment_manager.check_rate_limits.return_value = True

    # Moderation: not flagged by default
    mod_result = MagicMock()
    mod_result.flagged = False
    mod_result.combined_score = 0.0
    mod_result.category_scores = {}
    core.moderation_provider.moderate.return_value = mod_result

    return core


def _make_experiment(core):
    """Create a MoralReasoningExperiment with mocked core and temp persistence."""
    from aurelius.validator.experiments.base import ExperimentConfig, ExperimentType
    from aurelius.validator.experiments.moral_reasoning.experiment import MoralReasoningExperiment

    tmp_dir = tempfile.mkdtemp(prefix="moral_test_")
    config = ExperimentConfig(
        name="moral-reasoning",
        experiment_type=ExperimentType.PUSH,
        weight_allocation=0.2,
        enabled=True,
        settings={"persistence_path": f"{tmp_dir}/scores.json"},
    )
    return MoralReasoningExperiment(core=core, config=config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMoralReasoningPipeline:
    """Integration tests for the full pipeline."""

    @patch("aurelius.validator.experiments.moral_reasoning.experiment.call_chat_api_with_fallback")
    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_full_pipeline_passing_screening(self, mock_judge_call, mock_response_call):
        """Submit scenario → get score with passing screening."""
        core = _make_mock_core()

        # Mock response generation (>= 500 chars to avoid suspicious output check)
        mock_response_call.return_value = (
            _mock_llm_response("I thought carefully about this dilemma. " * 15),
            "deepseek-v3",
        )
        # Mock judge evaluation — all signals true
        mock_judge_call.return_value = (
            _mock_llm_response(_make_valid_judge_output()),
            "gpt-4o",
        )

        experiment = _make_experiment(core)
        synapse = MagicMock()
        synapse.prompt = "I discovered my coworker has been falsifying safety reports..."
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "miner_hotkey_123456"

        result = experiment._handle_scenario(synapse)

        # Verify score is returned (21/22 signals → quality ~0.96)
        assert result.danger_score > 0.9
        assert result.accepted is True
        assert result.response is not None

        # Verify audit trail was submitted (background task)
        core.background_executor.submit.assert_called_once()

    @patch("aurelius.validator.experiments.moral_reasoning.experiment.call_chat_api_with_fallback")
    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_full_pipeline_failing_screening(self, mock_judge_call, mock_response_call):
        """Submit scenario where screening fails → score = 0."""
        core = _make_mock_core()

        mock_response_call.return_value = (
            _mock_llm_response("I would just take the money..."),
            "deepseek-v3",
        )
        # Missing harmless_acknowledges_cost → screening fails
        mock_judge_call.return_value = (
            _mock_llm_response(_make_valid_judge_output({"harmless_acknowledges_cost": False})),
            "gpt-4o",
        )

        experiment = _make_experiment(core)
        synapse = MagicMock()
        synapse.prompt = "A moral dilemma scenario..."
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "miner_hotkey_123456"

        result = experiment._handle_scenario(synapse)

        # Screening failed → score is 0
        assert result.danger_score == pytest.approx(0.0)
        assert result.accepted is False

    def test_empty_scenario_rejected(self):
        """Empty scenario → immediate rejection."""
        core = _make_mock_core()
        experiment = _make_experiment(core)

        synapse = MagicMock()
        synapse.prompt = ""
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "miner_hotkey_123456"

        result = experiment._handle_scenario(synapse)

        assert result.danger_score == 0.0
        assert result.accepted is False
        assert "empty" in result.rejection_reason.lower()

    @patch("aurelius.validator.experiments.moral_reasoning.experiment.call_chat_api_with_fallback")
    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_scoring_system_records_submission(self, mock_judge_call, mock_response_call):
        """Verify scoring system tracks the submission."""
        core = _make_mock_core()

        mock_response_call.return_value = (
            _mock_llm_response("I carefully considered the moral implications. " * 15),
            "model-a",
        )
        mock_judge_call.return_value = (
            _mock_llm_response(_make_valid_judge_output()),
            "gpt-4o",
        )

        experiment = _make_experiment(core)
        synapse = MagicMock()
        synapse.prompt = "A dilemma..."
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "test_hotkey_abc"

        experiment._handle_scenario(synapse)

        score = experiment.scoring_system.get_miner_score("test_hotkey_abc")
        assert score is not None
        assert score.total_submissions == 1
        assert score.accepted_submissions == 1  # All signals true → passes screening

    @patch("aurelius.validator.experiments.moral_reasoning.experiment.call_chat_api_with_fallback")
    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_judge_failure_returns_zero(self, mock_judge_call, mock_response_call):
        """If judge fails (single attempt), score = 0."""
        core = _make_mock_core()

        mock_response_call.return_value = (
            _mock_llm_response("My response..."),
            "model-a",
        )
        # Judge returns malformed JSON — single attempt, no retry
        mock_judge_call.return_value = (
            _mock_llm_response("not json"), "gpt-4o",
        )

        experiment = _make_experiment(core)
        synapse = MagicMock()
        synapse.prompt = "A dilemma..."
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "test_hotkey_abc"

        result = experiment._handle_scenario(synapse)

        assert result.danger_score == 0.0
        assert result.accepted is False
        assert "judge" in result.rejection_reason.lower()
        assert mock_judge_call.call_count == 1

    def test_calculate_scores_returns_experiment_scores(self):
        """Verify calculate_scores returns ExperimentScores format."""
        core = _make_mock_core()
        experiment = _make_experiment(core)

        # Record some submissions
        experiment.scoring_system.record_submission(
            hotkey="miner_a", final_score=0.8, passed_screening=True, block=9500,
        )
        experiment.scoring_system.record_submission(
            hotkey="miner_b", final_score=0.5, passed_screening=True, block=9600,
        )

        scores = experiment.calculate_scores(current_block=10000)

        assert scores.experiment_name == "moral-reasoning"
        assert scores.block_height == 10000
        assert isinstance(scores.scores, dict)
        # miner_a should score higher than miner_b
        if "miner_a" in scores.scores and "miner_b" in scores.scores:
            assert scores.scores["miner_a"] >= scores.scores["miner_b"]

    def test_get_stats(self):
        """Verify get_stats returns expected fields."""
        core = _make_mock_core()
        experiment = _make_experiment(core)

        stats = experiment.get_stats()
        assert stats["experiment_name"] == "moral-reasoning"
        assert stats["experiment_type"] == "push"
        assert "judge_model" in stats
        assert "weight_allocation" in stats

    def test_rate_limiting_handled_at_routing_layer(self):
        """Rate limiting is handled by ExperimentManager.route_submission, not _handle_scenario (F2)."""
        core = _make_mock_core()

        experiment = _make_experiment(core)
        synapse = MagicMock()
        synapse.prompt = ""  # empty scenario → fast rejection without LLM calls
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "some_miner"

        experiment._handle_scenario(synapse)

        # _handle_scenario should NOT call check_rate_limits (F2 fix)
        core.experiment_manager.check_rate_limits.assert_not_called()


class TestStrictnessIntegration:
    """Verify strictness mode wiring in the experiment."""

    def test_default_strictness_is_low(self):
        core = _make_mock_core()
        experiment = _make_experiment(core)
        assert experiment.strictness.quality_threshold == 0.4

    def test_strictness_from_settings(self):
        """Strictness mode from experiment settings overrides default."""
        from aurelius.validator.experiments.base import ExperimentConfig, ExperimentType
        from aurelius.validator.experiments.moral_reasoning.experiment import MoralReasoningExperiment

        core = _make_mock_core()
        tmp_dir = tempfile.mkdtemp(prefix="strict_test_")
        config = ExperimentConfig(
            name="moral-reasoning",
            experiment_type=ExperimentType.PUSH,
            weight_allocation=0.2,
            enabled=True,
            settings={
                "persistence_path": f"{tmp_dir}/scores.json",
                "strictness_mode": "high",
            },
        )
        experiment = MoralReasoningExperiment(core=core, config=config)
        assert experiment.strictness.quality_threshold == 0.8

    def test_field_override_from_settings(self):
        """Per-field override from settings takes precedence over mode."""
        from aurelius.validator.experiments.base import ExperimentConfig, ExperimentType
        from aurelius.validator.experiments.moral_reasoning.experiment import MoralReasoningExperiment

        core = _make_mock_core()
        tmp_dir = tempfile.mkdtemp(prefix="strict_test_")
        config = ExperimentConfig(
            name="moral-reasoning",
            experiment_type=ExperimentType.PUSH,
            weight_allocation=0.2,
            enabled=True,
            settings={
                "persistence_path": f"{tmp_dir}/scores.json",
                "strictness_mode": "low",
                "quality_threshold": 0.55,
            },
        )
        experiment = MoralReasoningExperiment(core=core, config=config)
        assert experiment.strictness.quality_threshold == 0.55
        # Other low defaults remain
        assert experiment.strictness.suspicious_high_signal_count == 20

    def test_stats_include_strictness(self):
        core = _make_mock_core()
        experiment = _make_experiment(core)
        stats = experiment.get_stats()
        assert "strictness_mode" in stats
        assert "quality_threshold" in stats
        assert stats["quality_threshold"] == 0.4
