"""End-to-end tests for the moral reasoning experiment.

Tests: full pipeline with mocked LLM, registration enforcement,
per-experiment rate limit independence, and per-experiment novelty pool.
"""

import json
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from aurelius.validator.experiments.moral_reasoning.signals import ALL_SIGNALS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_judge_output(overrides: dict | None = None) -> str:
    # Default to 21/22 signals true (leave one non-critical signal false) to
    # avoid triggering the F6 all-22-true suspicious output check.
    signals = {s: True for s in ALL_SIGNALS}
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
    core = MagicMock()
    core._tracer = None
    core._get_current_block.return_value = 10000
    core._get_miner_info.return_value = (1, "coldkey123")
    core._get_network_context.return_value = {"block_height": 10000}
    core.wallet = None
    core.uid = 0

    # Enable embedding + novelty for novelty pool tests
    core.embedding_client.is_available.return_value = True
    core.embedding_client.get_embedding.return_value = [0.1] * 384
    core.novelty_client.is_available.return_value = True
    novelty_result = MagicMock()
    novelty_result.novelty_score = 0.85
    core.novelty_client.check_novelty.return_value = novelty_result

    core.experiment_manager.check_rate_limits.return_value = True

    mod_result = MagicMock()
    mod_result.flagged = False
    mod_result.combined_score = 0.0
    mod_result.category_scores = {}
    core.moderation_provider.moderate.return_value = mod_result

    return core


def _make_experiment(core):
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

class TestMoralReasoningE2E:
    """End-to-end tests for the moral reasoning experiment."""

    @patch("aurelius.validator.experiments.moral_reasoning.experiment.call_chat_api_with_fallback")
    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_registered_miner_gets_score(self, mock_judge, mock_response):
        """Registered miner submits scenario → receives quality score."""
        core = _make_mock_core()

        mock_response.return_value = (
            _mock_llm_response("I agonized over this decision and carefully considered all sides. " * 10),
            "deepseek-v3",
        )
        mock_judge.return_value = (
            _mock_llm_response(_make_valid_judge_output()),
            "gpt-4o",
        )

        experiment = _make_experiment(core)

        synapse = MagicMock()
        synapse.prompt = "I found out my boss is embezzling funds..."
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "registered_miner_key"

        result = experiment._handle_scenario(synapse)

        assert result.danger_score > 0.0  # Got a score
        assert result.response is not None  # Got a response

    def test_rate_limits_handled_at_routing_layer(self):
        """F2: Rate limits are checked at the routing layer, not in _handle_scenario.

        FR-015: ExperimentManager.route_submission checks per-(hotkey, experiment_id)
        rate limits before forwarding to the experiment.
        """
        core = _make_mock_core()
        experiment = _make_experiment(core)

        synapse = MagicMock()
        synapse.prompt = ""  # empty → fast rejection path
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "test_miner"

        experiment._handle_scenario(synapse)

        # _handle_scenario should NOT call check_rate_limits (F2 fix)
        core.experiment_manager.check_rate_limits.assert_not_called()

    @patch("aurelius.validator.experiments.moral_reasoning.experiment.call_chat_api_with_fallback")
    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_novelty_uses_experiment_id(self, mock_judge, mock_response):
        """FR-017: Novelty check must use experiment_id='moral-reasoning'.

        Assert the experiment_id argument in the novelty check call.
        """
        core = _make_mock_core()

        mock_response.return_value = (
            _mock_llm_response("I wrestled with this decision and carefully considered the ethical implications. " * 8),
            "model-a",
        )
        mock_judge.return_value = (
            _mock_llm_response(_make_valid_judge_output()),
            "gpt-4o",
        )

        experiment = _make_experiment(core)

        synapse = MagicMock()
        synapse.prompt = "A moral dilemma..."
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "test_miner"

        experiment._handle_scenario(synapse)

        # The background task is submitted via executor
        core.background_executor.submit.assert_called_once()

        # Extract the background function and call it manually
        submit_call = core.background_executor.submit.call_args
        bg_func = submit_call[0][0]
        bg_args = submit_call[0][1:]
        bg_func(*bg_args)

        # Verify novelty check was called with experiment_id="moral-reasoning"
        core.novelty_client.check_novelty.assert_called_once()
        novelty_call_kwargs = core.novelty_client.check_novelty.call_args
        assert novelty_call_kwargs[1].get("experiment_id") == "moral-reasoning" or \
               (len(novelty_call_kwargs[0]) > 2 and False), \
               f"Expected experiment_id='moral-reasoning' in novelty check, got {novelty_call_kwargs}"

    @patch("aurelius.validator.experiments.moral_reasoning.experiment.call_chat_api_with_fallback")
    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_audit_trail_logged(self, mock_judge, mock_response):
        """FR-019: Verify audit trail contains all required fields."""
        core = _make_mock_core()

        mock_response.return_value = (
            _mock_llm_response("I spent a long time thinking about this moral dilemma and its implications for everyone involved. " * 8),
            "deepseek-v3",
        )
        mock_judge.return_value = (
            _mock_llm_response(_make_valid_judge_output()),
            "gpt-4o",
        )

        experiment = _make_experiment(core)

        synapse = MagicMock()
        synapse.prompt = "A dilemma scenario..."
        synapse.experiment_id = "moral-reasoning"
        synapse.dendrite.hotkey = "test_miner"

        experiment._handle_scenario(synapse)

        # Execute background task to trigger logging
        submit_call = core.background_executor.submit.call_args
        bg_func = submit_call[0][0]
        bg_args = submit_call[0][1:]
        bg_func(*bg_args)

        # Verify dataset logger was called
        core.dataset_logger.log_entry.assert_called_once()
        log_kwargs = core.dataset_logger.log_entry.call_args[1]

        # Check audit trail fields
        assert log_kwargs["experiment_id"] == "moral-reasoning"
        assert log_kwargs["prompt"] == "A dilemma scenario..."
        assert "moral dilemma" in log_kwargs["response"]

        # Check model_config contains moral reasoning audit data
        model_config = log_kwargs["model_config"]
        assert "signals" in model_config
        assert "dimension_scores" in model_config
        assert "quality_score" in model_config
        assert "passed_screening" in model_config
        assert "raw_judge_output" in model_config

    @patch("aurelius.validator.experiments.moral_reasoning.experiment.call_chat_api_with_fallback")
    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_high_quality_scores_higher(self, mock_judge, mock_response):
        """SC-007: High-quality scenario scores higher than shallow one."""
        core = _make_mock_core()

        mock_response.return_value = (
            _mock_llm_response("I thought deeply about the consequences of this moral situation and what it means for everyone. " * 8),
            "model-a",
        )

        experiment = _make_experiment(core)

        # High quality: all signals true → quality 1.0
        mock_judge.return_value = (
            _mock_llm_response(_make_valid_judge_output()),
            "gpt-4o",
        )

        synapse_high = MagicMock()
        synapse_high.prompt = "A nuanced dilemma..."
        synapse_high.experiment_id = "moral-reasoning"
        synapse_high.dendrite.hotkey = "miner_a"

        result_high = experiment._handle_scenario(synapse_high)

        # Low quality: many signals false → lower quality
        mock_judge.return_value = (
            _mock_llm_response(_make_valid_judge_output({
                "identifying_third_party": False,
                "logical_alternatives": False,
                "logical_temporal": False,
                "clear_reconstructable": False,
                "helpful_relationship": False,
                "helpful_conditions_for_different_choice": False,
                "harmless_no_dismissal": False,
            })),
            "gpt-4o",
        )

        synapse_low = MagicMock()
        synapse_low.prompt = "A shallow dilemma..."
        synapse_low.experiment_id = "moral-reasoning"
        synapse_low.dendrite.hotkey = "miner_b"

        result_low = experiment._handle_scenario(synapse_low)

        assert result_high.danger_score > result_low.danger_score
