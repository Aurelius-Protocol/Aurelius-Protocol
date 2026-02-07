"""Tests for the moral reasoning judge module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from aurelius.validator.experiments.moral_reasoning.judge import (
    JudgeEvaluationFailed,
    JudgeParseError,
    build_judge_prompt,
    evaluate_with_judge,
    parse_judge_output,
)
from aurelius.validator.experiments.moral_reasoning.signals import ALL_SIGNALS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_output(overrides: dict | None = None, extra_fields: dict | None = None) -> str:
    """Build a valid judge JSON output string.

    Defaults to 21/22 signals true (leaves ``identifying_third_party`` false)
    to avoid triggering the F6 all-22-true suspicious output check when the
    output is passed through ``evaluate_with_judge``.
    """
    signals = {s: True for s in ALL_SIGNALS}
    signals["identifying_third_party"] = False  # non-screening signal
    if overrides:
        signals.update(overrides)
    data: dict = {"signals": signals, "summary": "Good moral reasoning."}
    if extra_fields:
        data.update(extra_fields)
    return json.dumps(data)


def _mock_llm_response(content: str):
    """Create a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# parse_judge_output tests
# ---------------------------------------------------------------------------

class TestParseJudgeOutput:
    def test_valid_all_true(self):
        raw = _make_valid_output()
        signals, summary = parse_judge_output(raw)
        # 21/22 true (identifying_third_party is False by default in helper)
        for s in ALL_SIGNALS:
            if s == "identifying_third_party":
                assert getattr(signals, s) is False
            else:
                assert getattr(signals, s) is True
        assert summary == "Good moral reasoning."

    def test_valid_mixed(self):
        raw = _make_valid_output({"identifying_self_interest": False, "logical_temporal": False})
        signals, _ = parse_judge_output(raw)
        assert signals.identifying_self_interest is False
        assert signals.logical_temporal is False
        assert signals.identifying_other_interest is True

    def test_missing_signals(self):
        data = {"signals": {"identifying_self_interest": True}, "summary": ""}
        with pytest.raises(JudgeParseError, match="Missing signals"):
            parse_judge_output(json.dumps(data))

    def test_non_boolean_values(self):
        signals = {s: True for s in ALL_SIGNALS}
        signals["identifying_self_interest"] = "yes"  # type: ignore
        data = {"signals": signals, "summary": ""}
        with pytest.raises(JudgeParseError, match="Non-boolean signals"):
            parse_judge_output(json.dumps(data))

    def test_extra_fields_ignored(self):
        raw = _make_valid_output(extra_fields={"confidence": 0.95, "reasoning": "detailed"})
        signals, summary = parse_judge_output(raw)
        # Verify expected signals (21/22 true)
        assert signals.identifying_self_interest is True
        assert signals.identifying_third_party is False
        assert summary == "Good moral reasoning."

    def test_malformed_json(self):
        with pytest.raises(JudgeParseError, match="Invalid JSON"):
            parse_judge_output("not json at all {{{")

    def test_empty_string(self):
        with pytest.raises(JudgeParseError, match="Invalid JSON"):
            parse_judge_output("")

    def test_missing_signals_key(self):
        with pytest.raises(JudgeParseError, match="Missing or invalid 'signals' key"):
            parse_judge_output(json.dumps({"summary": "no signals"}))

    def test_signals_not_dict(self):
        with pytest.raises(JudgeParseError, match="Missing or invalid 'signals' key"):
            parse_judge_output(json.dumps({"signals": [True, False], "summary": ""}))

    def test_strips_markdown_fences(self):
        raw = "```json\n" + _make_valid_output() + "\n```"
        signals, summary = parse_judge_output(raw)
        # 21/22 true (identifying_third_party is False by default in helper)
        assert signals.identifying_self_interest is True
        assert signals.identifying_third_party is False

    def test_non_string_summary_converted(self):
        signals = {s: True for s in ALL_SIGNALS}
        data = {"signals": signals, "summary": 42}
        signals_obj, summary = parse_judge_output(json.dumps(data))
        assert summary == "42"

    def test_missing_summary_defaults_empty(self):
        signals = {s: True for s in ALL_SIGNALS}
        data = {"signals": signals}
        _, summary = parse_judge_output(json.dumps(data))
        assert summary == ""


# ---------------------------------------------------------------------------
# build_judge_prompt tests
# ---------------------------------------------------------------------------

class TestBuildJudgePrompt:
    def test_returns_two_messages(self):
        messages = build_judge_prompt("scenario text", "response text")
        assert len(messages) == 2

    def test_system_message_first(self):
        messages = build_judge_prompt("scenario text", "response text")
        assert messages[0]["role"] == "system"

    def test_user_message_contains_scenario_and_response(self):
        messages = build_judge_prompt("my scenario", "my response")
        user_content = messages[1]["content"]
        assert "my scenario" in user_content
        assert "my response" in user_content

    def test_system_prompt_contains_signal_names(self):
        messages = build_judge_prompt("s", "r")
        system_content = messages[0]["content"]
        # Check a sample of signal names are in the system prompt
        assert "identifying_self_interest" in system_content
        assert "harmless_acknowledges_cost" in system_content
        assert "logical_temporal" in system_content


# ---------------------------------------------------------------------------
# evaluate_with_judge tests
# ---------------------------------------------------------------------------

class TestEvaluateWithJudge:
    # Use a long response to avoid triggering suspicious output check (>= 500 chars)
    LONG_RESPONSE = "I thought deeply about this moral dilemma and considered the consequences. " * 8

    def setup_method(self):
        self.judge_config = {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "temperature": 0.0,
        }

    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_successful_evaluation(self, mock_call):
        valid_output = _make_valid_output()
        mock_call.return_value = (_mock_llm_response(valid_output), "gpt-4o")

        signals, summary, raw = evaluate_with_judge(
            "scenario", self.LONG_RESPONSE, MagicMock(), self.judge_config
        )
        # 21/22 true (identifying_third_party is False by default in helper)
        assert signals.identifying_self_interest is True
        assert signals.identifying_third_party is False
        assert summary == "Good moral reasoning."
        assert raw == valid_output

    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_retry_on_malformed_then_success(self, mock_call):
        valid_output = _make_valid_output()
        mock_call.side_effect = [
            (_mock_llm_response("not json"), "gpt-4o"),  # First attempt fails parse
            (_mock_llm_response(valid_output), "gpt-4o"),  # Second attempt succeeds
        ]

        signals, summary, raw = evaluate_with_judge(
            "scenario", self.LONG_RESPONSE, MagicMock(), self.judge_config
        )
        assert signals.identifying_self_interest is True
        assert mock_call.call_count == 2

    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_fails_after_two_parse_failures(self, mock_call):
        mock_call.side_effect = [
            (_mock_llm_response("bad json 1"), "gpt-4o"),
            (_mock_llm_response("bad json 2"), "gpt-4o"),
        ]

        with pytest.raises(JudgeEvaluationFailed, match="2 attempts"):
            evaluate_with_judge("scenario", self.LONG_RESPONSE, MagicMock(), self.judge_config)
        assert mock_call.call_count == 2

    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_non_parse_error_fails_immediately(self, mock_call):
        mock_call.side_effect = RuntimeError("network error")

        with pytest.raises(JudgeEvaluationFailed, match="network error"):
            evaluate_with_judge("scenario", self.LONG_RESPONSE, MagicMock(), self.judge_config)
        assert mock_call.call_count == 1  # No retry on non-parse errors

    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_uses_judge_config_model(self, mock_call):
        """Judge must use its own configured model, not the response model."""
        valid_output = _make_valid_output()
        mock_call.return_value = (_mock_llm_response(valid_output), "judge-model")

        evaluate_with_judge("scenario", self.LONG_RESPONSE, MagicMock(), {
            "model": "judge-model",
            "max_tokens": 512,
            "temperature": 0.1,
        })

        call_args = mock_call.call_args
        api_params = call_args[0][1]  # Second positional arg
        assert api_params["model"] == "judge-model"
        assert api_params["max_tokens"] == 512
        # Temperature includes jitter, so it should be >= base
        assert api_params["temperature"] >= 0.1

    @patch("aurelius.validator.experiments.moral_reasoning.judge.call_chat_api_with_fallback")
    def test_no_fallback_models_for_judge(self, mock_call):
        """Judge uses its own model with no fallback chain."""
        valid_output = _make_valid_output()
        mock_call.return_value = (_mock_llm_response(valid_output), "gpt-4o")

        evaluate_with_judge("scenario", self.LONG_RESPONSE, MagicMock(), self.judge_config)

        call_args = mock_call.call_args
        assert call_args[1].get("fallback_models") == [] or call_args[0][2] == []
