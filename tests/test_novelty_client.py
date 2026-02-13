"""Tests for NoveltyClient per-experiment isolation (T086)."""

import json
from unittest.mock import MagicMock, patch

import pytest


def _get_request_body(call_args) -> dict:
    """Extract parsed request body from mock call args (supports both json= and data= kwargs)."""
    if "json" in call_args.kwargs:
        return call_args.kwargs["json"]
    if "data" in call_args.kwargs:
        return json.loads(call_args.kwargs["data"])
    raise AssertionError("No json= or data= kwarg found in call_args")


class TestNoveltyClientExperimentId:
    """Tests for per-experiment novelty isolation."""

    @pytest.fixture
    def novelty_client(self):
        """Create a NoveltyClient instance with mock endpoint."""
        from aurelius.shared.novelty_client import NoveltyClient

        return NoveltyClient(api_endpoint="http://test-api.example.com/api/novelty")

    def test_check_novelty_includes_experiment_id(self, novelty_client):
        """Test that check_novelty includes experiment_id in API request."""
        with patch.object(novelty_client._session, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "novelty_score": 0.85,
                "max_similarity": 0.15,
                "similar_count": 2,
            }
            mock_post.return_value = mock_response

            result = novelty_client.check_novelty(
                prompt="test prompt",
                prompt_embedding=[0.1] * 384,
                experiment_id="jailbreak-v1",
            )

            # Verify API was called with experiment_id
            mock_post.assert_called_once()
            request_json = _get_request_body(mock_post.call_args)
            assert request_json["experiment_id"] == "jailbreak-v1"

    def test_check_novelty_default_experiment_id(self, novelty_client):
        """Test that default experiment_id is 'prompt' for backward compatibility."""
        with patch.object(novelty_client._session, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "novelty_score": 0.9,
                "max_similarity": 0.1,
                "similar_count": 1,
            }
            mock_post.return_value = mock_response

            # Call without explicit experiment_id
            result = novelty_client.check_novelty(
                prompt="test prompt",
                prompt_embedding=[0.1] * 384,
            )

            # Should default to "prompt"
            request_json = _get_request_body(mock_post.call_args)
            assert request_json["experiment_id"] == "prompt"

    def test_experiment_a_novelty_independent_of_b(self, novelty_client):
        """Test that experiment A novelty doesn't affect experiment B."""
        with patch.object(novelty_client._session, "post") as mock_post:
            # Different responses for different experiments
            def mock_response_for_experiment(*args, **kwargs):
                response = MagicMock()
                response.status_code = 200
                body = json.loads(kwargs.get("data", "{}")) if "data" in kwargs else kwargs.get("json", {})
                exp_id = body.get("experiment_id", "prompt")

                if exp_id == "exp-a":
                    response.json.return_value = {
                        "novelty_score": 0.1,  # Low novelty (duplicate in exp-a)
                        "max_similarity": 0.9,
                        "similar_count": 10,
                    }
                else:
                    response.json.return_value = {
                        "novelty_score": 0.95,  # High novelty (novel in exp-b)
                        "max_similarity": 0.05,
                        "similar_count": 0,
                    }
                return response

            mock_post.side_effect = mock_response_for_experiment

            # Same prompt, different experiments
            result_a = novelty_client.check_novelty(
                prompt="test prompt",
                prompt_embedding=[0.1] * 384,
                experiment_id="exp-a",
            )
            result_b = novelty_client.check_novelty(
                prompt="test prompt",
                prompt_embedding=[0.1] * 384,
                experiment_id="exp-b",
            )

            # Results should be independent
            assert result_a.novelty_score == 0.1  # Duplicate in exp-a
            assert result_b.novelty_score == 0.95  # Novel in exp-b


class TestNoveltyClientBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    @pytest.fixture
    def novelty_client(self):
        """Create a NoveltyClient instance."""
        from aurelius.shared.novelty_client import NoveltyClient

        return NoveltyClient(api_endpoint="http://test.example.com/api/novelty")

    def test_check_novelty_works_without_experiment_id(self, novelty_client):
        """Test that existing code without experiment_id still works."""
        with patch.object(novelty_client._session, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "novelty_score": 0.75,
                "max_similarity": 0.25,
                "similar_count": 3,
            }
            mock_post.return_value = mock_response

            # Old-style call without experiment_id
            result = novelty_client.check_novelty(
                prompt="test",
                prompt_embedding=[0.1] * 384,
            )

            assert result is not None
            assert result.novelty_score == 0.75

    def test_result_structure_unchanged(self, novelty_client):
        """Test that NoveltyResult structure is unchanged."""
        from aurelius.shared.novelty_client import NoveltyResult

        result = NoveltyResult(
            novelty_score=0.8,
            max_similarity=0.2,
            similar_count=5,
            most_similar_id=123,
        )

        # All existing fields should work
        assert result.novelty_score == 0.8
        assert result.max_similarity == 0.2
        assert result.similar_count == 5
        assert result.most_similar_id == 123
