from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aurelius.common.constants import WEIGHT_FAIL
from aurelius.validator.pipeline import PipelineResult
from aurelius.config import Config
from aurelius.validator.validator import Validator

_CONSISTENCY_FLOOR = Config.CONSISTENCY_FLOOR
_MIN_CONSISTENCY_REPORTS = Config.MIN_CONSISTENCY_REPORTS


def _make_validator() -> Validator:
    """Create a Validator with all bittensor internals mocked out."""
    with (
        patch("aurelius.validator.validator.bt.Wallet") as mock_wallet_cls,
        patch("aurelius.validator.validator.bt.Subtensor"),
        patch("aurelius.validator.validator.bt.Client"),
        patch("aurelius.validator.validator.fetch_metagraph_blocking") as mock_fetch,
        patch("aurelius.validator.validator.CentralAPIClient"),
        patch("aurelius.validator.validator.LocalSubmissionQueue"),
    ):
        # Wallet mock
        mock_wallet = MagicMock()
        mock_wallet.hotkey.ss58_address = "validator_hotkey_ABC"
        mock_wallet.hotkey_str = "validator_hotkey_ABC"
        mock_wallet.name = "test_wallet"
        mock_wallet_cls.return_value = mock_wallet

        # Metagraph mock (bittensor 11 snapshot shape: a neurons list)
        mock_meta = MagicMock()
        neurons = [
            SimpleNamespace(uid=0, hotkey="miner_hotkey_1", total_stake=0.0, validator_permit=False, axon="10.0.0.1:8091"),
            SimpleNamespace(uid=1, hotkey="validator_hotkey_ABC", total_stake=100.0, validator_permit=True, axon=None),
            SimpleNamespace(uid=2, hotkey="miner_hotkey_2", total_stake=0.0, validator_permit=False, axon="10.0.0.2:8091"),
        ]
        mock_meta.neurons = neurons
        mock_meta.num_uids = 3
        mock_meta.hotkeys = [n.hotkey for n in neurons]
        mock_fetch.return_value = mock_meta

        validator = Validator()
    return validator


class TestSelfHotkeyExcluded:
    async def test_self_hotkey_excluded_from_weights(self):
        """Validator's own hotkey is excluded when building the weight list."""
        validator = _make_validator()

        # Escape the local-env bypass so the remote override takes effect, and
        # force burn_mode off explicitly so the per-miner weight distribution
        # path runs regardless of what the current local/remote default is.
        validator.remote_config._environment = "testnet"
        validator.remote_config._config["burn_mode"] = False

        # Set up results that include the validator's own hotkey
        validator.results = {
            "miner_hotkey_1": PipelineResult(weight=0.8, stages=[]),
            "validator_hotkey_ABC": PipelineResult(weight=0.9, stages=[]),
            "miner_hotkey_2": PipelineResult(weight=0.7, stages=[]),
        }
        # Ensure ramp-up does not skip these
        validator.validation_counts = {
            "miner_hotkey_1": 5,
            "validator_hotkey_ABC": 5,
            "miner_hotkey_2": 5,
        }
        validator.start_time = 0  # Long ago, so in_ramp_up is False

        # Mock the API client for consistency multiplier (returns 1.0)
        validator.api_client = AsyncMock()
        validator.api_client.get_consistency = AsyncMock(
            return_value={"agreement_rate": 0.9, "total_reports": 50}
        )

        # Mock bt.set_weights to capture the {uid: weight} mapping it receives
        with patch("aurelius.validator.validator.bt.set_weights") as mock_set:
            await validator._set_weights()

            assert mock_set.called
            weights_by_uid = mock_set.call_args[0][1]

            # UID for validator_hotkey_ABC is 1 — it must NOT be present
            assert 1 not in weights_by_uid, "Validator's own UID should be excluded from weights"
            # Miner UIDs 0 and 2 should be present
            assert 0 in weights_by_uid
            assert 2 in weights_by_uid


class TestConsistencyMultiplier:
    async def test_consistency_multiplier_default(self):
        """When API is unavailable, multiplier is 1.0."""
        validator = _make_validator()
        validator.api_client = None
        result = await validator._get_consistency_multiplier()
        assert result == 1.0

    async def test_consistency_multiplier_low_reports(self):
        """When total_reports < threshold, multiplier is 1.0 (not enough data)."""
        validator = _make_validator()
        validator.api_client = AsyncMock()
        validator.api_client.get_consistency = AsyncMock(
            return_value={"agreement_rate": 0.3, "total_reports": _MIN_CONSISTENCY_REPORTS - 1}
        )
        result = await validator._get_consistency_multiplier()
        assert result == 1.0

    async def test_consistency_multiplier_below_floor(self):
        """When agreement_rate < floor, multiplier is 0.0 (zeroed influence)."""
        validator = _make_validator()
        validator.api_client = AsyncMock()
        validator.api_client.get_consistency = AsyncMock(
            return_value={"agreement_rate": _CONSISTENCY_FLOOR - 0.1, "total_reports": 50}
        )
        result = await validator._get_consistency_multiplier()
        assert result == 0.0

    async def test_consistency_multiplier_normal(self):
        """When agreement_rate = 0.8, multiplier = (0.8 - 0.4) / (1.0 - 0.4)."""
        validator = _make_validator()
        validator.api_client = AsyncMock()
        validator.api_client.get_consistency = AsyncMock(
            return_value={"agreement_rate": 0.8, "total_reports": 50}
        )
        result = await validator._get_consistency_multiplier()
        expected = (0.8 - _CONSISTENCY_FLOOR) / (1.0 - _CONSISTENCY_FLOOR)
        assert abs(result - expected) < 1e-6

    async def test_consistency_multiplier_api_error(self):
        """When API call raises, multiplier falls back to 1.0."""
        validator = _make_validator()
        validator.api_client = AsyncMock()
        validator.api_client.get_consistency = AsyncMock(side_effect=Exception("network down"))
        result = await validator._get_consistency_multiplier()
        assert result == 1.0
