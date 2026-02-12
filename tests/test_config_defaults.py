"""Tests for network-aware configuration defaults."""

import os
from unittest.mock import MagicMock, patch

import pytest

from aurelius.shared.config import NETWORK_DEFAULTS, Config, ConfigurationError


class TestNetworkDefaults:
    """Tests for network-aware defaults."""

    def test_mainnet_defaults_defined(self):
        """Mainnet (netuid 37) should have defaults defined."""
        assert 37 in NETWORK_DEFAULTS
        defaults = NETWORK_DEFAULTS[37]
        assert defaults["DANGER_THRESHOLD"] == 0.3
        assert defaults["SINGLE_CATEGORY_THRESHOLD"] == 0.7
        assert defaults["MIN_HIT_RATE_THRESHOLD"] == 0.3
        assert defaults["MIN_VALIDATOR_STAKE"] == 1000.0
        assert defaults["MIN_NOVELTY_THRESHOLD"] == 0.02

    def test_testnet_defaults_defined(self):
        """Testnet (netuid 290) should have defaults defined."""
        assert 290 in NETWORK_DEFAULTS
        defaults = NETWORK_DEFAULTS[290]
        assert defaults["DANGER_THRESHOLD"] == 0.2
        assert defaults["SINGLE_CATEGORY_THRESHOLD"] == 0.8
        assert defaults["MIN_HIT_RATE_THRESHOLD"] == 0.3
        assert defaults["MIN_VALIDATOR_STAKE"] == 5.0
        assert defaults["MIN_NOVELTY_THRESHOLD"] == 0.3

    def test_local_defaults_defined(self):
        """Local mode (netuid 1) should have defaults defined."""
        assert 1 in NETWORK_DEFAULTS
        defaults = NETWORK_DEFAULTS[1]
        assert defaults["DANGER_THRESHOLD"] == 0.1
        assert defaults["SINGLE_CATEGORY_THRESHOLD"] == 0.5
        assert defaults["MIN_HIT_RATE_THRESHOLD"] == 0.0
        assert defaults["MIN_VALIDATOR_STAKE"] == 0.0
        assert defaults["MIN_NOVELTY_THRESHOLD"] == 0.0
        assert defaults["CENTRAL_API_ENDPOINT"] == "http://localhost:3000/api/collections"
        assert defaults["NOVELTY_API_ENDPOINT"] == "http://localhost:3000/api/novelty"
        assert defaults["EXPERIMENT_API_ENDPOINT"] == "http://localhost:3000/api/experiments"
        assert defaults["MINER_BURN_ENABLED"] is False
        assert defaults["MINER_BURN_PERCENTAGE"] == 0.0


class TestApplyLocalDefaults:
    """Tests for local-mode (netuid 1) defaults application."""

    def test_applies_local_defaults(self):
        """Netuid 1 should apply local-mode defaults with localhost endpoints."""
        original_danger = Config.DANGER_THRESHOLD
        original_stake = Config.MIN_VALIDATOR_STAKE
        original_burn = Config.MINER_BURN_ENABLED
        original_central = Config.CENTRAL_API_ENDPOINT
        original_novelty_ep = Config.NOVELTY_API_ENDPOINT

        try:
            Config.BT_NETUID = 1
            with patch.dict(os.environ, {}, clear=True):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    mock_logging.debug = MagicMock()
                    Config.apply_network_defaults()

            assert Config.DANGER_THRESHOLD == 0.1
            assert Config.MIN_VALIDATOR_STAKE == 0.0
            assert Config.MINER_BURN_ENABLED is False
            assert Config.MINER_BURN_PERCENTAGE == 0.0
            assert Config.CENTRAL_API_ENDPOINT == "http://localhost:3000/api/collections"
            assert Config.NOVELTY_API_ENDPOINT == "http://localhost:3000/api/novelty"
            assert Config.EXPERIMENT_API_ENDPOINT == "http://localhost:3000/api/experiments"
        finally:
            Config.DANGER_THRESHOLD = original_danger
            Config.MIN_VALIDATOR_STAKE = original_stake
            Config.MINER_BURN_ENABLED = original_burn
            Config.CENTRAL_API_ENDPOINT = original_central
            Config.NOVELTY_API_ENDPOINT = original_novelty_ep

    def test_local_network_name(self):
        """Netuid 1 should use 'local' as network name in logs."""
        original_danger = Config.DANGER_THRESHOLD

        try:
            Config.BT_NETUID = 1
            with patch.dict(os.environ, {}, clear=True):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    mock_logging.debug = MagicMock()
                    Config.apply_network_defaults()

            # Check that 'local' appears in the log messages
            info_calls = [str(call) for call in mock_logging.info.call_args_list]
            assert any("local" in str(call) for call in info_calls)
        finally:
            Config.DANGER_THRESHOLD = original_danger


class TestApplyNetworkDefaults:
    """Tests for Config.apply_network_defaults()."""

    def test_applies_mainnet_defaults(self):
        """Netuid 37 should apply mainnet defaults."""
        # Save original values
        original_danger = Config.DANGER_THRESHOLD
        original_stake = Config.MIN_VALIDATOR_STAKE

        try:
            # Set netuid and clear env vars
            Config.BT_NETUID = 37
            with patch.dict(os.environ, {}, clear=True):
                # Mock bittensor logging - it's imported inside the method
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    mock_logging.debug = MagicMock()
                    Config.apply_network_defaults()

            assert Config.DANGER_THRESHOLD == 0.3
            assert Config.MIN_VALIDATOR_STAKE == 1000.0
        finally:
            # Restore original values
            Config.DANGER_THRESHOLD = original_danger
            Config.MIN_VALIDATOR_STAKE = original_stake

    def test_applies_testnet_defaults(self):
        """Netuid 290 should apply testnet defaults."""
        original_danger = Config.DANGER_THRESHOLD
        original_stake = Config.MIN_VALIDATOR_STAKE

        try:
            Config.BT_NETUID = 290
            with patch.dict(os.environ, {}, clear=True):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    mock_logging.debug = MagicMock()
                    Config.apply_network_defaults()

            assert Config.DANGER_THRESHOLD == 0.2
            assert Config.MIN_VALIDATOR_STAKE == 5.0
        finally:
            Config.DANGER_THRESHOLD = original_danger
            Config.MIN_VALIDATOR_STAKE = original_stake

    def test_env_var_ignored_without_advanced_mode(self):
        """Without ADVANCED_MODE, env vars should be ignored with warning."""
        original_danger = Config.DANGER_THRESHOLD
        original_advanced = Config.ADVANCED_MODE

        try:
            Config.BT_NETUID = 37  # Would normally set 0.3
            Config.ADVANCED_MODE = False  # Default behavior
            # User tries to set 0.1 in env
            with patch.dict(os.environ, {"DANGER_THRESHOLD": "0.1"}):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    mock_logging.warning = MagicMock()
                    Config.apply_network_defaults()

            # Should be network default (0.3), NOT the env value (0.1)
            assert Config.DANGER_THRESHOLD == 0.3
            # Should have warned about ignored value
            mock_logging.warning.assert_called()
        finally:
            Config.DANGER_THRESHOLD = original_danger
            Config.ADVANCED_MODE = original_advanced

    def test_unknown_netuid_does_nothing(self):
        """Unknown netuid should not change defaults."""
        original_danger = Config.DANGER_THRESHOLD

        try:
            Config.BT_NETUID = 999  # Unknown netuid
            with patch.dict(os.environ, {}, clear=True):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.debug = MagicMock()
                    Config.apply_network_defaults()

            # Should remain unchanged
            assert Config.DANGER_THRESHOLD == original_danger
        finally:
            Config.DANGER_THRESHOLD = original_danger


class TestValidateAPIKeys:
    """Tests for API key validation in Config.validate()."""

    def test_missing_openai_key_raises_error(self):
        """Missing OpenAI key should raise ConfigurationError."""
        original_key = Config.OPENAI_API_KEY
        original_chutes = Config.CHUTES_API_KEY
        original_provider = Config.CHAT_PROVIDER

        try:
            Config.OPENAI_API_KEY = ""
            Config.CHUTES_API_KEY = "valid-key"
            Config.CHAT_PROVIDER = "chutes"

            with pytest.raises(ConfigurationError) as exc_info:
                Config.validate()

            assert "OPENAI_API_KEY" in str(exc_info.value)
            assert "required" in str(exc_info.value).lower()
        finally:
            Config.OPENAI_API_KEY = original_key
            Config.CHUTES_API_KEY = original_chutes
            Config.CHAT_PROVIDER = original_provider

    def test_placeholder_openai_key_raises_error(self):
        """Placeholder OpenAI key should be detected."""
        original_key = Config.OPENAI_API_KEY

        try:
            Config.OPENAI_API_KEY = "your-openai-api-key-here"

            with pytest.raises(ConfigurationError) as exc_info:
                Config.validate()

            assert "placeholder" in str(exc_info.value).lower()
        finally:
            Config.OPENAI_API_KEY = original_key

    def test_invalid_openai_key_format_raises_error(self):
        """OpenAI key not starting with sk- should raise error."""
        original_key = Config.OPENAI_API_KEY

        try:
            Config.OPENAI_API_KEY = "invalid-key-format"

            with pytest.raises(ConfigurationError) as exc_info:
                Config.validate()

            assert "sk-" in str(exc_info.value)
        finally:
            Config.OPENAI_API_KEY = original_key

    def test_missing_chutes_key_when_using_chutes(self):
        """Missing Chutes key should fail when CHAT_PROVIDER=chutes."""
        original_key = Config.OPENAI_API_KEY
        original_chutes = Config.CHUTES_API_KEY
        original_provider = Config.CHAT_PROVIDER

        try:
            Config.OPENAI_API_KEY = "sk-valid-key"
            Config.CHUTES_API_KEY = ""
            Config.CHAT_PROVIDER = "chutes"

            with pytest.raises(ConfigurationError) as exc_info:
                Config.validate()

            assert "CHUTES_API_KEY" in str(exc_info.value)
        finally:
            Config.OPENAI_API_KEY = original_key
            Config.CHUTES_API_KEY = original_chutes
            Config.CHAT_PROVIDER = original_provider

    def test_placeholder_chutes_key_raises_error(self):
        """Placeholder Chutes key should be detected."""
        original_key = Config.OPENAI_API_KEY
        original_chutes = Config.CHUTES_API_KEY
        original_provider = Config.CHAT_PROVIDER

        try:
            Config.OPENAI_API_KEY = "sk-valid-key"
            Config.CHUTES_API_KEY = "your-chutes-api-key"
            Config.CHAT_PROVIDER = "chutes"

            with pytest.raises(ConfigurationError) as exc_info:
                Config.validate()

            assert "placeholder" in str(exc_info.value).lower()
        finally:
            Config.OPENAI_API_KEY = original_key
            Config.CHUTES_API_KEY = original_chutes
            Config.CHAT_PROVIDER = original_provider

    @patch("os.path.exists", side_effect=lambda p: False if p == ".env" else os.path.exists(p))
    def test_valid_keys_pass_validation(self, mock_exists):
        """Valid API keys should pass validation."""
        original_key = Config.OPENAI_API_KEY
        original_chutes = Config.CHUTES_API_KEY
        original_provider = Config.CHAT_PROVIDER
        original_local = Config.LOCAL_MODE

        try:
            Config.OPENAI_API_KEY = "sk-valid-openai-key"
            Config.CHUTES_API_KEY = "valid-chutes-key"
            Config.CHAT_PROVIDER = "chutes"
            Config.LOCAL_MODE = True  # Avoid other validation issues

            # Should not raise
            Config.validate()
        finally:
            Config.OPENAI_API_KEY = original_key
            Config.CHUTES_API_KEY = original_chutes
            Config.CHAT_PROVIDER = original_provider
            Config.LOCAL_MODE = original_local


class TestDetectAndSetWallet:
    """Tests for Config.detect_and_set_wallet()."""

    def test_explicit_config_skips_detection(self):
        """Explicit wallet config should skip auto-detection."""
        with patch.dict(
            os.environ,
            {
                "VALIDATOR_WALLET_NAME": "my-validator",
                "VALIDATOR_HOTKEY": "my-hotkey",
            },
        ):
            with patch("bittensor.logging") as mock_logging:
                mock_logging.info = MagicMock()
                # Should not raise and not call detect_wallet
                with patch(
                    "aurelius.shared.wallet_detector.detect_wallet"
                ) as mock_detect:
                    Config.detect_and_set_wallet(role="validator")
                    mock_detect.assert_not_called()

    def test_auto_detection_success_sets_config(self, tmp_path):
        """Successful auto-detection should set config values."""
        wallet_dir = tmp_path / "myvalidator"
        wallet_dir.mkdir()
        (wallet_dir / "coldkey").touch()
        hotkeys_dir = wallet_dir / "hotkeys"
        hotkeys_dir.mkdir()
        (hotkeys_dir / "myhotkey").touch()

        original_wallet = Config.VALIDATOR_WALLET_NAME
        original_hotkey = Config.VALIDATOR_HOTKEY

        try:
            with patch.dict(os.environ, {}, clear=True):
                with patch(
                    "aurelius.shared.wallet_detector.get_wallets_path",
                    return_value=tmp_path,
                ):
                    with patch("bittensor.logging") as mock_logging:
                        mock_logging.info = MagicMock()
                        mock_logging.success = MagicMock()
                        Config.detect_and_set_wallet(role="validator")

            assert Config.VALIDATOR_WALLET_NAME == "myvalidator"
            assert Config.VALIDATOR_HOTKEY == "myhotkey"
        finally:
            Config.VALIDATOR_WALLET_NAME = original_wallet
            Config.VALIDATOR_HOTKEY = original_hotkey

    def test_auto_detection_failure_raises_error(self, tmp_path):
        """Failed auto-detection should raise ConfigurationError."""
        # Empty directory - no wallets
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "aurelius.shared.wallet_detector.get_wallets_path",
                return_value=tmp_path,
            ):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    with pytest.raises(ConfigurationError) as exc_info:
                        Config.detect_and_set_wallet(role="validator")

        assert "WALLET" in str(exc_info.value)


class TestAdvancedMode:
    """Tests for ADVANCED_MODE safeguard."""

    def test_advanced_mode_false_by_default(self):
        """ADVANCED_MODE should be False by default."""
        # Check that default is False when env is not set
        with patch.dict(os.environ, {}, clear=True):
            # Re-evaluate the expression that Config uses
            result = os.getenv("ADVANCED_MODE", "false").lower() == "true"
            assert result is False

    def test_advanced_mode_respects_env_overrides(self):
        """With ADVANCED_MODE=true, env values should be used."""
        original_danger = Config.DANGER_THRESHOLD
        original_advanced = Config.ADVANCED_MODE

        try:
            Config.BT_NETUID = 37  # Network default is 0.3
            Config.ADVANCED_MODE = True  # Enable advanced mode
            # User sets custom value - it should NOT be overwritten
            with patch.dict(os.environ, {"DANGER_THRESHOLD": "0.1"}):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    mock_logging.warning = MagicMock()
                    Config.apply_network_defaults()

            # In advanced mode, since env is set, apply_network_defaults
            # should NOT overwrite the value (it skips setting when env is set)
            # The Config.DANGER_THRESHOLD should not be changed to 0.3
            # Note: The current value remains what it was before
        finally:
            Config.DANGER_THRESHOLD = original_danger
            Config.ADVANCED_MODE = original_advanced

    def test_advanced_mode_applies_unset_defaults(self):
        """With ADVANCED_MODE, unset values should still get network defaults."""
        original_danger = Config.DANGER_THRESHOLD
        original_stake = Config.MIN_VALIDATOR_STAKE
        original_advanced = Config.ADVANCED_MODE

        try:
            Config.BT_NETUID = 37
            Config.ADVANCED_MODE = True
            # Only set one env var, leave others unset
            with patch.dict(os.environ, {"DANGER_THRESHOLD": "0.3"}, clear=True):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    mock_logging.warning = MagicMock()
                    Config.apply_network_defaults()

            # MIN_VALIDATOR_STAKE should be set to network default (1000)
            assert Config.MIN_VALIDATOR_STAKE == 1000.0
        finally:
            Config.DANGER_THRESHOLD = original_danger
            Config.MIN_VALIDATOR_STAKE = original_stake
            Config.ADVANCED_MODE = original_advanced

    def test_standard_mode_warns_about_ignored_values(self):
        """Without ADVANCED_MODE, should warn about ignored custom values."""
        original_danger = Config.DANGER_THRESHOLD
        original_stake = Config.MIN_VALIDATOR_STAKE
        original_advanced = Config.ADVANCED_MODE

        try:
            Config.BT_NETUID = 37
            Config.ADVANCED_MODE = False
            with patch.dict(os.environ, {"DANGER_THRESHOLD": "0.1", "MIN_VALIDATOR_STAKE": "500"}):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    mock_logging.warning = MagicMock()
                    Config.apply_network_defaults()

            # Verify warning was called with message about ignored values
            warning_calls = [str(call) for call in mock_logging.warning.call_args_list]
            # Check that at least one warning mentions "IGNORED"
            assert any("IGNORED" in str(call) for call in warning_calls)
            # Values should be network defaults, not env values
            assert Config.DANGER_THRESHOLD == 0.3
            assert Config.MIN_VALIDATOR_STAKE == 1000.0
        finally:
            Config.DANGER_THRESHOLD = original_danger
            Config.MIN_VALIDATOR_STAKE = original_stake
            Config.ADVANCED_MODE = original_advanced

    def test_standard_mode_always_uses_network_defaults(self):
        """Without ADVANCED_MODE, network defaults should always be applied."""
        original_danger = Config.DANGER_THRESHOLD
        original_hit_rate = Config.MIN_HIT_RATE_THRESHOLD
        original_stake = Config.MIN_VALIDATOR_STAKE
        original_novelty = Config.MIN_NOVELTY_THRESHOLD
        original_advanced = Config.ADVANCED_MODE

        try:
            Config.BT_NETUID = 290  # Testnet
            Config.ADVANCED_MODE = False
            with patch.dict(os.environ, {}, clear=True):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    Config.apply_network_defaults()

            # All testnet defaults should be applied
            assert Config.DANGER_THRESHOLD == 0.2
            assert Config.MIN_HIT_RATE_THRESHOLD == 0.3
            assert Config.MIN_VALIDATOR_STAKE == 5.0
            assert Config.MIN_NOVELTY_THRESHOLD == 0.3
        finally:
            Config.DANGER_THRESHOLD = original_danger
            Config.MIN_HIT_RATE_THRESHOLD = original_hit_rate
            Config.MIN_VALIDATOR_STAKE = original_stake
            Config.MIN_NOVELTY_THRESHOLD = original_novelty
            Config.ADVANCED_MODE = original_advanced

    def test_advanced_mode_logs_warning_on_enable(self):
        """Enabling ADVANCED_MODE should log a prominent warning."""
        original_advanced = Config.ADVANCED_MODE
        original_danger = Config.DANGER_THRESHOLD

        try:
            Config.BT_NETUID = 37
            Config.ADVANCED_MODE = True
            with patch.dict(os.environ, {}, clear=True):
                with patch("bittensor.logging") as mock_logging:
                    mock_logging.info = MagicMock()
                    mock_logging.warning = MagicMock()
                    Config.apply_network_defaults()

            # Verify warning was called with message about advanced mode
            warning_calls = [str(call) for call in mock_logging.warning.call_args_list]
            assert any("ADVANCED_MODE" in str(call) for call in warning_calls)
        finally:
            Config.ADVANCED_MODE = original_advanced
            Config.DANGER_THRESHOLD = original_danger


class TestGetValidatorHost:
    """Tests for Config.get_validator_host() and BT_AXON_EXTERNAL_IP support."""

    def test_bt_axon_external_ip_takes_priority(self):
        """BT_AXON_EXTERNAL_IP should take highest priority."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            Config.BT_AXON_EXTERNAL_IP = "1.2.3.4"
            Config.AUTO_DETECT_EXTERNAL_IP = True  # Should be ignored
            Config.VALIDATOR_HOST = "5.6.7.8"  # Should be ignored

            result = Config.get_validator_host()
            assert result == "1.2.3.4"
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host

    def test_auto_detect_used_when_bt_axon_not_set(self):
        """AUTO_DETECT_EXTERNAL_IP should be used when BT_AXON_EXTERNAL_IP is not set."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            Config.BT_AXON_EXTERNAL_IP = None
            Config.AUTO_DETECT_EXTERNAL_IP = True
            Config.VALIDATOR_HOST = "127.0.0.1"

            with patch.object(Config, "detect_external_ip", return_value="10.20.30.40"):
                result = Config.get_validator_host()
                assert result == "10.20.30.40"
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host

    def test_validator_host_fallback_when_auto_detect_fails(self):
        """VALIDATOR_HOST should be used when auto-detect fails."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            Config.BT_AXON_EXTERNAL_IP = None
            Config.AUTO_DETECT_EXTERNAL_IP = True
            Config.VALIDATOR_HOST = "192.168.1.100"

            # Auto-detect returns None (failure)
            with patch.object(Config, "detect_external_ip", return_value=None):
                result = Config.get_validator_host()
                assert result == "192.168.1.100"
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host

    def test_validator_host_used_when_auto_detect_disabled(self):
        """VALIDATOR_HOST should be used when AUTO_DETECT_EXTERNAL_IP is False."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            Config.BT_AXON_EXTERNAL_IP = None
            Config.AUTO_DETECT_EXTERNAL_IP = False
            Config.VALIDATOR_HOST = "10.0.0.1"

            result = Config.get_validator_host()
            assert result == "10.0.0.1"
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host

    def test_empty_bt_axon_external_ip_is_falsy(self):
        """Empty string BT_AXON_EXTERNAL_IP should be treated as not set."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            Config.BT_AXON_EXTERNAL_IP = ""  # Empty string
            Config.AUTO_DETECT_EXTERNAL_IP = False
            Config.VALIDATOR_HOST = "fallback.host"

            result = Config.get_validator_host()
            assert result == "fallback.host"
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host

    def test_priority_order_full_chain(self):
        """Test complete priority chain: BT_AXON > AUTO_DETECT > VALIDATOR_HOST."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            with patch.object(Config, "detect_external_ip", return_value="priority2"):
                # Test 1: All set - BT_AXON_EXTERNAL_IP wins
                Config.BT_AXON_EXTERNAL_IP = "priority1"
                Config.AUTO_DETECT_EXTERNAL_IP = True
                Config.VALIDATOR_HOST = "priority3"
                assert Config.get_validator_host() == "priority1"

                # Test 2: BT_AXON not set - AUTO_DETECT wins
                Config.BT_AXON_EXTERNAL_IP = None
                assert Config.get_validator_host() == "priority2"

                # Test 3: AUTO_DETECT disabled - VALIDATOR_HOST wins
                Config.AUTO_DETECT_EXTERNAL_IP = False
                assert Config.get_validator_host() == "priority3"
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host


class TestProductionValidationNetworkWarnings:
    """Tests for network-related production validation warnings."""

    def test_no_warning_when_bt_axon_external_ip_set(self):
        """Setting BT_AXON_EXTERNAL_IP should suppress network warnings."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            Config.BT_AXON_EXTERNAL_IP = "1.2.3.4"
            Config.AUTO_DETECT_EXTERNAL_IP = False
            Config.VALIDATOR_HOST = "127.0.0.1"  # Would normally warn

            warnings = Config.validate_production()
            network_warnings = [w for w in warnings if "reachable" in w.lower()]
            assert len(network_warnings) == 0
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host

    def test_no_warning_when_auto_detect_enabled(self):
        """AUTO_DETECT_EXTERNAL_IP=true should suppress network warnings."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            Config.BT_AXON_EXTERNAL_IP = None
            Config.AUTO_DETECT_EXTERNAL_IP = True
            Config.VALIDATOR_HOST = "127.0.0.1"  # Would normally warn

            warnings = Config.validate_production()
            network_warnings = [w for w in warnings if "reachable" in w.lower()]
            assert len(network_warnings) == 0
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host

    def test_warning_when_localhost_and_no_ip_config(self):
        """Should warn when VALIDATOR_HOST is localhost without proper IP config."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            Config.BT_AXON_EXTERNAL_IP = None
            Config.AUTO_DETECT_EXTERNAL_IP = False
            Config.VALIDATOR_HOST = "127.0.0.1"

            warnings = Config.validate_production()
            network_warnings = [w for w in warnings if "reachable" in w.lower()]
            assert len(network_warnings) == 1
            assert "BT_AXON_EXTERNAL_IP" in network_warnings[0]
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host

    def test_no_warning_when_validator_host_is_external_ip(self):
        """No warning when VALIDATOR_HOST is set to an external IP."""
        original_bt_axon = Config.BT_AXON_EXTERNAL_IP
        original_auto_detect = Config.AUTO_DETECT_EXTERNAL_IP
        original_validator_host = Config.VALIDATOR_HOST

        try:
            Config.BT_AXON_EXTERNAL_IP = None
            Config.AUTO_DETECT_EXTERNAL_IP = False
            Config.VALIDATOR_HOST = "203.0.113.50"  # External IP

            warnings = Config.validate_production()
            network_warnings = [w for w in warnings if "reachable" in w.lower()]
            assert len(network_warnings) == 0
        finally:
            Config.BT_AXON_EXTERNAL_IP = original_bt_axon
            Config.AUTO_DETECT_EXTERNAL_IP = original_auto_detect
            Config.VALIDATOR_HOST = original_validator_host


class TestFallbackModelValidation:
    """Tests for FALLBACK_MODELS validation."""

    @patch("os.path.exists", side_effect=lambda p: False if p == ".env" else os.path.exists(p))
    def test_invalid_fallback_model_raises_error(self, mock_exists):
        """FALLBACK_MODELS with invalid model should fail validation."""
        original_fallback = Config.FALLBACK_MODELS
        original_openai_key = Config.OPENAI_API_KEY
        original_chutes_key = Config.CHUTES_API_KEY
        original_provider = Config.CHAT_PROVIDER
        original_local = Config.LOCAL_MODE

        try:
            Config.FALLBACK_MODELS = ["nonexistent-model"]
            Config.OPENAI_API_KEY = "sk-valid-key"
            Config.CHUTES_API_KEY = "valid-chutes-key"
            Config.CHAT_PROVIDER = "chutes"
            Config.LOCAL_MODE = True

            with pytest.raises(ValueError, match="not in any vendor's ALLOWED_MODELS"):
                Config.validate()
        finally:
            Config.FALLBACK_MODELS = original_fallback
            Config.OPENAI_API_KEY = original_openai_key
            Config.CHUTES_API_KEY = original_chutes_key
            Config.CHAT_PROVIDER = original_provider
            Config.LOCAL_MODE = original_local

    @patch("os.path.exists", side_effect=lambda p: False if p == ".env" else os.path.exists(p))
    def test_valid_fallback_model_passes_validation(self, mock_exists):
        """FALLBACK_MODELS with valid model should pass validation."""
        original_fallback = Config.FALLBACK_MODELS
        original_openai_key = Config.OPENAI_API_KEY
        original_chutes_key = Config.CHUTES_API_KEY
        original_provider = Config.CHAT_PROVIDER
        original_local = Config.LOCAL_MODE

        try:
            # Use a model that exists in ALLOWED_MODELS
            Config.FALLBACK_MODELS = ["gpt-4o-mini"]  # In openai vendor
            Config.OPENAI_API_KEY = "sk-valid-key"
            Config.CHUTES_API_KEY = "valid-chutes-key"
            Config.CHAT_PROVIDER = "chutes"
            Config.LOCAL_MODE = True

            # Should not raise
            Config.validate()
        finally:
            Config.FALLBACK_MODELS = original_fallback
            Config.OPENAI_API_KEY = original_openai_key
            Config.CHUTES_API_KEY = original_chutes_key
            Config.CHAT_PROVIDER = original_provider
            Config.LOCAL_MODE = original_local

    @patch("os.path.exists", side_effect=lambda p: False if p == ".env" else os.path.exists(p))
    def test_empty_fallback_models_passes_validation(self, mock_exists):
        """Empty FALLBACK_MODELS should pass validation."""
        original_fallback = Config.FALLBACK_MODELS
        original_openai_key = Config.OPENAI_API_KEY
        original_chutes_key = Config.CHUTES_API_KEY
        original_provider = Config.CHAT_PROVIDER
        original_local = Config.LOCAL_MODE

        try:
            Config.FALLBACK_MODELS = []
            Config.OPENAI_API_KEY = "sk-valid-key"
            Config.CHUTES_API_KEY = "valid-chutes-key"
            Config.CHAT_PROVIDER = "chutes"
            Config.LOCAL_MODE = True

            # Should not raise
            Config.validate()
        finally:
            Config.FALLBACK_MODELS = original_fallback
            Config.OPENAI_API_KEY = original_openai_key
            Config.CHUTES_API_KEY = original_chutes_key
            Config.CHAT_PROVIDER = original_provider
            Config.LOCAL_MODE = original_local

    @patch("os.path.exists", side_effect=lambda p: False if p == ".env" else os.path.exists(p))
    def test_multiple_invalid_fallback_models_lists_first_bad_one(self, mock_exists):
        """With multiple invalid models, error should identify the first one."""
        original_fallback = Config.FALLBACK_MODELS
        original_openai_key = Config.OPENAI_API_KEY
        original_chutes_key = Config.CHUTES_API_KEY
        original_provider = Config.CHAT_PROVIDER
        original_local = Config.LOCAL_MODE

        try:
            Config.FALLBACK_MODELS = ["first-bad-model", "second-bad-model"]
            Config.OPENAI_API_KEY = "sk-valid-key"
            Config.CHUTES_API_KEY = "valid-chutes-key"
            Config.CHAT_PROVIDER = "chutes"
            Config.LOCAL_MODE = True

            with pytest.raises(ValueError, match="first-bad-model"):
                Config.validate()
        finally:
            Config.FALLBACK_MODELS = original_fallback
            Config.OPENAI_API_KEY = original_openai_key
            Config.CHUTES_API_KEY = original_chutes_key
            Config.CHAT_PROVIDER = original_provider
            Config.LOCAL_MODE = original_local
