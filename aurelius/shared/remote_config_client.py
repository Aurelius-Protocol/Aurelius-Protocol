"""Remote configuration client for fetching network-specific config from central API."""

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt
import requests

from aurelius.shared.config import Config


@dataclass
class RemoteConfigResult:
    """Result from fetching remote configuration.

    Attributes:
        success: Whether the fetch was successful
        config: The configuration dictionary (empty if not available)
        version: The config version number
        error: Error message if fetch failed
    """

    success: bool
    config: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    error: str | None = None


class RemoteConfigClient:
    """Client for fetching and applying remote configuration from the central API.

    This client periodically fetches network-specific configuration and applies
    allowed fields to the Config class. Secret fields, endpoints, and other
    security-sensitive settings are excluded from remote configuration.
    """

    # Fields that are NEVER allowed to be set remotely for security reasons
    EXCLUDED_FIELDS: frozenset[str] = frozenset({
        # Secrets - API keys must never be remotely configurable
        "CHUTES_API_KEY",
        "OPENAI_API_KEY",
        "CENTRAL_API_KEY",

        # Wallet configuration - prevents identity theft
        "VALIDATOR_WALLET_NAME",
        "VALIDATOR_HOTKEY",
        "MINER_WALLET_NAME",
        "MINER_HOTKEY",

        # Endpoints - prevents MITM attacks by redirecting traffic
        "CENTRAL_API_ENDPOINT",
        "NOVELTY_API_ENDPOINT",
        "TELEMETRY_TRACES_ENDPOINT",
        "TELEMETRY_LOGS_ENDPOINT",
        "TELEMETRY_REGISTRY_ENDPOINT",
        "SUBTENSOR_ENDPOINT",
        "CHUTES_API_BASE_URL",

        # Paths - prevents arbitrary file access
        "LOCAL_DATASET_PATH",
        "TELEMETRY_LOCAL_BACKUP_PATH",
        "VALIDATOR_TRUST_PERSISTENCE_PATH",
        "MINER_SCORES_PATH",

        # Network identity - prevents network confusion
        "BT_NETWORK",
        "BT_NETUID",
        "BT_PORT_VALIDATOR",
        "VALIDATOR_HOST",
        "BT_AXON_EXTERNAL_IP",

        # Mode flags - prevents disabling security features
        "LOCAL_MODE",
        "ADVANCED_MODE",
        "SKIP_WEIGHT_SETTING",
        "SKIP_ENDPOINT_VALIDATION",

        # Rate limiting - prevents disabling rate limits via remote config
        "GLOBAL_RATE_LIMIT_REQUESTS",

        # Provider selection - prevents switching to untrusted providers
        "CHAT_PROVIDER",
        "MODERATION_PROVIDER",

        # Model configuration - complex nested structures not suitable for remote config
        "ALLOWED_MODELS",
        "FALLBACK_MODELS",

        # Logging - could be used to leak sensitive data
        "LOG_SENSITIVE_DATA",

        # Telemetry enable flags - should be local decision
        "TELEMETRY_ENABLED",
        "TELEMETRY_TRACES_ENABLED",
        "TELEMETRY_LOGS_ENABLED",

        # Feature flags that affect security model
        "ENABLE_CONSENSUS",
        "ENABLE_LOCAL_BACKUP",
        "ENABLE_VALIDATOR_TRUST_TRACKING",
        "MODERATION_FAIL_MODE",
    })

    # Pattern for fields that should never be remotely configurable
    EXCLUDED_PATTERNS: tuple[re.Pattern, ...] = (
        re.compile(r".*_API_KEY$"),
        re.compile(r".*_ENDPOINT$"),
        re.compile(r".*_PATH$"),
        re.compile(r".*_URL$"),
    )

    def __init__(
        self,
        api_endpoint: str | None = None,
        netuid: int | None = None,
        timeout: int = 10,
        poll_interval: int = 300,
    ):
        """
        Initialize remote config client.

        Args:
            api_endpoint: Base URL for remote config API. If not provided,
                         derived from CENTRAL_API_ENDPOINT.
            netuid: Network UID to fetch config for. Defaults to Config.BT_NETUID.
            timeout: Request timeout in seconds.
            poll_interval: How often to poll for config updates (seconds).
        """
        # Derive endpoint from CENTRAL_API_ENDPOINT if not provided
        if api_endpoint:
            self.api_endpoint = api_endpoint
        elif Config.CENTRAL_API_ENDPOINT:
            # Replace /api/collections with /api/remote-config
            base = Config.CENTRAL_API_ENDPOINT.replace("/api/collections", "")
            self.api_endpoint = f"{base}/api/remote-config"
        else:
            self.api_endpoint = None

        self.netuid = netuid or Config.BT_NETUID
        self.timeout = timeout
        self.poll_interval = poll_interval

        # Cache for last successful config
        self._cached_config: dict[str, Any] = {}
        self._cached_version: int = 0
        self._lock = threading.Lock()

        if self.api_endpoint:
            bt.logging.info(f"Remote config client: endpoint at {self.api_endpoint}/{self.netuid}")
        else:
            bt.logging.info("Remote config client: No endpoint configured (disabled)")

    def is_available(self) -> bool:
        """Check if remote config is available (endpoint configured)."""
        return bool(self.api_endpoint)

    def _is_field_allowed(self, field_name: str) -> bool:
        """Check if a field is allowed to be remotely configured."""
        # Check explicit exclusion list
        if field_name in self.EXCLUDED_FIELDS:
            return False

        # Check exclusion patterns
        for pattern in self.EXCLUDED_PATTERNS:
            if pattern.match(field_name):
                return False

        # Check if field exists in Config (only configure known fields)
        if not hasattr(Config, field_name):
            return False

        return True

    def fetch_config(self) -> RemoteConfigResult:
        """
        Fetch remote configuration from the API.

        Returns:
            RemoteConfigResult with success status, config, and version.
            On 404 (no config exists), returns success=True with empty config.
            On error, returns success=False with cached config preserved.
        """
        if not self.api_endpoint:
            return RemoteConfigResult(
                success=False,
                error="No API endpoint configured",
            )

        try:
            response = requests.get(
                f"{self.api_endpoint}/{self.netuid}",
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                config = data.get("config", {})
                version = data.get("version", 0)

                # Update cache
                with self._lock:
                    self._cached_config = config
                    self._cached_version = version

                return RemoteConfigResult(
                    success=True,
                    config=config,
                    version=version,
                )

            elif response.status_code == 404:
                # No config exists for this netuid - this is OK
                bt.logging.debug(f"No remote config found for netuid {self.netuid}")
                return RemoteConfigResult(
                    success=True,
                    config={},
                    version=0,
                )

            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                bt.logging.warning(f"Remote config fetch failed: {error_msg}")
                return RemoteConfigResult(
                    success=False,
                    config=self._cached_config,
                    version=self._cached_version,
                    error=error_msg,
                )

        except requests.Timeout:
            bt.logging.warning(f"Remote config fetch timed out after {self.timeout}s")
            return RemoteConfigResult(
                success=False,
                config=self._cached_config,
                version=self._cached_version,
                error="Request timed out",
            )

        except requests.RequestException as e:
            bt.logging.warning(f"Remote config fetch request failed: {e}")
            return RemoteConfigResult(
                success=False,
                config=self._cached_config,
                version=self._cached_version,
                error=str(e),
            )

        except Exception as e:
            bt.logging.error(f"Remote config fetch unexpected error: {e}")
            return RemoteConfigResult(
                success=False,
                config=self._cached_config,
                version=self._cached_version,
                error=str(e),
            )

    def _convert_value(self, field_name: str, value: Any) -> Any:
        """
        Convert a value to the expected type based on existing Config field.

        Args:
            field_name: The config field name
            value: The value to convert

        Returns:
            Converted value, or original if conversion not needed/possible
        """
        if not hasattr(Config, field_name):
            return value

        current_value = getattr(Config, field_name)

        # Handle None case
        if current_value is None:
            return value

        current_type = type(current_value)

        # Already correct type
        if isinstance(value, current_type):
            return value

        # Allow int -> float conversion
        if current_type == float and isinstance(value, int):
            return float(value)

        # Allow float -> int conversion (with truncation)
        if current_type == int and isinstance(value, float):
            return int(value)

        # Allow string -> bool conversion
        if current_type == bool and isinstance(value, str):
            return value.lower() in ("true", "1", "yes")

        # Cannot convert
        bt.logging.warning(
            f"Remote config: Cannot convert {field_name} from {type(value).__name__} "
            f"to {current_type.__name__}, skipping"
        )
        return None

    def apply_to_config(self, remote_config: dict[str, Any]) -> list[str]:
        """
        Apply remote configuration values to the Config class.

        Only allowed fields are applied. Excluded fields are logged and skipped.
        Type validation is performed before applying.

        Args:
            remote_config: Dictionary of config field names to values

        Returns:
            List of field names that were successfully updated
        """
        updated_fields: list[str] = []
        skipped_fields: list[str] = []

        for field_name, value in remote_config.items():
            # Check if field is allowed
            if not self._is_field_allowed(field_name):
                skipped_fields.append(field_name)
                continue

            # Convert value to expected type
            converted_value = self._convert_value(field_name, value)
            if converted_value is None:
                continue

            # Get current value for comparison
            current_value = getattr(Config, field_name, None)

            # Only update if value has changed
            if current_value != converted_value:
                setattr(Config, field_name, converted_value)
                updated_fields.append(field_name)
                bt.logging.info(f"Remote config: {field_name} updated: {current_value} -> {converted_value}")

        if skipped_fields:
            bt.logging.debug(f"Remote config: Skipped excluded fields: {skipped_fields}")

        return updated_fields

    def get_cached_config(self) -> dict[str, Any]:
        """Get the last successfully fetched configuration."""
        with self._lock:
            return self._cached_config.copy()

    def get_cached_version(self) -> int:
        """Get the version of the cached configuration."""
        with self._lock:
            return self._cached_version


# Singleton instance
_remote_config_client: RemoteConfigClient | None = None


def get_remote_config_client() -> RemoteConfigClient:
    """Get singleton remote config client instance."""
    global _remote_config_client
    if _remote_config_client is None:
        _remote_config_client = RemoteConfigClient()
    return _remote_config_client
