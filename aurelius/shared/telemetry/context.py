"""Thread-local context for telemetry correlation and span tracking."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aurelius.shared.telemetry.span import Span


class TelemetryContext:
    """Thread-local context for correlation IDs and current span tracking."""

    _local = threading.local()

    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID for current thread."""
        cls._local.correlation_id = correlation_id

    @classmethod
    def get_correlation_id(cls) -> str | None:
        """Get current correlation ID."""
        return getattr(cls._local, "correlation_id", None)

    @classmethod
    def clear_correlation_id(cls) -> None:
        """Clear correlation ID for current thread."""
        cls._local.correlation_id = None

    @classmethod
    def set_current_span(cls, span: Span | None) -> None:
        """Set current span for current thread."""
        cls._local.current_span = span

    @classmethod
    def get_current_span(cls) -> Span | None:
        """Get current span for current thread."""
        return getattr(cls._local, "current_span", None)

    @classmethod
    def set_validator_identity(
        cls,
        hotkey: str | None = None,
        uid: int | None = None,
        coldkey: str | None = None,
    ) -> None:
        """Set validator identity for current thread."""
        cls._local.validator_hotkey = hotkey
        cls._local.validator_uid = uid
        cls._local.validator_coldkey = coldkey

    @classmethod
    def get_validator_identity(cls) -> dict:
        """Get validator identity for current thread."""
        return {
            "validator_hotkey": getattr(cls._local, "validator_hotkey", None),
            "validator_uid": getattr(cls._local, "validator_uid", None),
            "validator_coldkey": getattr(cls._local, "validator_coldkey", None),
        }

    @classmethod
    def set_network_context(
        cls,
        netuid: int | None = None,
        network: str | None = None,
        block_height: int | None = None,
    ) -> None:
        """Set network context for current thread."""
        cls._local.netuid = netuid
        cls._local.network = network
        cls._local.block_height = block_height

    @classmethod
    def get_network_context(cls) -> dict:
        """Get network context for current thread."""
        return {
            "netuid": getattr(cls._local, "netuid", None),
            "network": getattr(cls._local, "network", None),
            "block_height": getattr(cls._local, "block_height", None),
        }

    @classmethod
    def update_block_height(cls, block_height: int) -> None:
        """Update block height in network context."""
        cls._local.block_height = block_height
