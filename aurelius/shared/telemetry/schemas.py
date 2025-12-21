"""Data schemas for telemetry events."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


@dataclass
class TelemetryEvent:
    """Base schema for all telemetry events."""

    # Event identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Validator identity
    validator_hotkey: str | None = None
    validator_uid: int | None = None
    validator_coldkey: str | None = None

    # Correlation
    correlation_id: str | None = None
    parent_span_id: str | None = None

    # Network context
    netuid: int | None = None
    network: str | None = None  # "mainnet" | "testnet" | "local"
    block_height: int | None = None

    # Event classification
    event_type: str = "base"  # "error" | "log" | "span"
    event_name: str = ""

    # Optional metadata
    tags: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ErrorEvent(TelemetryEvent):
    """Schema for error events."""

    event_type: str = "error"

    # Exception details
    exception_type: str = ""
    exception_message: str = ""
    stack_trace: str = ""

    # Context
    operation: str = ""
    component: str = ""
    severity: str = "error"  # "critical" | "error" | "warning"

    # Optional rich context
    request_context: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None

    # Recovery info
    recovered: bool = False
    recovery_action: str | None = None


@dataclass
class LogEvent(TelemetryEvent):
    """Schema for structured log events."""

    event_type: str = "log"

    # Log details
    level: str = "INFO"  # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    message: str = ""
    logger_name: str = "validator"

    # Structured data
    structured_data: dict[str, Any] | None = None

    # Content (optional, controlled by config)
    prompt: str | None = None
    response: str | None = None

    # Performance context
    duration_ms: float | None = None


@dataclass
class SpanEvent(TelemetryEvent):
    """Schema for trace spans."""

    event_type: str = "span"

    # Span identity
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""

    # Span details
    operation_name: str = ""
    span_kind: str = "internal"  # "server" | "client" | "internal"
    status: str = "unset"  # "ok" | "error" | "unset"
    status_message: str | None = None

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_ms: float = 0.0

    # Attributes
    attributes: dict[str, Any] = field(default_factory=dict)

    # Events within span
    span_events: list[dict[str, Any]] | None = None

    # Links to other spans
    links: list[dict[str, Any]] | None = None
