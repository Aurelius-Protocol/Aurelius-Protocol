"""Telemetry package for validator observability."""

from aurelius.shared.telemetry.client import TelemetryClient
from aurelius.shared.telemetry.context import TelemetryContext
from aurelius.shared.telemetry.schemas import ErrorEvent, LogEvent, SpanEvent, TelemetryEvent
from aurelius.shared.telemetry.span import Span
from aurelius.shared.telemetry.otel_exporter import AureliusSpanExporter, AureliusLogExporter
from aurelius.shared.telemetry.otel_setup import setup_opentelemetry, get_tracer, shutdown_opentelemetry
from aurelius.shared.telemetry.http_client import (
    get_telemetry_session,
    get_circuit_breaker,
    CircuitBreaker,
    CircuitState,
)

__all__ = [
    "TelemetryClient",
    "TelemetryContext",
    "TelemetryEvent",
    "ErrorEvent",
    "LogEvent",
    "SpanEvent",
    "Span",
    # OpenTelemetry integration
    "AureliusSpanExporter",
    "AureliusLogExporter",
    "setup_opentelemetry",
    "get_tracer",
    "shutdown_opentelemetry",
    # HTTP client utilities
    "get_telemetry_session",
    "get_circuit_breaker",
    "CircuitBreaker",
    "CircuitState",
]
