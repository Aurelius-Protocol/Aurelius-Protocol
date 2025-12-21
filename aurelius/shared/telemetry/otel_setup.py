"""OpenTelemetry setup and configuration for Aurelius validators.

Provides a simple interface to configure OpenTelemetry with Aurelius custom exporters.
"""

from __future__ import annotations

import atexit
from typing import Any

import bittensor as bt
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry._logs import set_logger_provider

from aurelius.shared.telemetry.otel_exporter import AureliusSpanExporter, AureliusLogExporter


# Global providers (for cleanup on shutdown)
_tracer_provider: TracerProvider | None = None
_logger_provider: LoggerProvider | None = None
_initialized: bool = False


def setup_opentelemetry(
    service_name: str = "aurelius-validator",
    service_version: str | None = None,
    validator_hotkey: str | None = None,
    validator_uid: int | None = None,
    netuid: int | None = None,
    network: str | None = None,
    traces_endpoint: str | None = None,
    logs_endpoint: str | None = None,
    enable_traces: bool = True,
    enable_logs: bool = True,
    trace_batch_size: int = 100,
    log_batch_size: int = 200,
    flush_interval_ms: int = 5000,
    local_backup_path: str | None = None,
) -> dict[str, Any]:
    """Configure OpenTelemetry with Aurelius custom exporters.

    Args:
        service_name: Service name for resource attributes
        service_version: Service version (defaults to package version)
        validator_hotkey: Validator's SS58 hotkey
        validator_uid: Validator's UID
        netuid: Subnet UID
        network: Network name
        traces_endpoint: Custom endpoint for traces (overrides default)
        logs_endpoint: Custom endpoint for logs (overrides default)
        enable_traces: Whether to enable trace export
        enable_logs: Whether to enable log export
        trace_batch_size: Number of spans to batch
        log_batch_size: Number of logs to batch
        flush_interval_ms: Flush interval in milliseconds
        local_backup_path: Path for local backup on failure

    Returns:
        Dictionary with tracer instance (and logger if enabled)
    """
    global _tracer_provider, _logger_provider, _initialized

    if _initialized:
        bt.logging.warning("OpenTelemetry already initialized, skipping setup")
        return {"tracer": trace.get_tracer(service_name)}

    # Get version
    if service_version is None:
        try:
            from aurelius import __version__
            service_version = __version__
        except (ImportError, AttributeError):
            service_version = "0.1.0"

    # Create resource with service information
    resource_attributes = {
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "service.namespace": "aurelius",
    }

    if validator_hotkey:
        resource_attributes["validator.hotkey"] = validator_hotkey
    if validator_uid is not None:
        resource_attributes["validator.uid"] = str(validator_uid)
    if netuid is not None:
        resource_attributes["network.netuid"] = str(netuid)
    if network:
        resource_attributes["network.name"] = network

    resource = Resource.create(resource_attributes)

    result: dict[str, Any] = {}
    flush_interval_sec = flush_interval_ms / 1000

    # Setup tracing
    if enable_traces:
        span_exporter = AureliusSpanExporter(
            endpoint=traces_endpoint,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            netuid=netuid,
            network=network,
            batch_size=trace_batch_size,
            flush_interval=flush_interval_sec,
            local_backup_path=local_backup_path,
        )

        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(
                span_exporter,
                max_queue_size=10000,
                max_export_batch_size=trace_batch_size,
                schedule_delay_millis=flush_interval_ms,
            )
        )

        trace.set_tracer_provider(_tracer_provider)
        result["tracer"] = trace.get_tracer(service_name, service_version)
        bt.logging.info(f"OpenTelemetry tracing enabled for {service_name}")

    # Setup logging
    if enable_logs:
        log_exporter = AureliusLogExporter(
            endpoint=logs_endpoint,
            validator_hotkey=validator_hotkey,
            validator_uid=validator_uid,
            netuid=netuid,
            network=network,
            batch_size=log_batch_size,
            flush_interval=flush_interval_sec,
            local_backup_path=local_backup_path,
        )

        _logger_provider = LoggerProvider(resource=resource)
        _logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(
                log_exporter,
                max_queue_size=10000,
                max_export_batch_size=log_batch_size,
                schedule_delay_millis=flush_interval_ms,
            )
        )

        set_logger_provider(_logger_provider)
        result["logger"] = _logger_provider.get_logger(service_name, service_version)
        bt.logging.info(f"OpenTelemetry logging enabled for {service_name}")

    # Register shutdown handler
    atexit.register(shutdown_opentelemetry)

    _initialized = True
    return result


def shutdown_opentelemetry() -> None:
    """Shutdown OpenTelemetry providers and flush pending data."""
    global _tracer_provider, _logger_provider, _initialized

    if not _initialized:
        return

    if _tracer_provider:
        bt.logging.info("Shutting down OpenTelemetry trace provider...")
        try:
            _tracer_provider.shutdown()
        except Exception as e:
            bt.logging.error(f"Error shutting down trace provider: {e}")
        _tracer_provider = None

    if _logger_provider:
        bt.logging.info("Shutting down OpenTelemetry log provider...")
        try:
            _logger_provider.shutdown()
        except Exception as e:
            bt.logging.error(f"Error shutting down log provider: {e}")
        _logger_provider = None

    _initialized = False


def get_tracer(name: str = "aurelius.validator") -> trace.Tracer:
    """Get an OpenTelemetry tracer instance.

    Args:
        name: Tracer name (usually module name)

    Returns:
        Tracer instance (no-op if not initialized)
    """
    return trace.get_tracer(name)


def get_current_span() -> trace.Span:
    """Get the current active span.

    Returns:
        Current span or a no-op span if none active
    """
    return trace.get_current_span()


def is_initialized() -> bool:
    """Check if OpenTelemetry has been initialized.

    Returns:
        True if initialized, False otherwise
    """
    return _initialized
