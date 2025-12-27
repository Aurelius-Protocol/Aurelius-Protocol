"""OpenTelemetry setup and configuration for Aurelius validators.

Provides a simple interface to configure OpenTelemetry with Aurelius custom exporters.
"""

from __future__ import annotations

import atexit
import time
from typing import Any

import bittensor as bt
import requests
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry._logs import set_logger_provider

from aurelius.shared.telemetry.otel_exporter import (
    AureliusSpanExporter,
    AureliusLogExporter,
    PROTOCOL_VERSION,
)


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
    wallet: "bt.wallet | None" = None,
    heartbeat_interval_s: int = 300,
    register_on_startup: bool = True,
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
        wallet: Bittensor wallet for signing (enables heartbeat and registration)
        heartbeat_interval_s: Heartbeat interval in seconds (default 5 minutes)
        register_on_startup: Whether to register with telemetry API on startup

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
            wallet=wallet,
            heartbeat_interval=float(heartbeat_interval_s),
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
            wallet=wallet,
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

    # Register with telemetry API for health monitoring
    # Fix 4: Only register if we have all required info including validator_uid
    if register_on_startup and wallet and netuid is not None and network and validator_uid is not None:
        try:
            success = register_with_telemetry_api(
                wallet=wallet,
                validator_uid=validator_uid,
                netuid=netuid,
                network=network,
                heartbeat_interval_s=heartbeat_interval_s,
                telemetry_version=service_version or "1.0.0",
            )
            if not success:
                bt.logging.warning("Telemetry registration failed after all retries")
        except Exception as e:
            bt.logging.warning(f"Failed to register with telemetry API: {e}")
    elif register_on_startup and wallet and validator_uid is None:
        bt.logging.warning("Skipping telemetry registration: validator_uid is required")

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


def register_with_telemetry_api(
    wallet: bt.wallet,
    validator_uid: int | None,
    netuid: int,
    network: str,
    registry_endpoint: str | None = None,
    heartbeat_interval_s: int = 300,
    telemetry_version: str = "1.0.0",
    timeout: int = 10,
    max_retries: int = 3,
) -> bool:
    """Register validator with the central telemetry API for health monitoring.

    This function signs a message with the validator's hotkey and sends a registration
    request to the telemetry API. The API will track this validator's health and
    detect if it goes offline or experiences errors.

    Args:
        wallet: Bittensor wallet with hotkey for signing
        validator_uid: Validator's UID on the subnet (can be None if unknown)
        netuid: Subnet UID
        network: Network name (finney, test, local)
        registry_endpoint: Registry API endpoint (defaults to Config setting)
        heartbeat_interval_s: Expected heartbeat interval in seconds
        telemetry_version: Telemetry protocol version
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts (Fix 7)

    Returns:
        True if registration succeeded, False otherwise
    """
    from aurelius.shared.config import Config

    endpoint = registry_endpoint or f"{Config.TELEMETRY_REGISTRY_ENDPOINT}/register"
    hotkey = wallet.hotkey.ss58_address
    coldkey = wallet.coldkeypub.ss58_address if wallet.coldkeypub else None

    # Prepare registration payload
    payload = {
        "hotkey": hotkey,
        "uid": validator_uid,
        "coldkey": coldkey,
        "netuid": netuid,
        "network": network,
        "telemetry_version": telemetry_version,
        "heartbeat_interval_s": heartbeat_interval_s,
    }

    # Fix 7: Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            # Create signed message for authentication (fresh timestamp each attempt)
            timestamp = int(time.time())
            message = f"aurelius-telemetry:{timestamp}:{hotkey}"
            signature = wallet.hotkey.sign(message.encode()).hex()

            # Send registration with signature headers
            response = requests.post(
                endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Validator-Hotkey": hotkey,
                    "X-Validator-Signature": signature,
                    "X-Signature-Timestamp": str(timestamp),
                    "X-Protocol-Version": PROTOCOL_VERSION,
                },
                timeout=timeout,
            )

            if response.status_code in (200, 201):
                bt.logging.info(f"Registered with telemetry API: {hotkey[:16]}... (uid={validator_uid})")
                return True
            else:
                bt.logging.warning(
                    f"Telemetry registration attempt {attempt + 1}/{max_retries} failed: "
                    f"HTTP {response.status_code} - {response.text[:200]}"
                )

        except requests.RequestException as e:
            bt.logging.warning(f"Telemetry registration attempt {attempt + 1}/{max_retries} failed: {e}")
        except Exception as e:
            bt.logging.error(f"Telemetry registration error: {e}")
            return False  # Non-retryable error

        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            bt.logging.debug(f"Retrying registration in {wait_time}s...")
            time.sleep(wait_time)

    bt.logging.error(f"Telemetry registration failed after {max_retries} attempts")
    return False


def send_heartbeat(
    wallet: bt.wallet,
    registry_endpoint: str | None = None,
    timeout: int = 5,
) -> bool:
    """Send a heartbeat to the telemetry API to indicate validator is alive.

    Args:
        wallet: Bittensor wallet with hotkey for signing
        registry_endpoint: Registry API endpoint (defaults to Config setting)
        timeout: Request timeout in seconds

    Returns:
        True if heartbeat succeeded, False otherwise
    """
    from aurelius.shared.config import Config

    endpoint = registry_endpoint or f"{Config.TELEMETRY_REGISTRY_ENDPOINT}/heartbeat"
    hotkey = wallet.hotkey.ss58_address

    try:
        # Create signed message
        timestamp = int(time.time())
        message = f"aurelius-telemetry:{timestamp}:{hotkey}"
        signature = wallet.hotkey.sign(message.encode()).hex()

        # Send heartbeat
        response = requests.post(
            endpoint,
            json={"hotkey": hotkey},
            headers={
                "Content-Type": "application/json",
                "X-Validator-Hotkey": hotkey,
                "X-Validator-Signature": signature,
                "X-Signature-Timestamp": str(timestamp),
                "X-Protocol-Version": PROTOCOL_VERSION,
            },
            timeout=timeout,
        )

        return response.status_code in (200, 201)

    except requests.RequestException:
        return False
    except Exception as e:
        bt.logging.debug(f"Heartbeat error: {e}")
        return False
