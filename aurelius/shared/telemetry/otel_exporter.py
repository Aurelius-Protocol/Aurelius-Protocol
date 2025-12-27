"""OpenTelemetry custom exporters for Aurelius telemetry.

Exports traces and logs to the Aurelius collector API.
"""

from __future__ import annotations

# Protocol version for version metadata tracking
# Increment when making breaking changes to the telemetry protocol
PROTOCOL_VERSION = "1.0.0"

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Sequence

import bittensor as bt
import requests
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogExporter, LogExportResult
from opentelemetry.trace import SpanKind, StatusCode


class AureliusSpanExporter(SpanExporter):
    """Custom SpanExporter that sends spans to the Aurelius collector API.

    Features:
    - Batching with configurable batch size and flush interval
    - Retry logic with exponential backoff
    - Local file backup on API failure
    - Thread-safe queue-based submission
    """

    SPAN_KIND_MAP = {
        SpanKind.INTERNAL: "internal",
        SpanKind.SERVER: "server",
        SpanKind.CLIENT: "client",
        SpanKind.PRODUCER: "producer",
        SpanKind.CONSUMER: "consumer",
    }

    STATUS_MAP = {
        StatusCode.UNSET: "unset",
        StatusCode.OK: "ok",
        StatusCode.ERROR: "error",
    }

    def __init__(
        self,
        endpoint: str | None = None,
        validator_hotkey: str | None = None,
        validator_uid: int | None = None,
        netuid: int | None = None,
        network: str | None = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_retries: int = 3,
        timeout: int = 10,
        local_backup_path: str | None = None,
        wallet: "bt.wallet | None" = None,
        heartbeat_interval: float = 300.0,
        registry_endpoint: str | None = None,
    ):
        """Initialize the Aurelius span exporter.

        Args:
            endpoint: Telemetry API endpoint for traces
            validator_hotkey: Validator's SS58 hotkey
            validator_uid: Validator's UID on the subnet
            netuid: Subnet UID
            network: Network name (mainnet, testnet, local)
            batch_size: Number of spans to batch before sending
            flush_interval: Seconds between forced flushes
            max_retries: Maximum retry attempts for failed submissions
            timeout: Request timeout in seconds
            local_backup_path: Path for local backup files on failure
            wallet: Bittensor wallet for signing heartbeat requests
            heartbeat_interval: Seconds between heartbeats (default 5 minutes)
            registry_endpoint: Explicit registry endpoint (fixes fragile endpoint construction)
        """
        self.endpoint = endpoint or "https://collector.aureliusaligned.ai/api/telemetry/traces"
        self.validator_hotkey = validator_hotkey
        self.validator_uid = validator_uid
        self.netuid = netuid
        self.network = network
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.local_backup_path = local_backup_path
        self.wallet = wallet
        self.heartbeat_interval = heartbeat_interval
        self.registry_endpoint = registry_endpoint

        # Internal state
        self._queue: Queue[ReadableSpan] = Queue()
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._last_heartbeat: float = 0
        self._heartbeat_in_progress = False  # A9: Prevent concurrent heartbeat sends

        # Statistics
        self.spans_exported = 0
        self.spans_failed = 0
        self.batches_sent = 0

        # Start background worker
        self._start_worker()

    def _start_worker(self) -> None:
        """Start the background worker thread."""
        self._running = True
        # A19: Use daemon=False to ensure proper cleanup and data flushing on shutdown
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=False,
            name="AureliusSpanExporter"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker that batches and sends spans."""
        batch: list[ReadableSpan] = []
        last_flush = time.time()

        while self._running:
            try:
                # Try to get a span with timeout
                try:
                    span = self._queue.get(timeout=0.5)
                    batch.append(span)
                    self._queue.task_done()
                except Empty:
                    pass

                # Flush if batch is full or interval elapsed
                now = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and now - last_flush >= self.flush_interval)
                )

                if should_flush:
                    self._send_batch(batch)
                    batch = []
                    last_flush = now

                # A9: Send heartbeat if interval elapsed (fully thread-safe)
                # Check and set in-progress flag atomically to prevent race conditions
                with self._lock:
                    should_heartbeat = (
                        self.wallet and
                        not self._heartbeat_in_progress and
                        now - self._last_heartbeat >= self.heartbeat_interval
                    )
                    if should_heartbeat:
                        self._heartbeat_in_progress = True
                        self._last_heartbeat = now

                if should_heartbeat:
                    try:
                        self._send_heartbeat()
                    finally:
                        with self._lock:
                            self._heartbeat_in_progress = False

            except Exception as e:
                bt.logging.error(f"Span exporter worker error: {e}")

        # Final flush
        if batch:
            self._send_batch(batch)

    def _span_to_dict(self, span: ReadableSpan) -> dict[str, Any]:
        """Convert an OpenTelemetry span to our API format."""
        context = span.get_span_context()

        # Convert attributes to dict
        attributes = {}
        if span.attributes:
            for key, value in span.attributes.items():
                # Handle tuple values (OTel uses tuples for arrays)
                if isinstance(value, tuple):
                    value = list(value)
                attributes[key] = value

        # Convert events
        events = []
        if span.events:
            for event in span.events:
                event_attrs = {}
                if event.attributes:
                    for key, value in event.attributes.items():
                        if isinstance(value, tuple):
                            value = list(value)
                        event_attrs[key] = value
                events.append({
                    "timestamp_unix_nano": event.timestamp,
                    "name": event.name,
                    "attributes": event_attrs,
                })

        # Convert links
        links = []
        if span.links:
            for link in span.links:
                link_attrs = {}
                if link.attributes:
                    for key, value in link.attributes.items():
                        if isinstance(value, tuple):
                            value = list(value)
                        link_attrs[key] = value
                links.append({
                    "trace_id": format(link.context.trace_id, '032x'),
                    "span_id": format(link.context.span_id, '016x'),
                    "attributes": link_attrs,
                })

        # Resource attributes
        resource_attrs = {}
        if span.resource and span.resource.attributes:
            for key, value in span.resource.attributes.items():
                if isinstance(value, tuple):
                    value = list(value)
                resource_attrs[key] = value

        return {
            "trace_id": format(context.trace_id, '032x'),
            "span_id": format(context.span_id, '016x'),
            "parent_span_id": format(span.parent.span_id, '016x') if span.parent else None,
            "name": span.name,
            "kind": self.SPAN_KIND_MAP.get(span.kind, "internal"),
            "status": self.STATUS_MAP.get(span.status.status_code, "unset"),
            "status_message": span.status.description,
            "start_time_unix_nano": span.start_time,
            "end_time_unix_nano": span.end_time,
            "attributes": attributes,
            "events": events,
            "links": links,
            "resource_attributes": resource_attrs,
        }

    def _send_batch(self, spans: list[ReadableSpan]) -> bool:
        """Send a batch of spans to the API."""
        if not spans:
            return True

        payload = {
            "spans": [self._span_to_dict(span) for span in spans],
            "validator_hotkey": self.validator_hotkey,
            "validator_uid": self.validator_uid,
            "netuid": self.netuid,
            "network": self.network,
            "batch_id": str(uuid.uuid4()),
        }

        headers = {
            "Content-Type": "application/json",
            "X-Protocol-Version": PROTOCOL_VERSION,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if response.status_code in (200, 201):
                    with self._lock:
                        self.spans_exported += len(spans)
                        self.batches_sent += 1
                    bt.logging.debug(f"Exported {len(spans)} spans (total: {self.spans_exported})")
                    return True
                else:
                    bt.logging.warning(f"Span export failed: HTTP {response.status_code}")

            except requests.RequestException as e:
                bt.logging.warning(f"Span export attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed - save locally if configured
        with self._lock:
            self.spans_failed += len(spans)

        if self.local_backup_path:
            self._save_local_backup(payload)

        return False

    def _save_local_backup(self, payload: dict) -> None:
        """Save failed spans to local file."""
        try:
            backup_dir = Path(self.local_backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            filename = f"spans_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
            filepath = backup_dir / filename

            with open(filepath, 'w') as f:
                json.dump(payload, f)

            bt.logging.warning(f"Saved {len(payload['spans'])} spans to {filepath}")
        except Exception as e:
            bt.logging.error(f"Failed to save span backup: {e}")

    def _send_heartbeat(self) -> bool:
        """Send a heartbeat to the telemetry registry API."""
        import hashlib

        if not self.wallet or not self.validator_hotkey:
            return False

        try:
            from aurelius.shared.config import Config

            # Fix 3: Use explicit registry endpoint or fall back to Config
            if self.registry_endpoint:
                heartbeat_endpoint = f"{self.registry_endpoint}/heartbeat"
            else:
                heartbeat_endpoint = f"{Config.TELEMETRY_REGISTRY_ENDPOINT}/heartbeat"

            # A13: Include body hash in signature message to prevent body tampering
            body = {"hotkey": self.validator_hotkey}
            body_json = json.dumps(body, separators=(',', ':'), sort_keys=True)
            body_hash = hashlib.sha256(body_json.encode()).hexdigest()[:16]  # Use first 16 chars

            # Create signed message with body hash
            timestamp = int(time.time())
            message = f"aurelius-telemetry:{timestamp}:{self.validator_hotkey}:{body_hash}"
            signature = self.wallet.hotkey.sign(message.encode()).hex()

            response = requests.post(
                heartbeat_endpoint,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Validator-Hotkey": self.validator_hotkey,
                    "X-Validator-Signature": signature,
                    "X-Signature-Timestamp": str(timestamp),
                    "X-Body-Hash": body_hash,  # A13: Send body hash for server verification
                    "X-Protocol-Version": PROTOCOL_VERSION,
                },
                timeout=5,
            )

            if response.status_code in (200, 201):
                bt.logging.debug("Heartbeat sent successfully")
                return True
            else:
                # Fix 8: Elevate heartbeat failures to warning level
                bt.logging.warning(f"Heartbeat failed: HTTP {response.status_code} - {response.text[:200]}")
                return False

        except Exception as e:
            # Fix 8: Elevate heartbeat errors to warning level
            bt.logging.warning(f"Heartbeat error: {e}")
            return False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans (called by OpenTelemetry SDK)."""
        dropped = 0
        for span in spans:
            try:
                self._queue.put_nowait(span)
            except Exception:
                # A17: Log when spans are dropped due to queue being full
                dropped += 1

        if dropped > 0:
            bt.logging.warning(f"Dropped {dropped} spans - queue full (size: {self._queue.qsize()})")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter, flushing remaining spans."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        bt.logging.info(f"Span exporter shutdown. Exported: {self.spans_exported}, Failed: {self.spans_failed}")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending spans.

        A10: Uses timeout to prevent deadlock if flush can't complete.
        """
        import threading

        timeout_secs = timeout_millis / 1000.0

        # Use a thread with timeout instead of blocking join()
        def wait_for_queue():
            try:
                self._queue.join()
            except Exception:
                pass

        flush_thread = threading.Thread(target=wait_for_queue, daemon=True)
        flush_thread.start()
        flush_thread.join(timeout=timeout_secs)

        if flush_thread.is_alive():
            bt.logging.warning(f"Force flush timed out after {timeout_secs}s")
            return False
        return True


class AureliusLogExporter(LogExporter):
    """Custom LogExporter that sends logs to the Aurelius collector API.

    Similar architecture to AureliusSpanExporter with batching and retry.
    """

    SEVERITY_MAP = {
        1: "TRACE", 2: "TRACE", 3: "TRACE", 4: "TRACE",
        5: "DEBUG", 6: "DEBUG", 7: "DEBUG", 8: "DEBUG",
        9: "INFO", 10: "INFO", 11: "INFO", 12: "INFO",
        13: "WARN", 14: "WARN", 15: "WARN", 16: "WARN",
        17: "ERROR", 18: "ERROR", 19: "ERROR", 20: "ERROR",
        21: "FATAL", 22: "FATAL", 23: "FATAL", 24: "FATAL",
    }

    def __init__(
        self,
        endpoint: str | None = None,
        validator_hotkey: str | None = None,
        validator_uid: int | None = None,
        netuid: int | None = None,
        network: str | None = None,
        batch_size: int = 200,
        flush_interval: float = 5.0,
        max_retries: int = 3,
        timeout: int = 10,
        local_backup_path: str | None = None,
    ):
        """Initialize the Aurelius log exporter."""
        self.endpoint = endpoint or "https://collector.aureliusaligned.ai/api/telemetry/logs"
        self.validator_hotkey = validator_hotkey
        self.validator_uid = validator_uid
        self.netuid = netuid
        self.network = network
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.local_backup_path = local_backup_path

        # Internal state
        self._queue: Queue[ReadableLogRecord] = Queue()
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

        # Statistics
        self.logs_exported = 0
        self.logs_failed = 0

        # Start background worker
        self._start_worker()

    def _start_worker(self) -> None:
        """Start the background worker thread."""
        self._running = True
        # A19: Use daemon=False to ensure proper cleanup and data flushing on shutdown
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=False,
            name="AureliusLogExporter"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background worker that batches and sends logs."""
        batch: list[ReadableLogRecord] = []
        last_flush = time.time()

        while self._running:
            try:
                try:
                    log = self._queue.get(timeout=0.5)
                    batch.append(log)
                    self._queue.task_done()
                except Empty:
                    pass

                now = time.time()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and now - last_flush >= self.flush_interval)
                )

                if should_flush:
                    self._send_batch(batch)
                    batch = []
                    last_flush = now

            except Exception as e:
                bt.logging.error(f"Log exporter worker error: {e}")

        if batch:
            self._send_batch(batch)

    def _log_to_dict(self, log_record: ReadableLogRecord) -> dict[str, Any]:
        """Convert an OpenTelemetry log record to our API format."""
        # Convert attributes
        attributes = {}
        if log_record.attributes:
            for key, value in log_record.attributes.items():
                if isinstance(value, tuple):
                    value = list(value)
                attributes[key] = value

        # Resource attributes (instrumentation scope may not be directly available in newer SDK)
        resource_attrs = {}
        if hasattr(log_record, 'instrumentation_scope') and log_record.instrumentation_scope:
            resource_attrs["instrumentation.scope.name"] = log_record.instrumentation_scope.name
            if log_record.instrumentation_scope.version:
                resource_attrs["instrumentation.scope.version"] = log_record.instrumentation_scope.version

        # Get trace context if available
        trace_id = None
        span_id = None
        if log_record.trace_id:
            trace_id = format(log_record.trace_id, '032x')
        if log_record.span_id:
            span_id = format(log_record.span_id, '016x')

        # Get severity
        severity_number = 9  # Default to INFO
        if log_record.severity_number:
            severity_number = log_record.severity_number.value

        return {
            "timestamp_unix_nano": log_record.timestamp or int(time.time() * 1e9),
            "trace_id": trace_id,
            "span_id": span_id,
            "severity_number": severity_number,
            "severity_text": self.SEVERITY_MAP.get(severity_number, "INFO"),
            "body": str(log_record.body) if log_record.body else "",
            "attributes": attributes,
            "resource_attributes": resource_attrs,
        }

    def _send_batch(self, logs: list[ReadableLogRecord]) -> bool:
        """Send a batch of logs to the API."""
        if not logs:
            return True

        payload = {
            "logs": [self._log_to_dict(log) for log in logs],
            "validator_hotkey": self.validator_hotkey,
            "validator_uid": self.validator_uid,
            "netuid": self.netuid,
            "network": self.network,
            "batch_id": str(uuid.uuid4()),
        }

        headers = {
            "Content-Type": "application/json",
            "X-Protocol-Version": PROTOCOL_VERSION,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )

                if response.status_code in (200, 201):
                    with self._lock:
                        self.logs_exported += len(logs)
                    bt.logging.debug(f"Exported {len(logs)} logs (total: {self.logs_exported})")
                    return True
                else:
                    bt.logging.warning(f"Log export failed: HTTP {response.status_code}")

            except requests.RequestException as e:
                bt.logging.warning(f"Log export attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        with self._lock:
            self.logs_failed += len(logs)

        if self.local_backup_path:
            self._save_local_backup(payload)

        return False

    def _save_local_backup(self, payload: dict) -> None:
        """Save failed logs to local file."""
        try:
            backup_dir = Path(self.local_backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            filename = f"logs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
            filepath = backup_dir / filename

            with open(filepath, 'w') as f:
                json.dump(payload, f)

            bt.logging.warning(f"Saved {len(payload['logs'])} logs to {filepath}")
        except Exception as e:
            bt.logging.error(f"Failed to save log backup: {e}")

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogExportResult:
        """Export logs (called by OpenTelemetry SDK)."""
        dropped = 0
        for log in batch:
            try:
                self._queue.put_nowait(log)
            except Exception:
                # A17: Log when logs are dropped due to queue being full
                dropped += 1

        if dropped > 0:
            bt.logging.warning(f"Dropped {dropped} logs - queue full (size: {self._queue.qsize()})")
            return LogExportResult.FAILURE
        return LogExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        bt.logging.info(f"Log exporter shutdown. Exported: {self.logs_exported}, Failed: {self.logs_failed}")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending logs.

        A10: Uses timeout to prevent deadlock if flush can't complete.
        """
        import threading

        timeout_secs = timeout_millis / 1000.0

        # Use a thread with timeout instead of blocking join()
        def wait_for_queue():
            try:
                self._queue.join()
            except Exception:
                pass

        flush_thread = threading.Thread(target=wait_for_queue, daemon=True)
        flush_thread.start()
        flush_thread.join(timeout=timeout_secs)

        if flush_thread.is_alive():
            bt.logging.warning(f"Force flush timed out after {timeout_secs}s")
            return False
        return True
