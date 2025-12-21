"""Main telemetry client for validator observability."""

from __future__ import annotations

import atexit
import json
import os
import tempfile
import threading
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from time import sleep
from typing import Any, Generator

import bittensor as bt
import requests

from aurelius.shared.telemetry.context import TelemetryContext
from aurelius.shared.telemetry.schemas import ErrorEvent, LogEvent, SpanEvent, TelemetryEvent
from aurelius.shared.telemetry.span import Span


class TelemetryClient:
    """Central telemetry client following DatasetLogger patterns.

    Provides non-blocking telemetry submission with:
    - Error tracking with stack traces and context
    - Structured logging with prompt/response content
    - Distributed tracing with spans
    - Async queue-based submission with retry logic
    - Local file backup for reliability
    """

    def __init__(
        self,
        api_endpoint: str | None = None,
        api_key: str | None = None,
        local_path: str | None = None,
        enable_local_backup: bool = True,
        validator_hotkey: str | None = None,
        validator_uid: int | None = None,
        validator_coldkey: str | None = None,
        netuid: int | None = None,
        network: str | None = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_retries: int = 3,
        queue_size: int = 10000,
        shutdown_timeout: int = 30,
        include_prompts: bool = True,
        include_responses: bool = True,
        max_content_length: int = 10000,
        sample_rate: float = 1.0,
        min_log_level: str = "INFO",
    ):
        """
        Initialize telemetry client.

        Args:
            api_endpoint: URL for telemetry API
            api_key: API key for authentication
            local_path: Path for local backup files
            enable_local_backup: Whether to save local backups
            validator_hotkey: Validator's hotkey
            validator_uid: Validator's UID
            validator_coldkey: Validator's coldkey
            netuid: Subnet UID
            network: Network name ("mainnet", "testnet", "local")
            batch_size: Events to batch before submission
            flush_interval: Seconds between flush attempts
            max_retries: Max retry attempts for failed submissions
            queue_size: Max events to queue
            shutdown_timeout: Seconds to wait during shutdown
            include_prompts: Whether to include prompts in telemetry
            include_responses: Whether to include responses in telemetry
            max_content_length: Max length for prompt/response content
            sample_rate: Sampling rate (0.0-1.0)
            min_log_level: Minimum log level to submit
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.local_path = local_path
        self.enable_local_backup = enable_local_backup
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.shutdown_timeout = shutdown_timeout
        self.include_prompts = include_prompts
        self.include_responses = include_responses
        self.max_content_length = max_content_length
        self.sample_rate = sample_rate
        self.min_log_level = min_log_level

        # Validator identity
        self.validator_hotkey = validator_hotkey
        self.validator_uid = validator_uid
        self.validator_coldkey = validator_coldkey
        self.netuid = netuid
        self.network = network

        # Set identity in thread-local context
        TelemetryContext.set_validator_identity(
            validator_hotkey, validator_uid, validator_coldkey
        )
        TelemetryContext.set_network_context(netuid, network, None)

        # Queue for async processing
        self.queue: Queue[TelemetryEvent | None] = Queue(maxsize=queue_size)
        self.worker_thread: threading.Thread | None = None
        self.running = False

        # Thread lock for atomic file writes
        self._file_write_lock = threading.Lock()

        # Submission statistics
        self.events_submitted = 0
        self.events_failed = 0
        self.batches_submitted = 0
        self.batches_failed = 0

        # Log level priority for filtering
        self._log_levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

        # Create local directory if needed
        if self.enable_local_backup and self.local_path:
            Path(self.local_path).mkdir(parents=True, exist_ok=True)
            bt.logging.info(f"Telemetry: Local backup enabled at {self.local_path}")

        # Start background worker if API configured
        if self.api_endpoint:
            bt.logging.info(f"Telemetry: API enabled at {self.api_endpoint}")
            self._start_worker()
            atexit.register(self.stop)
        else:
            bt.logging.info("Telemetry: API not configured, local-only mode")

    def _start_worker(self) -> None:
        """Start background worker thread for async submissions."""
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=False,
            name="TelemetryWorker"
        )
        self.worker_thread.start()
        bt.logging.info("Telemetry: Background worker started")

    def _worker(self) -> None:
        """Background worker that processes the queue in batches."""
        batch: list[TelemetryEvent] = []
        last_flush = datetime.now(timezone.utc)

        while self.running:
            try:
                # Get event from queue with timeout
                try:
                    event = self.queue.get(timeout=0.5)
                    if event is None:  # Poison pill
                        bt.logging.info("Telemetry: Received stop signal")
                        break
                    batch.append(event)
                    self.queue.task_done()
                except Empty:
                    pass

                # Flush if batch is full or interval elapsed
                now = datetime.now(timezone.utc)
                time_since_flush = (now - last_flush).total_seconds()

                if batch and (len(batch) >= self.batch_size or time_since_flush >= self.flush_interval):
                    self._submit_batch(batch)
                    batch = []
                    last_flush = now

            except Exception as e:
                bt.logging.error(f"Telemetry worker error: {e}")

        # Final flush of remaining events
        if batch:
            self._submit_batch(batch)

    def _submit_batch(self, batch: list[TelemetryEvent]) -> bool:
        """
        Submit a batch of events to the API.

        Args:
            batch: List of events to submit

        Returns:
            True if successful, False otherwise
        """
        if not batch:
            return True

        # Save locally first (fast, blocking)
        if self.enable_local_backup:
            self._save_local(batch)

        if not self.api_endpoint:
            return False

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "events": [event.to_dict() for event in batch],
            "batch_id": str(uuid.uuid4()),
            "validator_version": self._get_version(),
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=10,
                )

                if response.status_code in [200, 201]:
                    self.events_submitted += len(batch)
                    self.batches_submitted += 1
                    bt.logging.debug(
                        f"Telemetry: Submitted {len(batch)} events "
                        f"(total: {self.events_submitted})"
                    )
                    return True
                else:
                    bt.logging.warning(
                        f"Telemetry API returned {response.status_code}: {response.text}"
                    )

            except requests.RequestException as e:
                bt.logging.warning(
                    f"Telemetry submission failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    sleep(2 ** attempt)  # Exponential backoff

        self.events_failed += len(batch)
        self.batches_failed += 1
        bt.logging.error(
            f"Telemetry: Failed to submit {len(batch)} events after all retries "
            f"(total failures: {self.events_failed})"
        )
        return False

    def _save_local(self, events: list[TelemetryEvent]) -> bool:
        """
        Save events to local JSONL file.

        Args:
            events: Events to save

        Returns:
            True if successful, False otherwise
        """
        if not self.enable_local_backup or not self.local_path:
            return False

        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filepath = Path(self.local_path) / f"telemetry_{date_str}.jsonl"

            # Serialize events outside lock
            lines = "\n".join(json.dumps(event.to_dict()) for event in events) + "\n"

            with self._file_write_lock:
                # Atomic write via temp file
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=self.local_path,
                    suffix=".tmp",
                    delete=False,
                ) as tmp_file:
                    tmp_file.write(lines)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                    tmp_path = tmp_file.name

                # Append to main file
                with open(filepath, "a") as main_file:
                    with open(tmp_path, "r") as tmp_read:
                        main_file.write(tmp_read.read())
                    main_file.flush()
                    os.fsync(main_file.fileno())

                os.unlink(tmp_path)

            return True

        except Exception as e:
            bt.logging.error(f"Telemetry: Failed to save local backup: {e}")
            return False

    def _submit_event(self, event: TelemetryEvent) -> None:
        """
        Queue an event for submission.

        Args:
            event: Event to submit
        """
        # Apply sampling
        if self.sample_rate < 1.0:
            import random
            if random.random() > self.sample_rate:
                return

        try:
            self.queue.put_nowait(event)
        except Exception:
            # Queue full - drop event
            bt.logging.warning("Telemetry: Queue full, dropping event")

    def _get_version(self) -> str:
        """Get validator version string."""
        try:
            from aurelius import __version__
            return __version__
        except Exception:
            return "unknown"

    def _truncate_content(self, content: str | None) -> str | None:
        """Truncate content to max length."""
        if content is None:
            return None
        if len(content) > self.max_content_length:
            return content[:self.max_content_length] + "...[truncated]"
        return content

    # ========== Error Tracking ==========

    def capture_exception(
        self,
        exception: Exception,
        operation: str,
        component: str,
        severity: str = "error",
        context: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        recovered: bool = False,
        recovery_action: str | None = None,
    ) -> str:
        """
        Capture and log an exception.

        Args:
            exception: The exception to capture
            operation: What operation was being performed
            component: Which component raised the error
            severity: "critical", "error", or "warning"
            context: Additional context dict
            tags: Key-value tags
            recovered: Whether we recovered from this error
            recovery_action: What recovery was attempted

        Returns:
            Event ID
        """
        identity = TelemetryContext.get_validator_identity()
        network = TelemetryContext.get_network_context()
        correlation_id = TelemetryContext.get_correlation_id()

        event = ErrorEvent(
            event_name=f"exception:{type(exception).__name__}",
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            operation=operation,
            component=component,
            severity=severity,
            request_context=context,
            recovered=recovered,
            recovery_action=recovery_action,
            tags=tags,
            correlation_id=correlation_id,
            **identity,
            **network,
        )

        self._submit_event(event)
        return event.event_id

    # ========== Structured Logging ==========

    def log(
        self,
        level: str,
        message: str,
        logger_name: str = "validator",
        structured_data: dict[str, Any] | None = None,
        prompt: str | None = None,
        response: str | None = None,
        duration_ms: float | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Log a structured event.

        Args:
            level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
            message: Log message
            logger_name: Source logger name
            structured_data: Arbitrary structured context
            prompt: Full prompt text (if include_prompts enabled)
            response: Full response text (if include_responses enabled)
            duration_ms: Operation duration
            tags: Key-value tags

        Returns:
            Event ID
        """
        # Filter by log level
        if self._log_levels.get(level, 0) < self._log_levels.get(self.min_log_level, 0):
            return ""

        identity = TelemetryContext.get_validator_identity()
        network = TelemetryContext.get_network_context()
        correlation_id = TelemetryContext.get_correlation_id()

        # Apply content settings
        final_prompt = self._truncate_content(prompt) if self.include_prompts else None
        final_response = self._truncate_content(response) if self.include_responses else None

        event = LogEvent(
            event_name=f"log:{logger_name}",
            level=level,
            message=message,
            logger_name=logger_name,
            structured_data=structured_data,
            prompt=final_prompt,
            response=final_response,
            duration_ms=duration_ms,
            tags=tags,
            correlation_id=correlation_id,
            **identity,
            **network,
        )

        self._submit_event(event)
        return event.event_id

    def debug(self, message: str, **kwargs) -> str:
        """Log at DEBUG level."""
        return self.log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs) -> str:
        """Log at INFO level."""
        return self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> str:
        """Log at WARNING level."""
        return self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> str:
        """Log at ERROR level."""
        return self.log("ERROR", message, **kwargs)

    # ========== Distributed Tracing ==========

    def start_span(
        self,
        operation_name: str,
        span_kind: str = "internal",
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Start a new trace span.

        Args:
            operation_name: Name of the operation
            span_kind: Type of span ("server", "client", "internal")
            parent_span: Optional parent span for nesting
            attributes: Initial span attributes

        Returns:
            Span object (use as context manager or call end() manually)
        """
        # Inherit trace ID and parent from context or explicit parent
        current_span = parent_span or TelemetryContext.get_current_span()
        trace_id = current_span.trace_id if current_span else None
        parent_span_id = current_span.span_id if current_span else None

        span = Span(
            client=self,
            operation_name=operation_name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            span_kind=span_kind,
            attributes=attributes,
        )

        return span

    @contextmanager
    def trace(
        self,
        operation_name: str,
        span_kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """
        Context manager for tracing an operation.

        Args:
            operation_name: Name of the operation
            span_kind: Type of span
            attributes: Initial span attributes

        Yields:
            Span object
        """
        span = self.start_span(operation_name, span_kind, attributes=attributes)
        with span:
            yield span

    # ========== Correlation Management ==========

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current thread."""
        TelemetryContext.set_correlation_id(correlation_id)

    def get_correlation_id(self) -> str | None:
        """Get current correlation ID."""
        return TelemetryContext.get_correlation_id()

    @contextmanager
    def correlation_context(self, correlation_id: str | None = None):
        """
        Context manager for correlation ID scope.

        Args:
            correlation_id: ID to use (generated if not provided)
        """
        cid = correlation_id or str(uuid.uuid4())
        previous = TelemetryContext.get_correlation_id()
        TelemetryContext.set_correlation_id(cid)
        try:
            yield cid
        finally:
            if previous:
                TelemetryContext.set_correlation_id(previous)
            else:
                TelemetryContext.clear_correlation_id()

    # ========== Lifecycle ==========

    def flush(self, timeout: float = 30.0) -> bool:
        """
        Flush pending events.

        Args:
            timeout: Max seconds to wait

        Returns:
            True if queue emptied, False if timeout
        """
        if not self.api_endpoint:
            return True

        try:
            self.queue.join()
            return True
        except Exception:
            return False

    def stop(self) -> None:
        """Stop the telemetry client, flush queue, persist unsent."""
        if not self.worker_thread or not self.running:
            return

        queue_size = self.queue.qsize()
        if queue_size > 0:
            bt.logging.info(f"Telemetry: Stopping... ({queue_size} events pending)")
        else:
            bt.logging.info("Telemetry: Stopping...")

        self.running = False

        # Wait for queue to drain
        bt.logging.info(f"Telemetry: Waiting up to {self.shutdown_timeout}s for queue to drain...")

        # Send poison pill
        try:
            self.queue.put(None, timeout=1.0)
        except Exception:
            pass

        # Wait for thread
        self.worker_thread.join(timeout=min(self.shutdown_timeout, 10))

        if self.worker_thread.is_alive():
            bt.logging.warning(
                f"Telemetry: Worker did not stop within {self.shutdown_timeout}s"
            )
            self._persist_queue()
        else:
            bt.logging.info("Telemetry: Stopped cleanly")

    def _persist_queue(self) -> None:
        """Persist remaining queue items to disk."""
        if not self.enable_local_backup or not self.local_path:
            bt.logging.warning("Telemetry: Cannot persist queue - local backup not enabled")
            return

        try:
            remaining: list[TelemetryEvent] = []
            while not self.queue.empty():
                try:
                    event = self.queue.get_nowait()
                    if event is not None:
                        remaining.append(event)
                except Exception:
                    break

            if remaining:
                persist_file = Path(self.local_path) / f"unsent_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
                with open(persist_file, "w") as f:
                    json.dump([event.to_dict() for event in remaining], f, indent=2)
                bt.logging.warning(
                    f"Telemetry: Persisted {len(remaining)} unsent events to {persist_file}"
                )
        except Exception as e:
            bt.logging.error(f"Telemetry: Failed to persist queue: {e}")

    def get_stats(self) -> dict:
        """
        Get telemetry statistics.

        Returns:
            Dictionary with current stats
        """
        return {
            "enabled": self.api_endpoint is not None,
            "local_backup_enabled": self.enable_local_backup,
            "local_path": self.local_path,
            "queue_size": self.queue.qsize(),
            "events_submitted": self.events_submitted,
            "events_failed": self.events_failed,
            "batches_submitted": self.batches_submitted,
            "batches_failed": self.batches_failed,
            "sample_rate": self.sample_rate,
        }


# Global telemetry instance (optional, for convenience)
_global_telemetry: TelemetryClient | None = None


def get_telemetry() -> TelemetryClient | None:
    """Get the global telemetry client instance."""
    return _global_telemetry


def set_telemetry(client: TelemetryClient) -> None:
    """Set the global telemetry client instance."""
    global _global_telemetry
    _global_telemetry = client
