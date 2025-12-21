"""Trace span implementation for distributed tracing."""

from __future__ import annotations

import traceback
import uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any

from aurelius.shared.telemetry.context import TelemetryContext
from aurelius.shared.telemetry.schemas import SpanEvent

if TYPE_CHECKING:
    from aurelius.shared.telemetry.client import TelemetryClient


class Span:
    """Represents a trace span with automatic timing."""

    def __init__(
        self,
        client: TelemetryClient,
        operation_name: str,
        span_id: str | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        span_kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ):
        """
        Initialize a span.

        Args:
            client: TelemetryClient instance for submitting span
            operation_name: Name of the operation being traced
            span_id: Unique span ID (generated if not provided)
            trace_id: Root trace ID (generated if not provided)
            parent_span_id: Parent span ID for nesting
            span_kind: Type of span ("server", "client", "internal")
            attributes: Initial span attributes
        """
        self.client = client
        self.operation_name = operation_name
        self.span_id = span_id or str(uuid.uuid4())
        self.trace_id = trace_id or str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.span_kind = span_kind
        self.attributes = attributes or {}

        self.status = "unset"
        self.status_message: str | None = None
        self.span_events: list[dict[str, Any]] = []
        self.links: list[dict[str, Any]] = []

        self._start_time: float | None = None
        self._start_timestamp: str | None = None
        self._end_time: float | None = None
        self._end_timestamp: str | None = None
        self._ended = False
        self._previous_span: Span | None = None

    def set_attribute(self, key: str, value: Any) -> Span:
        """
        Set a span attribute.

        Args:
            key: Attribute key
            value: Attribute value

        Returns:
            Self for chaining
        """
        self.attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> Span:
        """
        Set multiple span attributes.

        Args:
            attributes: Dictionary of attributes to set

        Returns:
            Self for chaining
        """
        self.attributes.update(attributes)
        return self

    def set_status(self, status: str, message: str | None = None) -> Span:
        """
        Set span status.

        Args:
            status: Status ("ok", "error", "unset")
            message: Optional status message

        Returns:
            Self for chaining
        """
        self.status = status
        self.status_message = message
        return self

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Add a point-in-time event to the span.

        Args:
            name: Event name
            attributes: Event attributes

        Returns:
            Self for chaining
        """
        self.span_events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        })
        return self

    def record_exception(
        self,
        exception: Exception,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Record an exception in the span.

        Args:
            exception: The exception to record
            attributes: Additional attributes

        Returns:
            Self for chaining
        """
        exc_attributes = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            "exception.stacktrace": traceback.format_exc(),
            **(attributes or {}),
        }
        self.add_event("exception", exc_attributes)
        return self

    def add_link(
        self,
        trace_id: str,
        span_id: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Add a link to another span.

        Args:
            trace_id: Linked span's trace ID
            span_id: Linked span's span ID
            attributes: Link attributes

        Returns:
            Self for chaining
        """
        self.links.append({
            "trace_id": trace_id,
            "span_id": span_id,
            "attributes": attributes or {},
        })
        return self

    def start(self) -> Span:
        """
        Start the span timer.

        Returns:
            Self for chaining
        """
        self._start_time = perf_counter()
        self._start_timestamp = datetime.now(timezone.utc).isoformat()

        # Save previous span and set self as current
        self._previous_span = TelemetryContext.get_current_span()
        TelemetryContext.set_current_span(self)

        return self

    def end(self, status: str | None = None) -> None:
        """
        End the span and submit to telemetry.

        Args:
            status: Optional final status to set
        """
        if self._ended:
            return

        self._ended = True
        self._end_time = perf_counter()
        self._end_timestamp = datetime.now(timezone.utc).isoformat()

        if status:
            self.status = status

        # Restore previous span as current
        TelemetryContext.set_current_span(self._previous_span)

        # Calculate duration
        duration_ms = 0.0
        if self._start_time is not None and self._end_time is not None:
            duration_ms = (self._end_time - self._start_time) * 1000

        # Get context
        identity = TelemetryContext.get_validator_identity()
        network = TelemetryContext.get_network_context()
        correlation_id = TelemetryContext.get_correlation_id()

        # Create span event
        span_event = SpanEvent(
            event_name=f"span:{self.operation_name}",
            span_id=self.span_id,
            trace_id=self.trace_id,
            parent_span_id=self.parent_span_id,
            operation_name=self.operation_name,
            span_kind=self.span_kind,
            status=self.status,
            status_message=self.status_message,
            start_time=self._start_timestamp or "",
            end_time=self._end_timestamp or "",
            duration_ms=duration_ms,
            attributes=self.attributes,
            span_events=self.span_events if self.span_events else None,
            links=self.links if self.links else None,
            correlation_id=correlation_id,
            **identity,
            **network,
        )

        # Submit to client
        self.client._submit_event(span_event)

    def __enter__(self) -> Span:
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - records exception if any."""
        if exc_val is not None:
            self.record_exception(exc_val)
            self.set_status("error", str(exc_val))
        elif self.status == "unset":
            self.set_status("ok")

        self.end()
        return False  # Don't suppress exceptions
