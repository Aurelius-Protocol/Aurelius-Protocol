"""Miner error classification and handling."""

import socket
import ssl
from dataclasses import dataclass, field
from enum import Enum, auto


class ErrorCategory(Enum):
    """Classification of miner errors by type."""

    # Retryable errors (transient)
    NETWORK_TIMEOUT = auto()
    CONNECTION_REFUSED = auto()
    DNS_RESOLUTION = auto()
    VALIDATOR_BUSY = auto()

    # Non-retryable errors (configuration/permanent)
    SSL_TLS_ERROR = auto()
    RATE_LIMITED = auto()
    INVALID_VALIDATOR_UID = auto()
    INVALID_WALLET = auto()
    VALIDATOR_MISCONFIGURED = auto()
    API_ERROR = auto()
    AUTHENTICATION_FAILED = auto()
    PROMPT_REJECTED = auto()

    # Unknown errors
    UNKNOWN = auto()


class ErrorSeverity(Enum):
    """Severity level for errors."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MinerError:
    """Structured miner error with diagnostics."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    cause: str
    suggestions: list[str] = field(default_factory=list)
    is_retryable: bool = False
    original_exception: Exception | None = None

    def format_message(self, context: dict) -> str:
        """Format message with context variables."""
        try:
            return self.message.format(**context)
        except KeyError:
            return self.message

    def format_cause(self, context: dict) -> str:
        """Format cause with context variables."""
        try:
            return self.cause.format(**context)
        except KeyError:
            return self.cause

    def format_suggestions(self, context: dict) -> list[str]:
        """Format suggestions with context variables."""
        formatted = []
        for suggestion in self.suggestions:
            try:
                formatted.append(suggestion.format(**context))
            except KeyError:
                formatted.append(suggestion)
        return formatted


# Error templates with detailed diagnostics
ERROR_TEMPLATES: dict[ErrorCategory, dict] = {
    ErrorCategory.CONNECTION_REFUSED: {
        "severity": ErrorSeverity.ERROR,
        "message": "Validator is unreachable",
        "cause": (
            "The validator at {host}:{port} refused the connection. "
            "This typically means the validator is not running or the port is blocked."
        ),
        "suggestions": [
            "Verify the validator is running: check validator logs or process list",
            "Confirm the correct port: BT_PORT_VALIDATOR should be {port}",
            "Check firewall rules: ensure port {port} is open",
            "If using LOCAL_MODE: verify VALIDATOR_HOST={host} is correct",
            "Test connectivity: nc -vz {host} {port}",
        ],
        "is_retryable": True,
    },
    ErrorCategory.NETWORK_TIMEOUT: {
        "severity": ErrorSeverity.WARNING,
        "message": "Request timed out after {timeout}s",
        "cause": (
            "The validator did not respond within the timeout period. "
            "This could indicate network congestion, high validator load, "
            "or the validator processing a slow request."
        ),
        "suggestions": [
            "Increase timeout: --timeout 60 (or set MINER_TIMEOUT=60)",
            "Check network latency: ping {host}",
            "Verify validator is responsive: check validator logs",
            "Try again - this may be a transient issue",
        ],
        "is_retryable": True,
    },
    ErrorCategory.DNS_RESOLUTION: {
        "severity": ErrorSeverity.ERROR,
        "message": "Cannot resolve hostname: {host}",
        "cause": (
            "DNS lookup failed for the validator host. "
            "This could be a network issue or an invalid hostname."
        ),
        "suggestions": [
            "Check your internet connection",
            "Verify the hostname is correct",
            "Try using an IP address instead of hostname",
            "Check DNS settings: nslookup {host}",
        ],
        "is_retryable": True,
    },
    ErrorCategory.SSL_TLS_ERROR: {
        "severity": ErrorSeverity.ERROR,
        "message": "SSL/TLS connection failed",
        "cause": (
            "Secure connection could not be established. "
            "This may indicate certificate issues or HTTPS misconfiguration."
        ),
        "suggestions": [
            "Check if the validator uses HTTPS",
            "Verify SSL certificates are valid",
            "Check system time (certificate validation requires correct time)",
        ],
        "is_retryable": False,
    },
    ErrorCategory.INVALID_VALIDATOR_UID: {
        "severity": ErrorSeverity.ERROR,
        "message": "Validator UID {uid} does not exist",
        "cause": (
            "The specified validator UID is not registered in the metagraph. "
            "This could mean the UID is wrong, or the validator has deregistered."
        ),
        "suggestions": [
            "List available validators: btcli subnet list --netuid {netuid}",
            "Use a different UID: --validator-uid <valid_uid>",
            "Verify you're on the correct network: BT_NETUID={netuid}",
        ],
        "is_retryable": False,
    },
    ErrorCategory.RATE_LIMITED: {
        "severity": ErrorSeverity.WARNING,
        "message": "Rate limit exceeded",
        "cause": (
            "The validator has rate-limited your requests. "
            "You have exceeded the allowed number of requests per time window."
        ),
        "suggestions": [
            "Wait for the rate limit window to reset",
            "Use a different validator: --validator-uid <other_uid>",
            "Check rejection reason for details on limit and window",
        ],
        "is_retryable": False,
    },
    ErrorCategory.INVALID_WALLET: {
        "severity": ErrorSeverity.CRITICAL,
        "message": "Invalid wallet credentials",
        "cause": "The wallet could not be loaded or is not properly configured.",
        "suggestions": [
            "Verify wallet exists: ls ~/.bittensor/wallets/",
            "Check wallet name: MINER_WALLET_NAME={wallet}",
            "Check hotkey name: MINER_HOTKEY={hotkey}",
            "Create wallet if needed: btcli wallet create --wallet.name {wallet}",
        ],
        "is_retryable": False,
    },
    ErrorCategory.API_ERROR: {
        "severity": ErrorSeverity.ERROR,
        "message": "Validator API error",
        "cause": (
            "The validator encountered an internal error processing your request. "
            "This could be a temporary issue with the chat or moderation API."
        ),
        "suggestions": [
            "Check validator logs for error details",
            "The prompt may have triggered an API issue",
            "Try a different prompt to verify connectivity",
            "Try again - this may be a transient issue",
        ],
        "is_retryable": True,
    },
    ErrorCategory.PROMPT_REJECTED: {
        "severity": ErrorSeverity.WARNING,
        "message": "Prompt was rejected",
        "cause": "The validator rejected the prompt: {reason}",
        "suggestions": [
            "Check the rejection reason above",
            "Ensure prompt meets length requirements (max {max_length} chars)",
            "If rate limited, wait before retrying",
        ],
        "is_retryable": False,
    },
    ErrorCategory.VALIDATOR_BUSY: {
        "severity": ErrorSeverity.WARNING,
        "message": "Validator is busy",
        "cause": (
            "The validator is currently processing too many requests. "
            "Try again in a moment."
        ),
        "suggestions": [
            "Wait a few seconds and retry",
            "Try a different validator: --validator-uid <other_uid>",
        ],
        "is_retryable": True,
    },
    ErrorCategory.AUTHENTICATION_FAILED: {
        "severity": ErrorSeverity.ERROR,
        "message": "Authentication failed",
        "cause": (
            "The validator rejected your request due to authentication issues. "
            "Your hotkey may not be properly registered."
        ),
        "suggestions": [
            "Verify your wallet is registered: btcli wallet overview",
            "Check you're using the correct network: BT_NETWORK={network}",
            "Ensure hotkey matches registration",
        ],
        "is_retryable": False,
    },
    ErrorCategory.UNKNOWN: {
        "severity": ErrorSeverity.ERROR,
        "message": "Unexpected error occurred",
        "cause": "An unexpected error occurred: {error}",
        "suggestions": [
            "Check the technical details below",
            "This may be a bug - please report if reproducible",
            "Try again in case it's a transient issue",
        ],
        "is_retryable": False,
    },
}


def create_error(
    category: ErrorCategory,
    context: dict | None = None,
    original_exception: Exception | None = None,
) -> MinerError:
    """Create a MinerError from a template with context."""
    context = context or {}
    template = ERROR_TEMPLATES.get(category, ERROR_TEMPLATES[ErrorCategory.UNKNOWN])

    return MinerError(
        category=category,
        severity=template["severity"],
        message=template["message"],
        cause=template["cause"],
        suggestions=template["suggestions"].copy(),
        is_retryable=template["is_retryable"],
        original_exception=original_exception,
    )


def classify_error(exception: Exception, context: dict | None = None) -> MinerError:
    """
    Classify an exception into a structured MinerError.

    Args:
        exception: The exception to classify
        context: Additional context for error messages (host, port, etc.)

    Returns:
        MinerError with appropriate category and diagnostics
    """
    context = context or {}
    exc_str = str(exception).lower()

    # Connection refused
    if isinstance(exception, ConnectionRefusedError) or "connection refused" in exc_str:
        return create_error(ErrorCategory.CONNECTION_REFUSED, context, exception)

    # Connection reset
    if isinstance(exception, ConnectionResetError) or "connection reset" in exc_str:
        return create_error(ErrorCategory.CONNECTION_REFUSED, context, exception)

    # Timeout errors
    if isinstance(exception, TimeoutError) or isinstance(exception, socket.timeout):
        context.setdefault("timeout", 30)
        return create_error(ErrorCategory.NETWORK_TIMEOUT, context, exception)

    if "timeout" in exc_str or "timed out" in exc_str:
        context.setdefault("timeout", 30)
        return create_error(ErrorCategory.NETWORK_TIMEOUT, context, exception)

    # DNS resolution errors
    if isinstance(exception, socket.gaierror):
        return create_error(ErrorCategory.DNS_RESOLUTION, context, exception)

    if "name resolution" in exc_str or "getaddrinfo" in exc_str:
        return create_error(ErrorCategory.DNS_RESOLUTION, context, exception)

    # SSL/TLS errors
    if isinstance(exception, ssl.SSLError):
        return create_error(ErrorCategory.SSL_TLS_ERROR, context, exception)

    if "ssl" in exc_str or "certificate" in exc_str or "tls" in exc_str:
        return create_error(ErrorCategory.SSL_TLS_ERROR, context, exception)

    # Index errors (often invalid UID)
    if isinstance(exception, IndexError):
        if "uid" in context:
            return create_error(ErrorCategory.INVALID_VALIDATOR_UID, context, exception)

    # Key errors (metagraph lookup)
    if isinstance(exception, KeyError):
        if "uid" in context or "metagraph" in exc_str:
            return create_error(ErrorCategory.INVALID_VALIDATOR_UID, context, exception)

    # OS errors
    if isinstance(exception, OSError):
        errno = getattr(exception, "errno", None)
        if errno == 111:  # Connection refused
            return create_error(ErrorCategory.CONNECTION_REFUSED, context, exception)
        if errno == 113:  # No route to host
            return create_error(ErrorCategory.CONNECTION_REFUSED, context, exception)

    # Default: unknown error
    context["error"] = str(exception)
    return create_error(ErrorCategory.UNKNOWN, context, exception)


class DiagnosticsFormatter:
    """Format detailed diagnostics for user display."""

    # ANSI color codes
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }

    SEVERITY_COLORS = {
        ErrorSeverity.INFO: "blue",
        ErrorSeverity.WARNING: "yellow",
        ErrorSeverity.ERROR: "red",
        ErrorSeverity.CRITICAL: "red",
    }

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors

    def colorize(self, text: str, color: str) -> str:
        """Apply color if enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def format_error(self, error: MinerError, context: dict | None = None) -> str:
        """Format error with full diagnostics."""
        context = context or {}
        lines = []

        # Header with severity
        severity_color = self.SEVERITY_COLORS[error.severity]
        severity_label = f"[{error.severity.value.upper()}]"
        message = error.format_message(context)
        header = f"{severity_label} {message}"
        lines.append("")
        lines.append(self.colorize(header, severity_color))
        lines.append("")

        # Cause
        lines.append(self.colorize("Cause:", "bold"))
        cause = error.format_cause(context)
        for line in cause.split(". "):
            line = line.strip()
            if line:
                if not line.endswith("."):
                    line += "."
                lines.append(f"  {line}")
        lines.append("")

        # Suggestions
        suggestions = error.format_suggestions(context)
        if suggestions:
            lines.append(self.colorize("Suggestions:", "bold"))
            for i, suggestion in enumerate(suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
            lines.append("")

        # Technical details
        if error.original_exception:
            lines.append(self.colorize("Technical Details:", "cyan"))
            exc_type = type(error.original_exception).__name__
            exc_msg = str(error.original_exception)
            lines.append(f"  Exception: {exc_type}")
            if exc_msg and exc_msg != exc_type:
                lines.append(f"  Message: {exc_msg}")
            lines.append("")

        return "\n".join(lines)

    def format_health_check(
        self,
        is_healthy: bool,
        latency_ms: float | None = None,
        error: MinerError | None = None,
        context: dict | None = None,
    ) -> str:
        """Format health check result."""
        if is_healthy:
            status = self.colorize("PASS", "green")
            latency = f" ({latency_ms:.0f}ms)" if latency_ms else ""
            return f"Pre-flight check: {status}{latency}"
        else:
            status = self.colorize("FAIL", "red")
            result = f"Pre-flight check: {status}"
            if error:
                result += "\n" + self.format_error(error, context)
            return result

    def format_retry_progress(
        self,
        attempt: int,
        max_attempts: int,
        delay: float,
        error: MinerError,
        context: dict | None = None,
    ) -> str:
        """Format retry progress message."""
        context = context or {}
        message = error.format_message(context)
        retry_label = self.colorize("Retry", "yellow")
        return f"{retry_label} [{attempt}/{max_attempts}] - {message} (waiting {delay:.1f}s...)"

    def format_success(self, message: str) -> str:
        """Format success message."""
        return self.colorize(f"[OK] {message}", "green")
