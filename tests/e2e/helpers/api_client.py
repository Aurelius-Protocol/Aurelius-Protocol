"""Collector API test client for E2E tests."""

import hashlib
import json
import time
from typing import Any

import requests

try:
    import bittensor as bt
except ImportError:
    bt = None


class CollectorAPIClient:
    """
    HTTP client for interacting with the collector API during E2E tests.

    Handles:
    - JSON serialization
    - SR25519 signature generation for telemetry
    - Proper authentication headers
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        api_key: str | None = None,
        timeout: int = 30,
    ):
        """
        Initialize the API client.

        Args:
            base_url: Collector API base URL
            api_key: Optional API key for authenticated endpoints
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self, auth: bool = False) -> dict[str, str]:
        """Get default headers, optionally with auth."""
        headers = {"Content-Type": "application/json"}
        if auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _sign_telemetry(
        self,
        wallet: "bt.Wallet",
        hotkey: str,
    ) -> dict[str, str]:
        """
        Create telemetry authentication headers with SR25519 signature.

        The signature format for telemetry is:
        message = "aurelius-telemetry:{timestamp}:{hotkey}"
        signature = wallet.hotkey.sign(message)

        Args:
            wallet: Bittensor wallet with hotkey
            hotkey: SS58 address of the hotkey

        Returns:
            Headers dict with X-Validator-Hotkey, X-Validator-Signature, X-Signature-Timestamp
        """
        timestamp = str(int(time.time()))
        message = f"aurelius-telemetry:{timestamp}:{hotkey}"

        signature = wallet.hotkey.sign(message.encode())

        return {
            "X-Validator-Hotkey": hotkey,
            "X-Validator-Signature": signature.hex(),
            "X-Signature-Timestamp": timestamp,
            "Content-Type": "application/json",
        }

    def _sign_submission(
        self,
        wallet: "bt.Wallet",
        hotkey: str,
    ) -> dict[str, str]:
        """
        Create submission authentication headers with SR25519 signature.

        The signature format for submissions is:
        message = "aurelius-submission:{timestamp}:{validator_hotkey}"
        signature = wallet.hotkey.sign(message)

        Args:
            wallet: Bittensor wallet with hotkey
            hotkey: SS58 address of the hotkey

        Returns:
            Headers dict with X-Signature, X-Timestamp
        """
        timestamp = str(int(time.time()))  # Use seconds, not milliseconds
        message = f"aurelius-submission:{timestamp}:{hotkey}"

        signature = wallet.hotkey.sign(message.encode())

        return {
            "X-Signature": signature.hex(),
            "X-Timestamp": timestamp,
            "Content-Type": "application/json",
        }

    # Health endpoints

    def health(self) -> requests.Response:
        """Check API health status."""
        return requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )

    def health_detailed(self) -> requests.Response:
        """Get detailed health with metrics."""
        return requests.get(
            f"{self.base_url}/health/detailed",
            timeout=self.timeout,
        )

    # Collection endpoints

    def submit_execution(
        self,
        prompt: str,
        response: str,
        danger_score: float,
        validator_hotkey: str,
        wallet: "bt.Wallet | None" = None,
        validator_uid: int | None = None,
        miner_hotkey: str | None = None,
        miner_uid: int | None = None,
        accepted: bool = True,
        consensus_verified: bool = False,
        prompt_embedding: list[float] | None = None,
        experiment_id: str = "prompt",
        category_scores: dict | None = None,
        **kwargs: Any,
    ) -> requests.Response:
        """
        Submit execution data to the collector.

        Args:
            prompt: The input prompt
            response: The LLM response
            danger_score: Danger score from moderation (0-1)
            validator_hotkey: Validator's SS58 address
            wallet: Optional wallet for signature auth
            miner_hotkey: Optional miner's SS58 address
            miner_uid: Optional miner's UID
            accepted: Whether the prompt was accepted
            consensus_verified: Whether consensus was verified
            prompt_embedding: Optional 384-dim embedding for novelty
            experiment_id: Experiment ID (default: "prompt")
            category_scores: Optional category scores from moderation
            **kwargs: Additional fields to include

        Returns:
            Response object from the API
        """
        data = {
            "prompt": prompt,
            "response": response,
            "danger_score": danger_score,
            "validator_hotkey": validator_hotkey,
            "accepted": accepted,
            "consensus_verified": consensus_verified,
            "experiment_id": experiment_id,
        }

        if validator_uid is not None:
            data["validator_uid"] = validator_uid
        if miner_hotkey:
            data["miner_hotkey"] = miner_hotkey
        if miner_uid is not None:
            data["miner_uid"] = miner_uid
        if prompt_embedding:
            data["prompt_embedding"] = prompt_embedding
        if category_scores:
            data["category_scores"] = category_scores

        data.update(kwargs)

        # Use signature auth if wallet provided
        if wallet:
            headers = self._sign_submission(wallet, validator_hotkey)
        else:
            headers = self._headers()

        return requests.post(
            f"{self.base_url}/api/collections",
            json=data,
            headers=headers,
            timeout=self.timeout,
        )

    def get_executions(
        self,
        validator_hotkey: str | None = None,
        miner_hotkey: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> requests.Response:
        """
        Get execution records with optional filters.

        Args:
            validator_hotkey: Filter by validator
            miner_hotkey: Filter by miner
            limit: Max records to return
            offset: Pagination offset

        Returns:
            Response with execution data
        """
        params = {"limit": limit, "offset": offset}
        if validator_hotkey:
            params["validator_hotkey"] = validator_hotkey
        if miner_hotkey:
            params["miner_hotkey"] = miner_hotkey

        return requests.get(
            f"{self.base_url}/api/collections/executions",
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )

    def get_execution_stats(
        self,
        validator_hotkey: str | None = None,
        since: str | None = None,
    ) -> requests.Response:
        """Get execution statistics."""
        params = {}
        if validator_hotkey:
            params["validator_hotkey"] = validator_hotkey
        if since:
            params["since"] = since

        return requests.get(
            f"{self.base_url}/api/collections/executions/stats",
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )

    # Novelty endpoints

    def check_novelty(
        self,
        prompt: str,
        embedding: list[float],
        experiment_id: str = "prompt",
        include_similar: bool = False,
    ) -> requests.Response:
        """
        Check novelty of a prompt.

        Args:
            prompt: The prompt text
            embedding: 384-dimensional embedding
            experiment_id: Experiment ID for isolation
            include_similar: Whether to include most similar prompt text

        Returns:
            Response with novelty score and similarity info
        """
        data = {
            "prompt": prompt,
            "prompt_embedding": embedding,
            "experiment_id": experiment_id,
            "include_similar_prompt": include_similar,
        }

        return requests.post(
            f"{self.base_url}/api/novelty/check",
            json=data,
            headers=self._headers(),
            timeout=self.timeout,
        )

    def get_novelty_stats(self) -> requests.Response:
        """Get global novelty statistics."""
        return requests.get(
            f"{self.base_url}/api/novelty/stats",
            headers=self._headers(),
            timeout=self.timeout,
        )

    def get_miner_novelty(
        self,
        hotkey: str,
        experiment_id: str = "prompt",
    ) -> requests.Response:
        """
        Get novelty statistics for a specific miner.

        Requires authentication.

        Args:
            hotkey: Miner's SS58 address
            experiment_id: Optional experiment ID filter

        Returns:
            Response with miner novelty stats
        """
        params = {"experiment_id": experiment_id}

        return requests.get(
            f"{self.base_url}/api/novelty/miner/{hotkey}",
            params=params,
            headers=self._headers(auth=True),
            timeout=self.timeout,
        )

    # Telemetry endpoints

    def submit_traces(
        self,
        spans: list[dict],
        validator_hotkey: str,
        wallet: "bt.Wallet",
        validator_uid: int | None = None,
        netuid: int | None = None,
        network: str | None = None,
    ) -> requests.Response:
        """
        Submit trace spans with signature authentication.

        Args:
            spans: List of span objects
            validator_hotkey: Validator's SS58 address
            wallet: Wallet for signing
            validator_uid: Optional validator UID
            netuid: Optional network UID
            network: Optional network name

        Returns:
            Response with submission result
        """
        data = {
            "spans": spans,
            "validator_hotkey": validator_hotkey,
        }

        if validator_uid is not None:
            data["validator_uid"] = validator_uid
        if netuid is not None:
            data["netuid"] = netuid
        if network:
            data["network"] = network

        headers = self._sign_telemetry(wallet, validator_hotkey)

        return requests.post(
            f"{self.base_url}/api/telemetry/traces",
            json=data,
            headers=headers,
            timeout=self.timeout,
        )

    def submit_logs(
        self,
        logs: list[dict],
        validator_hotkey: str,
        wallet: "bt.Wallet",
        validator_uid: int | None = None,
        netuid: int | None = None,
        network: str | None = None,
    ) -> requests.Response:
        """
        Submit log records with signature authentication.

        Args:
            logs: List of log record objects
            validator_hotkey: Validator's SS58 address
            wallet: Wallet for signing
            validator_uid: Optional validator UID
            netuid: Optional network UID
            network: Optional network name

        Returns:
            Response with submission result
        """
        data = {
            "logs": logs,
            "validator_hotkey": validator_hotkey,
        }

        if validator_uid is not None:
            data["validator_uid"] = validator_uid
        if netuid is not None:
            data["netuid"] = netuid
        if network:
            data["network"] = network

        headers = self._sign_telemetry(wallet, validator_hotkey)

        return requests.post(
            f"{self.base_url}/api/telemetry/logs",
            json=data,
            headers=headers,
            timeout=self.timeout,
        )

    def get_trace(self, trace_id: str) -> requests.Response:
        """
        Get all spans for a trace.

        Args:
            trace_id: 32-character hex trace ID

        Returns:
            Response with trace data
        """
        return requests.get(
            f"{self.base_url}/api/telemetry/traces/{trace_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )

    def query_logs(
        self,
        validator_hotkey: str | None = None,
        trace_id: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> requests.Response:
        """
        Query telemetry logs with filters.

        Args:
            validator_hotkey: Filter by validator
            trace_id: Filter by trace
            severity: Filter by severity level
            limit: Max records
            offset: Pagination offset

        Returns:
            Response with log records
        """
        params = {"limit": limit, "offset": offset}
        if validator_hotkey:
            params["validator_hotkey"] = validator_hotkey
        if trace_id:
            params["trace_id"] = trace_id
        if severity:
            params["severity"] = severity

        return requests.get(
            f"{self.base_url}/api/telemetry/logs",
            params=params,
            headers=self._headers(),
            timeout=self.timeout,
        )

    def get_telemetry_stats(self) -> requests.Response:
        """
        Get telemetry statistics.

        Requires authentication.
        """
        return requests.get(
            f"{self.base_url}/api/telemetry/stats",
            headers=self._headers(auth=True),
            timeout=self.timeout,
        )


def generate_random_embedding(dimensions: int = 384) -> list[float]:
    """
    Generate a random normalized embedding for testing.

    Args:
        dimensions: Embedding dimensions (default 384 for MiniLM)

    Returns:
        Normalized embedding vector
    """
    import random
    import math

    raw = [random.gauss(0, 1) for _ in range(dimensions)]
    magnitude = math.sqrt(sum(x * x for x in raw))
    if magnitude == 0:
        magnitude = 1  # Avoid division by zero
    return [x / magnitude for x in raw]


def generate_deterministic_embedding(text: str, dimensions: int = 384) -> list[float]:
    """
    Generate a deterministic embedding from text for testing.

    Uses hash of text to seed random generator for reproducibility.

    Args:
        text: Input text
        dimensions: Embedding dimensions

    Returns:
        Normalized embedding vector
    """
    import random
    import math

    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    raw = [rng.gauss(0, 1) for _ in range(dimensions)]
    magnitude = math.sqrt(sum(x * x for x in raw))
    return [x / magnitude for x in raw]


def create_test_span(
    name: str,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    status: str = "ok",
    duration_ns: int = 1_000_000,  # 1ms default
    attributes: dict | None = None,
) -> dict:
    """
    Create a test span object for telemetry tests.

    Args:
        name: Span name
        trace_id: 32-char hex trace ID (generated if not provided)
        span_id: 16-char hex span ID (generated if not provided)
        parent_span_id: Optional parent span ID
        status: Span status (unset, ok, error)
        duration_ns: Span duration in nanoseconds
        attributes: Optional span attributes

    Returns:
        Span dict matching telemetry schema
    """
    import secrets

    if trace_id is None:
        trace_id = secrets.token_hex(16)
    if span_id is None:
        span_id = secrets.token_hex(8)

    start_time = int(time.time() * 1_000_000_000)
    end_time = start_time + duration_ns

    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "kind": "internal",
        "status": status,
        "start_time_unix_nano": str(start_time),
        "end_time_unix_nano": str(end_time),
        "attributes": attributes or {},
        "events": [],
        "links": [],
        "resource_attributes": {},
    }


def create_test_log(
    body: str,
    severity: str = "INFO",
    trace_id: str | None = None,
    span_id: str | None = None,
    attributes: dict | None = None,
) -> dict:
    """
    Create a test log record for telemetry tests.

    Args:
        body: Log message body
        severity: Severity level (TRACE, DEBUG, INFO, WARN, ERROR, FATAL)
        trace_id: Optional trace correlation
        span_id: Optional span correlation
        attributes: Optional log attributes

    Returns:
        Log record dict matching telemetry schema
    """
    severity_map = {
        "TRACE": 1,
        "DEBUG": 5,
        "INFO": 9,
        "WARN": 13,
        "ERROR": 17,
        "FATAL": 21,
    }

    timestamp = int(time.time() * 1_000_000_000)

    return {
        "timestamp_unix_nano": str(timestamp),
        "trace_id": trace_id,
        "span_id": span_id,
        "severity_number": severity_map.get(severity, 9),
        "severity_text": severity,
        "body": body,
        "attributes": attributes or {},
        "resource_attributes": {},
    }


# Additional helper methods for edge case testing


class RawAPIClient:
    """
    Low-level API client for edge case and boundary tests.

    Provides raw access to make requests without automatic validation
    or signature generation, useful for testing invalid inputs.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        timeout: int = 30,
    ):
        """
        Initialize raw API client.

        Args:
            base_url: Collector API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def submit_execution_raw(
        self,
        data: dict,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """
        Submit raw JSON data to collections endpoint.

        No validation or signature generation - useful for boundary tests.

        Args:
            data: Raw JSON data to submit
            headers: Optional custom headers

        Returns:
            Response object
        """
        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)

        return requests.post(
            f"{self.base_url}/api/collections",
            json=data,
            headers=default_headers,
            timeout=self.timeout,
        )

    def submit_with_custom_headers(
        self,
        endpoint: str,
        data: dict,
        headers: dict[str, str],
    ) -> requests.Response:
        """
        Submit data with arbitrary custom headers.

        Useful for testing authentication edge cases.

        Args:
            endpoint: API endpoint path (e.g., "/api/telemetry/traces")
            data: JSON data to submit
            headers: Custom headers (replaces defaults)

        Returns:
            Response object
        """
        return requests.post(
            f"{self.base_url}{endpoint}",
            json=data,
            headers=headers,
            timeout=self.timeout,
        )

    def submit_with_timeout(
        self,
        endpoint: str,
        data: dict,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """
        Submit data with custom timeout.

        Useful for timeout testing.

        Args:
            endpoint: API endpoint path
            data: JSON data to submit
            timeout: Custom timeout in seconds
            headers: Optional custom headers

        Returns:
            Response object
        """
        default_headers = {"Content-Type": "application/json"}
        if headers:
            default_headers.update(headers)

        return requests.post(
            f"{self.base_url}{endpoint}",
            json=data,
            headers=default_headers,
            timeout=timeout,
        )

    def submit_raw_body(
        self,
        endpoint: str,
        body: str | bytes,
        headers: dict[str, str],
    ) -> requests.Response:
        """
        Submit raw string/bytes body (not JSON).

        Useful for testing malformed request handling.

        Args:
            endpoint: API endpoint path
            body: Raw body content
            headers: Headers to send

        Returns:
            Response object
        """
        return requests.post(
            f"{self.base_url}{endpoint}",
            data=body,
            headers=headers,
            timeout=self.timeout,
        )

    def get_with_params(
        self,
        endpoint: str,
        params: dict,
    ) -> requests.Response:
        """
        Make GET request with query parameters.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Response object
        """
        return requests.get(
            f"{self.base_url}{endpoint}",
            params=params,
            timeout=self.timeout,
        )


def create_invalid_embedding(
    variation: str = "wrong_dimensions",
    dimensions: int = 384,
) -> list:
    """
    Create invalid embeddings for boundary testing.

    Args:
        variation: Type of invalid embedding:
            - "wrong_dimensions": 100 dimensions instead of 384
            - "zero_vector": All zeros
            - "contains_nan": Contains NaN value
            - "contains_inf": Contains infinity
            - "strings": Contains string values
            - "too_large": Contains very large values
        dimensions: Base dimensions (for valid-like variations)

    Returns:
        Invalid embedding list
    """
    import math

    if variation == "wrong_dimensions":
        return [0.1] * 100

    if variation == "zero_vector":
        return [0.0] * dimensions

    if variation == "contains_nan":
        embedding = generate_random_embedding(dimensions)
        embedding[0] = float("nan")
        return embedding

    if variation == "contains_inf":
        embedding = generate_random_embedding(dimensions)
        embedding[0] = float("inf")
        return embedding

    if variation == "strings":
        return ["not", "valid", "floats"] * (dimensions // 3 + 1)

    if variation == "too_large":
        return [1e37] * dimensions

    # Default: wrong dimensions
    return [0.1] * 100


def create_edge_case_prompt(case: str = "empty") -> str:
    """
    Create edge case prompts for boundary testing.

    Args:
        case: Type of edge case:
            - "empty": Empty string
            - "whitespace": Only whitespace
            - "very_long": 10,000 characters
            - "unicode": Contains emoji, CJK, RTL
            - "control_chars": Contains control characters
            - "sql_injection": SQL injection attempt
            - "xss": XSS attempt
            - "null_byte": Contains null byte

    Returns:
        Edge case prompt string
    """
    cases = {
        "empty": "",
        "whitespace": "   \t\n   ",
        "very_long": "A" * 10000,
        "unicode": "Hello \U0001F600 \u4F60\u597D \u0645\u0631\u062D\u0628\u0627",
        "control_chars": "Test\x00with\x01control\x02chars",
        "sql_injection": "'; DROP TABLE executions; --",
        "xss": "<script>alert('xss')</script>",
        "null_byte": "test\x00null",
    }

    return cases.get(case, "")
