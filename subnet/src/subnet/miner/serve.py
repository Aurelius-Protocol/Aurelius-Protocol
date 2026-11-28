"""Miner serve (plan §1.8; pulled forward at 1.7): miner-owned candidate selection plus a FRESH,
real-sr25519-signed work_id per query.

CONTRACT (capsule header)
- In ``staging`` mode the validator requests only an operator-defined task type and schema
  version, and the miner selects its one eligible window-scoped item from its own validated JSONL
  pool. The validator never supplies the candidate. The selected config is canonicalized, assigned
  its shared cost identity, and signed with an exact v2 work authorization bound to the
  authenticated validator audience, its 64-hex request ID, and the authoritative fanout window.
  The nonce is OS-random, so the same content still yields a DIFFERENT verifying work_id across
  restarts.
- No lane import (the miner treats config as opaque data — contract `platform-not-lane`); no charge
  path (contract `charge-path-single-import-site`). The coldkey is never touched by serving.
- Query bodies have a hard pre-read cap and socket deadline. A bounded server rejects saturation
  instead of spawning unbounded threads. Every query is authenticated: the server refuses to
  expose ``/v1/query`` without a request authorizer. The companion HTTP server on the health port
  is health-only and loopback-bound.
- Startup is posture-checked: serving requires a hotkey-only Bittensor wallet, an endpoint address
  the policy admits, and a readable signed configuration.
"""

from __future__ import annotations

import http.server
import json
import os
import secrets
import threading
import time
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from subnet.bittensor_v11 import (
    BittensorV11Error,
    load_wallet,
    require_bittensor_v11,
    validate_subtensor_endpoint,
)
from subnet.miner.candidates import (
    CandidateConfigUnavailable,
    NoCandidateAvailable,
    candidate_for_request,
)
from subnet.miner.config_schema import ServingConfig, SchemaUnavailable, config_source
from subnet.shared.canonicalization import canonical_bytes, content_hash, parse_canonical_json
from subnet.shared.clock import now_ts
from subnet.shared.cost_model import cost_model_version, derive_cost_class
from subnet.shared.network_policy import AxonEndpointPolicy, AxonEndpointPolicyError
from subnet.shared.signing import work_id_v2_signed_bytes
from subnet.shared.wallet_view import assert_hotkey_only_wallet_view


MAX_HTTP_QUERY_BYTES = 69_632  # 64KiB config cap plus bounded request metadata
DEFAULT_MAX_CONCURRENT_REQUESTS = 32
HARD_MAX_CONCURRENT_REQUESTS = 256
DEFAULT_SOCKET_TIMEOUT_S = Decimal("5")
HARD_MAX_SOCKET_TIMEOUT_S = Decimal("60")


class QueryAuthorizationError(ValueError):
    """A transport authenticator refused a query before candidate selection."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


DEFAULT_STATUS_INTERVAL_S = 300.0


class ServingStats:
    """Thread-safe serving telemetry: what a silent access log would have said, aggregated.

    Observation only — never authority over money, permits, or serving decisions.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._started = time.time()
        self._queries_served = 0
        self._auth_rejected = 0
        self._client_errors = 0
        self._no_candidate = 0
        self._saturated = 0
        self._rate_limited = 0
        self._last_caller = None
        self._last_activity = None

    def note_served(self, caller: str | None) -> None:
        with self._lock:
            self._queries_served += 1
            self._last_caller = caller
            self._last_activity = time.time()

    def note_no_candidate(self, caller: str | None) -> None:
        with self._lock:
            self._no_candidate += 1
            self._last_caller = caller
            self._last_activity = time.time()

    def note_auth_rejected(self) -> None:
        with self._lock:
            self._auth_rejected += 1
            self._last_activity = time.time()

    def note_client_error(self) -> None:
        with self._lock:
            self._client_errors += 1
            self._last_activity = time.time()

    def note_saturated(self) -> None:
        with self._lock:
            self._saturated += 1
            self._last_activity = time.time()

    def note_rate_limited(self) -> None:
        with self._lock:
            self._rate_limited += 1
            self._last_activity = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": round(time.time() - self._started, 3),
                "queries_served": self._queries_served,
                "auth_rejected": self._auth_rejected,
                "client_errors": self._client_errors,
                "no_candidate": self._no_candidate,
                "saturated": self._saturated,
                "rate_limited": self._rate_limited,
                "last_caller_hotkey": self._last_caller,
                "last_activity_at": (
                    None if self._last_activity is None else round(self._last_activity, 3)
                ),
            }

    def line(self) -> str:
        snap = self.snapshot()
        last = (
            "never"
            if snap["last_activity_at"] is None
            else f"{round(time.time() - snap['last_activity_at'])}s ago"
        )
        caller = snap["last_caller_hotkey"] or "none"
        return (
            f"served={snap['queries_served']} auth_rejected={snap['auth_rejected']} "
            f"no_candidate={snap['no_candidate']} "
            f"client_errors={snap['client_errors']} saturated={snap['saturated']} "
            f"rate_limited={snap['rate_limited']} "
            f"last_caller={caller[:16]}… last_activity={last} uptime={round(snap['uptime_s'])}s"
        )


DEFAULT_MAX_QUERIES_PER_CALLER = 60
DEFAULT_CALLER_WINDOW_S = Decimal("60")


class CallerRateLimiter:
    """A bounded sliding-window query quota per caller (hotkey when authenticated, else peer IP).

    Hygiene only — it complements the concurrency ceiling with a rate ceiling so one caller
    cannot keep every request slot hot or harvest envelopes at line rate. Never authority
    over money, permits, or candidate selection.
    """

    def __init__(self, max_per_window: int, window_s: float, clock=time.time):
        if type(max_per_window) is not int or max_per_window < 0:
            raise ValueError("max_per_window must be a non-negative integer (0 disables)")
        if isinstance(window_s, bool) or not isinstance(window_s, (int, float)) or window_s <= 0:
            raise ValueError("window_s must be positive seconds")
        self._max = max_per_window
        self._window_s = float(window_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._calls: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        if self._max == 0:
            return True
        now = self._clock()
        with self._lock:
            history = [stamp for stamp in self._calls.get(key, ()) if now - stamp < self._window_s]
            if len(history) >= self._max:
                self._calls[key] = history
                return False
            history.append(now)
            self._calls[key] = history
            if len(self._calls) > 4096:  # bound memory under key churn
                cutoff = now - self._window_s
                self._calls = {
                    candidate: stamps
                    for candidate, stamps in self._calls.items()
                    if any(stamp >= cutoff for stamp in stamps)
                }
            return True


def caller_rate_limiter() -> CallerRateLimiter:
    """Build the serving rate limiter from explicit operator configuration (0 disables)."""
    raw_max = os.environ.get(
        "AURELIUS_MINER_MAX_QUERIES_PER_CALLER", str(DEFAULT_MAX_QUERIES_PER_CALLER)
    )
    if not isinstance(raw_max, str) or not raw_max.isascii() or not raw_max.isdecimal():
        raise ValueError("AURELIUS_MINER_MAX_QUERIES_PER_CALLER must be decimal digits")
    raw_window = os.environ.get("AURELIUS_MINER_CALLER_WINDOW_S", str(DEFAULT_CALLER_WINDOW_S))
    try:
        window = Decimal(raw_window)
    except InvalidOperation:
        raise ValueError("AURELIUS_MINER_CALLER_WINDOW_S must be decimal seconds") from None
    if not window.is_finite() or not Decimal("0.1") <= window <= Decimal("3600"):
        raise ValueError("AURELIUS_MINER_CALLER_WINDOW_S must be 0.1..3600 seconds")
    return CallerRateLimiter(int(raw_max), float(window))


def status_interval_s() -> float:
    """The periodic operator status-line interval; 0 disables the line entirely."""
    raw = os.environ.get("AURELIUS_MINER_STATUS_INTERVAL_S", str(DEFAULT_STATUS_INTERVAL_S))
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise ValueError(
            "AURELIUS_MINER_STATUS_INTERVAL_S must be decimal seconds (0 disables)"
        ) from None
    if not value.is_finite() or not Decimal("0") <= value <= Decimal("86400"):
        raise ValueError("AURELIUS_MINER_STATUS_INTERVAL_S must be 0..86400 seconds")
    return float(value)


def start_status_line(stats: ServingStats, prefix: str, *, interval_s: float | None = None) -> None:
    """Print one aggregated status line per interval so a live miner is never silent."""

    if interval_s is None:
        interval_s = status_interval_s()
    if interval_s <= 0:
        return

    def run():
        while True:
            time.sleep(interval_s)
            print(f"{prefix} status: {stats.line()}", flush=True)

    threading.Thread(target=run, daemon=True).start()


class WorkIdMinter:
    """Mints fresh v2 work authorizations with an OS-random nonce for one hotkey."""

    def __init__(self, *, hotkey_ss58: str, signer: Callable[[bytes], bytes]):
        if not isinstance(hotkey_ss58, str) or not hotkey_ss58:
            raise ValueError("hotkey_ss58 is required")
        if not callable(signer):
            raise ValueError("signer must be callable")
        # The mounted Bittensor wallet's hotkey signer is passed in directly. Keeping it a
        # callable avoids ever extracting or copying private key material.
        self.hotkey_ss58 = hotkey_ss58
        self._signer = signer

    def sign_observation(self, message: bytes) -> bytes:
        """Sign a domain-separated, read-only observation request using only the hotkey."""
        from subnet.shared.miner_observation import DOMAIN

        if not message.startswith(DOMAIN):
            raise ValueError("observation signature domain required")
        return self._signer(message)

    def sign_economics(self, message: bytes) -> bytes:
        """Sign only the separate read-only economics domain using the hotkey."""
        from subnet.shared.miner_economics import DOMAIN

        if not message.startswith(DOMAIN):
            raise ValueError("economics signature domain required")
        return self._signer(message)

    def sign_terms(self, message: bytes) -> bytes:
        """Sign only a scoped acceptance of explicitly selected submission terms."""
        from subnet.shared.submission_terms import DOMAIN

        if not message.startswith(DOMAIN):
            raise ValueError("terms acceptance signature domain required")
        return self._signer(message)

    def sign_appeal(self, message: bytes) -> bytes:
        """Sign only explicit appeal requests, never a charge authorization."""
        from subnet.shared.miner_appeal import DOMAIN

        if not message.startswith(DOMAIN):
            raise ValueError("appeal signature domain required")
        return self._signer(message)

    def sign_miner_funding_terms(self, message: bytes) -> bytes:
        """Sign only the read-only ``terms_version: 2`` funding-terms request (gamma §11.1)."""
        from subnet.miner.gamma_funding import TERMS_V2_DOMAIN

        if not message.startswith(TERMS_V2_DOMAIN + b"\x00"):
            raise ValueError("funding terms signature domain required")
        return self._signer(message)

    def sign_gamma_funding_auth(self, message: bytes) -> bytes:
        """Sign only the purpose-bound gamma quote authorization; never a spend or a charge."""
        from subnet.miner.gamma_funding import GAMMA_AUTH_DOMAIN

        if not message.startswith(GAMMA_AUTH_DOMAIN + b"\x00"):
            raise ValueError("gamma funding auth signature domain required")
        return self._signer(message)

    def _next_nonce(self) -> str:
        return secrets.token_hex(32)

    def envelope(
        self,
        task: dict,
        *,
        validator_hotkey_ss58: str,
        request_id: str,
        fanout_window: int,
    ) -> dict:
        """Mint the exact v2 audience/window-bound work authorization.

        The three request-bound fields are required keyword arguments: there is no v1 fallback
        and no miner-selected default for validator audience, request identity, or chain window.
        """
        task_type = task["task_type"]
        schema_version = int(task["schema_version"])
        config = task["config"]
        selected_cost_model_version = cost_model_version(task_type)
        cost_class = derive_cost_class(task_type, config)
        ch = content_hash(
            task_type, schema_version, config, selected_cost_model_version, cost_class
        )
        nonce = self._next_nonce()
        # The wall clock. Freshness comes from the nonce and the window, never from a fixed time.
        ts = now_ts()
        signature = self._signer(
            work_id_v2_signed_bytes(
                content_hash=ch,
                hotkey_ss58=self.hotkey_ss58,
                validator_hotkey_ss58=validator_hotkey_ss58,
                request_id=request_id,
                fanout_window=fanout_window,
                timestamp=ts,
                nonce=nonce,
            )
        )
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise ValueError("hotkey signer must return a 64-byte sr25519 signature")
        from subnet.miner.journal import observe_envelope

        observe_envelope(task, self.hotkey_ss58, validator_hotkey_ss58, fanout_window)
        return {
            "task_type": task_type,
            "schema_version": schema_version,
            "canonical_config": canonical_bytes(config).decode("utf-8"),
            "cost_model_version": selected_cost_model_version,
            "cost_class": cost_class,
            "content_hash": ch,
            "work_id": {
                "version": 2,
                "hotkey_ss58": self.hotkey_ss58,
                "validator_hotkey_ss58": validator_hotkey_ss58,
                "request_id": request_id,
                "fanout_window": fanout_window,
                "timestamp": ts,
                "nonce": nonce,
                "signature_hex": signature.hex(),
            },
        }


def load_bittensor_wallet(bt_module=None):
    """Load the mounted wallet named by the chain env, without importing Bittensor on dev paths."""
    wallet_name = os.environ.get("AURELIUS_BT_WALLET", "miner")
    hotkey_name = os.environ.get("AURELIUS_BT_HOTKEY", "default")
    wallet_path = os.environ.get("AURELIUS_BT_WALLET_PATH")
    guard = os.environ.get("AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET", "false")
    if guard not in {"true", "false"}:
        raise ValueError("AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET must be true or false")
    if guard == "true":
        if not wallet_path:
            raise ValueError("guarded Bittensor wallet loading requires AURELIUS_BT_WALLET_PATH")
        assert_hotkey_only_wallet_view(wallet_path, wallet_name, hotkey_name)
    if bt_module is None:
        try:
            bt_module = require_bittensor_v11()
        except ImportError as e:
            raise NotImplementedError(
                "the miner's identity needs the optional `bt` extra (the chain-bt image)"
            ) from e
    else:
        bt_module = require_bittensor_v11(bt_module)
    return load_wallet(
        bt_module,
        name=wallet_name,
        hotkey=hotkey_name,
        path=wallet_path,
    )


def minter_from_bittensor_wallet(wallet) -> WorkIdMinter:
    """Bind work-id authorship to the exact hotkey the Bittensor wallet selected."""
    hotkey = wallet.hotkey
    ss58 = str(hotkey.ss58_address)
    signer = getattr(hotkey, "sign", None)
    if not ss58 or not callable(signer):
        raise ValueError("Bittensor wallet hotkey must expose ss58_address and sign(message)")
    return WorkIdMinter(hotkey_ss58=ss58, signer=signer)


def minter_from_env() -> WorkIdMinter:
    """The miner's identity: the hotkey of the mounted Bittensor wallet."""
    return minter_from_bittensor_wallet(load_bittensor_wallet())


def validate_miner_startup_posture(environ=None) -> None:
    """Refuse to serve with an incomplete chain identity, an unguarded wallet, an endpoint
    address the policy cannot admit, or no way to read the signed configuration.

    There is one posture. Every independently checkable problem is collected first, so one
    restart reports the whole misconfiguration instead of one missing key per restart.
    """

    env = os.environ if environ is None else environ
    problems: list[str] = []
    for key in (
        "AURELIUS_BITTENSOR_NONCE_DB",
        "AURELIUS_AXON_PUBLISH_STATE_DIR",
        "AURELIUS_AXON_EXPECTED_CHAIN_NAME",
        "AURELIUS_AXON_EXPECTED_GENESIS_HASH",
        "AURELIUS_RUNTIME_SPEC_PIN",
        "CHAIN_ID",
        "SUBSTRATE_URL",
    ):
        if not env.get(key):
            problems.append(f"{key} is required to serve")
    if env.get("SUBSTRATE_URL"):
        try:
            validate_subtensor_endpoint(env["SUBSTRATE_URL"])
        except BittensorV11Error as exc:
            problems.append(f"SUBSTRATE_URL is invalid: {exc}")
    runtime = env.get("AURELIUS_RUNTIME_SPEC_PIN")
    if runtime and (not runtime.isascii() or not runtime.isdecimal() or int(runtime) <= 0):
        problems.append("AURELIUS_RUNTIME_SPEC_PIN must be a positive decimal integer")
    genesis = env.get("AURELIUS_AXON_EXPECTED_GENESIS_HASH")
    if genesis and (
        len(genesis) != 66
        or not genesis.startswith("0x")
        or any(character not in "0123456789abcdefABCDEF" for character in genesis[2:])
    ):
        problems.append("AURELIUS_AXON_EXPECTED_GENESIS_HASH must be an exact 32-byte hash")
    if env.get("AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET") != "true":
        problems.append("AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET must be 'true' to serve")
    try:
        AxonEndpointPolicy.build(allowed_cidrs=env.get("AURELIUS_AXON_ALLOWED_CIDRS"))
    except AxonEndpointPolicyError as exc:
        problems.append(str(exc))
    try:
        config_source(env)
    except (SchemaUnavailable, ValueError) as exc:
        problems.append(str(exc))
    if problems:
        raise ValueError("; ".join(problems))


def _handler(
    minter: WorkIdMinter,
    status=None,
    candidate_resolver=None,
    *,
    query_enabled: bool = True,
    max_query_bytes: int = MAX_HTTP_QUERY_BYTES,
    request_authorizer=None,
    stats: ServingStats | None = None,
    rate_limiter: CallerRateLimiter | None = None,
):
    if query_enabled and request_authorizer is None:
        raise ValueError("the query route requires a request authorizer")
    resolve_candidate = candidate_resolver or candidate_for_request

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                if status is None:
                    body = {"status": "ok", "hotkey": minter.hotkey_ss58}
                    if stats is not None:
                        body["stats"] = stats.snapshot()
                    self._json(200, body)
                else:
                    # readiness seam (btauth mode): 503 until the endpoint is PUBLISHED on chain, so
                    # a compose healthcheck means "discoverable", not merely "socket open"
                    code, doc = status()
                    if code == 200 and stats is not None:
                        doc = {**doc, "stats": stats.snapshot()}
                    self._json(code, doc)
            else:
                if stats is not None:
                    stats.note_client_error()
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/query" or not query_enabled:
                if stats is not None:
                    stats.note_client_error()
                self._json(404, {"error": "not found"})
                return
            try:
                raw_length = self.headers.get("content-length")
                if raw_length is None or not raw_length.isascii() or not raw_length.isdecimal():
                    if stats is not None:
                        stats.note_client_error()
                    self._json(400, {"error": "valid content-length required"})
                    return
                length = int(raw_length)
                if length > max_query_bytes:
                    if stats is not None:
                        stats.note_client_error()
                    self._json(413, {"error": "request body too large"})
                    return
                try:
                    raw_body = self.rfile.read(length)
                except TimeoutError:
                    if stats is not None:
                        stats.note_client_error()
                    self._json(408, {"error": "request body deadline exceeded"})
                    return
                if len(raw_body) != length:
                    if stats is not None:
                        stats.note_client_error()
                    self._json(400, {"error": "incomplete request body"})
                    return
                caller_hotkey = request_authorizer(
                    self.headers, raw_body, method="POST", path=self.path
                )
                if not isinstance(caller_hotkey, str) or not caller_hotkey:
                    raise QueryAuthorizationError(401, "authenticated caller is unavailable")
                if rate_limiter is not None and not rate_limiter.allow(caller_hotkey):
                    if stats is not None:
                        stats.note_rate_limited()
                    self._json(429, {"error": "caller query rate exceeded"})
                    return
                request = parse_canonical_json(raw_body)
                if not isinstance(request, dict):
                    raise ValueError("task request must be an object")
                if request.get("validator_hotkey_ss58") != caller_hotkey:
                    raise QueryAuthorizationError(
                        403, "declared validator audience differs from authenticated caller"
                    )
                candidate = resolve_candidate(request)
                envelope = minter.envelope(
                    candidate,
                    validator_hotkey_ss58=request["validator_hotkey_ss58"],
                    request_id=request["request_id"],
                    fanout_window=request["fanout_window"],
                )
            except NoCandidateAvailable:
                if stats is not None:
                    stats.note_no_candidate(caller_hotkey)
                self.send_response(204)
                self.end_headers()
                return
            except CandidateConfigUnavailable:
                if stats is not None:
                    stats.note_client_error()
                self._json(503, {"error": "serving config unavailable"})
                return
            except QueryAuthorizationError as e:
                if stats is not None:
                    stats.note_auth_rejected()
                self._json(e.status, {"error": str(e)})
                return
            except (UnicodeError, ValueError, KeyError, TypeError) as e:
                if stats is not None:
                    stats.note_client_error()
                self._json(422, {"error": f"bad task: {e}"})
                return
            if stats is not None:
                stats.note_served(caller_hotkey)
            self._json(200, envelope)

        def _json(self, code: int, body: dict):
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):  # silence access logs
            pass

    return Handler


def _server_limit_int(name: str, default: int, hard_max: int) -> int:
    raw = os.environ.get(name, str(default))
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{name} must be canonical decimal digits")
    value = int(raw)
    if not 1 <= value <= hard_max:
        raise ValueError(f"{name} must be between 1 and {hard_max}")
    return value


def _server_limit_seconds(name: str, default: Decimal, hard_max: Decimal) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = Decimal(raw)
    except (InvalidOperation, TypeError):
        raise ValueError(f"{name} must be decimal seconds") from None
    if not value.is_finite() or value <= 0 or value > hard_max:
        raise ValueError(f"{name} must be greater than zero and at most {hard_max} seconds")
    return float(value)


class BoundedThreadingHTTPServer(http.server.ThreadingHTTPServer):
    """Thread-per-request HTTP with an exact ceiling and an accepted-socket deadline."""

    daemon_threads = True

    def __init__(
        self,
        server_address,
        handler_class,
        *,
        max_concurrent_requests: int,
        socket_timeout_s: float,
        stats: ServingStats | None = None,
    ):
        self._request_slots = threading.BoundedSemaphore(max_concurrent_requests)
        self.socket_timeout_s = socket_timeout_s
        self._stats = stats
        self.request_queue_size = min(max_concurrent_requests, 64)
        super().__init__(server_address, handler_class)

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(self.socket_timeout_s)
        return request, client_address

    def process_request(self, request, client_address):
        stats = getattr(self, "_stats", None)
        if not self._request_slots.acquire(blocking=False):
            if stats is not None:
                stats.note_saturated()
            payload = b'{"error":"server saturated"}'
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
                + payload
            )
            try:
                request.sendall(response)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


def make_server(
    minter: WorkIdMinter,
    port: int,
    status=None,
    candidate_resolver=None,
    *,
    query_enabled: bool = True,
    bind_host: str = "0.0.0.0",
    max_query_bytes: int = MAX_HTTP_QUERY_BYTES,
    request_authorizer=None,
    max_concurrent_requests: int | None = None,
    socket_timeout_s: float | None = None,
    stats: ServingStats | None = None,
    rate_limiter: CallerRateLimiter | None = None,
    serving_config: ServingConfig | None = None,
) -> BoundedThreadingHTTPServer:
    """The HTTP surface: health, and ``/v1/query`` behind the request authorizer. It is THE
    server on the advertised port, and a health-only sidecar on the health port (ADR-0030)."""
    if max_concurrent_requests is None:
        max_concurrent_requests = _server_limit_int(
            "AURELIUS_MINER_MAX_CONCURRENT_REQUESTS",
            DEFAULT_MAX_CONCURRENT_REQUESTS,
            HARD_MAX_CONCURRENT_REQUESTS,
        )
    elif (
        type(max_concurrent_requests) is not int
        or not 1 <= max_concurrent_requests <= HARD_MAX_CONCURRENT_REQUESTS
    ):
        raise ValueError(
            f"max_concurrent_requests must be between 1 and {HARD_MAX_CONCURRENT_REQUESTS}"
        )
    if socket_timeout_s is None:
        socket_timeout_s = _server_limit_seconds(
            "AURELIUS_MINER_SOCKET_TIMEOUT_S",
            DEFAULT_SOCKET_TIMEOUT_S,
            HARD_MAX_SOCKET_TIMEOUT_S,
        )
    elif (
        isinstance(socket_timeout_s, bool)
        or not isinstance(socket_timeout_s, (int, float))
        or not 0 < socket_timeout_s <= float(HARD_MAX_SOCKET_TIMEOUT_S)
    ):
        raise ValueError(
            f"socket_timeout_s must be greater than zero and at most {HARD_MAX_SOCKET_TIMEOUT_S}"
        )
    if serving_config is not None:

        def candidate_resolver(request):
            return candidate_for_request(request, schema_loader=serving_config.schema)

        original_status = status

        def configured_status():
            code, body = serving_config.readiness()
            if code != 200:
                return code, body
            if original_status is None:
                return code, {**body, "hotkey": minter.hotkey_ss58}
            base_code, base_body = original_status()
            return base_code, {**base_body, "config": body["config"]}

        status = configured_status
    shared = stats if stats is not None else ServingStats()
    return BoundedThreadingHTTPServer(
        (bind_host, port),
        _handler(
            minter,
            status,
            candidate_resolver,
            query_enabled=query_enabled,
            max_query_bytes=max_query_bytes,
            request_authorizer=request_authorizer,
            stats=shared,
            rate_limiter=(rate_limiter if rate_limiter is not None else caller_rate_limiter()),
        ),
        max_concurrent_requests=max_concurrent_requests,
        socket_timeout_s=socket_timeout_s,
        stats=shared,
    )
