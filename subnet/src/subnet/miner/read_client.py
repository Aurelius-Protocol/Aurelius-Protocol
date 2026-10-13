"""Bounded miner read transport. HTTPS (loopback HTTP for development), no redirects/proxies."""

import json
import math
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class ReadUnavailable(RuntimeError):
    def __init__(self, code: str, status: int = 0):
        self.code, self.status = code, status
        super().__init__(code)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(url: str, *, body: dict | None = None) -> dict:
    parsed = urlsplit(url)
    if (
        parsed.username
        or parsed.password
        or parsed.fragment
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})
    ):
        raise ReadUnavailable("secure_endpoint_required")
    encoded = (
        None if body is None else json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
    )
    if encoded is not None and len(encoded) > 16_384:
        raise ReadUnavailable("request_too_large")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=15) as response:
            raw = response.read(2 * 1024**2 + 1)
    except urllib.error.HTTPError as exc:
        code = "http_error"
        try:
            error = json.loads(exc.read(4097))
            if error == {"error": "epoch_not_found"}:
                code = "epoch_not_found"
        except (ValueError, OSError):
            pass
        raise ReadUnavailable(code, exc.code) from None
    except (OSError, ValueError) as exc:
        raise ReadUnavailable("network_unavailable") from exc
    if len(raw) > 2 * 1024**2:
        raise ReadUnavailable("response_too_large")
    try:

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result

        def constant(_):
            raise ValueError("nonfinite JSON")

        def finite_float(value):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("nonfinite JSON")
            return result

        result = json.loads(
            raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float
        )
    except (ValueError, RecursionError):
        raise ReadUnavailable("invalid_response") from None
    if not isinstance(result, dict):
        raise ReadUnavailable("invalid_response")
    return result
