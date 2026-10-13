"""Step-11 tests: the aurelius-sae-upload miner client (ADR-0205 §2 wire contract).

No wall clock, no network, no randomness beyond sr25519's deliberate signature
randomization: HTTP is exercised through an injected fake opener, the wallet is the
deterministic testlab dev pair, and the timestamp is a frozen constant. The cross-package case — that the client's bytes satisfy the sidecar's ``UploadVerifier``
unchanged (step-11 acceptance) — lives in
``sidecars/artifact-ingress/tests/test_http.py`` instead. It was here, importing
``artifact_ingress.auth``, until the feature gate found that the D5 public projection copies
``subnet/tests/**`` wholesale but copies no ``sidecars/`` at all, so the projected suite could
not collect this module. Step 11's own acceptance pre-authorised the move. What stays here is
the pure client: wire framing, headers and typed error mapping, all public-safe.
"""

from __future__ import annotations

import hashlib
import http.client
import inspect
import io
import json
from pathlib import Path
import urllib.error
import urllib.request

import pytest


from subnet.lanes.interp_sae import upload as upload_module
from subnet.lanes.interp_sae.bundle_digest import bundle_digest
from subnet.lanes.interp_sae.upload import (
    HEADER_DIGEST,
    HEADER_HOTKEY,
    HEADER_NONCE,
    HEADER_SIGNATURE,
    HEADER_SIZE,
    HEADER_SLOT_ID,
    HEADER_TIMESTAMP,
    UPLOAD_PATH,
    BundleError,
    PreparedBundle,
    UploadError,
    _refused,
    _upload_url,
    body_chunks,
    prepare_upload,
    upload,
)
from subnet.miner.wallet import load_hotkey
from subnet.shared.signing import keypair_from_seed, verify_artifact_upload
from subnet.shared.ss58 import ss58_decode, ss58_encode

pytestmark = pytest.mark.miner

CFG: bytes = b'{"activation": "relu", "bins": 16}'
WEIGHTS: bytes = bytes(range(256)) * 4  # 1024 bytes
SLOT_ID = "slot-1"
NONCE = "ab" * 32
NOW = 1_700_000_000
BASE_URL = "http://127.0.0.1:8095"  # testlab-local: the one plain-http allowance
SEED = hashlib.sha256(b"aurelius-testlab-wallet:tester").digest()
WALLET_KEYPAIR = keypair_from_seed(SEED)
WALLET_HOTKEY = ss58_encode(WALLET_KEYPAIR[0])


def _bundle_dir(tmp_path: Path, *, cfg: bytes = CFG, weights: bytes = WEIGHTS) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "cfg.json").write_bytes(cfg)
    (bundle / "sae_weights.safetensors").write_bytes(weights)
    return bundle


def _prepared(tmp_path: Path, **kwargs: bytes) -> PreparedBundle:
    return prepare_upload(_bundle_dir(tmp_path, **kwargs))


def _framed(cfg: bytes, weights: bytes) -> bytes:
    """The ADR-0205 §2 golden framing: two u64 BE lengths around the exact S4 preimage."""
    return len(cfg).to_bytes(8, "big") + cfg + len(weights).to_bytes(8, "big") + weights


class _JsonResponse:
    """Minimal 200/201 stand-in: the tiny JSON ack the ingress actually sends."""

    def __init__(self, status: int, payload: dict[str, str]) -> None:
        self.status = status
        self._payload = json.dumps(payload).encode()

    def read(self, amount: int = -1) -> bytes:
        return self._payload if amount < 0 else self._payload[:amount]

    def close(self) -> None:
        pass


class _FakeOpener:
    """Records the exact Request and answers with one canned response or refusal."""

    def __init__(
        self,
        response: _JsonResponse | None = None,
        error: Exception | None = None,
        handlers: tuple[object, ...] = (),
    ) -> None:
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float | None] = []
        self.handlers: tuple[object, ...] = handlers  # vetted by upload() like a real opener
        self._response = response
        self._error = error

    def open(self, request: urllib.request.Request, timeout: float | None = None) -> object:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _wire_headers(request: urllib.request.Request) -> dict[str, str]:
    """The request's headers as they leave the client (urllib capitalizes the names)."""
    return {name.lower(): value for name, value in request.header_items()}


class _EarlyReplySocket:
    """Real urllib/http.client parsing; deterministic socket with an early server reply."""

    def __init__(self, reply):
        self.reply = reply
        self.writes = 0

    def sendall(self, data):
        self.writes += 1
        if self.writes > 1:  # headers arrived; server closed its read side before the body
            raise BrokenPipeError("peer answered before body")

    def makefile(self, *args):
        return io.BytesIO(self.reply)

    def close(self):
        pass


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize(
    "status,payload", [(200, {"status": "exists"}), (403, {"code": "slot_not_admitted"}), (307, {})]
)
def test_early_response_after_broken_pipe_is_read_without_redirecting(
    tmp_path, monkeypatch, scheme, status, payload
):
    prepared = _prepared(tmp_path)
    data = json.dumps(payload).encode()
    wire = (
        f"HTTP/1.1 {status} Reply\r\nContent-Length: {len(data)}\r\nLocation: https://other.invalid/\r\n\r\n".encode()
        + data
    )
    connections = []

    def connect(connection):
        connection.sock = _EarlyReplySocket(wire)
        connections.append(connection)

    monkeypatch.setattr(http.client.HTTPConnection, "connect", connect)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", connect)
    kwargs = dict(
        url=f"{scheme}://127.0.0.1:8095",
        prepared=prepared,
        slot_id=SLOT_ID,
        keypair=WALLET_KEYPAIR,
        hotkey_ss58=WALLET_HOTKEY,
        nonce=NONCE,
        timestamp=NOW,
    )
    if status == 200:
        assert upload(**kwargs) == payload
    else:
        with pytest.raises(UploadError) as caught:
            upload(**kwargs)
        assert (caught.value.status, caught.value.code) == (
            status,
            "redirect" if status == 307 else "slot_not_admitted",
        )
    assert len(connections) == 1


@pytest.mark.parametrize("failure", ["no_response", "changed_bytes", "grown_bytes"])
def test_early_reply_never_hides_transport_failure_or_local_bundle_drift(
    tmp_path, monkeypatch, failure
):
    prepared = _prepared(tmp_path)
    wire = (
        b""
        if failure == "no_response"
        else b'HTTP/1.1 200 OK\r\nContent-Length: 19\r\n\r\n{"status":"exists"}'
    )

    def connect(connection):
        connection.sock = _EarlyReplySocket(wire)
        if failure == "changed_bytes":
            prepared.weights_path.write_bytes(b"x" * prepared.weights_size)
        elif failure == "grown_bytes":
            prepared.weights_path.write_bytes(b"x" * (prepared.weights_size + 1))

    monkeypatch.setattr(http.client.HTTPConnection, "connect", connect)
    expected = UploadError if failure == "no_response" else BundleError
    with pytest.raises(expected) as caught:
        upload(
            url=BASE_URL,
            prepared=prepared,
            slot_id=SLOT_ID,
            keypair=WALLET_KEYPAIR,
            hotkey_ss58=WALLET_HOTKEY,
            nonce=NONCE,
            timestamp=NOW,
        )
    if failure == "no_response":
        assert caught.value.code == "network"
    elif failure == "changed_bytes":
        assert str(caught.value) == "digest_changed"


# ------------------------------------------------------------------- prepare_upload


def test_prepare_upload_digest_and_size_match_the_s4_contract(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    assert prepared.cfg == CFG
    assert prepared.weights_path.name == "sae_weights.safetensors"
    assert prepared.weights_size == len(WEIGHTS)
    assert prepared.digest == bundle_digest(CFG, WEIGHTS)
    assert prepared.size == 16 + len(CFG) + len(WEIGHTS)


def test_prepare_upload_refuses_a_missing_bundle_directory(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="directory"):
        prepare_upload(tmp_path / "never-created")


def test_prepare_upload_refuses_a_directory_with_extra_files(tmp_path: Path) -> None:
    bundle = _bundle_dir(tmp_path)
    (bundle / "notes.txt").write_bytes(b"extra")
    with pytest.raises(BundleError, match="exactly"):
        prepare_upload(bundle)


@pytest.mark.parametrize("victim", ["cfg.json", "sae_weights.safetensors"])
def test_prepare_upload_refuses_a_symlinked_bundle_file(tmp_path: Path, victim: str) -> None:
    bundle = _bundle_dir(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside the bundle")
    (bundle / victim).unlink()
    (bundle / victim).symlink_to(outside)
    with pytest.raises(BundleError, match="symlink"):
        prepare_upload(bundle)


def test_prepare_upload_refuses_a_cfg_over_the_64kib_cap(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="65536"):
        _prepared(tmp_path, cfg=b"x" * 65537)


# ---------------------------------------------------------------------- body_chunks


def test_body_chunks_emits_the_adr0205_framing_golden(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    expected = _framed(CFG, WEIGHTS)
    assert b"".join(body_chunks(prepared, chunk_bytes=7)) == expected
    assert b"".join(body_chunks(prepared)) == expected  # default 1 MiB chunking
    assert len(expected) == prepared.size


def test_body_chunks_frames_an_empty_weights_file(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, weights=b"")
    assert b"".join(body_chunks(prepared)) == _framed(CFG, b"")


def test_body_chunks_refuses_a_weights_file_that_drifts_from_its_prepared_size(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    prepared.weights_path.write_bytes(b"short")
    with pytest.raises(BundleError, match="prepared size"):
        list(body_chunks(prepared))


def test_body_chunks_refuses_weights_bytes_that_changed_after_preparation(
    tmp_path: Path,
) -> None:
    """A same-length mutation between prepare and stream is a typed refusal after the last chunk."""
    prepared = _prepared(tmp_path)
    drifted = bytearray(WEIGHTS)
    drifted[0] ^= 0xFF  # same length, different content: only the re-hash can catch this
    prepared.weights_path.write_bytes(bytes(drifted))
    with pytest.raises(BundleError, match="digest_changed"):
        list(body_chunks(prepared))


def test_body_chunks_refuses_a_non_positive_chunk_bytes(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    for bad in (0, -8):
        with pytest.raises(ValueError, match="chunk_bytes"):
            list(body_chunks(prepared, chunk_bytes=bad))


# --------------------------------------------------------------------------- upload


def test_mounted_hotkey_signer_preserves_ingress_wire_signature(tmp_path: Path) -> None:
    from subnet.shared.signing import sign

    prepared = _prepared(tmp_path)
    opener = _FakeOpener(_JsonResponse(201, {"status": "stored"}))
    upload(
        url="https://artifacts.example",
        prepared=prepared,
        slot_id=SLOT_ID,
        signer=lambda message: sign(message, WALLET_KEYPAIR),
        hotkey_ss58=WALLET_HOTKEY,
        nonce=NONCE,
        timestamp=NOW,
        opener=opener,
    )
    [request] = opener.requests
    headers = _wire_headers(request)
    assert verify_artifact_upload(
        ss58_decode(WALLET_HOTKEY),
        bytes.fromhex(headers[HEADER_SIGNATURE.lower()]),
        hotkey_ss58=WALLET_HOTKEY,
        slot_id=SLOT_ID,
        digest=prepared.digest,
        size=prepared.size,
        nonce=NONCE,
        timestamp=NOW,
    )
    assert b"".join(request.data) == _framed(CFG, WEIGHTS)


@pytest.mark.parametrize("both", [True, False])
def test_upload_refuses_ambiguous_or_missing_signer_before_request(tmp_path: Path, both) -> None:
    opener = _FakeOpener(_JsonResponse(201, {"status": "stored"}))
    with pytest.raises(UploadError, match="invalid_field"):
        upload(
            url=BASE_URL,
            prepared=_prepared(tmp_path),
            slot_id=SLOT_ID,
            keypair=WALLET_KEYPAIR if both else None,
            signer=(lambda _: b"x" * 64) if both else None,
            hotkey_ss58=WALLET_HOTKEY,
            nonce=NONCE,
            timestamp=NOW,
            opener=opener,
        )
    assert opener.requests == []


def test_bittensor_upload_requires_hotkey_only_custody_before_wallet_load(monkeypatch, tmp_path):
    monkeypatch.delenv("AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET", raising=False)
    assert (
        upload_module.main(
            [
                "--bundle",
                str(tmp_path),
                "--slot-id",
                SLOT_ID,
                "--url",
                BASE_URL,
                "--wallet",
                "registered-miner",
                "--bittensor-wallet",
            ]
        )
        == 2
    )


def test_upload_sends_exact_headers_body_and_a_verifying_signature(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    ack = {"status": "stored", "digest": prepared.digest, "ref": "aurelius-artifact://x/y"}
    opener = _FakeOpener(_JsonResponse(201, ack))
    result = upload(
        url=BASE_URL,
        prepared=prepared,
        slot_id=SLOT_ID,
        keypair=WALLET_KEYPAIR,
        hotkey_ss58=WALLET_HOTKEY,
        nonce=NONCE,
        timestamp=NOW,
        opener=opener,
    )
    assert result == ack
    [request] = opener.requests
    assert opener.timeouts == [upload_module._OPEN_TIMEOUT_S]
    assert request.get_method() == "PUT"
    assert request.full_url == BASE_URL + UPLOAD_PATH
    headers = _wire_headers(request)
    expected = {
        HEADER_HOTKEY.lower(): WALLET_HOTKEY,
        HEADER_SLOT_ID.lower(): SLOT_ID,
        HEADER_DIGEST.lower(): prepared.digest,
        HEADER_SIZE.lower(): str(prepared.size),
        HEADER_NONCE.lower(): NONCE,
        HEADER_TIMESTAMP.lower(): str(NOW),
        HEADER_SIGNATURE.lower(): headers[HEADER_SIGNATURE.lower()],  # verified below
        "content-length": str(prepared.size),
        "content-type": "application/octet-stream",
    }
    assert headers == expected
    assert b"".join(request.data) == _framed(CFG, WEIGHTS)
    assert verify_artifact_upload(
        ss58_decode(WALLET_HOTKEY),
        bytes.fromhex(headers[HEADER_SIGNATURE.lower()]),
        hotkey_ss58=WALLET_HOTKEY,
        slot_id=SLOT_ID,
        digest=prepared.digest,
        size=prepared.size,
        nonce=NONCE,
        timestamp=NOW,
    )


@pytest.mark.parametrize(
    ("status", "body", "expected_code"),
    [
        (401, b'{"code": "replayed"}', "replayed"),
        (413, b'{"code": "too_large"}', "too_large"),
        (500, b"boom", "http_error"),
    ],
)
def test_upload_maps_non_201_answers_to_typed_upload_errors(
    tmp_path: Path, status: int, body: bytes, expected_code: str
) -> None:
    prepared = _prepared(tmp_path)
    error = urllib.error.HTTPError(BASE_URL + UPLOAD_PATH, status, "refused", {}, io.BytesIO(body))
    with pytest.raises(upload_module.UploadError) as refused:
        upload(
            url=BASE_URL,
            prepared=prepared,
            slot_id=SLOT_ID,
            keypair=WALLET_KEYPAIR,
            hotkey_ss58=WALLET_HOTKEY,
            nonce=NONCE,
            timestamp=NOW,
            opener=_FakeOpener(error=error),
        )
    assert (refused.value.code, refused.value.status) == (expected_code, status)


def test_upload_refuses_a_redirect_without_following_it(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    error = urllib.error.HTTPError(BASE_URL + UPLOAD_PATH, 302, "Found", {}, io.BytesIO(b""))
    with pytest.raises(upload_module.UploadError) as refused:
        upload(
            url=BASE_URL,
            prepared=prepared,
            slot_id=SLOT_ID,
            keypair=WALLET_KEYPAIR,
            hotkey_ss58=WALLET_HOTKEY,
            nonce=NONCE,
            timestamp=NOW,
            opener=_FakeOpener(error=error),
        )
    assert (refused.value.code, refused.value.status) == ("redirect", 302)


def test_upload_refuses_an_injected_opener_with_a_redirect_following_handler(
    tmp_path: Path,
) -> None:
    """P2: an opener that declares a stock redirect handler could ship the bundle elsewhere."""
    prepared = _prepared(tmp_path)
    opener = _FakeOpener(
        _JsonResponse(201, {"status": "stored"}),
        handlers=(urllib.request.HTTPRedirectHandler(),),
    )
    with pytest.raises(upload_module.UploadError) as refused:
        upload(
            url=BASE_URL,
            prepared=prepared,
            slot_id=SLOT_ID,
            keypair=WALLET_KEYPAIR,
            hotkey_ss58=WALLET_HOTKEY,
            nonce=NONCE,
            timestamp=NOW,
            opener=opener,
        )
    assert (refused.value.code, refused.value.status) == ("insecure_opener", 0)
    assert opener.requests == [], "a redirect-capable opener must be refused before any request"


def test_upload_accepts_an_injected_opener_with_only_no_redirect_handlers(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    opener = _FakeOpener(
        _JsonResponse(201, {"status": "stored"}),
        handlers=(upload_module._NoRedirectHandler(),),
    )
    result = upload(
        url=BASE_URL,
        prepared=prepared,
        slot_id=SLOT_ID,
        keypair=WALLET_KEYPAIR,
        hotkey_ss58=WALLET_HOTKEY,
        nonce=NONCE,
        timestamp=NOW,
        opener=opener,
    )
    assert result == {"status": "stored"}
    assert len(opener.requests) == 1


class _TruncatedReadResponse:
    """A 200 whose body read fails mid-stream, as ``http.client.IncompleteRead`` does."""

    def __init__(self, status: int) -> None:
        self.status = status

    def read(self, amount: int = -1) -> bytes:
        del amount
        raise http.client.IncompleteRead(b"partial")

    def close(self) -> None:
        pass


def test_upload_maps_a_mid_read_failure_to_a_typed_network_error(tmp_path: Path) -> None:
    """P1: a broken body read after a 200 must not escape as a raw traceback."""
    prepared = _prepared(tmp_path)
    opener = _FakeOpener(_TruncatedReadResponse(200))
    with pytest.raises(upload_module.UploadError) as refused:
        upload(
            url=BASE_URL,
            prepared=prepared,
            slot_id=SLOT_ID,
            keypair=WALLET_KEYPAIR,
            hotkey_ss58=WALLET_HOTKEY,
            nonce=NONCE,
            timestamp=NOW,
            opener=opener,
        )
    assert (refused.value.code, refused.value.status) == ("network", 200)


def test_no_redirect_handler_converts_redirects_into_http_errors() -> None:
    handler = upload_module._NoRedirectHandler()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://evil.invalid/x") is None


@pytest.mark.parametrize(
    "url",
    [
        "http://ingress.example.invalid",
        "ftp://127.0.0.1",
        "https://user:pass@ingress.example.invalid",
    ],
)
def test_upload_refuses_insecure_urls_before_any_request(tmp_path: Path, url: str) -> None:
    prepared = _prepared(tmp_path)
    opener = _FakeOpener(_JsonResponse(201, {"status": "stored"}))
    with pytest.raises(upload_module.UploadError) as refused:
        upload(
            url=url,
            prepared=prepared,
            slot_id=SLOT_ID,
            keypair=WALLET_KEYPAIR,
            hotkey_ss58=WALLET_HOTKEY,
            nonce=NONCE,
            timestamp=NOW,
            opener=opener,
        )
    assert (refused.value.code, refused.value.status) == ("insecure_url", 0)
    assert opener.requests == [], "an insecure URL must be refused before any request"


@pytest.mark.parametrize(
    "url",
    [
        "https://sn455-artifacts.aureliusaligned.ai",
        "https://sn455-artifacts.aureliusaligned.ai/",
        "http://localhost:8095",
        "http://127.0.0.1:8095",
    ],
)
def test_upload_accepts_https_and_local_http_urls(tmp_path: Path, url: str) -> None:
    prepared = _prepared(tmp_path)
    opener = _FakeOpener(_JsonResponse(201, {"status": "stored"}))
    upload(
        url=url,
        prepared=prepared,
        slot_id=SLOT_ID,
        keypair=WALLET_KEYPAIR,
        hotkey_ss58=WALLET_HOTKEY,
        nonce=NONCE,
        timestamp=NOW,
        opener=opener,
    )
    [request] = opener.requests
    assert request.full_url == url.rstrip("/") + UPLOAD_PATH


def test_production_upload_opener_is_proxyless_and_no_redirect(monkeypatch) -> None:
    captured: list[object] = []
    sentinel = object()

    def build_opener(*handlers: object) -> object:
        captured.extend(handlers)
        return sentinel

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient-proxy.invalid:8080")
    assert upload_module._default_opener() is sentinel
    proxy = next(h for h in captured if isinstance(h, urllib.request.ProxyHandler))
    assert proxy.proxies == {}
    assert any(type(h).__name__ == "_NoRedirectHandler" for h in captured)
    assert any(isinstance(h, urllib.request.HTTPSHandler) for h in captured)


# ------------------------------------------------------------------------ main/CLI


def _run_main_with_fake_opener(
    monkeypatch: pytest.MonkeyPatch, opener: _FakeOpener, bundle: Path, url: str = BASE_URL
) -> int:
    real_upload = upload_module.upload

    def with_fake_opener(**kwargs: object) -> dict[str, object]:
        assert "opener" not in kwargs, "main must let the test inject the transport"
        return real_upload(opener=opener, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(upload_module, "upload", with_fake_opener)
    return upload_module.main(
        [
            "--bundle",
            str(bundle),
            "--slot-id",
            SLOT_ID,
            "--url",
            url,
            "--wallet",
            "tester",
            "--allow-dev-wallet",
        ]
    )


def test_main_uploads_via_the_dev_wallet_and_prints_the_ingress_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared = _prepared(tmp_path)
    assert load_hotkey("tester") == (WALLET_KEYPAIR, WALLET_HOTKEY)
    opener = _FakeOpener(
        _JsonResponse(
            201,
            {
                "status": "stored",
                "digest": prepared.digest,
                "ref": f"aurelius-artifact://{WALLET_HOTKEY}/{prepared.digest}",
            },
        )
    )
    exit_code = _run_main_with_fake_opener(monkeypatch, opener, tmp_path / "bundle")
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "ref": f"aurelius-artifact://{WALLET_HOTKEY}/{prepared.digest}",
        "digest": prepared.digest,
        "status": "stored",
    }
    [request] = opener.requests
    signature = _wire_headers(request)[HEADER_SIGNATURE.lower()]
    output = captured.out + captured.err
    assert SEED.hex() not in output, "the wallet seed must never be printed"
    assert WALLET_KEYPAIR[1].hex() not in output, "the secret key must never be printed"
    assert signature not in output, "the raw signature must never be printed"


def test_main_reports_an_ingress_refusal_on_stderr_with_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refused(**kwargs: object) -> dict[str, object]:
        raise upload_module.UploadError("replayed", 401)

    monkeypatch.setattr(upload_module, "upload", refused)
    exit_code = upload_module.main(
        [
            "--bundle",
            str(_bundle_dir(tmp_path)),
            "--slot-id",
            SLOT_ID,
            "--url",
            BASE_URL,
            "--wallet",
            "tester",
            "--allow-dev-wallet",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "replayed" in captured.err
    assert captured.out == ""


def test_upload_module_never_touches_the_coldkey() -> None:
    source = inspect.getsource(upload_module)
    assert "load_coldkey" not in source
    assert "ColdkeyLocked" not in source


@pytest.mark.parametrize("base", ["https://ingress.example?x=1", "https://ingress.example/#frag"])
def test_upload_url_refuses_query_and_fragment(base: str) -> None:
    with pytest.raises(UploadError) as raised:
        _upload_url(base)
    assert raised.value.code == "insecure_url"


def test_refused_ignores_a_server_code_that_is_not_a_plain_token() -> None:
    body = json.dumps({"code": "\x1b[31mevil\x1b[0m"}).encode()
    error = urllib.error.HTTPError("https://x", 403, "Forbidden", hdrs=None, fp=io.BytesIO(body))  # type: ignore[arg-type]
    assert _refused(error).code == "http_error"


def test_main_refuses_the_dev_wallet_seam_for_a_remote_url_without_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = upload_module.main(
        [
            "--bundle",
            str(_bundle_dir(tmp_path)),
            "--slot-id",
            SLOT_ID,
            "--url",
            "https://ingress.example",
            "--wallet",
            "tester",
        ]
    )
    assert exit_code == 2
    assert "dev_wallet_refused_for_remote_url" in capsys.readouterr().err


def test_refused_tolerates_a_truncated_refusal_body() -> None:
    class _Truncated(io.BytesIO):
        def read(self, size: int = -1) -> bytes:  # noqa: ARG002 -- the fake never yields a body
            raise http.client.IncompleteRead(b"partial")

    error = urllib.error.HTTPError("https://x", 403, "Forbidden", hdrs=None, fp=_Truncated())  # type: ignore[arg-type]
    refused = _refused(error)
    assert (refused.code, refused.status) == ("http_error", 403)


def test_body_chunks_validates_chunk_bytes_eagerly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_bytes"):
        body_chunks(_prepared(tmp_path), chunk_bytes=0)


def test_body_chunks_refuses_a_weights_file_that_grew_after_prepare(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    prepared.weights_path.write_bytes(WEIGHTS + b"\x00")  # longer than the signed size
    with pytest.raises(BundleError, match="grew beyond its prepared size"):
        list(body_chunks(prepared))


class _OversizeAckResponse:
    """A 200 whose body exceeds the bounded ack read by exactly one byte."""

    status = 200

    def read(self, amount: int = -1) -> bytes:
        return b"{" + b" " * 65536  # 65537 bytes: one past the cap, never parsed

    def close(self) -> None:
        pass


def test_upload_refuses_an_oversize_ack_body(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    opener = _FakeOpener(_OversizeAckResponse())  # type: ignore[arg-type]
    with pytest.raises(upload_module.UploadError) as refused:
        upload(
            url=BASE_URL,
            prepared=prepared,
            slot_id=SLOT_ID,
            keypair=WALLET_KEYPAIR,
            hotkey_ss58=WALLET_HOTKEY,
            nonce=NONCE,
            timestamp=NOW,
            opener=opener,
        )
    assert (refused.value.code, refused.value.status) == ("bad_response", 200)
