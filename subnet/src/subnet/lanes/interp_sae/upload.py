"""Miner-side uploader for the SN455 artifact ingress (ADR-0205 §2): ``PUT /v1/artifacts``.

CONTRACT (capsule header)
- Signs the exact step-3 artifact-upload bytes with the miner HOTKEY
  (`subnet.shared.signing.sign_artifact_upload`); the coldkey is never read here and no key
  material is ever printed or logged.
- Body framing is the S4 preimage without the domain:
  ``[cfg_len u64 BE][cfg.json bytes][weights_len u64 BE][sae_weights.safetensors bytes]`` —
  streamed straight off disk so a multi-GB bundle never sits in memory, with ``Content-Length``
  pinned to the signed ``size`` (framing overhead + cfg + weights). The stream is re-hashed
  while it is framed; the re-hash can only be compared after the LAST chunk, i.e. after the
  body has already left, so a drift raises typed to abort the connection before the ack is
  consumed — the ingress's own digest check refuses the mismatched body regardless.
- An injected transport is vetted before any request: an opener whose declared handlers include
  a redirect-following handler is refused (only a ``_NoRedirectHandler`` may carry redirects).
- Transport fails closed: https only (plain http solely for the localhost/127.0.0.1 testlab),
  no ambient proxy, no redirects (every 3xx is a typed ``UploadError``), and every non-200/201
  answer is a typed ``UploadError(code, status)`` carrying the ingress refusal code.
- An early HTTP reply may close the peer's read side during upload. A broken pipe then
  permits reading that reply, never assuming success. The remaining local body is still
  hashed and size-checked before consuming the acknowledgement.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
import http.client
import json
import os
import re
from pathlib import Path
import secrets
import ssl
import stat
import sys
import time
from typing import Final
import urllib.error
import urllib.parse
import urllib.request

from subnet.lanes.interp_sae.bundle_digest import (
    CFG_FILENAME,
    MAX_CFG_BYTES,
    S4BundleHasher,
    WEIGHTS_FILENAME,
    digests_equal,
)
from subnet.miner.wallet import load_hotkey
from subnet.shared.signing import artifact_upload_signed_bytes, sign_artifact_upload

UPLOAD_PATH: Final = "/v1/artifacts"
DEFAULT_CHUNK_BYTES: Final = 1 << 20  # 1 MiB — the streamed weights read size
HEADER_HOTKEY: Final = "X-Aurelius-Hotkey"
HEADER_SLOT_ID: Final = "X-Aurelius-Slot-Id"
HEADER_DIGEST: Final = "X-Aurelius-Digest"
HEADER_SIZE: Final = "X-Aurelius-Size"
HEADER_NONCE: Final = "X-Aurelius-Nonce"
HEADER_TIMESTAMP: Final = "X-Aurelius-Timestamp"
HEADER_SIGNATURE: Final = "X-Aurelius-Signature"
_FRAMING_OVERHEAD_BYTES: Final = 16  # two u64 big-endian length prefixes
_HASH_CHUNK_BYTES: Final = 1 << 20
_OPEN_TIMEOUT_S: Final = 600.0
_MAX_RESPONSE_BYTES: Final = 65536  # the ingress ack is a tiny JSON object
_LOCAL_HTTP_HOSTS: Final = frozenset({"localhost", "127.0.0.1"})  # testlab only


class BundleError(ValueError):
    """The local bundle directory is not an exact, plain two-file S4 bundle."""


class UploadError(Exception):
    """The ingress refused the upload, or the transport failed; never carries key material."""

    def __init__(self, code: str, status: int) -> None:
        super().__init__(code, status)
        self.code = code
        self.status = status

    def __str__(self) -> str:
        return f"{self.code} (HTTP status {self.status})"


@dataclass(frozen=True, slots=True)
class PreparedBundle:
    """Everything one upload needs, prepared once: cfg bytes, weights path, digest, framed size."""

    cfg: bytes
    weights_path: Path
    weights_size: int
    digest: str
    size: int


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Turn every redirect into ``HTTPError`` so ``upload`` can refuse it typed."""

    def redirect_request(  # noqa: PLR0913 -- the base-class override signature
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _require_plain_regular_file(path: Path) -> None:
    if path.is_symlink():
        raise BundleError(f"bundle file must not be a symlink: {path.name}")
    if not stat.S_ISREG(path.stat().st_mode):
        raise BundleError(f"bundle file must be a regular file: {path.name}")


def prepare_upload(bundle_dir: Path) -> PreparedBundle:
    """Authenticate the exact two-file S4 bundle and compute its digest and framed size.

    Refuses anything but a directory holding exactly ``cfg.json`` and ``sae_weights.safetensors``
    as plain regular files (no symlinks), a cfg above the 64 KiB S4 cap, and a weights file whose
    size drifts while it is hashed. The digest is the S4 bundle digest, streamed — never the
    whole weights file in memory.
    """
    if not bundle_dir.is_dir():
        raise BundleError(f"not an existing bundle directory: {bundle_dir}")
    entries = sorted(entry.name for entry in bundle_dir.iterdir())
    expected = sorted((CFG_FILENAME, WEIGHTS_FILENAME))
    if entries != expected:
        raise BundleError(
            f"bundle must contain exactly {CFG_FILENAME} and {WEIGHTS_FILENAME}, got {entries}"
        )
    cfg_path = bundle_dir / CFG_FILENAME
    weights_path = bundle_dir / WEIGHTS_FILENAME
    _require_plain_regular_file(cfg_path)
    _require_plain_regular_file(weights_path)
    with cfg_path.open("rb") as handle:
        cfg = handle.read(MAX_CFG_BYTES + 1)
    if len(cfg) > MAX_CFG_BYTES:
        raise BundleError(f"cfg.json exceeds the {MAX_CFG_BYTES}-byte S4 cap")
    if not cfg:
        raise BundleError("cfg.json is empty")
    weights_size = weights_path.stat().st_size
    try:
        hasher = S4BundleHasher(cfg, weights_size)
        with weights_path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    except ValueError as error:
        raise BundleError(f"bundle fails the S4 digest contract: {error}") from None
    return PreparedBundle(
        cfg=cfg,
        weights_path=weights_path,
        weights_size=weights_size,
        digest=digest,
        size=_FRAMING_OVERHEAD_BYTES + len(cfg) + weights_size,
    )


def body_chunks(
    prepared: PreparedBundle, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> Iterator[bytes]:
    """The ADR-0205 §2 body: cfg length + cfg + weights length + weights, streamed off disk.

    Raises ``BundleError`` the moment the weights file no longer totals exactly its prepared
    size, so a shrinking or growing artifact can never produce a body that disagrees with the
    signed ``Content-Length`` (earlier chunks have already been sent by then; the raise aborts
    the connection). While streaming, the same bytes are absorbed into a second
    ``S4BundleHasher``: a same-length mutation between :func:`prepare_upload` and the stream is
    detected after the final chunk — necessarily after the whole body has left — and raised
    typed (``digest_changed``) so the client never consumes an ack for bytes it did not sign;
    the ingress refuses the mismatched body on its own digest check regardless.
    ``chunk_bytes`` is validated eagerly, at call time.
    """
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
        raise ValueError("chunk_bytes must be a positive integer")
    return _body_chunks(prepared, chunk_bytes)


def _body_chunks(prepared: PreparedBundle, chunk_bytes: int) -> Iterator[bytes]:
    hasher = S4BundleHasher(prepared.cfg, prepared.weights_size)
    yield len(prepared.cfg).to_bytes(8, "big")
    yield prepared.cfg
    yield prepared.weights_size.to_bytes(8, "big")
    read_total = 0
    with prepared.weights_path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            read_total += len(chunk)
            if read_total > prepared.weights_size:
                raise BundleError("weights file grew beyond its prepared size")
            hasher.update(chunk)
            yield chunk
    if read_total != prepared.weights_size:
        raise BundleError("weights file no longer matches its prepared size")
    if not digests_equal(hasher.hexdigest(), prepared.digest):
        raise BundleError("digest_changed")


def _is_local_url(base_url: str) -> bool:
    """True only for the localhost/127.0.0.1 testlab targets (the dev-wallet seam's home)."""
    host = urllib.parse.urlsplit(base_url).hostname
    return host is not None and host.lower() in _LOCAL_HTTP_HOSTS


def _upload_url(base_url: str) -> str:
    """The exact ``PUT`` target: https, or plain http solely for the localhost testlab."""
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname
    https = parsed.scheme == "https"
    local_http = parsed.scheme == "http" and host is not None and host.lower() in _LOCAL_HTTP_HOSTS
    if not https and not local_http:
        raise UploadError("insecure_url", 0)
    if parsed.username is not None or parsed.password is not None:
        raise UploadError("insecure_url", 0)
    if not host:
        raise UploadError("insecure_url", 0)
    if parsed.query or parsed.fragment:
        raise UploadError("insecure_url", 0)  # a query/fragment would mangle the exact target
    return base_url.rstrip("/") + UPLOAD_PATH


class _EarlyResponseRequest:
    """Keep http.client's response path available when a peer answers before the body ends."""

    def request(self, method, url, body=None, headers=None, *, encode_chunked=False):
        try:
            super().request(method, url, body, headers or {}, encode_chunked=encode_chunked)
        except BrokenPipeError:
            if not isinstance(body, Iterator):
                raise
            # Do not send more bytes, but finish the iterator's size/digest checks. A drift
            # remains a BundleError; absent/malformed HTTP replies still fail in getresponse.
            for _chunk in body:
                pass


class _UploadHTTPConnection(_EarlyResponseRequest, http.client.HTTPConnection):
    pass


class _UploadHTTPSConnection(_EarlyResponseRequest, http.client.HTTPSConnection):
    pass


class _UploadHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, request):
        return self.do_open(_UploadHTTPConnection, request)


class _UploadHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(_UploadHTTPSConnection, request, context=self._context)


def _default_opener() -> urllib.request.OpenerDirector:
    """Proxyless, redirect-refusing opener with an explicitly constructed TLS context."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    context.load_default_certs(ssl.Purpose.SERVER_AUTH)
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
        _UploadHTTPHandler(),
        _UploadHTTPSHandler(context=context),
    )


def _refuse_redirect_capable_opener(opener: urllib.request.OpenerDirector) -> None:
    """Refuse an injected opener whose declared handlers could follow a redirect.

    ``OpenerDirector.handlers`` is the declared handler list; a stock ``HTTPRedirectHandler``
    in it could silently ship the bundle to a 3xx target, so the only accepted redirect handler
    is our own refusing ``_NoRedirectHandler``. An opener without a ``handlers`` attribute
    declares no redirect handler and is accepted.
    """
    handlers = getattr(opener, "handlers", ())
    for handler in handlers:
        if isinstance(handler, urllib.request.HTTPRedirectHandler) and not isinstance(
            handler, _NoRedirectHandler
        ):
            raise UploadError("insecure_opener", 0)


_REFUSAL_CODE = re.compile(r"[a-z0-9_]{1,64}")


def _refused(error: urllib.error.HTTPError) -> UploadError:
    """Map one HTTP refusal to ``UploadError``; the ingress ``code`` when it sent one."""
    status = int(error.code)
    if 300 <= status < 400:
        return UploadError("redirect", status)
    code = "http_error"
    body = b""
    if error.fp is not None:
        try:
            body = error.read(_MAX_RESPONSE_BYTES + 1)
        except (http.client.HTTPException, OSError, ValueError):
            body = b""  # a refusal body is best-effort (even truncated); the status is the contract
        finally:
            with contextlib.suppress(Exception):
                error.close()  # never leak the refusal socket, even on a one-shot CLI
    if len(body) <= _MAX_RESPONSE_BYTES:
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        if (
            isinstance(parsed, dict)
            and isinstance(parsed.get("code"), str)
            and _REFUSAL_CODE.fullmatch(parsed["code"])
        ):
            code = parsed["code"]  # server-controlled: only a plain token ever reaches the terminal
    return UploadError(code, status)


def upload(
    *,
    url: str,
    prepared: PreparedBundle,
    slot_id: str,
    keypair: tuple[bytes, bytes] | None = None,
    hotkey_ss58: str,
    nonce: str,
    timestamp: int,
    signer: Callable[[bytes], bytes] | None = None,
    opener: urllib.request.OpenerDirector | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Sign the step-3 bytes and stream one framed bundle to ``PUT {url}/v1/artifacts``.

    Returns the ingress JSON on 200/201; raises ``UploadError(code, status)`` otherwise —
    including for insecure URLs (before any request), redirect attempts, and transport
    failures. The signature covers exactly ``hotkey|slot_id|digest|size|timestamp|nonce`` under
    the ``aurelius/artifact_upload/v1`` domain, so the ingress verifier (step 5) accepts the
    request unchanged.
    """
    target = _upload_url(url)
    if opener is not None:
        _refuse_redirect_capable_opener(opener)  # before signing: an unsafe transport refuses first
    try:
        fields = dict(
            hotkey_ss58=hotkey_ss58,
            slot_id=slot_id,
            digest=prepared.digest,
            size=prepared.size,
            nonce=nonce,
            timestamp=timestamp,
        )
        if (keypair is None) == (signer is None):
            raise ValueError("exactly one upload signing source is required")
        signature = (
            signer(artifact_upload_signed_bytes(**fields))
            if signer is not None
            else sign_artifact_upload(keypair, **fields)
        )
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise ValueError("upload signer must return a 64-byte signature")
    except ValueError as error:  # a field the signing grammar refuses is a typed refusal too
        raise UploadError("invalid_field", 0) from error
    headers = {
        HEADER_HOTKEY: hotkey_ss58,
        HEADER_SLOT_ID: slot_id,
        HEADER_DIGEST: prepared.digest,
        HEADER_SIZE: str(prepared.size),
        HEADER_NONCE: nonce,
        HEADER_TIMESTAMP: str(timestamp),
        HEADER_SIGNATURE: signature.hex(),
        "Content-Length": str(prepared.size),
        "Content-Type": "application/octet-stream",
    }

    def stream():
        sent = 0
        for chunk in body_chunks(prepared):
            yield chunk
            sent += len(chunk)
            if progress is not None:
                progress(sent, prepared.size)

    request = urllib.request.Request(
        target,
        data=stream(),
        headers=headers,
        method="PUT",
    )
    director = opener if opener is not None else _default_opener()
    try:
        response = director.open(request, timeout=_OPEN_TIMEOUT_S)
    except urllib.error.HTTPError as error:
        raise _refused(error) from None
    except BundleError:
        raise  # a weights file that drifted mid-stream is a local bundle fault, not "network"
    except (http.client.HTTPException, OSError, ValueError) as error:
        # URLError and TimeoutError are OSError subclasses: this tuple is the whole surface.
        raise UploadError("network", 0) from error
    try:
        status = int(response.status)
        try:
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
        except (http.client.HTTPException, OSError, ValueError) as error:
            # a truncated or broken body after a 200 is a transport fault, not a bad ack
            raise UploadError("network", status) from error
    finally:
        response.close()
    if status not in (200, 201):
        raise UploadError("unexpected_status", status)
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise UploadError("bad_response", status)
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise UploadError("bad_response", status) from None
    if not isinstance(parsed, dict):
        raise UploadError("bad_response", status)
    return parsed


def main(argv: list[str] | None = None) -> int:
    """``aurelius-sae-upload --bundle DIR --slot-id S --url URL --wallet NAME`` (ADR-0205 §2)."""
    parser = argparse.ArgumentParser(
        prog="aurelius-sae-upload",
        description="Upload a staged S4 SAE-Lens bundle to the artifact ingress (ADR-0205 §2).",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="directory with exactly cfg.json and sae_weights.safetensors",
    )
    parser.add_argument("--slot-id", required=True, help="ingress-admitted slot identifier")
    parser.add_argument(
        "--url",
        required=True,
        help="ingress base URL (https; plain http only for localhost/127.0.0.1 testlab use)",
    )
    parser.add_argument(
        "--wallet",
        required=True,
        help=(
            "miner wallet name (hotkey only; never the coldkey). The current wallet seam derives the dev/testlab hotkey from the name — not encrypted keyfile custody"
        ),
    )
    parser.add_argument(
        "--allow-dev-wallet",
        action="store_true",
        help=(
            "required for any non-localhost --url: acknowledges that the wallet seam derives the "
            "hotkey from the wallet NAME (testlab custody), not from an encrypted keyfile"
        ),
    )
    parser.add_argument(
        "--bittensor-wallet",
        action="store_true",
        help="sign with the mounted Bittensor hotkey; requires the hotkey-only wallet guard",
    )
    parser.add_argument("--progress", action="store_true", help="show bytes sent on stderr")
    parser.add_argument(
        "--journal", help="save a verified upload acknowledgement in the local journal"
    )
    args = parser.parse_args(argv)
    try:
        signer = None
        keypair = None
        if args.bittensor_wallet:
            if args.allow_dev_wallet:
                raise UploadError("ambiguous_wallet_source", 0)
            if os.environ.get("AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET") != "true":
                raise UploadError("hotkey_only_wallet_required", 0)
            from subnet.miner.serve import load_bittensor_wallet

            os.environ["AURELIUS_BT_WALLET"] = args.wallet
            wallet = load_bittensor_wallet()
            hotkey_ss58 = str(wallet.hotkey.ss58_address)
            signer = wallet.hotkey.sign
        else:
            if not _is_local_url(args.url) and not args.allow_dev_wallet:
                raise UploadError("dev_wallet_refused_for_remote_url", 0)
            keypair, hotkey_ss58 = load_hotkey(args.wallet)
        prepared = prepare_upload(Path(args.bundle))
        result = upload(
            url=args.url,
            prepared=prepared,
            slot_id=args.slot_id,
            keypair=keypair,
            hotkey_ss58=hotkey_ss58,
            nonce=secrets.token_hex(32),
            timestamp=int(time.time()),
            signer=signer,
            progress=(
                lambda sent, total: print(
                    f"\rUpload {sent:,}/{total:,} bytes", end="", file=sys.stderr
                )
            )
            if args.progress
            else None,
        )
        if args.progress:
            print(file=sys.stderr)
        from subnet.shared.artifact_ingress_ref import parse_aurelius_artifact_ref

        parsed_ref = parse_aurelius_artifact_ref(result.get("ref"))
        if (parsed_ref.hotkey, parsed_ref.digest) != (hotkey_ss58, prepared.digest) or result.get(
            "digest"
        ) != prepared.digest:
            raise UploadError("bad_response", 200)
        if args.journal:
            from subnet.miner.journal import AttemptJournal

            AttemptJournal(args.journal).upload(
                hotkey_ss58,
                args.slot_id,
                prepared.digest,
                {key: result.get(key) for key in ("ref", "digest", "status")},
            )
    except (UploadError, ValueError, OSError) as exc:
        print(f"aurelius-sae-upload failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ref": result.get("ref"),
                "digest": result.get("digest"),
                "status": result.get("status"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
