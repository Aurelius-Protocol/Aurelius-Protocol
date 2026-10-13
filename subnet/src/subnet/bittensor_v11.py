# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Small, explicit compatibility boundary for the pinned Bittensor v11 SDK.

The v11 release is a rewrite, not a continuation of the v10 neuron stack.  Keeping the few
construction/shape rules used by both miners and validators here prevents permissive fallbacks
(``bt.wallet`` vs ``bt.Wallet``, legacy ``Subtensor`` methods, AxonInfo objects) from silently
reappearing in individual call sites.

This module deliberately imports Bittensor lazily.  The ordinary hermetic HTTP lane and the
default workspace environment remain SDK-free; only the ``subnet[bt]`` runtime profile loads it.
"""

from __future__ import annotations
import ipaddress
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from urllib.parse import urlsplit

BITTENSOR_SDK_VERSION = "11.0.0"


BITTENSOR_V11_CALLABLE_SURFACES = (
    "AddStake",
    "AddStake.build",
    "AddStakeLimit",
    "AddStakeLimit.build",
    "Balance",
    "Balance.from_rao",
    "ExtrinsicResult",
    "Intent",
    "Intent.build",
    "Policy",
    "ServeAxon",
    "ServeAxon.build",
    "SetWeights",
    "SetWeights.build",
    "Snapshot",
    "Snapshot.at",
    "Snapshot.block_info",
    "Snapshot.constant",
    "Snapshot.query",
    "Snapshot.read",
    "Snapshot.runtime",
    "Subtensor",
    "Subtensor.at",
    "Subtensor.block_info",
    "Subtensor.close",
    "Subtensor.constant",
    "Subtensor.decode_scale",
    "Subtensor.plan",
    "Subtensor.prepare_call",
    "Subtensor.query",
    "Subtensor.read",
    "Subtensor.runtime",
    "Subtensor.submit_signature",
    "Transfer",
    "Transfer.build",
    "TransferStake",
    "TransferStake.build",
    "UnsignedExtrinsic",
    "UnsignedExtrinsic.from_dict",
    "UnsignedExtrinsic.to_dict",
    "Wallet",
    "WalletSigner",
    "WalletSigner.sign",
    "calls.AdminUtils.sudo_set_commit_reveal_weights_enabled",
    "calls.SubtensorModule.register_limit",
    "http_auth.AuthError",
    "http_auth.InMemoryNonceStore",
    "http_auth.sign",
    "http_auth.verify",
)


BITTENSOR_V11_CLIENT_CALLABLE_SURFACES = (
    "balances.get",
    "staking.get",
    "subnets.metagraph",
)


BITTENSOR_V11_DESCRIPTOR_SURFACES = (
    "constants.SubtensorModule.InitialMinStake",
    "http_auth.HEADER_HOTKEY",
    "http_auth.HEADER_NONCE",
    "http_auth.HEADER_RECEIVER",
    "http_auth.HEADER_SIGNATURE",
    "runtime_api.StakeInfoRuntimeApi.get_stake_availability_for_coldkeys",
    "runtime_api.SwapRuntimeApi.current_alpha_price",
    "runtime_api.SwapRuntimeApi.sim_swap_alpha_for_tao",
    "storage.SubtensorModule.Active",
    "storage.SubtensorModule.Axons",
    "storage.SubtensorModule.Burn",
    "storage.SubtensorModule.CommitRevealWeightsEnabled",
    "storage.SubtensorModule.Keys",
    "storage.SubtensorModule.LastUpdate",
    "storage.SubtensorModule.MaxAllowedUids",
    "storage.SubtensorModule.MaxAllowedValidators",
    "storage.SubtensorModule.MaxWeightsLimit",
    "storage.SubtensorModule.MinAllowedWeights",
    "storage.SubtensorModule.NetworksAdded",
    "storage.SubtensorModule.Owner",
    "storage.SubtensorModule.StakeThreshold",
    "storage.SubtensorModule.SubnetOwner",
    "storage.SubtensorModule.SubnetworkN",
    "storage.SubtensorModule.Tempo",
    "storage.SubtensorModule.TotalHotkeyAlpha",
    "storage.SubtensorModule.Uids",
    "storage.SubtensorModule.ValidatorPermit",
    "storage.SubtensorModule.Weights",
    "storage.SubtensorModule.WeightsSetRateLimit",
    "storage.SubtensorModule.WeightsVersionKey",
    "storage.System.Account",
    "storage.System.Events",
)


class BittensorV11Error(RuntimeError):
    """The configured SDK or a v11 response does not satisfy Aurelius' pinned contract."""


def validate_subtensor_endpoint(endpoint: object) -> str:
    """Return an exact WS endpoint or refuse before SDK/network access.

    The configured endpoint is an authority boundary: credentials, paths, query strings and
    fragments could make two callers interpret the same setting differently. The mainnet operator expressly
    selects credential-bearing TLS OnFinality paths/queries; their full endpoint is preserved.
    Public SDK fallback is disabled separately by :func:`open_subtensor`.
    """

    if (
        not isinstance(endpoint, str)
        or not endpoint
        or not endpoint.isascii()
        or endpoint != endpoint.strip()
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in endpoint)
        or any(character.isspace() for character in endpoint)
        or len(endpoint) > 2048
        or any(character in endpoint for character in ("\\", "#"))
        or not endpoint.startswith(("ws://", "wss://"))
    ):
        raise BittensorV11Error(
            "Subtensor endpoint must be an exact printable-ASCII ws:// or wss:// authority"
        )
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise BittensorV11Error("Subtensor endpoint has a malformed authority or port") from exc
    hostname = parsed.hostname
    onfinality = (
        parsed.scheme == "wss"
        and hostname is not None
        and (hostname == "onfinality.io" or hostname.endswith(".onfinality.io"))
        and port in (None, 443)
    )
    valid_hostname = False
    if hostname is not None:
        try:
            ipaddress.ip_address(hostname)
            valid_hostname = True
        except ValueError:
            labels = hostname.split(".")
            valid_hostname = (
                len(hostname) <= 253
                and not hostname.isdecimal()
                and not all(character.isdecimal() or character == "." for character in hostname)
                and all(
                    1 <= len(label) <= 63
                    and label[0].isalnum()
                    and label[-1].isalnum()
                    and all(character.isalnum() or character == "-" for character in label)
                    for label in labels
                )
            )
    if (
        parsed.scheme not in {"ws", "wss"}
        or not valid_hostname
        or parsed.username is not None
        or parsed.password is not None
        or not onfinality
        and (parsed.path not in {"", "/"} or parsed.query or "%" in endpoint or "?" in endpoint)
        or parsed.fragment
        or parsed.netloc.endswith(":")
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise BittensorV11Error(
            "Subtensor endpoint must contain only one valid WS authority with no credentials, "
            "path, query, or fragment"
        )
    return endpoint


class DurableNonceStore:
    """Crash-safe btauth replay state for a single host or shared local volume.

    Bittensor's default store is intentionally process-local.  A miner restart inside the short
    request-freshness window would therefore forget an already accepted request.  This SQLite
    implementation commits the ``(sender, nonce)`` pair before candidate extraction continues;
    ``BEGIN IMMEDIATE`` plus the primary key also makes separate worker processes sharing this
    exact local file atomic.  Multi-host replicas must use an equivalently atomic shared
    ``NonceStore`` (for example Redis SET-NX-with-expiry), not a network-mounted SQLite file.
    """

    def __init__(self, path: str | Path, *, retention: float = 60.0, clock_ns=None):
        if (
            isinstance(retention, bool)
            or not isinstance(retention, (int, float))
            or retention < 12.0
            or retention > 86_400.0
        ):
            raise BittensorV11Error("btauth nonce retention must be between 12 and 86400 seconds")
        self.retention = float(retention)
        self._clock_ns = time.time_ns if clock_ns is None else clock_ns
        self._lock = threading.Lock()
        self.path = Path(path)
        if not self.path.name or str(self.path) == ":memory:":
            raise BittensorV11Error("durable btauth nonce store requires a filesystem path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            parent_info = self.path.parent.lstat()
            if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
                raise BittensorV11Error("btauth nonce-store parent must be a real directory")
            if self.path.exists() or self.path.is_symlink():
                info = self.path.lstat()
                if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    raise BittensorV11Error("btauth nonce store must be a regular file")
        except OSError as exc:
            raise BittensorV11Error(f"cannot inspect btauth nonce-store path: {exc}") from exc
        try:
            self._db = sqlite3.connect(
                self.path,
                timeout=5.0,
                isolation_level=None,
                check_same_thread=False,
            )
            self._db.execute("PRAGMA busy_timeout = 5000")
            self._db.execute("PRAGMA synchronous = FULL")
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS accepted_btauth_nonces (
                    hotkey_ss58 TEXT NOT NULL,
                    nonce_ns INTEGER NOT NULL,
                    expires_at_ns INTEGER NOT NULL CHECK (expires_at_ns > 0),
                    PRIMARY KEY (hotkey_ss58, nonce_ns)
                ) WITHOUT ROWID
                """
            )
            os.chmod(self.path, 0o600)
        except (OSError, sqlite3.Error) as exc:
            raise BittensorV11Error(f"cannot initialize durable btauth nonce store: {exc}") from exc

    def check_and_store(self, hotkey_ss58: str, nonce_ns: int) -> bool:
        now_ns = int(self._clock_ns())
        expires_at_ns = now_ns + int(self.retention * 1_000_000_000)
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute(
                    "DELETE FROM accepted_btauth_nonces WHERE expires_at_ns <= ?", (now_ns,)
                )
                cursor = self._db.execute(
                    """
                    INSERT OR IGNORE INTO accepted_btauth_nonces
                        (hotkey_ss58, nonce_ns, expires_at_ns)
                    VALUES (?, ?, ?)
                    """,
                    (hotkey_ss58, int(nonce_ns), expires_at_ns),
                )
                accepted = cursor.rowcount == 1
                self._db.execute("COMMIT")
                return accepted
            except (OverflowError, sqlite3.Error, ValueError) as exc:
                try:
                    self._db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise BittensorV11Error(f"durable btauth nonce update failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            self._db.close()


@dataclass(frozen=True)
class FinalizedChainSnapshot:
    head_hash: str
    block_number: int
    genesis_hash: str
    chain: str
    runtime_spec_version: int
    transaction_version: int


class RpcUnavailable(BittensorV11Error):
    """Finalized RPC transport exhausted its bounded reconnect; no authority was observed."""


class FinalizedRpc:
    """Minimal public JSON-RPC reader for evidence that must be explicitly finalized.

    Bittensor v11 intentionally hides the old public ``Subtensor.substrate`` escape hatch, while
    its default typed reads target the current head.  Aurelius' mutation guards still need a
    finalized hash/height, genesis and runtime tuple.  Keeping this read-only client independent
    of SDK internals preserves those evidence semantics across SDK releases.
    """

    def __init__(self, endpoint: str, timeout: float = 10.0):
        self.endpoint = validate_subtensor_endpoint(endpoint)
        self.timeout = timeout
        self._ws = None
        self._request_id = 0

    def _connect(self) -> None:
        from websocket import create_connection

        self._ws = create_connection(self.endpoint, timeout=self.timeout)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001 — cleanup cannot mask rejected RPC authority
                pass
            finally:
                self._ws = None

    def call(self, method: str, params: list) -> object:
        from websocket import (
            WebSocketAddressException,
            WebSocketBadStatusException,
            WebSocketConnectionClosedException,
            WebSocketProxyException,
            WebSocketTimeoutException,
        )

        last = None
        for attempt in range(2):
            try:
                if self._ws is None:
                    self._connect()
                self._request_id += 1
                self._ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": self._request_id,
                            "method": method,
                            "params": params,
                        },
                        separators=(",", ":"),
                    )
                )
                raw = self._ws.recv()
                try:
                    response = json.loads(raw)
                except (ValueError, TypeError) as exc:
                    raise BittensorV11Error(f"{method} returned malformed JSON") from exc
                if (
                    not isinstance(response, dict)
                    or response.get("jsonrpc") != "2.0"
                    or type(response.get("id")) is not int
                    or response["id"] != self._request_id
                    or response.get("error") is not None
                ):
                    raise BittensorV11Error(f"{method} failed: {response!r}")
                if "result" not in response:
                    raise BittensorV11Error(f"{method} returned no result")
                return response["result"]
            except BittensorV11Error:
                raise
            except (
                OSError,
                WebSocketAddressException,
                WebSocketBadStatusException,
                WebSocketConnectionClosedException,
                WebSocketProxyException,
                WebSocketTimeoutException,
            ) as exc:
                # Only transport failures qualify for a last-good discovery fallback.
                last = exc
                self.close()
                if attempt == 1:
                    break
        raise RpcUnavailable(f"{method} RPC failed: {last}") from last

    def snapshot(self) -> FinalizedChainSnapshot:
        head = self.call("chain_getFinalizedHead", [])
        if not isinstance(head, str) or not head.startswith("0x") or len(head) != 66:
            raise BittensorV11Error(f"invalid finalized head {head!r}")
        header = self.call("chain_getHeader", [head])
        runtime = self.call("state_getRuntimeVersion", [head])
        genesis = self.call("chain_getBlockHash", [0])
        chain = self.call("system_chain", [])
        try:
            block_number = int(header["number"], 16)
            runtime_spec_version = int(runtime["specVersion"])
            transaction_version = int(runtime["transactionVersion"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BittensorV11Error("finalized header/runtime response is malformed") from exc
        if not isinstance(genesis, str) or not isinstance(chain, str):
            raise BittensorV11Error("chain identity response is malformed")
        return FinalizedChainSnapshot(
            head_hash=head.lower(),
            block_number=block_number,
            genesis_hash=genesis.lower(),
            chain=chain,
            runtime_spec_version=runtime_spec_version,
            transaction_version=transaction_version,
        )


def _sdk_surface(bt: object, path: str) -> object | None:
    value = bt
    for component in path.split("."):
        try:
            value = getattr(value, component)
        except (AttributeError, RuntimeError):
            return None
    return value


def _probe_cold_bittensor_v11_client(bt: object) -> None:
    """Probe SDK11's instance-only namespaces without opening a socket or a wallet.

    In the pinned v11 client contract, construction is explicitly cold: the connection opens on
    the first read/``connect()``, not in ``Subtensor.__init__``.  An unroutable loopback endpoint
    and empty fallback/archive sets make that contract explicit here.  This probe is only needed
    for the real imported runtime; injected unit doubles are covered by their own narrow tests.
    """

    try:
        client = bt.Subtensor(
            network="ws://127.0.0.1:1",
            fallback_endpoints=[],
            archive_endpoints=[],
        )
    except Exception as exc:  # noqa: BLE001 - normalize any constructor shape drift
        raise BittensorV11Error(
            f"Bittensor v11 cold client construction failed before wallet/network use: {exc}"
        ) from exc
    missing = [
        path
        for path in BITTENSOR_V11_CLIENT_CALLABLE_SURFACES
        if not callable(_sdk_surface(client, path))
    ]
    if missing:
        raise BittensorV11Error(
            "Bittensor v11 cold client namespaces are incomplete: "
            f"missing/non-callable {sorted(missing)!r}"
        )


def require_bittensor_v11(bt_module=None):
    """Return the SDK only when it is the exact reviewed v11 build.

    ``uv.lock`` already pins the package.  The runtime check catches images or operator-created
    environments that bypass the lock, before a wallet is opened or a chain mutation is built.
    """

    imported_runtime = bt_module is None
    bt = import_module("bittensor") if imported_runtime else bt_module
    version = getattr(bt, "__version__", None)
    if version != BITTENSOR_SDK_VERSION:
        raise BittensorV11Error(
            f"Aurelius requires bittensor=={BITTENSOR_SDK_VERSION}; loaded {version!r}"
        )
    missing = [path for path in BITTENSOR_V11_DESCRIPTOR_SURFACES if _sdk_surface(bt, path) is None]
    missing.extend(
        path for path in BITTENSOR_V11_CALLABLE_SURFACES if not callable(_sdk_surface(bt, path))
    )
    if missing:
        raise BittensorV11Error(
            "Bittensor v11 surface is incomplete or has the wrong shape: "
            f"missing/non-callable {sorted(missing)!r}"
        )
    if imported_runtime:
        _probe_cold_bittensor_v11_client(bt)
    return bt


def load_wallet(bt, *, name: str, hotkey: str, path: str | None = None):
    """Load the native v11 wallet, omitting ``path`` when it is not configured.

    In v11 ``bittensor.wallet`` is a module (not the old lowercase constructor), and explicitly
    passing ``None`` as ``path`` reaches ``Path(None)``.  Both old compatibility idioms therefore
    fail and are intentionally absent here.
    """

    kwargs = {"name": name, "hotkey": hotkey}
    if path:
        kwargs["path"] = path
    return bt.Wallet(**kwargs)


def open_subtensor(bt, endpoint: str):
    """Construct the v11 blocking/async-unified client against an explicit endpoint."""

    return bt.Subtensor(
        network=validate_subtensor_endpoint(endpoint),
        fallback_endpoints=[],
        archive_endpoints=[],
    )


@dataclass(frozen=True)
class PublishedEndpoint:
    ip: str
    port: int

    @property
    def url(self) -> str:
        host = f"[{self.ip}]" if ":" in self.ip else self.ip
        return f"http://{host}:{self.port}"


def parse_published_endpoint(value: object) -> PublishedEndpoint | None:
    """Parse v11 ``MetagraphNeuron.axon`` (``ip:port`` or ``[ipv6]:port``).

    A missing/zero endpoint means unserved.  Malformed non-empty data fails loudly so discovery
    cannot quietly route around an SDK/runtime shape drift.
    """

    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise BittensorV11Error("v11 metagraph axon endpoint must be a string or None")
    try:
        parsed = urlsplit(f"http://{value}")
        ip = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise BittensorV11Error(f"invalid v11 metagraph axon endpoint {value!r}") from exc
    if not ip or port is None or not 1 <= port <= 65_535 or parsed.path not in {"", "/"}:
        raise BittensorV11Error(f"invalid v11 metagraph axon endpoint {value!r}")
    return PublishedEndpoint(ip=ip, port=port)
