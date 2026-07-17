"""End-to-end tests for the bittensor-11 HTTP transport (aurelius.transport).

Real cryptography, real FastAPI app, no sockets: the validator-side
query_miner talks to the miner app through httpx's ASGI transport, so the
full sign → verify → blacklist → forward → sign-response → verify-response
path is exercised exactly as it runs in production.
"""

from types import SimpleNamespace

import bittensor as bt
import httpx

from aurelius.protocol import ScenarioConfigSynapse
from aurelius.transport import ROUTE, create_miner_app, query_miner


def _wallet(seed_byte: int) -> SimpleNamespace:
    """Wallet-shaped test double (bittensor's WalletLike is a duck type:
    coldkey/coldkeypub/hotkey suffice) with deterministic in-memory keys."""
    hot = bt.sp_core.Keypair.create_from_seed(bytes([seed_byte]) * 32)
    cold = bt.sp_core.Keypair.create_from_seed(bytes([seed_byte + 1]) * 32)
    return SimpleNamespace(hotkey=hot, coldkey=cold, coldkeypub=cold)


VALIDATOR_WALLET = _wallet(0x11)
MINER_WALLET = _wallet(0x33)
VALIDATOR_HOTKEY = VALIDATOR_WALLET.hotkey.ss58_address
MINER_HOTKEY = MINER_WALLET.hotkey.ss58_address


class FakeMiner:
    """Minimal object satisfying create_miner_app's miner contract."""

    def __init__(self, allowed: set[str] | None = None, wallet=None):
        self.wallet = wallet or MINER_WALLET
        self.hotkey_ss58 = MINER_HOTKEY
        self._allowed = {VALIDATOR_HOTKEY} if allowed is None else allowed

    def blacklist(self, caller: str) -> tuple[bool, str]:
        if caller not in self._allowed:
            return True, f"Hotkey {caller} not allowed"
        return False, ""

    def forward(self, synapse: ScenarioConfigSynapse) -> ScenarioConfigSynapse:
        synapse.scenario_config = {"name": "test-scenario"}
        synapse.work_id = "w" * 64
        synapse.miner_protocol_version = synapse.protocol_version
        return synapse


def _client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app))


def _request() -> ScenarioConfigSynapse:
    return ScenarioConfigSynapse(request_id="req-1", validator_version="0.1.0")


class TestTransportRoundTrip:
    async def test_signed_round_trip(self):
        app = create_miner_app(FakeMiner())
        async with _client_for(app) as http:
            resp = await query_miner(
                http, VALIDATOR_WALLET, "testserver", MINER_HOTKEY, _request(), timeout=5.0
            )
        assert resp.success, resp.error
        assert resp.hotkey == MINER_HOTKEY
        assert resp.synapse.scenario_config == {"name": "test-scenario"}
        # Immutable request fields echo back unchanged
        assert resp.synapse.request_id == "req-1"
        assert resp.synapse.validator_version == "0.1.0"

    async def test_unsigned_request_rejected(self):
        app = create_miner_app(FakeMiner())
        body = _request().model_dump_json().encode()
        async with _client_for(app) as http:
            r = await http.post(
                f"http://testserver{ROUTE}",
                content=body,
                headers={"content-type": "application/json"},
            )
        assert r.status_code == 401

    async def test_tampered_body_rejected(self):
        app = create_miner_app(FakeMiner())
        body = _request().model_dump_json().encode()
        headers = bt.http_auth.sign(
            VALIDATOR_WALLET, method="POST", path=ROUTE, body=body, receiver_ss58=MINER_HOTKEY
        )
        headers["content-type"] = "application/json"
        tampered = body.replace(b"req-1", b"req-2")
        async with _client_for(app) as http:
            r = await http.post(f"http://testserver{ROUTE}", content=tampered, headers=headers)
        assert r.status_code == 401

    async def test_blacklisted_caller_rejected(self):
        app = create_miner_app(FakeMiner(allowed=set()))
        async with _client_for(app) as http:
            resp = await query_miner(
                http, VALIDATOR_WALLET, "testserver", MINER_HOTKEY, _request(), timeout=5.0
            )
        assert not resp.success
        assert "403" in resp.error

    async def test_response_from_wrong_hotkey_rejected(self):
        # Miner signs responses with keys that don't match its chain-published
        # hotkey — the validator must reject, not trust the payload.
        imposter = _wallet(0x55)
        app = create_miner_app(FakeMiner(wallet=imposter))
        async with _client_for(app) as http:
            resp = await query_miner(
                http, VALIDATOR_WALLET, "testserver", MINER_HOTKEY, _request(), timeout=5.0
            )
        assert not resp.success
        assert "signed by" in resp.error

    async def test_transport_error_is_soft_failure(self):
        # No server at all — query_miner must return success=False, not raise.
        async with httpx.AsyncClient() as http:
            resp = await query_miner(
                http, VALIDATOR_WALLET, "127.0.0.1:1", MINER_HOTKEY, _request(), timeout=0.2
            )
        assert not resp.success
        assert resp.error
