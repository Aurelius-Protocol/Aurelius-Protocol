"""Subnet HTTP transport replacing the removed bittensor axon/dendrite stack.

bittensor 11 has no neuron networking layer; its guidance is to keep your own
HTTP layer, authenticate with `bittensor.http_auth`, and publish endpoints
on-chain with the ServeAxon intent. This module implements both sides:

- Validator side: `query_miner` POSTs a `ScenarioConfigSynapse` (JSON) to a
  miner endpoint with http_auth-signed request headers, and verifies the
  miner's http_auth signature over the response body. Both directions are
  therefore authenticated — an upgrade over the pre-11 Synapse protocol,
  whose responses were unauthenticated (legacy bittensor issue #3406).
- Miner side: `create_miner_app` builds a FastAPI app exposing ROUTE,
  verifying the validator's request signature (with nonce replay
  protection), applying the miner's blacklist, and signing the response.

Wire compatibility: this protocol is NOT compatible with the pre-11 Synapse
HTTP protocol. Validators and miners must upgrade together.
"""

import logging
from dataclasses import dataclass

import bittensor as bt
import httpx

from aurelius.protocol import ScenarioConfigSynapse

logger = logging.getLogger(__name__)

# Single route for the scenario-config exchange. Versioned so a future
# breaking transport change can add /v2 while still serving /v1.
ROUTE = "/aurelius/v1/scenario_config"

# Signature freshness window (seconds). http_auth defaults to 10s; we allow
# more slack for slow links since the nonce store already prevents replays.
MAX_AGE = 30.0


@dataclass
class MinerResponse:
    """Outcome of querying one miner. Mirrors what the validator pipeline
    needs from the old dendrite response: who answered, whether transport
    succeeded, and the payload."""

    hotkey: str
    endpoint: str  # "ip:port"
    success: bool
    synapse: ScenarioConfigSynapse | None = None
    error: str = ""


async def query_miner(
    http: httpx.AsyncClient,
    wallet,
    endpoint: str,
    miner_hotkey: str,
    request: ScenarioConfigSynapse,
    *,
    timeout: float,
    nonce_store=None,
) -> MinerResponse:
    """POST a signed scenario-config request to one miner and verify the reply.

    Never raises: transport, auth, and parse failures come back as
    `MinerResponse(success=False)` so one bad miner can't break the fan-out.
    """
    body = request.model_dump_json().encode()
    headers = bt.http_auth.sign(
        wallet,
        method="POST",
        path=ROUTE,
        body=body,
        receiver_ss58=miner_hotkey,
    )
    headers["content-type"] = "application/json"
    url = f"http://{endpoint}{ROUTE}"
    try:
        resp = await http.post(url, content=body, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # The miner signs its response body the same way requests are signed
        # (receiver = us). Verify before parsing so a MITM or wrong-hotkey
        # response is rejected outright.
        caller = bt.http_auth.verify(
            resp.headers,
            resp.content,
            method="POST",
            path=ROUTE,
            self_hotkey_ss58=wallet.hotkey.ss58_address,
            max_age=MAX_AGE,
            nonce_store=nonce_store,
        )
        if caller.hotkey_ss58 != miner_hotkey:
            return MinerResponse(
                hotkey=miner_hotkey,
                endpoint=endpoint,
                success=False,
                error=f"response signed by {caller.hotkey_ss58[:8]}, expected {miner_hotkey[:8]}",
            )
        synapse = ScenarioConfigSynapse.model_validate_json(resp.content)
        return MinerResponse(hotkey=miner_hotkey, endpoint=endpoint, success=True, synapse=synapse)
    except Exception as e:
        return MinerResponse(hotkey=miner_hotkey, endpoint=endpoint, success=False, error=str(e))


def create_miner_app(miner):
    """Build the miner's FastAPI app.

    `miner` must provide:
      - wallet (bt.Wallet, hotkey unlocked)
      - hotkey_ss58 (str)
      - blacklist(caller_hotkey: str) -> tuple[bool, str]
      - forward(synapse: ScenarioConfigSynapse) -> ScenarioConfigSynapse
    """
    from fastapi import FastAPI, Request, Response

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    nonce_store = bt.http_auth.InMemoryNonceStore(retention=MAX_AGE * 2)

    @app.post(ROUTE)
    async def scenario_config(request: Request) -> Response:
        body = await request.body()
        try:
            caller = bt.http_auth.verify(
                request.headers,
                body,
                method="POST",
                path=ROUTE,
                self_hotkey_ss58=miner.hotkey_ss58,
                max_age=MAX_AGE,
                nonce_store=nonce_store,
            )
        except bt.http_auth.AuthError as e:
            logger.debug("Rejected request: auth failed: %s", e)
            return Response(status_code=401, content=str(e))

        blocked, reason = miner.blacklist(caller.hotkey_ss58)
        if blocked:
            logger.debug("Blacklisted %s: %s", caller.hotkey_ss58[:8], reason)
            return Response(status_code=403, content=reason)

        try:
            synapse = ScenarioConfigSynapse.model_validate_json(body)
        except ValueError as e:
            return Response(status_code=422, content=f"invalid payload: {e}")

        synapse = miner.forward(synapse)

        resp_body = synapse.model_dump_json().encode()
        resp_headers = bt.http_auth.sign(
            miner.wallet,
            method="POST",
            path=ROUTE,
            body=resp_body,
            receiver_ss58=caller.hotkey_ss58,
        )
        return Response(content=resp_body, media_type="application/json", headers=resp_headers)

    return app
