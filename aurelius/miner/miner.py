import asyncio
import logging
import socket
from pathlib import Path

import bittensor as bt
import uvicorn

import aurelius
from aurelius.common.chain import fetch_metagraph_blocking, neuron_for_hotkey
from aurelius.common.version import PROTOCOL_VERSION
from aurelius.config import Config
from aurelius.miner.config_store import ConfigStore
from aurelius.miner.work_token import generate_work_id
from aurelius.protocol import ScenarioConfigSynapse
from aurelius.transport import create_miner_app

logger = logging.getLogger(__name__)


class Miner:
    def __init__(self):
        self.config = Config
        self.should_exit = False

        # Warn if wallet is still on defaults (easy identity collision)
        if self.config.WALLET_NAME == "default" and self.config.WALLET_HOTKEY == "default":
            logger.warning(
                "WALLET_NAME and WALLET_HOTKEY are both 'default'. "
                "Set explicit wallet names to avoid identity collisions between operators."
            )

        # Guard against TESTLAB_MODE on mainnet — disables validator-permit
        # checks, allowing any registered hotkey to query the miner (CS-H2).
        if self.config.TESTLAB_MODE and self.config.NETWORK == "finney":
            raise RuntimeError(
                "TESTLAB_MODE=1 is not allowed on mainnet (finney). "
                "This disables validator-permit checks and exposes the miner to "
                "unauthorized queries. Remove TESTLAB_MODE or set ENVIRONMENT=testnet."
            )

        self.wallet = bt.Wallet(name=self.config.WALLET_NAME, hotkey=self.config.WALLET_HOTKEY)
        self.hotkey_ss58 = self.wallet.hotkey.ss58_address
        self.subtensor = bt.Subtensor(network=self.config.NETWORK)
        self.metagraph = fetch_metagraph_blocking(self.config.NETWORK, self.config.NETUID)
        if self.metagraph is None:
            raise RuntimeError(f"Subnet {self.config.NETUID} does not exist on network {self.config.NETWORK}")

        config_dir = self.config.MINER_CONFIG_DIR
        if not Path(config_dir).is_dir():
            raise ValueError(
                f"MINER_CONFIG_DIR={config_dir!r} does not exist. "
                "Create the directory and add scenario JSON files, or set MINER_CONFIG_DIR to an existing path."
            )
        self.config_store = ConfigStore(config_dir)
        logger.info("Config store: %d configs loaded from %s", self.config_store.count, config_dir)

        external_ip = self.config.AXON_EXTERNAL_IP
        if external_ip == "auto":
            external_ip = self._detect_external_ip()
            logger.info("Auto-detected external IP: %s", external_ip)
        self._external_ip = external_ip

        # bittensor 11 removed the Axon server; the miner runs its own HTTP
        # app (see aurelius.transport) and publishes its endpoint on-chain
        # with the ServeAxon intent so validators can discover it.
        self.app = create_miner_app(self)
        self._publish_endpoint()

        logger.info("Miner started | wallet=%s hotkey=%s", self.wallet.name, self.wallet.hotkey_str)

        # Fetch and display deposit address for operator convenience.
        # Best-effort: a flaky API must never block miner startup; fall back
        # to the `aurelius-deposit` CLI if this banner can't be printed.
        try:
            from aurelius.common.central_api import CentralAPIClient, CentralAPIError

            with CentralAPIClient(self.config.CENTRAL_API_URL, timeout=5) as client:
                addr = client.get_designated_address().address
            if addr:
                logger.info("Work-token deposit address: %s", addr)
                logger.info(
                    "To deposit: btcli stake transfer --origin-netuid %d --dest-netuid %d"
                    " --dest %s --amount <AMOUNT> --network %s",
                    self.config.NETUID,
                    self.config.NETUID,
                    addr,
                    self.config.NETWORK,
                )
        except CentralAPIError as e:
            logger.debug("Could not fetch deposit address banner: %s", e)

    def _publish_endpoint(self) -> None:
        """Publish (or confirm) this miner's endpoint on-chain via ServeAxon.

        Skips the extrinsic when the chain already carries the current
        ip:port — re-serving identical values every restart would burn fees
        and can trip the serve rate limit.
        """
        target = f"{self._external_ip}:{self.config.AXON_EXTERNAL_PORT}"
        me = neuron_for_hotkey(self.metagraph, self.hotkey_ss58)
        if me is None:
            logger.error(
                "Hotkey %s is NOT registered on subnet %d. "
                "Register first: btcli subnet register --netuid %d --network %s",
                self.hotkey_ss58[:16],
                self.config.NETUID,
                self.config.NETUID,
                self.config.NETWORK,
            )
            return
        if me.axon is not None and str(me.axon) == target:
            logger.info("Endpoint %s already published on-chain, skipping ServeAxon", target)
            return
        logger.info("Publishing endpoint %s on netuid %d (ServeAxon)...", target, self.config.NETUID)
        try:
            self.subtensor.execute(
                bt.ServeAxon(
                    netuid=self.config.NETUID,
                    ip=self._external_ip,
                    port=self.config.AXON_EXTERNAL_PORT,
                ),
                self.wallet,
            )
            logger.info("Endpoint published")
        except bt.ChainError as e:
            logger.error(
                "Failed to publish endpoint via ServeAxon: %s — validators will "
                "not discover this miner until it succeeds (retried next restart).",
                e,
            )

    def forward(self, synapse: ScenarioConfigSynapse) -> ScenarioConfigSynapse:
        scenario_config = self.config_store.next()
        if scenario_config is None:
            logger.warning("No configs available to serve")
            return synapse

        result = generate_work_id(scenario_config, self.hotkey_ss58, wallet=self.wallet)

        synapse.scenario_config = scenario_config
        synapse.work_id = result.work_id
        synapse.work_id_nonce = result.nonce
        synapse.work_id_time_ns = result.time_ns
        synapse.work_id_signature = result.signature
        synapse.miner_version = aurelius.__version__
        synapse.miner_protocol_version = PROTOCOL_VERSION

        logger.debug("Serving config '%s' with work_id %s", scenario_config.get("name", "?"), result.work_id[:16])
        return synapse

    def blacklist(self, caller: str) -> tuple[bool, str]:
        """Reject callers that aren't permitted validators on this subnet.

        `caller` is the http_auth-verified hotkey of the requester (the
        transport rejects unsigned/badly-signed requests before this runs).
        """
        neuron = neuron_for_hotkey(self.metagraph, caller)
        if neuron is None:
            return True, f"Hotkey {caller} not in metagraph"

        # On testnet, validator_permit may not be set for low-stake validators.
        # Allow any registered hotkey to query in testlab mode.
        if not Config.TESTLAB_MODE and not neuron.validator_permit:
            return True, f"UID {neuron.uid} lacks validator permit"

        return False, ""

    @staticmethod
    def _detect_external_ip() -> str:
        """Detect external IP using UDP socket trick (no actual traffic sent).

        Falls back to gethostbyname if that fails. Raises RuntimeError if
        all methods return a loopback address.
        """
        # Method 1: UDP connect to public DNS — reveals the default route IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except OSError:
            pass

        # Method 2: hostname resolution
        try:
            ip = socket.gethostbyname(socket.gethostname())
            if ip and not ip.startswith("127."):
                return ip
        except socket.gaierror:
            pass

        raise RuntimeError(
            "Could not detect a non-loopback external IP. Set AXON_EXTERNAL_IP explicitly in your environment."
        )

    async def _metagraph_sync_loop(self):
        """Refresh the metagraph snapshot periodically for blacklist checks."""
        sync_interval = self.config.METAGRAPH_SYNC_INTERVAL
        async with bt.Client(network=self.config.NETWORK) as client:
            while not self.should_exit:
                await asyncio.sleep(sync_interval)
                try:
                    fresh = await asyncio.wait_for(
                        bt.metagraph.fetch(client, self.config.NETUID),
                        timeout=max(sync_interval, 60),
                    )
                    if fresh is not None:
                        self.metagraph = fresh
                        logger.debug("Metagraph synced: %d neurons", fresh.num_uids)
                except asyncio.TimeoutError:
                    logger.warning("Metagraph sync timed out — skipping this cycle")
                except Exception as e:
                    logger.warning("Metagraph sync failed: %s — keeping previous snapshot", e)

    async def run_async(self):
        logger.info("Serving on port %d (netuid %d). Press Ctrl+C to exit.", self.config.AXON_PORT, self.config.NETUID)
        server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=self.config.AXON_PORT,
                log_level="warning",
            )
        )
        sync_task = asyncio.create_task(self._metagraph_sync_loop())
        try:
            # uvicorn installs its own SIGINT/SIGTERM handlers and exits
            # serve() gracefully on either.
            await server.serve()
        finally:
            self.should_exit = True
            sync_task.cancel()
            self.stop()

    def run(self):
        asyncio.run(self.run_async())

    def stop(self):
        logger.info("Stopping miner...")
        self.subtensor.close()


def _configure_logging():
    log_format = Config.LOG_FORMAT
    if log_format == "json":
        try:
            from pythonjsonlogger import jsonlogger

            handler = logging.StreamHandler()
            handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            logging.root.handlers = [handler]
            logging.root.setLevel(logging.INFO)
            return
        except ImportError:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def main():
    _configure_logging()
    miner = Miner()
    miner.run()


if __name__ == "__main__":
    main()
