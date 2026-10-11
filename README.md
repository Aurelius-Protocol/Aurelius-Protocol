# Aurelius Protocol — v5 miner

Aurelius is Bittensor subnet **37**. Miners build tools that help
explain and work with language models, submit them for evaluation, and compete for rewards based
on their usefulness. Each tool type has its own **lane**, evaluation rules and emission allocation.
Sparse autoencoders (SAEs) are the **first lane**, rather than the purpose of mining itself.
Other interpretability tools, such as probes, steering vectors and logit readouts, can have their
own lanes as their evaluation rules are approved and enabled. This release includes the miner,
SAE submission tools and the shared protocol they need.

Competition guides, scoring rules, training instructions and the incentive mechanism live in
the **[lanes folder](lanes/README.md)**. Start with [Lane 1: SAEs](lanes/lane1.md).

## Start from scratch

Use Linux, Python **3.11 or 3.12**, [uv](https://docs.astral.sh/uv/), a publicly reachable
IPv4 address and persistent disk. Serving and bundle checks run on CPU; training your own SAE
requires separate compute. Have a Bittensor wallet, TAO for registration/fees and an evaluation
budget. Before funding, obtain Aurelius's current **config signing public key**, supported slot,
chain/runtime pins and work-token purchase instructions. The public key must come from a trusted
operator channel; downloading a config does not establish its signing key.

### 1. Install and register

```sh
git clone https://github.com/Aurelius-Protocol/Aurelius-Protocol.git
cd Aurelius-Protocol
uv sync --all-packages --extra bt --no-dev --frozen
source .venv/bin/activate
```

On your wallet machine, create or restore a wallet and register its hotkey on subnet 37:

```sh
btcli wallet create --wallet miner --wallet-hotkey default
btcli tx burned-register --network finney --netuid 37 --wallet miner --wallet-hotkey default --dry-run
btcli tx burned-register --network finney --netuid 37 --wallet miner --wallet-hotkey default
btcli wallet show --wallet miner --wallet-hotkey default
```

Save the seed phrases privately and record the **hotkey SS58 address**. Registration has a
changing chain fee and is separate from work-token funding. See the
[Bittensor mining guide](https://www.bittensor.com/docs/guides/mining) for registration details.

Create the runtime wallet view, then transfer that directory to the serving machine:

```sh
python scripts/prepare_wallet_view.py --output ./miner-wallets --hotkey-ss58 YOUR_HOTKEY_SS58
```

The runtime view contains `miner/coldkeypub.txt`, `miner/hotkeys/default` and
`miner/hotkeys/defaultpub.txt`. It excludes the private coldkey. Use directory permissions
`0700` and file permissions `0600`; the miner checks them before loading its hotkey.

### 2. Configure and serve

```sh
cp .env.example .env
# Edit .env: replace every REPLACE_WITH value and confirm the current chain/runtime pins.
set -a
. ./.env
set +a
mkdir -p "$AURELIUS_MINER_DATA" "$AURELIUS_AXON_PUBLISH_STATE_DIR"
chmod 700 "$AURELIUS_AXON_PUBLISH_STATE_DIR"
aurelius-miner serve --netuid "$NETUID" --port 8091
```

Set absolute paths in `.env`, publish your own IPv4 address and allow inbound **TCP 8091**.
If using NAT, forward the advertised port to the bind port. Keep the local readiness port
**8092** private. The miner publishes its endpoint on-chain and accepts authenticated requests
from the authorized evaluator. Until you stage an eligible candidate, it serves an idle response.
A process supervisor should restart the same command and retain the wallet and state directories.

Keep the axon publication journal, nonce database, staging pool and attempt journal across
restarts. A chain upgrade or ambiguous endpoint publication requires investigating the retained
evidence and updating approved pins; deleting the journal to force another submission is unsafe.

### 3. Choose a lane and submit your work

Follow the [lane guides](lanes/README.md) for the required artifact, training workflow,
submission commands, scoring and competition rules. [Lane 1](lanes/lane1.md) covers the current
SAE competition and links to the [incentive mechanism](INCENTIVES.md).

Use `aurelius-miner dashboard` for the local journal at `http://127.0.0.1:8093`.

Code and documentation are governed by [LICENSE](LICENSE).
