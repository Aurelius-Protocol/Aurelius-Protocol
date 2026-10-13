# Aurelius Protocol — v5 miner

Aurelius is Bittensor subnet **37**. Miners build tools that help
explain and work with language models, submit them for evaluation, and compete for rewards based
on their usefulness. Each tool type has its own **lane**, evaluation rules and emission allocation.
Sparse autoencoders (SAEs) are the **first lane**, rather than the purpose of mining itself.
Other interpretability tools, such as probes, steering vectors and logit readouts, can have their
own lanes as their evaluation rules are approved and enabled. This release includes the miner,
SAE submission tools and the shared protocol they need.

**Launch status (2026-10-13):** installation and CPU bundle checks are available, but this release
does not yet provide a complete path for a new miner to compete. Pricing changes require 7,200 finalized blocks of notice. Public economics and deposit-address reads are live. Work-token funding and the
backend rollout for registered-miner admission remain incomplete. See [current blockers](lanes/lane1.md#current-participation-blockers)
before training or transferring funds.

Competition guides, scoring rules, training instructions and the incentive mechanism live in
the **[lanes folder](lanes/README.md)**. Start with [Lane 1: SAEs](lanes/lane1.md).

## Start from scratch

Use Linux, Git, curl, Bash, Python **3.11 or 3.12**, [uv](https://docs.astral.sh/uv/), a publicly reachable
IPv4 address and persistent disk. Serving and bundle checks run on CPU; training your own SAE
requires separate compute. Have a Bittensor wallet, TAO for registration/fees and an evaluation
budget. The miner includes Aurelius's **public config and receipt verification keys** and authenticates
configuration and purchase confirmations automatically. Before funding, check the supported slot, current chain/runtime
pins and active work-token purchase instructions.

### 1. Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. These commands use
uv to install Python 3.12 if needed and install the serving dependencies from the supplied lockfile:

```sh
git clone https://github.com/Aurelius-Protocol/Aurelius-Protocol.git
cd Aurelius-Protocol
uv python install 3.12
uv sync --all-packages --extra bt --no-dev --frozen --python 3.12
source .venv/bin/activate
```

### 2. Provide your wallet

Use an existing Bittensor wallet with a **hotkey registered on subnet 37**, and record its SS58
address. Keep enough TAO available for chain fees; registration and evaluation funding are
separate. The serving machine needs the hotkey signing file and public wallet documents. Keep
the private coldkey and recovery phrases on your wallet machine.
If you do not yet have a registered hotkey, follow the [Bittensor mining guide](https://www.bittensor.com/docs/guides/mining)
to create/register one on subnet 37 before continuing.

Point `AURELIUS_BT_WALLET_PATH` at a runtime wallet root containing only
`<wallet>/coldkeypub.txt`, `<wallet>/hotkeys/<hotkey>` and `<wallet>/hotkeys/<hotkey>pub.txt`.
Directories must have permissions `0700`, files `0600`. The included
[wallet-view helper](scripts/prepare_wallet_view.py) copies this layout from an existing wallet.
On the wallet machine, with this repository and its environment installed, run:

```sh
python scripts/prepare_wallet_view.py --source "$HOME/.bittensor/wallets" \
  --output "$PWD/miner-wallets" --wallet miner --hotkey default \
  --hotkey-ss58 YOUR_REGISTERED_HOTKEY_SS58
```

Substitute your wallet/hotkey names and matching hotkey address. The destination must be new;
the helper refuses to overwrite it. Transfer only the generated `miner-wallets` directory to
the serving machine, preserving permissions, then set `AURELIUS_BT_WALLET_PATH` to its absolute
path and the wallet/hotkey environment variables to the same names.

### 3. Configure and serve

```sh
cp .env.example .env
# Edit .env: replace every REPLACE_WITH value and confirm the current chain/runtime pins.
set -a
. ./.env
set +a
umask 077
mkdir -p "$AURELIUS_MINER_DATA" "$AURELIUS_AXON_PUBLISH_STATE_DIR" \
  "$(dirname "$AURELIUS_BITTENSOR_NONCE_DB")" "$(dirname "$AURELIUS_MINER_JOURNAL")"
chmod 700 "$AURELIUS_AXON_PUBLISH_STATE_DIR"
aurelius-miner serve --netuid "$NETUID" --port 8091
```

Set absolute paths in `.env`, publish your own IPv4 address and allow inbound **TCP 8091**.
If using NAT, forward the advertised port to the bind port. Keep the local readiness port
**8092** private. The miner publishes its endpoint on-chain and accepts authenticated requests
from the authorized evaluator. Until you stage an eligible candidate, it serves an idle response.
A process supervisor should restart the same command and retain the wallet and state directories.

`serve` stays in the foreground. Leave it running; use a second terminal with the same activated
environment and exported `.env` for submission and monitoring. Check readiness there:

```sh
curl --fail http://127.0.0.1:8092/health
```

Expect HTTP 200, `status: "ok"` and your registered hotkey. From a different machine, also check
`http://YOUR_PUBLIC_IPV4:8091/health` (use your advertised port) to verify firewall/NAT reachability.
A healthy listener does not prove that you have an eligible candidate, funded evaluations or rewards.

Keep the axon publication journal, nonce database, staging pool and attempt journal across
restarts. A chain upgrade or ambiguous endpoint publication requires investigating the retained
evidence and updating approved pins; deleting the journal to force another submission is unsafe.

### 4. Choose a lane and submit your work

Follow the [lane guides](lanes/README.md) for the required artifact, training workflow,
submission commands, scoring and competition rules. [Lane 1](lanes/lane1.md) covers the current
SAE competition and links to the [incentive mechanism](INCENTIVES.md). You need to train or obtain
a compatible SAE bundle; starting an empty miner earns no rewards, and no pretrained submission
bundle is included.

## Environment variables

Start with [.env.example](.env.example), replace the placeholders, and export it as shown above.
The miner does **not** load `.env` automatically. Values below describe the supplied configuration;
compare the supplied pins with the [finalized chain read](lanes/lane1.md#read-the-finalized-chain-head)
before starting. Use a release that supports the observed runtime.

### Wallet, network and state

| Variable | Value / action | Purpose |
| --- | --- | --- |
| `NETUID` | `37` | Subnet used by `serve`, `economics` and `observe` unless `--netuid` overrides it. |
| `CENTRAL_API_URL` | `https://central.aureliusaligned.ai` | Signed configuration and accounting; uploads use the separate artifact URL in the lane guide. |
| `SUBSTRATE_URL` | `wss://archive.chain.opentensor.ai:443` | Chain RPC for endpoint publication and finalized chain reads. |
| `AURELIUS_BT_WALLET` | `miner`, or your wallet name | Directory name under the runtime wallet root. |
| `AURELIUS_BT_HOTKEY` | `default`, or your hotkey name | Actual signing key loaded from that wallet. |
| `AURELIUS_BT_WALLET_PATH` | **Supply** absolute directory | Runtime wallet root, containing the wallet directory. |
| `MINER_HOTKEY` | `default`; keep aligned with your hotkey name | Compatibility identity argument; `AURELIUS_BT_HOTKEY` selects the actual Bittensor signing key. |
| `AURELIUS_AXON_EXTERNAL_IP` | **Supply** public IPv4 | Address advertised on-chain. |
| `AURELIUS_AXON_EXTERNAL_PORT` | `8091` | Advertised port; map it to the `serve --port` bind port when using NAT. |
| `AURELIUS_MINER_HTTP_PORT` | `8092`; `0` disables | Local readiness listener, bound to loopback for this configuration. |
| `AURELIUS_AXON_PUBLISH_STATE_DIR` | **Supply** absolute directory | Persistent endpoint-publication and recovery evidence; directory permissions `0700`. |
| `AURELIUS_BITTENSOR_NONCE_DB` | **Supply** absolute SQLite path | Persistent replay protection for authenticated requests. |
| `AURELIUS_MINER_DATA` | **Supply** absolute directory | Staging pool and candidate-selection history. |
| `AURELIUS_MINER_JOURNAL` | **Supply** absolute SQLite path | Attempt, upload and accounting observations. |

Keep all four state paths across restarts. If `AURELIUS_MINER_JOURNAL` is omitted, it uses
`attempts.sqlite3` inside `AURELIUS_MINER_DATA`; the default data directory is `~/.aurelius/miner`.

### Protocol and chain pins

| Variable | Supplied value | Purpose |
| --- | --- | --- |
| `AURELIUS_ENV` | `mainnet` | Production environment selection for subnet 37. |
| `CHAIN_ID` | See `.env.example` | Chain identity bound to the endpoint-publication journal. |
| `AURELIUS_AXON_EXPECTED_CHAIN_NAME` | `Bittensor` | Required chain name before publication. |
| `AURELIUS_AXON_EXPECTED_GENESIS_HASH` | See `.env.example` | Required genesis hash before publication. |
| `AURELIUS_RUNTIME_SPEC_PIN` | `477`; confirm current pin | Required runtime specification version. |
| `AURELIUS_RUNTIME_TRANSACTION_VERSION_PIN` | `1`; confirm current pin | Required transaction version. |
| `AURELIUS_TRANSPORT` | `btauth` | Authenticated serving transport. |
| `AURELIUS_IDENTITY_SOURCE` | `bittensor` | Sign with your mounted Bittensor hotkey. |
| `AURELIUS_CLOCK` | `real` | Use current timestamps in work IDs. |
| `AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET` | `true` | Reject runtime wallets containing a private coldkey or unexpected files. |
| `AURELIUS_AXON_ENDPOINT_PROFILE` | `public` | Require a publicly routable advertised endpoint. |

The template also contains `AURELIUS_CHAIN=mainnet` and `AURELIUS_CANDIDATE_SOURCE=staging` as
compatibility labels. This release does not read those variables; production candidate selection
uses the staging pool.

### Optional controls

These can be omitted to use the defaults shown.

| Variable | Default | Purpose / range |
| --- | --- | --- |
| `AURELIUS_AXON_MAX_FEE_TAO` | `0.01` | Maximum endpoint-publication fee, from `0` to `0.01` TAO in whole rao. |
| `AURELIUS_MINER_MAX_CONCURRENT_REQUESTS` | `32` | Concurrent requests per listener; `1`–`256`. |
| `AURELIUS_MINER_SOCKET_TIMEOUT_S` | `5` | Socket timeout in seconds; greater than `0`, at most `60`. |
| `AURELIUS_MINER_MAX_QUERIES_PER_CALLER` | `60` | Per-caller query limit within the window below; `0` disables the limit. |
| `AURELIUS_MINER_CALLER_WINDOW_S` | `60` | Rate-limit window in seconds; `0.1`–`3600`. |
| `AURELIUS_MINER_METAGRAPH_SYNC_S` | `30` | Refresh interval for finalized caller permits; `1`–`3600` seconds. |
| `AURELIUS_MINER_STATUS_INTERVAL_S` | `300` | Console status interval; `0` disables, maximum `86400` seconds. |
| `AURELIUS_CONFIG_SNAPSHOT` | Unset | Read and verify config from a local signed snapshot file instead of fetching `/api/config`; service commands still use `CENTRAL_API_URL`. |
| `CONFIG_SIGNING_PUBKEY` | Bundled subnet 37 key | Optional 64-character Ed25519 hex public-key override for a different deployment. Omit for subnet 37; an empty or malformed override is rejected. |
| `MINER_HOTKEY_SS58` | Loaded wallet address | Optional address override for the public `balance` read only. |

## Miner commands

Run `aurelius-miner <command> --help` for its full options. All commands below use the exported
environment. Set `NETUID=37` or pass `--netuid 37` explicitly to commands that accept it.

| Command | Options | What it does |
| --- | --- | --- |
| `serve` | `--netuid 37`, `--port 8091` | Publish the endpoint and serve staged candidates. `--port` is the bind port; the advertised port comes from the environment. |
| `economics` | `--netuid 37` | Read your balance, signed prices and recent receipts using hotkey proof. |
| `fund` | **Required:** `--alpha AMOUNT`, `--wallet-name NAME`, `--wallet-path PATH`, `--intent-dir PATH`; optional `--fee-cap-tao 0.01`, `--execute`, `--central URL`, `--endpoint WSS_URL` | Preview a WT purchase with a separately held coldkey. `--execute` submits once when funding terms are available. Reuse the same intent directory to check confirmation. |
| `balance` | `--task-type interp_sae`, `--hotkey SS58` | Public affordability check and deposit-address lookup. Pass the lane task type; its default is `synthetic`. |
| `validate` | No additional options | Check all staged candidates against their published schemas; does not score or submit them. |
| `ready` | **Required:** `--task-type interp_sae`, `--fanout-window WINDOW_ID` | Use `fanout_window` from SAE preview, not a block height. Check local eligibility; does not establish funding or discovery. |
| `pool list` | `--pool PATH` | List staged items, IDs and enabled states. |
| `pool enable ITEM_ID` / `pool disable ITEM_ID` | `--pool PATH` | Toggle selection eligibility while preserving history. |
| `status` | `--journal PATH` | Print the local attempt journal. |
| `observe` | `--netuid 37`, `--central URL`, `--journal PATH`, `--limit 20`, `--validator HOTKEY=URL` | Refresh accounting observations; optionally query the evaluator's supplied progress endpoint. |
| `dashboard` | `--port 8093`, `--journal PATH`, `--pool PATH`, `--experiments DIR` | Serve the local, read-only journal dashboard on loopback. |

`--journal` selects a SQLite file; `--pool` selects a staging JSONL file. Omit them to use your
configured paths. The dashboard reads recorded observations; run `observe` to refresh them.

```sh
aurelius-miner economics --netuid 37
aurelius-miner pool list
aurelius-miner dashboard --port 8093
```

### Fund evaluations

The `fund` command supports the reviewed runtimes475 and477. The temporary launch
purchases **1WT per alpha**, with exact fractional carry, and charges **10WT per SAE
evaluation**. Full retention premium activation and the complete recovery drill are
deferred. The backend is deploying this profile and preserves the original notice
activation at finalized block9271373. The command refuses to sign while funding
terms are unavailable. Once funding opens, run it
on your wallet machine with alpha staked on subnet 37 under your registered serving hotkey.
The funding wallet's coldkey must own that hotkey; keep TAO available for chain fees.
The command checks unlocked alpha and registration collateral, and requires a transfer worth
at least the chain's 0.0001 TAO minimum at the current price.
Use your hotkey wallet view for `AURELIUS_BT_WALLET_PATH` and pass the separate full wallet
root with `--wallet-path`:

```sh
aurelius-miner fund --alpha 10 --wallet-name miner \
  --wallet-path "$HOME/.bittensor/wallets" --intent-dir "$PWD/funding/purchase-001"
# Review the amount, treasury and fee before adding --execute to the same command.
```

Alpha is transferred to the published treasury. Credit is confirmed by a signed WT receipt;
chain inclusion alone does not confirm credit. Keep the intent directory. If the response is
lost or status is pending, rerun the same command with that directory to inspect the original
transfer. Use a new directory only for a deliberate new purchase. Purchase rates, retention
and evaluation rewards are explained in the [lane guide](lanes/lane1.md).

The SAE commands `aurelius-sae check`, `preview`, `prepare` and `status`, plus
`aurelius-sae-upload` and `aurelius-sae-stage`, are documented with their submission options in
[Lane 1](lanes/lane1.md).

The public verification key is pinned in [the miner release](subnet/src/subnet/miner/trust.py).
Configuration signatures remain mandatory; a downloaded config cannot replace the trusted key.

## If startup or submission fails

| Symptom | Next check |
| --- | --- |
| Python/dependency installation fails | Use Python 3.11/3.12 and the frozen lockfile; confirm the `bt` extra was installed. |
| Wallet guard rejects startup | Check the root/name/hotkey layout, `0700` directories, `0600` files and absence of a private coldkey. |
| Chain/runtime pin mismatch | Compare the read-only chain snapshot in [Lane 1](lanes/lane1.md#read-the-finalized-chain-head). Confirm the endpoint and approved pins before retrying. |
| Cannot reach `/health` | Check the bind port, advertised IPv4/port, firewall/NAT and startup logs. HTTP 503 means readiness failed; check publication/config evidence. |
| Config signature/key error | For subnet 37, omit `CONFIG_SIGNING_PUBKEY`; check config connectivity. Do not replace the key with one received from the server. |
| `invalid signed cadence` | The signed cadence or offset is invalid or unaligned with the commit window; refresh authenticated config. |
| Economics unavailable, or affordability is false/unknown | Funding is insufficient or unconfirmed. See funding status in the lane guide; do not send funds to an unavailable address. |
| Healthy miner, no submissions | Check `pool list`, bundle validation, the previewed window and `ready --fanout-window WINDOW_ID`. Uploading alone does not submit. |

Code and documentation are governed by [LICENSE](LICENSE).
