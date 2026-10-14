# Aurelius Protocol — subnet 37 miner

Aurelius is Bittensor subnet **37**. Miners build tools that help explain and work with language
models, submit them for evaluation, and compete for rewards based on their usefulness. Each tool
type has its own **lane** with its own evaluation rules. Sparse autoencoders (SAEs) are the first
lane; probes, steering vectors and other tools can follow as their rules are published.

This repository contains the miner, the SAE submission tools and the shared protocol they need.

- **Incentives are in open beta** and change frequently. Rewards depend on winning your lane's
  evaluation, scaled by the subnet 37 alpha you hold. Read [INCENTIVES.md](INCENTIVES.md) before
  you start; the miner asks you to confirm you have.
- **Lane guides:** competition rules, training and submission live in the
  [lanes folder](lanes/README.md). Start with [Lane 1: SAEs](lanes/lane1.md).

## What you need

- Linux with Git, curl, Bash, Python **3.11 or 3.12** and [uv](https://docs.astral.sh/uv/)
- A publicly reachable IPv4 address and persistent disk
- A Bittensor wallet with a hotkey **registered on subnet 37**, and TAO for fees
- Subnet 37 alpha for evaluation fees (1 WT per alpha; one SAE evaluation costs 10 WT)
- A trained SAE bundle. Serving and checks run on CPU; training needs your own GPU compute.

## 1. Install

```sh
git clone https://github.com/Aurelius-Protocol/Aurelius-Protocol.git
cd Aurelius-Protocol
uv python install 3.12
uv sync --all-packages --extra bt --no-dev --frozen --python 3.12
source .venv/bin/activate
```

## 2. Prepare your wallet

The serving machine needs only your hotkey, never your private coldkey. If you don't have a
registered hotkey yet, follow the [Bittensor mining guide](https://www.bittensor.com/docs/guides/mining).

On your wallet machine, create a hotkey-only copy of the wallet:

```sh
python scripts/prepare_wallet_view.py --source "$HOME/.bittensor/wallets" \
  --output "$PWD/miner-wallets" --wallet miner --hotkey default \
  --hotkey-ss58 YOUR_REGISTERED_HOTKEY_SS58
```

Use your own wallet and hotkey names. Copy the generated `miner-wallets` directory to the serving
machine, keeping its permissions (`0700` directories, `0600` files), and set
`AURELIUS_BT_WALLET_PATH` to its absolute path.

## 3. Configure and serve

```sh
cp .env.example .env
# Edit .env: replace every REPLACE_WITH value.
set -a
. ./.env
set +a
umask 077
mkdir -p "$AURELIUS_MINER_DATA" "$AURELIUS_AXON_PUBLISH_STATE_DIR" \
  "$(dirname "$AURELIUS_BITTENSOR_NONCE_DB")" "$(dirname "$AURELIUS_MINER_JOURNAL")"
chmod 700 "$AURELIUS_AXON_PUBLISH_STATE_DIR"
aurelius-miner serve --netuid "$NETUID" --port 8091
```

- Allow inbound **TCP 8091**. Behind NAT, forward your advertised port to it. Keep port **8092**
  (local health check) private.
- The first `serve` or `fund` shows the [incentive terms](INCENTIVES.md) and asks you to confirm
  you have read and agree to them. Under a process supervisor or Docker with no terminal, set
  `AURELIUS_ACCEPT_INCENTIVES=yes` or pass `--accept-incentives`.
- `serve` runs in the foreground. Leave it running and use a second terminal, with the same
  `.env` exported, for everything else. Run it under a supervisor in production.
- Keep the wallet and state directories across restarts. Never delete the journals to force a
  resubmission.

Check that it is up:

```sh
curl --fail http://127.0.0.1:8092/health                 # on the miner
curl --fail http://YOUR_PUBLIC_IPV4:8091/health          # from another machine
```

Expect HTTP 200, `status: "ok"` and your hotkey. Until you stage a candidate, the miner serves an
idle response; a healthy miner alone earns nothing.

## 4. Fund evaluations

Evaluations are paid in work-tokens (WT), bought with subnet 37 alpha. Run `fund` on your
**wallet machine**; the coldkey must own your registered hotkey.

```sh
aurelius-miner fund --alpha 10 --wallet-name miner \
  --wallet-path "$HOME/.bittensor/wallets" --intent-dir "$PWD/funding/purchase-001"
# Review the amount, destination and fee, then rerun the same command with --execute.
```

- Credit is confirmed by a signed WT receipt, not by the chain transfer alone.
- If the result is lost or pending, rerun the same command with the same `--intent-dir` to check
  it. Use a new directory only for a new purchase.
- Check your balance with `aurelius-miner economics --netuid 37`.

Buying WT does not improve your score, and the alpha you spend no longer counts toward your
payout multiplier. See [INCENTIVES.md](INCENTIVES.md#work-tokens).

## 5. Build and submit

Follow [Lane 1](lanes/lane1.md) to train a compatible SAE, check the bundle, upload it and stage
it for the next window. No pretrained bundle is included. **Uploading alone is not submitting:**
the evaluator pulls your staged candidate, and the WT charge commits it to the epoch.

After the epoch closes, check your result and charges:

```sh
aurelius-sae status --refresh --accounting --netuid 37
```

This shows your verdict and its explanation, measured scores, your ladder rank, and whether a
charge or refund was recorded.

## Environment variables

Start from [.env.example](.env.example). The miner does **not** load `.env` by itself; export it
as shown above.

### You must set

| Variable | Value | Purpose |
| --- | --- | --- |
| `SUBSTRATE_URL` | Your Bittensor mainnet `wss://` endpoint | Chain reads and endpoint publication. Plain `wss://host` URLs and OnFinality API-key URLs are accepted. An archive node is recommended. |
| `AURELIUS_BT_WALLET` | Your wallet name | Wallet directory under `AURELIUS_BT_WALLET_PATH`. |
| `AURELIUS_BT_HOTKEY` / `MINER_HOTKEY` | Your hotkey name | Signing hotkey; keep both the same. |
| `AURELIUS_BT_WALLET_PATH` | Absolute path | The hotkey-only wallet copy from step 2. |
| `AURELIUS_AXON_EXTERNAL_IP` | Your public IPv4 | Address advertised on-chain. |
| `AURELIUS_AXON_EXTERNAL_PORT` | `8091` | Advertised port. |
| `AURELIUS_AXON_PUBLISH_STATE_DIR` | Absolute directory (`0700`) | Endpoint-publication state. |
| `AURELIUS_BITTENSOR_NONCE_DB` | Absolute SQLite path | Replay protection. |
| `AURELIUS_MINER_DATA` | Absolute directory | Staging pool and incentive acceptance. |
| `AURELIUS_MINER_JOURNAL` | Absolute SQLite path | Your submission and accounting history. |
| `AURELIUS_ACCEPT_INCENTIVES` | `yes`, only after reading [INCENTIVES.md](INCENTIVES.md) | Agreement for runs without a terminal. |

### Leave as supplied

`NETUID`, `CENTRAL_API_URL`, `AURELIUS_ENV`, `CHAIN_ID`, `AURELIUS_AXON_EXPECTED_CHAIN_NAME`,
`AURELIUS_AXON_EXPECTED_GENESIS_HASH`, `AURELIUS_TRANSPORT`, `AURELIUS_IDENTITY_SOURCE`,
`AURELIUS_CLOCK`, `AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET`, `AURELIUS_AXON_ENDPOINT_PROFILE` and
`AURELIUS_MINER_HTTP_PORT`.

`AURELIUS_RUNTIME_SPEC_PIN` (`478`) and `AURELIUS_RUNTIME_TRANSACTION_VERSION_PIN` (`1`) must
match the chain.
After a chain upgrade, update this repository and its pins together; the miner refuses a mismatch.

### Optional

| Variable | Default | Purpose |
| --- | --- | --- |
| `AURELIUS_AXON_MAX_FEE_TAO` | `0.01` | Maximum endpoint-publication fee in TAO. |
| `AURELIUS_MINER_MAX_CONCURRENT_REQUESTS` | `32` | Concurrent requests (`1`–`256`). |
| `AURELIUS_MINER_SOCKET_TIMEOUT_S` | `5` | Socket timeout in seconds (up to `60`). |
| `AURELIUS_MINER_MAX_QUERIES_PER_CALLER` | `60` | Per-caller query limit per window; `0` disables it. |
| `AURELIUS_MINER_CALLER_WINDOW_S` | `60` | Rate-limit window in seconds. |
| `AURELIUS_MINER_STATUS_INTERVAL_S` | `300` | Console status interval; `0` disables it. |

## Miner commands

Run `aurelius-miner <command> --help` for all options.

| Command | What it does |
| --- | --- |
| `serve` | Publish your endpoint and serve staged candidates. |
| `fund --alpha N ...` | Buy WT with alpha (preview by default; `--execute` to send). |
| `economics --netuid 37` | Your WT balance, prices and recent receipts. |
| `validate` | Check staged candidates against the published schema. |
| `ready --task-type interp_sae --fanout-window ID` | Check a staged candidate is eligible for a window. |
| `pool list` / `pool enable ID` / `pool disable ID` | Manage staged candidates. |
| `status` | Print your local submission journal. |
| `observe --netuid 37` | Refresh charge and refund observations. |
| `dashboard --port 8093` | Local read-only dashboard of your submissions. |

The SAE commands (`aurelius-sae check`, `preview`, `prepare`, `status`, `aurelius-sae-upload`
and `aurelius-sae-stage`) are documented in [Lane 1](lanes/lane1.md).

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Install fails | Use Python 3.11/3.12 and the frozen lockfile with the `bt` extra. |
| Wallet rejected at startup | Wallet layout and names, `0700`/`0600` permissions, and no private coldkey on the server. |
| Runtime pin mismatch | The chain was upgraded. Update this repository and the pins in `.env`. |
| Incentive terms not accepted | Run `serve` in a terminal and answer `y`, or set `AURELIUS_ACCEPT_INCENTIVES=yes`. |
| `/health` unreachable | Bind port, advertised IP/port, firewall/NAT and startup logs. HTTP 503 means startup checks failed. |
| Config signature error | Don't set `CONFIG_SIGNING_PUBKEY`; check you can reach `CENTRAL_API_URL`. |
| `fund` refuses or balance is unknown | Check the preview output and your receipts; don't send funds to an address the command hasn't confirmed. |
| Miner healthy but no submissions | `pool list`, `validate`, the previewed window and `ready --fanout-window ID`. Uploading alone does not submit. |

Code and documentation are governed by [LICENSE](LICENSE).
