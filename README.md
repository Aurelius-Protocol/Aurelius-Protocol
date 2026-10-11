# Aurelius Protocol — v5 miner

Aurelius is Bittensor subnet **37** (mainnet; **455** on testnet). Miners submit sparse
autoencoders (SAEs) that help explain a language model's internal activations. This repository
contains the miner, artifact submission tools and the shared protocol they need.

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

On your wallet machine, create or restore a wallet and register its hotkey on SN37:

```sh
btcli wallet create --wallet miner --wallet-hotkey default
btcli tx burned-register --network finney --netuid 37 --wallet miner --wallet-hotkey default --dry-run
btcli tx burned-register --network finney --netuid 37 --wallet miner --wallet-hotkey default
btcli wallet show --wallet miner --wallet-hotkey default
```

Save the seed phrases privately and record the **hotkey SS58 address**. Registration has a
changing chain fee and is separate from work-token funding. See the
[Bittensor mining guide](https://www.bittensor.com/docs/guides/mining) for registration details.
For SN455 use `--network test --netuid 455` and the operator's testnet configuration.

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
from permitted validators. Until you stage an eligible candidate, it serves an idle response.
A process supervisor should restart the same command and retain the wallet and state directories.

Keep the axon publication journal, nonce database, staging pool and attempt journal across
restarts. A chain upgrade or ambiguous endpoint publication requires investigating the retained
evidence and updating approved pins; deleting the journal to force another submission is unsafe.

### 3. Prepare and submit your SAE

Train your own SAE against a slot's exact published model revision, activation hook, dimensions
and serialization ABI. Export a directory containing **only** `cfg.json` and
`sae_weights.safetensors`; the current native ABI uses SAE-Lens **6.46.0**. Use your own training
corpus: the validator's evaluation draws are private. The signed config defines the admitted
architectures, activation settings, size limits and commit cadence. This package checks and
submits exports; it does not bundle a training corpus or model checkpoint.

In another shell with the same environment, set `SLOT_ID`, `BUNDLE`, `HOTKEY_SS58` and a freshly
observed finalized block height `HEAD`, then run:

```sh
aurelius-sae check --slot-id "$SLOT_ID" --bundle "$BUNDLE"
aurelius-sae preview --slot-id "$SLOT_ID" --head "$HEAD"
aurelius-sae-upload --bundle "$BUNDLE" --slot-id "$SLOT_ID" \
  --url https://artifacts.aureliusaligned.ai --bittensor-wallet --progress
```

Keep the upload's returned `ref`. Set `ARTIFACT_REF` to that value and `SLOT_EPOCH` to the epoch
from `preview`; set `SPARSITY_TARGET` to your declared sparsity value in `[0,1]`:

```sh
aurelius-sae prepare --slot-id "$SLOT_ID" --bundle "$BUNDLE" --head "$HEAD" \
  --slot-epoch "$SLOT_EPOCH" --artifact-ref "$ARTIFACT_REF" \
  --hotkey "$HOTKEY_SS58" --sparsity-target "$SPARSITY_TARGET" --stage
aurelius-miner validate
aurelius-miner pool list
aurelius-miner balance --task-type interp_sae
aurelius-sae status --refresh --accounting --netuid "$NETUID"
```

`check` checks bytes, metadata and tensors; it does not predict a scientific pass. `preview`
reports when a commit window opens and closes. Stage before that window, refresh `HEAD` before
preparing, and keep the miner online through it: **uploading alone does not submit a candidate**.
Validators pull the staged candidate and its signed work authorization. Each candidate is bound
to one epoch/window; stage a new eligible attempt for later epochs. Public ladder/results and
own charge/refund observations are separate. A pending or unavailable observation is not a pass.
Use `aurelius-miner dashboard` for the local journal at `http://127.0.0.1:8093`.

## Work-tokens and incentives

**Work-tokens (WT) buy evaluation, not rewards.** They are prepaid credits in Aurelius's
coordination ledger, attached to your `(coldkey, hotkey, netuid)`. The supported purchase path
transfers subnet alpha stake to the designated work-token wallet; credits appear only after the
transfer is finalized and attributed to your account. Obtain the current destination and signed
purchase terms before transferring. The miner's legacy `deposit` command is a development seam,
not a production funding command. Test credits are separate from real deposits.

The signed price policy specifies the evaluation class, WT charge and effective window. One
logical candidate is charged **once per content hash and fanout window across validators**;
multiple validators do not multiply that WT charge. Miner code never deducts tokens. Charged
content/quality failures can still spend the evaluation budget. Operator infrastructure failures
are refund-eligible; ambiguous cases require review, and a refund is established only when the
ledger records it. Registration, training compute and transaction fees are additional costs.

**SAE rewards follow scientific competition.** Trusted validators inspect the artifact as data,
check compatibility, sparsity, dead features and copying, then evaluate reconstruction and the
interpretability battery on a finalized-block-seeded draw. A challenger must satisfy the signed
quality gates and incumbent comparison. A pass does not necessarily take the crown. The current
winner-only policy pays the eligible **rank-1** SAE; lower ranks can receive zero. A fresh eligible
result and all payout gates must remain valid, so keep submitting eligible renewals as required.

**Capital affects payout separately from scientific ranking.** Where the signed policy enables
capital conditioning, the winner's allocated lane budget is scaled by `K / (K + H)`: `K` is
eligible effective capital after provenance, aging, spending and debt; `H` is the policy's
reference scale. Fresh retained capital ages more slowly than emission-derived capital. The
reference fresh-principal floor scales up to 20% of the available lane budget; it does not
promise every miner a 20% reward. Unawarded weight goes to the burn destination, rather than
being redistributed to losing miners. Validator weights and Bittensor consensus determine actual
alpha emissions to the registered hotkey. Burn-only or no-eligible-winner periods pay no miner
reward from that lane.

The adopted retention-based purchase design starts fresh capital at **1×** and rises to **20×**
after 2,628,000 blocks; emitted/untracked alpha remains **1×**. The initial mainnet denomination
of **1 base WT/alpha** and measured-cost tariffs are calibration decisions, not a universal live
quote. Capital-conditioned payout and production purchase pricing require explicit signed
activation. Read current policy and receipts; neither these design figures nor the testnet credit
schedule guarantee an active rate, reward or profitability.

Code and documentation are governed by [LICENSE](LICENSE).
