# Aurelius Protocol — v5 miner

Aurelius is Bittensor subnet **37** (mainnet; **455** on testnet). Miners build tools that help
explain and work with language models, submit them for evaluation, and compete for rewards based
on their usefulness. Each tool type has its own **lane**, evaluation rules and emission allocation.
Sparse autoencoders (SAEs) are the **first lane**, rather than the purpose of mining itself.
Other interpretability tools, such as probes, steering vectors and logit readouts, can have their
own lanes as their evaluation rules are approved and enabled. This release includes the miner,
SAE submission tools and the shared protocol they need; the onboarding below covers the SAE lane.

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
from the authorized evaluator. Until you stage an eligible candidate, it serves an idle response.
A process supervisor should restart the same command and retain the wallet and state directories.

Keep the axon publication journal, nonce database, staging pool and attempt journal across
restarts. A chain upgrade or ambiguous endpoint publication requires investigating the retained
evidence and updating approved pins; deleting the journal to force another submission is unsafe.

### 3. Prepare and submit your SAE

Train your own SAE against a slot's exact published model revision, activation hook, dimensions
and serialization ABI. Export a directory containing **only** `cfg.json` and
`sae_weights.safetensors`; the current native ABI uses SAE-Lens **6.46.0**. Use your own training
corpus: the epoch's evaluation draw is not known before the commit window closes. The signed
config defines the admitted architectures, activation settings, size limits and commit cadence. This package checks and
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
The evaluator pulls the staged candidate and its signed work authorization. Each candidate is bound
to one epoch/window; stage a new eligible attempt for later epochs. Public ladder/results and
own charge/refund observations are separate. A pending or unavailable observation is not a pass.
Use `aurelius-miner dashboard` for the local journal at `http://127.0.0.1:8093`.

## Work-tokens: your evaluation budget

**Work-tokens (WT) buy evaluation, not rewards.** They are prepaid credits in Aurelius's
coordination ledger, attached to your `(coldkey, hotkey, netuid)`. The supported purchase path
transfers subnet alpha stake to the designated work-token wallet; credits appear only after the
transfer is finalized and attributed to your account. Obtain the current destination and signed
purchase terms before transferring. The miner's legacy `deposit` command is a development seam,
not a production funding command. Test credits are separate from real deposits.

The signed price policy specifies the evaluation class, WT charge and effective window. One
logical candidate is charged **once per content hash and evaluation window**; retries of that
same commitment do not create another charge. Re-presenting an artifact in a later epoch is a new
paid commitment. Miner code never deducts tokens. Charged content/quality failures can still
spend the evaluation budget. Operator infrastructure failures
are refund-eligible; ambiguous cases require review, and a refund is established only when the
ledger records it. Registration, training compute and transaction fees are additional costs.

## How an SAE is scored and wins rewards

The objective is to preserve the model's useful behavior with a sparse, interpretable feature
representation. A large reconstruction score alone is insufficient: the artifact must meet the
slot's compatibility, sparsity, novelty and interpretability requirements.

1. **Commit, then evaluate.** The work-token charge commits your artifact to a slot epoch. After
   the window closes, a finalized chain block seeds the rotating evaluation draw. The evaluator
   compares challengers and the present incumbent on that same draw, rather than comparing a new
   score against a historical score from easier data. Miner artifacts are loaded as data in the
   trusted harness; uploaded Python code is not executed.
2. **Pass compatibility, sparsity and novelty checks.** The harness checks the pinned model/hook, architecture,
   tensor dimensions and finite weights. It measures **L0** (active SAE features per token) and
   the fraction of dead features, enforcing the signed bounds. This prevents recovering behavior
   simply by using a dense representation. Decoder similarity against prior other-hotkey artifacts
   checks for copying; a legitimate warm start must still win the scientific comparison.
3. **Measure cross-entropy loss recovered.** The model runs with its original activations
   (`L_clean`), with those activations replaced by zeros (`L_zero`), and with your SAE's
   reconstruction (`L_SAE`). Its next-token prediction losses give the principal score:

   ```text
   S = (L_zero - L_SAE) / (L_zero - L_clean)
   ```

   Higher is better: `S = 0` recovers none of the behavior lost by zeroing the activations;
   `S = 1` recovers all of it. For example, `S = 0.8` recovers 80% of that loss gap, not 80%
   prediction accuracy. An admitted score must exceed the signed random-decoder baseline plus
   margin. Local bundle checks and training loss do not establish this evaluation score.
4. **Protect interpretability.** A challenger that could take the crown also runs the concept
   battery. Sparse probing measures how well SAE features identify known concepts; feature
   absorption measures how often a concept is hidden in other features instead of represented
   cleanly. Probing is higher-is-better and absorption is lower-is-better. Probing must meet the
   retained model-activation baseline, and both measures must stay within the signed regression
   bounds of the best crowned profiles. The battery is a floor and regression safeguard, not
   a bonus that overrides a weak reconstruction score or breaks ties.
5. **Beat the incumbent decisively.** Let `S_i` be the incumbent's score on the same draw. With
   the current `incumbent_holds` policy, a challenger must pass all gates and have a positive
   score improvement at least:

   ```text
   S_challenger - S_i >= max(q_decisive * (1 - S_i), epsilon_rotation)
   ```

   `epsilon_rotation` protects against noise-sized improvements; `q_decisive` requires recovering
   a share of the incumbent's remaining loss gap. Both come from signed slot policy. Smaller
   improvements leave the incumbent in place. If the throne is vacant, a candidate must meet
   the first-incumbent score floor and the applicable quality gates.

For illustration only, if `S_i = 0.60`, `q_decisive = 0.25` and `epsilon_rotation = 0.003`,
the required improvement is `0.10`: a gate-clean `0.70` challenger can take the crown, while
`0.61` cannot. Use the published policy for actual thresholds.

**Winning determines eligibility; the economy determines the payout.** The current winner-only
policy gives the SAE lane's eligible **rank-1** artifact its configured share; other ranks receive
zero from that lane. Passing evaluation does not guarantee winning or earning. The incumbent must
also make a paid commitment in the current epoch: absence vacates the throne. Keep submitting
eligible renewals and check both the public result and your charge receipt.

## How rewards are allocated

Signed policy sets the subnet's burn fraction, each enabled lane's emission allocation and its
payout cap. Winning a slot therefore does not entitle a miner to all subnet emissions. Different
tool lanes can reward their own useful contributions under their own scoring rules.

**Capital affects payout separately from scientific ranking.** Where the signed policy enables
capital conditioning, the winner's allocated lane budget is scaled by `K / (K + H)`: `K` is
eligible effective capital after provenance, aging, spending and debt; `H` is the policy's
reference scale. Fresh retained capital ages more slowly than emission-derived capital. The
reference fresh-principal floor scales up to 20% of the available lane budget; it does not
promise every miner a 20% reward. Unawarded weight goes to the burn destination, rather than
being redistributed to losing miners. Published reward weights determine actual on-chain alpha
emissions to the registered hotkey.
Burn-only or no-eligible-winner periods pay no miner
reward from that lane.

The adopted retention-based purchase design starts fresh capital at **1×** and rises to **20×**
after 2,628,000 blocks; emitted/untracked alpha remains **1×**. The initial mainnet denomination
of **1 base WT/alpha** and measured-cost tariffs are calibration decisions, not a universal live
quote. Capital-conditioned payout and production purchase pricing require explicit signed
activation. Read current policy and receipts; neither these design figures nor the testnet credit
schedule guarantee an active rate, reward or profitability.

Code and documentation are governed by [LICENSE](LICENSE).
