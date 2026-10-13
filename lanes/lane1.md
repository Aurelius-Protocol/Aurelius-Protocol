# Lane 1 — sparse autoencoder competition

**Task type:** `interp_sae`. Build an SAE that represents a language model's internal activations
with sparse, useful features while preserving its predictive behavior. Submit the trained weights
and metadata, not a training service or executable model plugin. A candidate earns by winning
the slot's scientific competition and remaining eligible under the payout policy.

SAEs are Aurelius's first tool lane. The miner can serve other published task types as additional
lanes become available. For setup, see the [miner README](../README.md); for WT charges, reward
allocation and the adopted capital mechanisms, see [INCENTIVES.md](../INCENTIVES.md).

## Current competition and model

The values below describe [public config](https://central.aureliusaligned.ai/api/config)
**version 24**, retrieved on **2026-10-11**. The miner authenticates the current snapshot with
the public verification key included in its release before preparing a submission.

| Parameter | Current subnet 37 slot |
| --- | --- |
| Slot ID | `olmo2-7b-l16-sn37-proof` |
| Status | `enabled` |
| Base model | [`allenai/OLMo-2-1124-7B`](https://huggingface.co/allenai/OLMo-2-1124-7B) — the base model, not an Instruct/SFT variant |
| Model revision | `7df9a82518afdecae4e8c026b27adccc8c1f0032` |
| Activation site | TransformerLens `blocks.16.hook_resid_post` — residual stream after block 16, indexed from zero |
| Input dimension, `d_in` | 4,096 |
| SAE width, `d_sae` | 32,768 — an 8× expansion |
| Model precision / processing | BF16; `model_load_processing = none` |
| SAE weights | FP32 |
| Serialization ABI | SAE-Lens 6.46.0 |
| Activation runtime | TransformerLens 3.7.2 |
| Concept battery | SAE-Bench 0.6.0 |
| Reference training context | 2,048 tokens |
| Evaluation CE context | 128 tokens; pinned native packing uses 800 batches of 4 contexts |
| Activation normalization / reshape | `none` / `none` |
| Subtract decoder bias before encoding | `apply_b_dec_to_input = false` |
| TopK activation rescaling | `rescale_acts_by_decoder_norm = true` |
| Published training conventions | Prepend BOS, exclude special tokens, sequence-position slice `[null]` |
| Bundle byte limit | 2,147,483,648 bytes (2 GiB); config JSON is separately limited to 64 KiB |
| Competition phase / tiebreak | `throne_only` / `incumbent_holds` |
| Scientific parameter provenance | `operator_accepted`; not a claim of completed empirical calibration |

The 2,048-token training context is a reference recipe, not the length of the CE scoring stream.
Training-context/BOS/special-token provenance must be recorded correctly; it is not evidence that
the artifact passed the measured sparsity or scientific gates. Model identity, hook, width,
precision and inference semantics must match the slot. Do not train on a different checkpoint
and change the metadata to make it look compatible.

The older `olmo2-7b-l16-mvp` slot in the same snapshot is **disabled**. Use the enabled
`olmo2-7b-l16-sn37-proof` slot above.

## What you submit

A bundle is a directory containing exactly:

```text
bundle/
  cfg.json
  sae_weights.safetensors
```

Both must be regular files. Extra files, links, malformed or duplicate-key JSON, nonfinite
weights, wrong tensor names/shapes/dtypes and incompatible metadata can be rejected. Keep
checkpoints, optimizer state, logs and your training data outside this directory. The candidate
binds the bundle digest, immutable artifact reference, hotkey, slot and epoch. After uploading,
do not mutate the local bundle used for preparation.

The native inference allowlist includes `standard`, `gated`, `jumprelu` and `topk`. The protocol
also records `batchtopk` and `matryoshka_batchtopk` training provenance; SAE-Lens exports those
training architectures as **JumpReLU inference models**. Preserve the genuine saved architecture
and provenance. The reference recipe below uses plain TopK, so its exported architecture is
`topk`. A custom training implementation must export the same admitted native ABI; arbitrary
Python callbacks are not accepted.

## How effectiveness is measured

### 1. Sparse features and reconstruction

For an activation row `x` in 4,096 dimensions, an SAE encodes a sparse feature row `z` in
32,768 dimensions and decodes it back to `x_hat`. A simplified TopK form is:

```text
z = TopK_positive(x * W_enc + b_enc, k)
x_hat = z * W_dec + b_dec

W_enc: 4096 x 32768
W_dec: 32768 x 4096
```

This expresses the basic representation; the pinned library's decoder-norm rescaling is also
part of inference. Training typically minimizes reconstruction loss, with the architecture's
sparsity mechanism and auxiliary losses. The competition ultimately measures the effect of
`x_hat` on the language model, rather than accepting training MSE as the final score.

For `N` evaluated non-special-token activations, let `z_(t,j)` be feature `j` at token `t`:

```text
L0_t = sum_j 1[z_(t,j) > 0]
mean_L0 = (1/N) * sum_t L0_t
dead_fraction = count_j(sum_t 1[z_(t,j) > 0] == 0) / d_sae
```

The current draw-level requirements are **40 ≤ mean_L0 ≤ 120** and
**dead_fraction ≤ 0.5**. Equality passes these bounds. TopK `k` is a training/inference
setting, not a substitute for measured L0: zero/negative feature preactivations and distribution
shift can change what actually fires.

The harness also records family-level means, the draw's 95th-percentile L0, maximum context mean
L0 and activation/reconstruction mass. Current family bands are `[0,32768]`, with tail/context
ceilings `32768`; these are nonrestrictive bounds, not calibrated distribution-specific limits.
Mass-ratio diagnostics are advisory and are not an additional headline score.

### 2. Cross-entropy loss recovered

The evaluator runs the pinned model on the same token stream in three conditions:

| Loss | Intervention at the signed hook |
| --- | --- |
| `L_clean` | Keep the original activation |
| `L_zero` | Replace the activation with zero |
| `L_SAE` | Replace it with your SAE's decoded reconstruction |

Each is mean next-token cross-entropy over the admitted non-special input positions. The
principal score is:

```text
S = (L_zero - L_SAE) / (L_zero - L_clean)
```

Higher is better. `S = 0` recovers none of the prediction performance lost by zeroing the site;
`S = 1` recovers all of it. It is not prediction accuracy or explained variance. For example:

```text
L_clean = 2.0, L_zero = 5.0, L_SAE = 2.6
S = (5.0 - 2.6) / (5.0 - 2.0) = 0.8
```

The SAE recovers 80% of that cross-entropy gap. Nonfinite, degenerate or inadmissible evaluation
outputs do not become valid scores. The current random-decoder floor requires:

```text
S > random_floor + random_floor_delta = 0.26 + 0.10 = 0.36
```

Equality at `0.36` fails this strict ordinary gate. A good score still needs all other gates and
the crown comparison; `S > 0.36` alone does not earn rewards.

### 3. Rotating, paired evaluation

After commitments close, a finalized chain block seeds the epoch's draw. The CE pool covers
books, code, dialogue, reference, scientific and web text. Current policy requests 4,500,000
characters per family with 8,000-character chunks; the native tokenizer/packer selects the
scored token prefix. Those character counts are not a number of scored tokens or a training quota.

The present incumbent and challengers are evaluated on the **same epoch draw**. Historical scores
from a different draw cannot establish that a challenger is better. The text and exact rotating
selection are not a training corpus supplied by this repository. Use your own diverse corpus
and separate held-out development data; do not train directly on the private evaluation draws
or concept-test answers.

### 4. Novelty and copying

The referee compares candidate decoder directions against the slot's retained prior decoders,
excluding the candidate's own hotkey. For candidate row `u_i` and prior row `v_j`, it measures:

```text
c_ij = abs(dot(u_i, v_j)) / (norm(u_i) * norm(v_j))
duplicate_fraction = count_i(max_j c_ij > 0.95) / d_sae
```

A fraction **strictly greater than 0.5** is a duplicate rejection. Signs, feature reordering or
small cosmetic changes are not a useful new contribution. The comparison uses the complete
admitted decoder archive; an unproven sketch shortcut is not used. Commit order determines
archive precedence. Own-hotkey re-presentations can maintain presence, but still pay for the
new epoch and must remain scientifically eligible. Warm-starting does not exempt an artifact
from copying checks or the decisive-improvement rule.

### 5. Interpretability battery

Candidates that reach the eligible positive-improvement band, and candidates for a vacant throne,
run the deterministic concept battery. It uses SAE-Bench sparse probing and feature absorption.
The seeded concept program rotates profession/sentiment concepts and an absorption-letter subset
from the pinned benchmark assets. Results describe measured concept behavior, not a general proof
that every feature is human-interpretable.

- `P` is sparse-probe top-1 accuracy: can a sparse feature representation identify the concept?
  Higher is better.
- `A` is feature absorption: how often is the concept captured in other features instead of
  its expected clean representation? Lower is better.
- `P_LLM` is the retained model-activation probe baseline established by the first crown.
- `P_best` and `A_best` are the best crowned probing and absorption values retained in the
  competition's high-water state.

The gates for a later challenger are:

```text
P >= P_LLM
P >= P_best - delta_cat_probe
A <= A_best + delta_cat_abs

delta_cat_probe = 0.07457
delta_cat_abs   = 0.31942
```

Equality passes these battery gates. The first crown initializes the baseline and high-water
profile; later crowns update `P_best = max(P_best, P)` and `A_best = min(A_best, A)`. A candidate
that does not become champion cannot relax the retained best values. The baseline evidence must
remain consistent, rather than silently replacing it with a more favorable run.

The battery is an interpretability floor and regression guard. It is not blended into `S`,
and the current policy does not use a battery win to break a close CE comparison. Feature-injection
diagnostics remain advisory under the accepted current mechanism; do not treat an advisory flag
as a separate activated rejection threshold.

## Taking and keeping the crown

Current scientific constants are:

| Parameter | Value | Role |
| --- | --- | --- |
| `screen_margin_W` | 0.0037 | Cheap screen against the stored incumbent score |
| `epsilon_rotation` | 0.003 | Minimum meaningful positive CE improvement |
| `q_decisive` | 0.25 | Required share of the incumbent's remaining loss gap |
| `coronation_floor_ce` | 0.36 | First-incumbent floor; the ordinary strict random-floor gate still applies |
| `tiebreak` | `incumbent_holds` | A non-decisive challenger leaves the crown unchanged |

A challenger below `S_stored - screen_margin_W` is screened out. Passing that screen does not
win: the actual crown comparison uses the incumbent's score `S_i` on the current paired draw.

```text
gap = S_challenger - S_i
required_gap = max(q_decisive * (1 - S_i), epsilon_rotation)
```

The decision sequence is:

1. All applicable artifact, sparsity, dead-feature, novelty and ordinary CE gates must pass.
2. `gap <= 0`: the incumbent holds.
3. `0 < gap < epsilon_rotation`: the incumbent holds without spending a battery run.
4. A challenger with `gap >= epsilon_rotation` must pass the battery floors/regression gates.
5. `gap >= required_gap`: a gate-clean challenger takes the crown. Equality passes.
6. A positive gap below `required_gap` leaves the incumbent in place under `incumbent_holds`.

For `S_i = 0.60`, the required gap is `max(0.25 * 0.40, 0.003) = 0.10`.
A gate-clean `0.70` challenger can win; `0.61` cannot. At `S_i = 0.99`, the epsilon floor
dominates: `max(0.0025, 0.003) = 0.003`.

If the throne is vacant, eligible candidates are considered by descending CE, with deterministic
commit-order tie handling. A first incumbent must meet the coronation floor and pass its battery
floor; there is no free crown for being the only submission. A present incumbent keeps competing
on the fresh draw. An incumbent that does not make a paid commitment for the new epoch vacates.

The current `throne_only` phase gives a positive lane score only to the eligible rank-1 champion;
lower ranks have no reward share. A candidate can be evaluated successfully and still earn zero.
Payout then follows [the incentive mechanism](../INCENTIVES.md#from-scientific-winner-to-reward-weight),
including freshness, ownership, charge evidence, burn and the signed lane-health cap. The current
policy does not impose the adopted capital-conditioned payout curve on this lane.

## How to train a compatible SAE

### Prepare data and compute

Keep training separate from the serving process and wallet. Use a CUDA training environment with
enough GPU memory for the BF16 7B model, FP32 SAE weights, gradients, optimizer state and activation
buffer. These have different memory costs; reducing forward batch/buffer sizes can help without
changing the required model or width. CPU serving does not require these training dependencies.

Prepare your own UTF-8 JSONL corpus, one object with a `text` field per line. Deduplicate it,
reserve independent development text, and record a digest, tokenization convention and shuffle
seed. Cover varied domains rather than repeatedly fitting a small set of prompts. Monitor actual
tokens consumed: a requested training budget does not prove unique-data coverage, and a streaming
loader can recycle an exhausted corpus.

Use SAE-Lens **6.46.0** and TransformerLens **3.7.2** for compatibility with this native ABI.
On the training machine, create a separate environment:

```sh
uv venv --python 3.12 .venv-sae
uv pip install --python .venv-sae/bin/python \
  'sae-lens==6.46.0' 'transformer-lens==3.7.2' 'datasets==3.6.0'
```

Use a CUDA-enabled PyTorch installation compatible with that environment and your GPU; retain
the resolved training dependency lock for reproducibility. The miner's production environment
does not include a trainer. The pinned [SAE-Lens runner configuration](https://github.com/decoderesearch/SAELens/blob/v6.46.0/sae_lens/config.py)
and [training runner](https://github.com/decoderesearch/SAELens/blob/v6.46.0/sae_lens/llm_sae_training_runner.py)
define the v6 API used below.

### A reference TopK80 starting recipe

The following recipe illustrates the compatible training setup: TopK80, learning rate `3e-4`,
2,048-token contexts, 4,096 activation rows per optimizer batch, 1,000 warm-up steps and
100,003,840 training tokens. These are starting training choices, **not competition-imposed
training budgets or a guarantee of winning**. Improve data, optimizer schedule and admitted
architecture using your development evaluations.

Save this as `train_sae.py` on your training machine, beside your `train.jsonl`:

```python
from pathlib import Path
import shutil

from datasets import load_dataset
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

MODEL = "allenai/OLMo-2-1124-7B"
REVISION = "7df9a82518afdecae4e8c026b27adccc8c1f0032"
SEED = 42

torch.set_default_dtype(torch.float32)
torch.manual_seed(SEED)
cfg = LanguageModelSAERunnerConfig.from_dict(
    {
        "model_name": MODEL,
        "hook_name": "blocks.16.hook_resid_post",
        "model_from_pretrained_kwargs": {"revision": REVISION, "dtype": "bfloat16"},
        "context_size": 2048,
        "prepend_bos": True,
        "exclude_special_tokens": True,
        "seqpos_slice": [None],
        "dataset_path": str(Path("train.jsonl").resolve()),
        "dataset_trust_remote_code": False,
        "is_dataset_tokenized": False,
        "streaming": True,
        "training_tokens": 100_003_840,
        "store_batch_size_prompts": 8,
        "n_batches_in_buffer": 32,
        "train_batch_size_tokens": 4096,
        "prefetch_llm_batches": False,
        "lr": 3e-4,
        "lr_scheduler_name": "constant",
        "lr_warm_up_steps": 1000,
        "device": "cuda",
        "dtype": "float32",
        "autocast": True,
        "autocast_lm": False,
        "seed": SEED,
        "checkpoint_path": "training-checkpoints",
        "save_final_checkpoint": True,
        "output_path": "training-output",
        "logger": {"log_to_wandb": False, "wandb_id": "lane1-topk80"},
        "sae": {
            "architecture": "topk",
            "d_in": 4096,
            "d_sae": 32768,
            "k": 80,
            "device": "cuda",
            "dtype": "float32",
            "normalize_activations": "none",
            "reshape_activations": "none",
            "apply_b_dec_to_input": False,
            "rescale_acts_by_decoder_norm": True,
        },
    }
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    revision=REVISION,
    trust_remote_code=False,
    use_fast=True,
    add_bos_token=False,
)
model = HookedTransformer.from_pretrained_no_processing(
    MODEL,
    revision=REVISION,
    tokenizer=tokenizer,
    trust_remote_code=False,
    use_safetensors=True,
    dtype=torch.bfloat16,
    device="cuda",
    n_devices=1,
    default_prepend_bos=True,
    default_padding_side="right",
)
model.eval()
model.requires_grad_(False)
dataset = load_dataset(
    "json",
    data_files="train.jsonl",
    split="train",
    streaming=True,
).shuffle(seed=SEED, buffer_size=10_000)

LanguageModelSAETrainingRunner(
    cfg,
    override_model=model,
    override_dataset=dataset,
).run()

bundle = Path("bundle")
bundle.mkdir(exist_ok=False)
for name in ("cfg.json", "sae_weights.safetensors"):
    shutil.copyfile(Path("training-output") / name, bundle / name)
```

Run with `.venv-sae/bin/python train_sae.py`. The model is frozen; training updates the SAE,
not the base language model. The runner records model/hook and training provenance, then saves
an inference model. Only the two copied files belong in the submission bundle. Use fresh output
directories/run IDs for later experiments and preserve their configs and checkpoints.

For a TopK SAE, reconstruction MSE is the main training objective, sparsity is structurally
controlled by `k`, and the library's auxiliary training losses help with underused features.
An L1-penalized architecture instead trades reconstruction against its sparsity penalty. Tuning
that trade-off must still land inside the measured competition bounds. BatchTopK and JumpReLU
can have different training behavior; consult their pinned implementation and verify the
exported inference semantics before submitting.

### Measure before spending an evaluation

On your held-out development data, monitor loss recovered using the three interventions above,
mean/tail L0, dead-feature fraction and stability across domains. A lower activation MSE can
still worsen next-token loss; a higher loss-recovered score can still fail the concept battery.
Keep a genuinely separate development split when comparing recipes. Approximate local probes
help iterate, but do not reproduce the private rotating battery or establish an official pass.

Use the packaged `aurelius-sae check` on the exported bundle. It validates layout, metadata and
tensor ABI on CPU; `compatible: true` is not a scientific verdict. There is no bundled GPU
training or official scientific evaluation command in this public miner package.

## Prepare, upload and commit

### Current participation blockers

Checked on **2026-10-13**:

| Requirement | Current gap |
| --- | --- |
| Compatible signed schedule | Fixed in this branch: adjacent `C = W = 100` windows are supported. Overlapping or unaligned schedules remain refused. |
| Public evaluation funding | The approved temporary launch uses10WT per SAE evaluation and1WT per alpha with exact carry. Charges use real WT first, then enabled sufficient synthetic WT, without combining balances. Premium and the full recovery drill are deferred; the original notice matured at9271373 and hotkey-authenticated funding terms report available. See [INCENTIVES.md](../INCENTIVES.md#retention-based-work-token-purchase-design). |
| Registered-miner admission | Registered-hotkey admission is deployed with finalized ownership and paid-result verification. No extra signup or trusted-key setup is required. |
| Economics read | Live: a hotkey-signed `POST /api/miner/economics` returned HTTP 200 and passed the miner client's signed-config and account checks. Unsigned requests return HTTP 401. The funding read reports base conversion available. An existing miner’s signed purchase receipt confirms10alpha credited10realWT with zero carry; this is not a fresh-wallet scored proof. |
| Deposit-address lookup | Live: `GET /api/deposit_address` returns HTTP 200 with the subnet 37 treasury address. Address discovery alone does not enable work-token purchases; real deposits have been unpaused after notice maturity. |

The config and public verdict routes return JSON, and artifact ingress has a live backend.
Those checks do not establish a successful funded submission from a new wallet. A fresh real-funded scored proof remains outstanding. Confirm
a signed WT purchase receipt before preparing a paid submission.

### Read the finalized chain head

With the miner environment exported, this read-only command prints the finalized block and
chain/runtime identity without loading your wallet:

```sh
python - <<'PY'
from dataclasses import asdict
import json
import os

from subnet.bittensor_v11 import FinalizedRpc

rpc = FinalizedRpc(os.environ["SUBSTRATE_URL"])
try:
    print(json.dumps(asdict(rpc.snapshot()), indent=2))
finally:
    rpc.close()
PY
```

Compare `chain`, `genesis_hash`, `runtime_spec_version` and `transaction_version` with your
configured pins. Use `block_number` as `HEAD` below. An unexpected identity requires checking
the endpoint; an upgrade requires confirmed supported pins, rather than blindly copying values.

### Check and stage a bundle

First complete [installation and serving](../README.md). In a shell with the same miner
environment, set:

```sh
export SLOT_ID=olmo2-7b-l16-sn37-proof
export BUNDLE=/absolute/path/to/bundle
export HOTKEY_SS58=YOUR_REGISTERED_HOTKEY_SS58
# Set HEAD to a fresh finalized block height observed from your subnet 37 chain endpoint.
export HEAD=YOUR_FINALIZED_BLOCK_HEIGHT
```

Then inspect the bundle and schedule:

```sh
aurelius-sae check --slot-id "$SLOT_ID" --bundle "$BUNDLE"
aurelius-sae preview --slot-id "$SLOT_ID" --head "$HEAD"
```

The timing comes from chain blocks, not wall-clock estimates. With cadence `C`, phase offset
`O`, epoch `e` and commitment-window width `W`:

```text
B_e = O + e*C
commit_window = [B_e - 2*W, B_e - W)
seed_block = B_e + 8
```

The end is exclusive. `preview` reports the eligible epoch, open/close blocks and whether the
window is upcoming, open or expired. The seed is finalized after the commitment closes. Refresh
`HEAD` before preparation; do not substitute a wall-clock timestamp.

**Current schedule compatibility:** config 24 publishes `C = 100` and `W = 100`, while this
ported miner's preview requires `C >= 2*W`. That combination currently refuses with
`invalid signed cadence`. Obtain a compatible operator-published schedule or an approved miner
update before using `preview`/`prepare`; do not alter authenticated policy locally to bypass it.
The model and scientific requirements above do not resolve this schedule mismatch.

When schedule preview succeeds, upload with your hotkey:

```sh
aurelius-sae-upload --bundle "$BUNDLE" --slot-id "$SLOT_ID" \
  --url https://artifacts.aureliusaligned.ai --wallet "$AURELIUS_BT_WALLET" \
  --bittensor-wallet --journal "$AURELIUS_MINER_JOURNAL" --progress
```

Save the returned `ref` as `ARTIFACT_REF` and the previewed epoch as `SLOT_EPOCH`. Set
`SPARSITY_TARGET` to your declared decimal value in `[0,1]`; it is not the integer TopK `k`
and does not replace measured sparsity. The declaration is bound into the candidate.

```sh
aurelius-sae prepare --slot-id "$SLOT_ID" --bundle "$BUNDLE" --head "$HEAD" \
  --slot-epoch "$SLOT_EPOCH" --artifact-ref "$ARTIFACT_REF" \
  --hotkey "$HOTKEY_SS58" --sparsity-target "$SPARSITY_TARGET" --stage
aurelius-miner validate
aurelius-miner pool list
aurelius-miner balance --task-type interp_sae
```

Save preview's integer `fanout_window` as `WINDOW_ID` and verify local selection with
`aurelius-miner ready --task-type interp_sae --fanout-window "$WINDOW_ID"`. This is a window
identifier, not `HEAD` or the epoch number. Check that affordability is explicitly `true` before
expecting paid admission; an unavailable address or accounting response is not funding evidence.

Stage before the eligible commitment window and keep the miner reachable throughout it.
**Uploading alone is not submission.** The evaluator pulls the staged candidate and signed
work authorization; the recorded charge commits it to the epoch. The bundle must remain
retrievable for evaluation. Capacity limits can refuse admission before a charge; the structural
epoch ceiling is 256 distinct charged commitments, including re-presentations.

Afterward, inspect result and accounting separately:

```sh
aurelius-sae status --refresh --accounting --netuid "$NETUID"
```

Look for your exact artifact/epoch, verdict reason, ladder rank and charge/refund evidence.
`first_incumbent` or `decisive_ce` can establish a crown; `ce_not_better`,
`ce_gap_below_epsilon` or `tiebreak_incumbent_holds` can be completed evaluations with no crown.
Pending, unavailable or an infrastructure error is not a pass. Keep the attempt journal and
artifact files across restarts. Each later epoch needs its own eligible paid commitment; a
past crown is not a perpetual reward entitlement. Current result freshness is 3,600 blocks,
but freshness does not substitute for the lane's epoch-presence requirement.

Return to [lane guides](README.md), [incentives](../INCENTIVES.md) or [miner setup](../README.md).
