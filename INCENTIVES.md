# Incentive mechanism

Miners compete by producing useful tools. A lane's scientific evaluation determines who is
eligible to earn; the subnet's economic policy determines the reward allocation. Work-tokens
pay for evaluation. Buying more evaluation budget does not improve a scientific score.

For the current competition, see [Lane 1: SAEs](lanes/lane1.md). Other lanes can use different
effectiveness measures while sharing the miner, accounting and reward-allocation machinery.

## Current policy versus adopted economic design

The [public configuration](https://central.aureliusaligned.ai/api/config), version **25** retrieved
on **2026-10-13**, declares the following subnet 37 policy:

| Setting | Published value |
| --- | --- |
| Enabled earning lane | `interp_sae` |
| Lane allocation | `1` — all of the non-burn lane budget |
| Configured base burn fraction | `0` |
| Competition phase | `throne_only` — only the eligible champion has a positive lane score |
| Lane-health single-miner cap | `1.0` |
| Capital conditioning | `none` — capital does not scale the current lane payout |
| Unused share disposition | `to_burn` |

This is a dated reading of published policy, not a fixed future reward promise or evidence of
a particular finalized payout. The miner authenticates config against its bundled public key.
Check the current authenticated policy, result, charge receipt and on-chain payment separately.

The capital and purchase-retention formulas below describe the adopted economic design. They
apply only where the relevant signed policy and funding path enable them. In particular, they
are **not additional capital requirements for the current unconditioned SAE competition**.

## Work-tokens: paying for evaluation

**Work-tokens (WT)** are prepaid credits in the coordination ledger, attached to
`(coldkey, hotkey, netuid)`. They are distinct from TAO registration fees, subnet alpha and
training compute. Your miner authorizes the candidate; only the server records the deduction.

The supported purchase mechanism transfers subnet alpha stake to the designated work-token
wallet. Credit appears only after the transfer is finalized and attributed to your beneficiary.
Before sending funds, obtain the current destination, active purchase terms and confirmation
that deposit intake is enabled. Base purchases are open at1WT per independently verified
alpha with exact carry; SAE evaluations cost10WT. Premium activation and the full recovery
drill are deferred. Test or synthetic credits do not establish real purchase or capital evidence.

The miner's legacy `deposit` command is a development seam, not a production funding command.
Read your balance with:

```sh
aurelius-miner balance --task-type interp_sae
```

### Charges, retries and refunds

The signed price policy binds the evaluation class, WT price and effective window. In the SAE
lane, the server-side charge is the commitment; scientific evaluation follows the closed window.
One logical candidate is charged once for its content hash in that evaluation window. Replaying
that same commitment does not create another charge. A new candidate, or a later-epoch
re-presentation of the same artifact, is a new paid commitment.

A scientific failure can spend WT: evaluation is work even when the artifact does not win.
Content failures can also be chargeable under the lane's admission rules. Infrastructure
failures are refund-eligible; ambiguous outcomes require review. A refund exists only when the
ledger records it, not merely because a request timed out or the local journal says pending.

Track the result and accounting evidence with:

```sh
aurelius-sae status --refresh --accounting --netuid "$NETUID"
```

Upload success, charge success, scientific acceptance, crown ownership, published weight and
observed emissions are separate facts. Registration, GPU training, hosting and transaction fees
remain additional costs.

## From scientific winner to reward weight

Let:

- `b` be the configured base burn fraction.
- `a_t` be lane `t`'s allocation, normalized over enabled lanes.
- `q_(m,t)` be miner `m`'s positive, eligible lane score after scientific evaluation.
- `f_(m,t) = q_(m,t) / sum_j q_(j,t)` be the normalized scientific share.
- `p_(m,t)` be the economic payout multiplier; it is `1` in an unconditioned lane.

Before payout caps and chain encoding, the contribution of a healthy lane is:

```text
B_t = (1 - b) * a_t
w_(m,t) = B_t * f_(m,t) * p_(m,t)
```

Only eligible, fresh results supported by the required real charge and artifact evidence can
enter `q`. Submitting many copies does not multiply ownership of the winning result. A local
training score or a bare approval label cannot create an eligible share.

For Lane 1's current `throne_only` phase, only the champion has a positive score. The internal
ladder score cap is **not** a second absolute payout cap: the allocation engine normalizes the
eligible scores. With only the champion eligible, its scientific share is `1`. The separate
lane-health `max_single_miner_share` caps its fraction of the lane budget. Version 24 declares
that cap as `1.0`, with base burn `0` and capital mode `none`; an otherwise fully eligible
champion can therefore receive the full allocated lane budget in the reward vector.

Missing or stale results, unavailable required evidence and failed payout gates can leave no
payable champion. Under the current `to_burn` policy, unused weight goes to the burn destination.
Capital-withheld weight, if that mechanism is enabled, also goes to burn and is not redistributed
to losing miners. A configured zero base burn does not guarantee that all computed weight pays
miners. Published weights and actual on-chain alpha emissions must be observed separately.

## Capital-conditioned payout: adopted, currently disabled for Lane 1

Capital conditioning leaves the scientific winner unchanged. It limits the share of an allocated
lane budget the eligible winner can draw. For eligible effective capital `K`, a reference scale
`H > 0`, qualifying retained fresh principal `F` and reference principal `P > 0`, the adopted
reference multiplier is:

```text
p = max(K / (K + H), (1/5) * min(1, F/P))
p = 0 if payout eligibility fails
```

The second term is a fresh-principal floor that rises to 20% of the available lane share; it
does not give every registered miner 20% of emissions. The reference sets `H = K_ref / 19`,
where `K_ref` is the effective capital of `P` fresh principal at 30 days. Consequently,
`K = K_ref` gives `K/(K+H) = 0.95`. Neither `P` nor a live capital requirement is chosen by
this document; enabled signed policy supplies them.

### How effective capital ages

Capital belongs to a `(coldkey, netuid)` position and is attributed in retained lots. For a lot
of amount `A` and age `a` in blocks, the adopted unit values are:

```text
v_fresh(a) = (2 + 8 * min(1, a/216000)) * (0.25 + 0.75 * 2^(-a/2628000))

v_emitted(a) = 2^(-a/216000),  if a < 1296000
             = 0,            otherwise

K = sum_eligible_lots A_lot * v_source(age_lot)
```

The schedule uses 7,200 blocks/day as its design convention: 216,000 blocks corresponds to
30 days, 2,628,000 to 365 days, and 1,296,000 to 180 days. Wall-clock estimates are not the
valuation authority. Fresh unit value starts at 2, reaches about 9.58 at 30 days, then approaches
2.5; emitted value halves every 30 days and expires at 180 days. Aging changes economic credit,
not the amount of alpha that exists on chain.

Only attributed, retained, owner-matching and seasoned lots contribute. Untracked stake does not
become mining capital merely because it is visible on chain. Outstanding exit debt makes the
position's payable capital and qualifying fresh principal zero. A top-up has its own birth block;
moving or changing identity does not legitimately relabel old principal as new. Purchases consume
principal, so the same capital cannot fund both a purchase and retained mining credit.

If several conditioned lanes are enabled, qualifying capital and fresh principal are divided
equally across those signed lanes before allocation to eligible owner-matching hotkeys. Neutral
lanes are excluded from that divisor. Unused allocations are not moved to the lane where the
miner happens to be winning. One full signed valuation cadence of seasoning is required for
mining payout; purchase retention uses its own lot-age calculation.

## Retention-based work-token purchase design

The adopted purchase multiplier for a fresh lot is:

```text
rho(a) = 1 + 19 * min(1, a/2628000)^2
```

It starts at **1×**, rises quadratically to **20×** after 2,628,000 blocks of retention, and
then stays at 20×. Emitted and untracked alpha stays at 1× for purchases. This is separate from
the mining-capital aging function above.

| Fresh lot age | Purchase multiplier |
| --- | --- |
| Arrival | 1× |
| 216,000 blocks (design 30 days) | About 1.128× |
| 1,314,000 blocks (half the retention period) | 5.75× |
| 2,628,000 blocks or more | 20× |

For signed base conversion rate `r` in WT/alpha, consumed fresh lots `A_i` with ages `a_i`,
consumed emitted/untracked amount `A_other`, and existing fractional carry `c`:

```text
WT_exact = r * (sum_i A_i * rho(a_i) + A_other) + c
WT_credit = floor(WT_exact)
carry_new = WT_exact - WT_credit
```

Whole spendable tokens, the retained fractional remainder and the purchase identity commit
atomically. Carry belongs to the WT beneficiary, and replay changes neither balance nor carry.
The receipt fixes the rate and consumed lots at purchase time; later aging does not reprice a
past purchase. A lot that exits and returns starts a new retention clock; retained internal moves
and supported coldkey swaps preserve its history.

The operator selected a temporary base launch on2026-10-13: **1WT per alpha**,
**10WT per SAE evaluation**, and exact fractional carry per beneficiary. Full
retention premium activation and the complete recovery drill are deferred; history
and encrypted backups continue. Base-mode purchases receive base credit and do
not promise retroactive premium. Active authenticated funding terms must report
`base_rate_v1`; the original7200-finalized-block notice matured at9271373. Base
purchases are now enabled. The
retention design above describes the later premium policy, not this temporary rate.

### Evaluation funding and price changes

Compute and associated operating costs are covered by the operator from external
sources, using owned hardware. WT pays for evaluation access; the selected **10 WT
per SAE evaluation** does not depend on recovering GPU costs from deposited alpha.

Initial price activation and future changes require at least **7,200 finalized
mainnet blocks** of notice—about 24 hours at 12 seconds per block. Historical
purchases, balances and signed receipts are not repriced.

Budget with the actual quote and receipts, including training, failed attempts, renewals and
operating costs. Neither the capital formulas nor the retention schedule guarantee profit.

Return to [lane guides](lanes/README.md) or the [miner setup](README.md).
