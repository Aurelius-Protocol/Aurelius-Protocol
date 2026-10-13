# Incentive mechanism

Miners compete by producing useful tools. A lane's scientific evaluation determines who is
eligible to earn; the subnet's economic policy determines the reward allocation. Work-tokens
pay for evaluation. Buying more evaluation budget does not improve a scientific score.

For the current competition, see [Lane 1: SAEs](lanes/lane1.md). Other lanes can use different
effectiveness measures while sharing the miner, accounting and reward-allocation machinery.

## Current launch policy

Work-token purchases are open at **1 WT per alpha** with exact fractional carry.
SAE evaluation costs **10 WT**. Real WT is charged first; sufficient enabled
synthetic WT is a fallback. The two balances are never combined for one charge.

Retention tracking and sell-off penalties are suspended while RPC capacity is upgraded.
The immutable original histories, debt records, prior grant and encrypted backups survive;
the original anchor remains9261069. Neither current stake nor WT purchases invent past
retention. Restarting or upgrading RPC does not automatically resume penalties.

Signed snapshot28 enables **current owned subnet37 alpha**, with reference principal
**10,000 alpha**, without an age requirement or exit-debt penalty. The matching validator authority is schema6; historical snapshot27 and its
retention grant remain inactive. Check [authenticated current config](https://central.aureliusaligned.ai/api/config),
charge receipts, published chain weights and realized emissions separately.

The scientific lane remains `throne_only`: only an eligible scored champion has a positive
scientific share. Base burn is0, the single-miner cap is1.0, and unused or economically withheld
weight goes to burn. No wallet/UID allowlist or additional signup is required.

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

Only eligible, fresh results supported by the required signed charge and artifact evidence can
enter `q`. Submitting many copies does not multiply ownership of the winning result. A local
training score or a bare approval label cannot create an eligible share.

For Lane 1's current `throne_only` phase, only the champion has a positive score. The internal
ladder score cap is **not** a second absolute payout cap: the allocation engine normalizes the
eligible scores. With only the champion eligible, its scientific share is `1`. The separate
lane-health `max_single_miner_share` caps its fraction of the lane budget. The selected
cap is `1.0`, with base burn `0`. The champion's
scientific share is multiplied by its verified economic payout factor; a zero
factor withholds its entire lane share to burn.

Missing or stale results, unavailable required evidence and failed payout gates can leave no
payable champion. Under the current `to_burn` policy, unused weight goes to the burn destination.
Capital-withheld weight, if that mechanism is enabled, also goes to burn and is not redistributed
to losing miners. A configured zero base burn does not guarantee that all computed weight pays
miners. Published weights and actual on-chain alpha emissions must be observed separately.

## Current-alpha payout multiplier

For current owned eligible subnet alpha `S` and reference `P = 10,000 alpha`:

```text
p = S / (S + P/19)
```

| Owned alpha | Multiplier before scientific share and chain encoding |
| --- | --- |
| 0 | 0% |
| 100 | About 16.0% |
| 1,000 | About 65.5% |
| 10,000 | 95% |
| 100,000 | About 99.5% |

Only alpha owned by the registered hotkey's coldkey on that eligible hotkey counts.
Delegated third-party stake, TAO-valued totals, treasury custody and WT balances do not.
Eligible hotkeys under one coldkey share its grouped multiplier in proportion to their owned
alpha. Adding hotkeys cannot multiply the same observed stake. With several conditioned lanes,
stake is divided equally over those lanes before calculating each multiplier.

The validator reads exact finalized positions in one bounded batch at the signed10-block
valuation cadence. It does not scan sell history, create aged lots or enforce exit debt.
Missing, stale or mismatched current evidence withholds the conditioned lane. Selling alpha
can reduce the next current-stake multiplier; it creates no additional sell-off penalty in this
mode. This is a payout condition, not a staking lock or a registration minimum.

This formula scales the eligible scientific share; buying stake alone cannot win a score,
claim the crown or earn without a valid paid result. Unpaid and stale candidates remain
ineligible. Actual earnings depend on available subnet emissions and finalized chain weights.

## Retention-based work-token purchase design

The adopted purchase multiplier for a fresh lot is:

```text
rho(a) = 1 + 19 * min(1, a/2628000)^2
```

It starts at **1×**, rises quadratically to **20×** after 2,628,000 blocks of retention, and
then stays at 20×. Emitted and untracked alpha stays at 1× for purchases. This is separate from
the current-alpha payout multiplier above.

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
retention premium activation and the complete recovery drill are deferred; history capture is paused and
existing encrypted backups are preserved. Base-mode purchases receive base credit and do
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
