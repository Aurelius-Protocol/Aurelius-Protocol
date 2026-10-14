# Incentive mechanism

Miners compete by producing useful tools. A lane's scientific evaluation decides who is eligible
to earn; the subnet's economic policy decides how much of the lane's budget that miner receives.
Work-tokens (WT) pay for evaluation. Neither stake nor WT improves a scientific score.

For the current competition, see [Lane 1: SAEs](lanes/lane1.md).

> **Open beta.** The current version is an open beta. The values and mechanics below can and
> will change frequently, at any moment. The signed [public configuration](https://central.aureliusaligned.ai/api/config)
> is authoritative for current values.
>
> **The spirit of the rules matters more than these facts.** The intent is to reward useful
> work by miners who hold real stake in the subnet. Exploiting gaps in the implementation,
> staking briefly around valuations, borrowing stake, or counting the same stake twice is out
> of bounds, even where the mechanics below would allow it.
>
> **Audits.** We periodically audit rewards against chain data and may adjust retroactively.
>
> The miner asks you to agree to these terms before `serve` or `fund`, and asks again when the
> terms are re-versioned.

## How rewards are decided

1. **Science.** Only eligible, fresh results with the required signed charge and artifact
   evidence get a lane score. In Lane 1's current `throne_only` phase, only the champion does.
2. **Lane budget.** A base burn is taken first (currently **70%**). The remaining budget is
   divided between enabled lanes, then between eligible miners by normalized score.
3. **Stake.** Each miner's share is scaled by a payout multiplier `p` between 0 and 1, based on
   the alpha they currently hold on the subnet (see below).
4. **Burn.** Anything not paid out goes to burn. This includes the part held back by
   `p`, unused allocation and rounding. It is never redistributed to other miners.

```text
B_t     = (1 - b) * a_t                  # lane budget: base burn b, lane allocation a_t
w_(m,t) = B_t * f_(m,t) * p_m            # f: normalized score share, p: payout multiplier
```

Submitting copies does not multiply a winning share. A local score or approval label cannot
create one. Published weights and realized on-chain emissions are separate facts to check.

## How stake affects your reward

The multiplier `p` grows with the subnet 37 alpha you currently hold, with diminishing returns:

| Owned alpha | Multiplier `p` | Weight share as sole champion (after 70% burn) |
| --- | --- | --- |
| 0 | 0% | 0% |
| 100 | about 16% | about 4.8% |
| 1,000 | about 65.5% | about 19.7% |
| 10,000 | 95% | 28.5% |
| 100,000 | about 99.5% | about 29.8% |

- Stake is read from finalized chain state at each valuation, every 10 blocks. Only the
  current balance matters: how long you have held it, and where it came from, do not.
- **Only your own alpha counts:** alpha your coldkey owns on its registered hotkey. Stake
  delegated to you by others, TAO, and WT balances do not count.
- **Your position is your coldkey.** Stake is grouped by coldkey and shared across that
  coldkey's eligible registered hotkeys in proportion to their stake, so running more hotkeys
  does not multiply it.
- Selling or moving stake simply lowers your current balance; there is no exit penalty.
  Buying work-tokens transfers alpha away, so it lowers your stake too.
- If your stake cannot be read or verified for a valuation, that lane's share is withheld to
  burn for that valuation.

Stake is an economic reward condition. It is not a registration minimum and does not lock
your stake.

## Work-tokens

WT are prepaid evaluation credits recorded per `(coldkey, hotkey, netuid)`. They are separate
from registration fees, alpha and training compute.

- **Buying.** Transfer subnet alpha stake to the designated work-token wallet. Credit appears
  only after the transfer is finalized and attributed to you. Before sending, confirm the
  current destination, terms and that deposits are enabled. The current base rate is
  **1 WT per alpha**, with fractional remainders carried exactly per beneficiary.
- **Spending.** An SAE evaluation costs **10 WT**. One logical candidate is charged once per
  evaluation window; replaying the same commitment does not charge again. A new candidate, or
  the same artifact in a later epoch, is a new charge.
- **Failures and refunds.** A scientific loss still spends WT. Infrastructure failures are
  refund-eligible and ambiguous outcomes are reviewed. A refund exists only when the ledger
  records it.
- **Price changes.** Price changes take effect after at least **7,200 finalized blocks** of notice
  (about 24 hours). Past purchases and receipts are not repriced.

```sh
aurelius-miner balance --task-type interp_sae
aurelius-sae status --refresh --accounting --netuid "$NETUID"
```

The legacy `deposit` command is a development seam, not a production funding path. Budget
for training, failed attempts, renewals and operating costs; nothing here guarantees profit.

## Reference formula

For current owned stake `S` and reference principal `P = 10,000 alpha`:

```text
p = S / (S + P/19)
p = 0 if ineligible
```

## Current published settings

Open beta values, read from config version **29** on **2026-10-14**. They will change.

| Setting | Value |
| --- | --- |
| Earning lane | `interp_sae`, allocation `1` |
| Base burn fraction | `0.7` |
| Competition phase | `throne_only` |
| Single-miner share cap | `1.0` |
| Stake conditioning | `current_stake`, reference principal 10,000 alpha |
| Unused share | `to_burn` |

Return to [lane guides](lanes/README.md) or the [miner setup](README.md).
