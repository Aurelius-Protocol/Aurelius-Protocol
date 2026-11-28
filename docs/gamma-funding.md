# Gamma funding

**Status: not open.** Subnet 37 does not accept gamma funding yet. The commands are in the
miner so that nothing changes for you when it opens. Until then, `fund --method gamma` reports
what is missing, and `--prepare` refuses with the server's reason.

## What it is

`fund` buys the work-tokens (WT) that evaluations are paid in. The default method pays with
subnet 37 alpha. The gamma method pays with **gamma**, a balance the chain holds for each
account on the subnet:

- The payment is one direct `spend_gamma` call, signed by your funding coldkey. The gamma is
  burned; nothing is transferred.
- WT credit is confirmed only by a signed funding receipt from the server. A successful chain
  call, or a balance that changed, does not prove credit.
- It reads only the hotkey and the chain to tell you whether you are ready. It unlocks the
  coldkey once, to sign one prepared payment, and never for a read.

Buying WT does not improve your score. See [INCENTIVES.md](../INCENTIVES.md#work-tokens).

## What has to be true

`fund --method gamma` checks all of this and names what fails. It does not change any of it.

To buy WT with gamma:

- **The chain** runs a runtime the miner has reviewed (today 490). Another runtime refuses.
- **The server** offers gamma funding and publishes a signed purchase policy. Its packs are the
  only amounts you can buy.
- **A separate gamma API hotkey** holds your gamma deposit. It must differ from your registered
  hotkey and from both coldkeys.
- **The deposit** belongs to your funding coldkey, which must also be its *spender*. The chain
  names the subnet's service spender by default, so the deposit owner has to set this. The
  miner never changes a spender for you.
- **The deposit can pay**: enough gamma for the pack, and room under its daily spend cap if it
  has one.
- **Your funding coldkey** owns your registered hotkey, has TAO for the fee and no pending
  transaction.

To mint gamma, the chain must also allow it, on the network and on subnet 37, publish a gamma
price for the subnet, and have a TAO/USD price that is at most 300 blocks old.

Like the default method, this one needs your agreement to the
[incentive terms](../INCENTIVES.md#work-tokens) first.

## Check readiness

No wallet is loaded and nothing is signed.

```sh
aurelius-miner fund --method gamma
```

This reports the chain identity, whether the server offers gamma funding and why not, the
policy with its packs, and whether minting is possible.

Add your gamma API hotkey to check your own deposit and account, and a pack to check it against
that pack's price:

```sh
aurelius-miner fund --method gamma --gamma-hotkey GAMMA_HOTKEY_SS58 --pack PACK_ID
```

The report's `spend_readiness.ready` is `true` when everything above holds. `reasons` lists
what does not, for example `gamma_deposit_absent`, `spender_mismatch` or
`insufficient_gamma_balance`.

## Prepare, then submit once

Run both on your **wallet machine**. Prepare obtains a quote from the server, verifies it
against the signed policy, and signs exactly one payment. Nothing is sent yet:

```sh
aurelius-miner fund --method gamma --prepare --intent-dir "$PWD/funding/gamma-001" \
  --pack PACK_ID --gamma-hotkey GAMMA_HOTKEY_SS58 \
  --wallet-name miner --wallet-path "$HOME/.bittensor/wallets"
```

Review the printed amount, fee cap and validity. Then submit what was signed, from any machine,
with no wallet present:

```sh
aurelius-miner fund --method gamma --execute --intent-dir "$PWD/funding/gamma-001"
```

- Keep the `--intent-dir`. It holds the journal of this one purchase.
- A new purchase gets a new directory. Reusing the directory with another pack or hotkey refuses.
- `--fee-cap-tao` (default `0.01`, at most `0.1`) and `--era-period-blocks` (default `64`) bound
  the payment. The era must be a power of two from 4 to 65536 blocks, and it must fit inside
  the quote's lifetime.

## Observe the result

Read-only, and safe to rerun as often as you like:

```sh
aurelius-miner fund --method gamma --status --intent-dir "$PWD/funding/gamma-001"
```

| `status` | What it means |
| --- | --- |
| `PREPARED_NOT_SUBMITTED` | Signed, never sent. Only `--execute` sends it. |
| `OUTCOME_UNKNOWN` | Sent, and not yet knowable. Rerun `--status` later. |
| `OBSERVED_ON_CHAIN` | In a finalized block. `spend_event` reports spent or skipped. |
| `NOT_ON_CHAIN_AFTER_ERA` | Too late to be included; unpaid once the server reports no settlement. |
| `CENTRAL_CREDITED` | A signed funding receipt was verified. This is the only proof of WT credit. |

**Never delete the journal to try again.** After a submission was attempted, `--execute` and
`--prepare` only observe; they never sign or send a second payment for the same directory. An
unknown outcome is not a failure: wait for the finalized chain to advance and rerun `--status`.

## Mint gamma

`--mint-setup` burns TAO to add gamma to your deposit. It is separate from buying WT and never
runs on its own. Without `--execute` it is a readiness report and shows the deposit's actual
spender:

```sh
aurelius-miner fund --method gamma --mint-setup --intent-dir "$PWD/funding/gamma-mint-001" \
  --gamma-hotkey GAMMA_HOTKEY_SS58 --wallet-name miner --wallet-path "$HOME/.bittensor/wallets"
```

To mint, add the amounts and both confirmations. `--mint-tao` is the TAO to burn;
`--mint-min-gamma-out` is the least gamma you accept, in atoms; `--mint-daily-cap` optionally
sets the deposit's daily spend cap, in atoms. The exact gamma you receive is computed by the
chain, never by the miner.

```sh
aurelius-miner fund --method gamma --mint-setup --execute --confirm-mint-burns-tao \
  --intent-dir "$PWD/funding/gamma-mint-001" --gamma-hotkey GAMMA_HOTKEY_SS58 \
  --wallet-name miner --wallet-path "$HOME/.bittensor/wallets" \
  --mint-tao 1 --mint-min-gamma-out 100000000
```

The mint is submitted at most once per `--intent-dir`. If its outcome is unknown, read the
deposit balance with the readiness check; do not repeat the mint.

## Refusals

Every refusal prints `{"status": "refused", "reason": "..."}` and exits with code 2. A refusal
in a check signed nothing. A refusal in `--prepare` may come after the coldkey signed, but
nothing was sent and no submission was attempted. After an attempted submission, rerun
`--status` and read the table above.
