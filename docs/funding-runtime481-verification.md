# Runtime481 funding client verification

The public `aurelius-miner fund` command admits the exact reviewed runtime481
metadata and `:code` storage hash through its shared funding profile table.
Retained475/477/479 profiles remain supported. Unknown identities still refuse
before wallet unlock; a signed chain submission still requires its own verified
Central receipt before the client reports credited funding.

Validation on 2026-10-20 used an isolated GH200 arm64 Kubernetes Job:

- Full public source suite: **463 passed**.
- Full suite against the built wheel in a separate environment: **463 passed**.
  The imported funding module came from the installed wheel's `site-packages`.
- Both import contracts, Ruff checks/format and warnings-as-errors source
  compilation passed. The installed `fund --help` command succeeded.
- Runtime481 transfer bytes match independent Python scalecodec and the
  Bittensor11 native codec. Signed extensions and mortal payload layout match.
- Wallet guards, ambiguous one-shot submission, retained intent observation,
  signed receipts, exact code/metadata refusal and runtime-change refusal remain
  covered. The source and fixture hashes match the files tested in the cluster.
- At **03:50:42 UTC**, the installed wheel's read-only `chain_scope` admitted
  finalized Bittensor mainnet block **9322132**,
  `0x4a655c6ac551465aca8f60ced190223a0e81b42c3fe9f5295bcc9516e764921c`,
  with spec481/transaction1 and exact reviewed code/metadata.

No wallet was loaded and no real transfer was signed or submitted. The recorded
native481 fixture is a foreign subnet93 transfer and is explicitly rejected as an
SN37 funding success. Existing479 fixtures retain the positive SN37 path. This
verification does not claim a new native481 SN37 payment or a package release.
