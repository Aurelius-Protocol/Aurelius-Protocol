# Runtime488 funding client verification

The public `aurelius-miner fund` command admits runtime488/transaction1 using
the exact reviewed metadata SHA256 and `:code` storage BLAKE2b-256 hash.
The retained475/477/479/481 funding profiles remain available.

On 2026-11-07, an isolated GH200 arm64 Kubernetes Job verified:

- **506 tests passed** against source and **506 passed** against a separately
  installed wheel. The wheel imported funding from its own `site-packages`.
- Both import contracts, Ruff lint/format, warnings-as-errors source compilation,
  wheel construction and the installed `fund --help` command passed.
- Independent Python scalecodec and native Bittensor11 codecs agree on the
  treasury-bound transfer bytes and mortal signing payload. Tests retain exact
  code/metadata, transaction version, genesis, runtime-change, event, receipt and
  no-resubmit refusals.
- Finney and archive RPCs independently agreed on the canonical finalized
  block9450723 and its metadata, raw code and storage hash. Public observations
  and metadata are retained in `subnet/tests/fixtures/runtime488/`.
- The installed wheel's keyless `chain_scope` admitted finalized block9450764,
  `0x9c2e83616141ad82e6783eb8f0490a805ff465a0fced97174ad7e6af92352f2a`,
  using the exact488/transaction1 identity.

The client requires Central's executable funding terms to agree with the
finalized runtime before wallet unlock. Unknown identities still refuse. A chain
inclusion or a refund-related balance increase cannot establish purchase credit:
only the exact signed Central deposit receipt can do so.

No wallet was loaded, alpha transfer signed or chain mutation submitted during
these checks. This does not claim a new real purchase or a PyPI release.
