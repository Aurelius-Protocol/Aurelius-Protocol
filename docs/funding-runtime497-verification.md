# Runtime497 serving and funding client verification

The public miner admits runtime497 at transaction version 2: `aurelius-miner serve` for the
ServeAxon call, and `aurelius-miner fund` using the exact reviewed metadata SHA256 and `:code`
storage BLAKE2b-256 hash. The retained475/477/479/481/488 funding profiles remain available,
each at transaction version 1.

Runtime497 is the first reviewed runtime at transaction version 2. The chain's version became
2 at block 9,570,596 (2026-11-23T16:27:00Z) under runtime494. The miner used to compare it
with a fixed 1, so it refused every runtime from 494 on. Each reviewed runtime now carries the
transaction version its own code reports, and a chain, a retained intent or a server naming
another version for one of them is refused.

On 2026-11-28, an isolated Linux container (arm64, Python 3.12, no wallet, no GPU) verified:

- **803 tests passed** against the source tree and **803 passed** against a separately built
  wheel. The wheel was put ahead of the source on the import path, and the package was
  confirmed to load from it.
- Ruff lint and format at the lockfile's version, warnings-as-errors source compilation,
  wheel construction, and the built wheel's `fund --help` and `serve --help` passed. Both
  import contracts were kept (run at the container's import-linter 2.12; the lockfile names
  2.15).
- Independent Python scalecodec and the native Bittensor11 codecs agree on the ServeAxon
  endpoint bytes, the treasury-bound transfer bytes and the mortal signing payload, in which
  the signed data carries 497 and 2. The signed-extension list and the extrinsic version are
  the same as at runtime479; a test compares the two metadata documents.
- Tests retain exact code/metadata, transaction version, genesis, runtime-change, event,
  receipt and no-resubmit refusals, on every reviewed funding runtime. Transaction version 2
  does not admit an unreviewed runtime: 496 and 498 refuse with it as 478 and 480 do with 1.
- Finney and archive RPCs independently agreed on the canonical finalized block9602450,
  `0x99057c78d3da30f9dcf71dd5ff059355c7afa828054402c97a9fb60c30ed9635`, and on its metadata,
  raw code and storage hash. Public observations and metadata are retained in
  `subnet/tests/fixtures/runtime497/`.
- With no key, through each of the two providers, the client's funding scope admitted
  finalized block9602567, `0x6e632afb882a080f9d5b252a9c2a48a5bc766e0915be15e8e76648bb6735ddee`,
  under the exact497/transaction2 identity, and its serving preflight admitted the same block
  at pins 497 and 2. The previous pins, 479 and 488, were refused.

The client requires Central's executable funding terms to name the same finalized runtime and
transaction version before wallet unlock; terms naming another version, and unknown
identities, still refuse. A chain inclusion or a refund-related balance increase cannot
establish purchase credit: only the exact signed Central deposit receipt can do so.
`.env.example` pins 497 and 2, and a test keeps the example's pins on a reviewed runtime.

`fund --method gamma` is unchanged. It remains reviewed for runtime490 at transaction
version 1 and refuses on any other runtime.

No wallet was loaded, endpoint published, alpha transfer signed or chain mutation submitted
during these checks. This does not claim a real purchase or a PyPI release.
