# Runtime490 gamma funding fixture

Captured from finalized bittensor mainnet state on 2026-11-13, at block 9496593. Runtime 490
became executable at block 9478217.
`capture.json` records the genesis hash, the block hash, runtime/transaction versions, the
exact metadata SHA256, raw code SHA256, the recomputed BLAKE2b-256 `:code` storage hash and the
runtime's constant APIs, corroborated through two independent providers.

The metadata is public chain data and contains no wallet material. Tests decode the miner's
hand-composed `spend_gamma` call, signed extrinsic and storage layouts against it through
scalecodec, and pin the reviewed runtime identity that gamma funding checks before it signs.
This fixture does not claim a real funding purchase.
