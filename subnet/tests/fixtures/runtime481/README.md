# Runtime481 funding evidence

`metadata.scale` and `capture.json` describe finalized Bittensor mainnet block
9320995 (`0x99e1a0637e9fbc3d6ad949a6f5db9e4b411c128809f261d1c6161361d9d7335d`).
The metadata SHA256 is recorded in `capture.json`. Its `code_sha256` identifies the
raw runtime WASM; the funding profile separately pins the node’s BLAKE2b-256
`:code` storage hash, not that SHA256.

`native-transfer.json` contains public raw extrinsic and System.Events bytes from
finalized block 9321866, extrinsic 15. This is a foreign subnet93 transfer, not an
SN37 treasury deposit. It proves native runtime481 decoding and rejection as an
SN37 funding success. Its alpha amount is 2,751,908,400 rao; the event’s TAO
valuation must not be interpreted as that alpha amount. Existing runtime479
fixtures retain the positive SN37 receipt path. No live transfer was submitted
when collecting or testing this evidence.
