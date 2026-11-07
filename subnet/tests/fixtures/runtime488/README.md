# Runtime488 funding fixture

Captured on GH200 from finalized mainnet block 9450723 on 2026-11-07.
`capture.json` records independent Finney and archive RPC observations of the same
canonical block, runtime/transaction versions, exact metadata SHA256, raw code
SHA256 and the independently recomputed BLAKE2b-256 `:code` storage hash.

The metadata is public chain data. No wallet was loaded, transfer signed or chain
mutation submitted. Tests compare the native Bittensor11 codec and independent
Python scalecodec against the client's exact treasury-bound transfer and mortal
signing payload; this fixture does not claim a real funding purchase.
