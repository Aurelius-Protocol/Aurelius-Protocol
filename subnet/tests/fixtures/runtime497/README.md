# Runtime497 fixture

Captured from finalized mainnet block 9602450 on 2026-11-28, about four and a half days
after the chain's transaction version became 2 (block 9570596, runtime494).
`capture.json` records independent Finney and archive RPC observations of the same
canonical block, runtime/transaction versions, exact metadata SHA256, raw code SHA256 and
the independently recomputed BLAKE2b-256 `:code` storage hash.

The metadata is public chain data. No wallet was loaded, transfer signed or chain
mutation submitted. Tests compare the native Bittensor11 codec and independent Python
scalecodec against the client's exact ServeAxon endpoint bytes, treasury-bound transfer and
mortal signing payload, and compare the signing extensions with runtime479's; this fixture
does not claim a real endpoint publication or funding purchase.
