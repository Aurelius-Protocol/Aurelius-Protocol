# Miner dependency subset extracted from Aurelius v5 (02375403).
"""fanout_window derivation (shared library, spec §3.2).

CONTRACT (capsule header)
- Owns (§12): 31 (window half: `fanout_window = floor(finalized_block / window_blocks)`,
  CHAIN-derived, never wall-clock).
- Guarantees: pure function of a finalized block number — wall-clock cannot leak in by
  construction. The finalized block arrives through a seam (`FinalizedHeadSource` on the Python
  side, `ChainView` on the Elixir side); this module never reads a clock or the network.

The charge/fanout idempotency key is `(miner, content_hash, fanout_window)`; the server's own
derivation at charge time is authoritative (§8.3), the validator's local derivation is the §3.2
advisory pre-computation.
"""

from __future__ import annotations

AUTHORITATIVE_WINDOW_BLOCKS = 100
