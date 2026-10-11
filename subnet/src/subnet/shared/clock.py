# Miner dependency subset extracted from Aurelius v5 (02375403).
"""The clock seam (testlab design §4): every timestamp the subnet mints flows through here.

CONTRACT (capsule header)
- `AURELIUS_CLOCK=real` ⇒ wall clock; anything else (default `frozen`) ⇒ the fixed testlab
  timestamp (`AURELIUS_FROZEN_TS`, default 1782900000). Read PER CALL, never at import time, so
  long-running processes and tests see env changes without re-imports.
- stdlib-only (the `shared-independent` import contract): both the miner (work_id timestamps) and
  the validator (pipeline `started_at`) depend on this seam; neither may drag a dependency in.
- Freshness interplay: `stages/reference.is_fresh` compares the miner work_id ts against the
  validator's `started_at` (±3600s/±300s). Both sides on the same mode keeps the delta ~0; the
  seam is honest only if BOTH flip together (the compose env does exactly that).
"""

from __future__ import annotations
import os
import time

FROZEN_TS_DEFAULT = 1_782_900_000


def now_ts(clock: str | None = None) -> int:
    """The subnet's one timestamp source. `clock` overrides the env (callers that resolved the
    mode at construction pass it through); None reads `AURELIUS_CLOCK` fresh."""
    mode = clock or os.environ.get("AURELIUS_CLOCK", "frozen")
    if mode == "real":
        return int(time.time())
    return int(os.environ.get("AURELIUS_FROZEN_TS", str(FROZEN_TS_DEFAULT)))
