"""The miner's one timestamp source.

Every timestamp the miner mints flows through here, as whole seconds since the epoch from the
wall clock. The validator compares a work-id's timestamp with its own within a freshness window,
so a miner and the validator it serves must both read the wall clock.

stdlib-only (the `shared-independent` import contract).
"""

from __future__ import annotations

import time


def now_ts() -> int:
    """Seconds since the epoch, from the wall clock."""
    return int(time.time())
