"""Miner wallet access (plan §1.8, invariant 1 support).

CONTRACT (capsule header)
- The HOTKEY is loaded freely — `serve` signs a fresh work_id with it per query. The COLDKEY is
  local-only and gated: `load_coldkey` refuses unless the caller explicitly unlocked it (the
  `deposit` command is the sole unlocker). The coldkey signs a chain transfer LOCALLY; it never
  enters a network payload — `deposit` is the only command that touches it and it is never sent.
- Testlab dev wallets are the committed deterministic seeds (`sha256("aurelius-testlab-wallet:"+
  name)`); a real deployment loads an encrypted keyfile and unlocks with a passphrase (seam).
"""

from __future__ import annotations

import hashlib

from subnet.shared.signing import keypair_from_seed
from subnet.shared.ss58 import ss58_encode


class ColdkeyLocked(Exception):
    """The coldkey was requested without an explicit unlock — refused (it must never leak)."""


def _dev_pair(name: str) -> tuple[tuple[bytes, bytes], str]:
    seed = hashlib.sha256(f"aurelius-testlab-wallet:{name}".encode()).digest()
    pair = keypair_from_seed(seed)
    return pair, ss58_encode(pair[0])


def load_hotkey(name: str) -> tuple[tuple[bytes, bytes], str]:
    """The signing hotkey — loaded freely (public identity; used to sign work_ids)."""
    return _dev_pair(name)


def load_coldkey(name: str, *, unlocked: bool) -> tuple[tuple[bytes, bytes], str]:
    """The coldkey — refused unless explicitly unlocked. Used LOCALLY only (deposit signing);
    never placed in a payload."""
    if not unlocked:
        raise ColdkeyLocked(
            "coldkey access requires an explicit unlock (deposit --unlock) — it is local-only"
        )
    return _dev_pair(name)
