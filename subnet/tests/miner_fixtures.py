"""Test identities for the miner.

The package derives no key from a name: a signer is built from an address and a signing
callable, and in production both come from the mounted Bittensor wallet. Tests build theirs
from fixed seeds, so a test identity has the same address on every run and on every machine.

Nothing here is named like a test, so pytest never collects it.
"""

from __future__ import annotations

import hashlib

from subnet.miner import serve
from subnet.miner.serve import WorkIdMinter
from subnet.shared.signing import keypair_from_seed, sign
from subnet.shared.ss58 import ss58_encode

_SALT = b"aurelius-miner-test-identity/v1:"


def fixed_keypair(name: str) -> tuple[tuple[bytes, bytes], str]:
    """A fixed sr25519 pair and its address for a test identity."""
    pair = keypair_from_seed(hashlib.sha256(_SALT + name.encode()).digest())
    return pair, ss58_encode(pair[0])


def fixed_minter(name: str) -> WorkIdMinter:
    """A signing identity for a test."""
    pair, address = fixed_keypair(name)
    return WorkIdMinter(hotkey_ss58=address, signer=lambda message: sign(message, pair))


def mount_wallet(monkeypatch, name: str):
    """Stand in for the mounted, hotkey-only Bittensor wallet with the fixed identity ``name``.

    Every command that signs loads the wallet, so a test that drives one mounts this first.
    """
    pair, address = fixed_keypair(name)

    class Hotkey:
        ss58_address = address
        sign = staticmethod(lambda message: sign(message, pair))

    class Wallet:
        hotkey = Hotkey()

    monkeypatch.setenv("AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET", "true")
    monkeypatch.setattr(serve, "load_bittensor_wallet", lambda: Wallet())
    return Wallet()
