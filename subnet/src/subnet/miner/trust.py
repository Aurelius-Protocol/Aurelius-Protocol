"""Release-pinned public configuration authority for the subnet 37 miner."""

from __future__ import annotations

from collections.abc import Mapping
import os
import re

# Public Ed25519 key from the deployed SN37 miner's CONFIG_SIGNING_PUBKEY.
# Verify release updates against a signed production snapshot before changing this pin.
# This key is public verification material, never a wallet or signing secret.
SN37_CONFIG_SIGNING_PUBKEY = "42cc02c98d4cea3f899bdabb2a8a335c2e6e96ac34d9da9a6cbf5fdc07305d46"
# Verified against the deployed SN37 receipt authority; never accept a key from a response.
SN37_RECEIPT_SIGNING_PUBKEY = "523b928b627171a363bcd0f6cee7e283fc5d04d47fd85af9a5bbf334fef9bd07"


def config_signing_pubkey(environ: Mapping[str, str] | None = None) -> str:
    """Use the shipped key unless an explicit, valid override selects another authority.

    Missing means the release default; empty or malformed overrides fail closed. No server
    response may supply or replace this trust anchor.
    """
    env = os.environ if environ is None else environ
    key = env.get("CONFIG_SIGNING_PUBKEY", SN37_CONFIG_SIGNING_PUBKEY)
    if not isinstance(key, str) or re.fullmatch(r"[0-9a-fA-F]{64}", key) is None:
        raise ValueError("CONFIG_SIGNING_PUBKEY must be a 32-byte hex key")
    return key
