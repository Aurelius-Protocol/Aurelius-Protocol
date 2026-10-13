# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Domain-separated, short-lived authorization for one miner-owned read. No session or spend."""

import re
from .canonicalization import canonical_bytes
from .ss58 import ss58_decode

FIELDS = frozenset(
    {"hotkey", "netuid", "content_hash", "fanout_window", "config_hash", "timestamp", "nonce"}
)


DOMAIN = b"aurelius/miner-observation/v1\x00"


def signed_bytes(body: dict) -> bytes:
    if not isinstance(body, dict) or set(body) != FIELDS:
        raise ValueError("miner observation fields are not closed")
    ss58_decode(body["hotkey"])
    for name in ("content_hash", "config_hash", "nonce"):
        if not isinstance(body[name], str) or re.fullmatch(r"[0-9a-f]{64}", body[name]) is None:
            raise ValueError("observation requires exact hashes/nonce")
    for name in ("netuid", "fanout_window", "timestamp"):
        if type(body[name]) is not int or not 0 <= body[name] < 2**63:
            raise ValueError("observation requires bounded integers")
    return DOMAIN + canonical_bytes(body)
