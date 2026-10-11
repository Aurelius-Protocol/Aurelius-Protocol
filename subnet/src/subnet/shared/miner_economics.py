# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Short-lived, miner-owned economics read capability. No session or spending authority."""

import re
from .canonicalization import canonical_bytes
from .ss58 import ss58_decode

DOMAIN = b"aurelius/miner-economics/v1\x00"


FIELDS = frozenset({"hotkey", "netuid", "config_hash", "timestamp", "nonce"})


def signed_bytes(body):
    if type(body) is not dict or set(body) != FIELDS:
        raise ValueError("economics fields are not closed")
    ss58_decode(body["hotkey"])
    for key in ("config_hash", "nonce"):
        if type(body[key]) is not str or not re.fullmatch("[0-9a-f]{64}", body[key]):
            raise ValueError("economics requires exact hash and nonce")
    for key in ("netuid", "timestamp"):
        if type(body[key]) is not int or not 0 <= body[key] < 2**63:
            raise ValueError("economics requires bounded integer")
    return DOMAIN + canonical_bytes(body)
