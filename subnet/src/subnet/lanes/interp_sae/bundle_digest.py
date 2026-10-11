"""Dependency-free S4 bundle digest: domain-separated SHA-256 over cfg bytes + streamed weights.

This is a lane LEAF module: standard library only, no ``subnet.*`` imports, so it can be imported
by the sidecar (and any future embedder) without pulling in the adapter or any heavy lane module.
The digest construction is frozen by the S4 contract — changing it invalidates every published
artifact hash.
"""

from __future__ import annotations

import hashlib
import hmac

BUNDLE_DOMAIN: bytes = b"AURELIUS-SAE-LENS-BUNDLE-v1\0"
CFG_FILENAME = "cfg.json"
WEIGHTS_FILENAME = "sae_weights.safetensors"
MAX_CFG_BYTES = 65536


class S4BundleHasher:
    """Incremental S4 bundle digest: feed the cfg once, then stream the exact weights bytes.

    The hasher binds the domain separator, the exact cfg length and bytes, and the declared
    weights size up front; weight chunks may then arrive in any chunking but must total exactly
    ``weights_size`` bytes before :meth:`hexdigest` is legal.
    """

    def __init__(self, cfg: bytes, weights_size: int) -> None:
        if type(cfg) is not bytes:
            raise ValueError("cfg must be bytes")
        if not cfg:
            raise ValueError("cfg is empty")
        if len(cfg) > MAX_CFG_BYTES:
            raise ValueError(f"cfg exceeds {MAX_CFG_BYTES} bytes")
        if type(weights_size) is not int or weights_size < 0:
            raise ValueError("weights_size must be a non-negative int")
        self._weights_size = weights_size
        self._streamed = 0
        self._digest = hashlib.sha256()
        self._digest.update(BUNDLE_DOMAIN)
        self._digest.update(len(cfg).to_bytes(8, "big"))
        self._digest.update(cfg)
        self._digest.update(weights_size.to_bytes(8, "big"))

    def update(self, chunk: bytes) -> None:
        """Absorb one non-empty weights chunk; refuse to stream past the declared size."""

        if type(chunk) is not bytes or not chunk:
            raise ValueError("weight chunk must be a non-empty bytes value")
        if self._streamed + len(chunk) > self._weights_size:
            raise ValueError("weight stream exceeds its declared size")
        self._streamed += len(chunk)
        self._digest.update(chunk)

    def hexdigest(self) -> str:
        """Return the S4 bundle digest; refuse unless exactly ``weights_size`` bytes streamed."""

        if self._streamed != self._weights_size:
            raise ValueError("weight stream differs from its declared size")
        return self._digest.hexdigest()


def bundle_digest(cfg: bytes, weights: bytes) -> str:
    """One-shot S4 bundle digest over the exact cfg and weights bytes (tests/CLI convenience)."""

    if type(weights) is not bytes:
        raise ValueError("weights must be bytes")
    hasher = S4BundleHasher(cfg, len(weights))
    if weights:
        hasher.update(weights)
    return hasher.hexdigest()


def digests_equal(a: str, b: str) -> bool:
    """Constant-time digest comparison."""

    return hmac.compare_digest(a, b)
