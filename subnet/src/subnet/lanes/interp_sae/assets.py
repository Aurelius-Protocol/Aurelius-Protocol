# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Offline, content-addressed import for the M6 SAE referee asset set.

This operator command has no network client.  It accepts only a local preseed whose four
components are already described by broker-compatible ``aurelius-asset-manifest-v1.json`` files,
verifies the complete byte closure, and atomically publishes it below the SHA-256 of the exact root
descriptor bytes.  Missing real production bytes are therefore an explicit deployment
prerequisite, never a reason to add a downloader or follow an object-store redirect here.

The publication also carries one derived broker-compatible root manifest.  The broker mounts this
whole immutable publication at ``/assets`` so ``assets.json`` and all four components belong to the
same verified mount; it never assembles an unbound descriptor inside an evaluator process.
"""

from __future__ import annotations

MODEL_REVISION = "7df9a82518afdecae4e8c026b27adccc8c1f0032"


MODEL_DTYPE = "bfloat16"
