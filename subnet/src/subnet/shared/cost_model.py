# Miner dependency subset extracted from Aurelius v5 (02375403).
"""cost_class / cost_model_version derivation (shared library, spec §3.2).

CONTRACT (capsule header)
- Owns (§12): 2 (cost authority: locally derivable, server-verified) jointly with the server's
  re-derivation at novelty/charge (Elixir `CentralApi.Shared.CostModel` mirrors this).
- Guarantees: deterministic — every validator, the miner CLI, and the server derive the same
  `(cost_class, cost_model_version)` for the same input, so `content_hash` (which folds both in)
  is identical everywhere. Conformance vectors: `fixtures/canonicalization/cost_class_vectors.json`.

Legacy lanes retain the spec's sanctioned trivial form: ONE config-independent bucket
(``standard``) under ``cost_model_version = 1``. The ``interp_sae`` lane starts at version 2 with
the config-independent ``gpu_sae_verdict`` bucket reserved by C5. This is identity versioning, not
a production price-model claim: version activation is selected by task type at the local lane
boundary and the server re-derives the same selection.
"""

from __future__ import annotations
from subnet.shared.canonicalization import CanonicalizationError, JsonValue

COST_MODEL_VERSION = 1


_INTERP_SAE_COST_MODEL_VERSION = 2


def cost_model_version(task_type: str) -> int:
    """Return the locally shipped cost-policy version for ``task_type``.

    Unknown future task names retain the v1 trivial policy until their locally shipped shared
    derivation explicitly opts into a later version. Remote config cannot inject this mapping.
    """
    _validate_task_type(task_type)
    return _INTERP_SAE_COST_MODEL_VERSION if task_type == "interp_sae" else COST_MODEL_VERSION


def derive_cost_class(task_type: str, config: JsonValue) -> str:
    """Return the deterministic coarse cost bucket for a work item."""
    _validate_task_type(task_type)
    _ = config  # both shipped policies are config-independent; argument pins the future signature
    return "gpu_sae_verdict" if task_type == "interp_sae" else "standard"


def _validate_task_type(task_type: str) -> None:
    if not isinstance(task_type, str) or not task_type:
        raise CanonicalizationError("bad_field", "task_type must be a non-empty string")
