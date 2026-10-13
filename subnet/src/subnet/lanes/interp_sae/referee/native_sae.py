# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Closed SAE-Lens 6.46 native-forge ABI and descriptor-only inference loader.

This deliberately admits the reviewed native export profile, not every historical SAE-Lens
configuration. BatchTopK/Matryoshka training exports use the real JumpReLU inference class.
No submitted path, import name, model kwargs, or training implementation is executed.
"""

from __future__ import annotations
from collections.abc import Mapping
import json
import math
from types import MappingProxyType
from typing import Any
from ..assets import MODEL_DTYPE, MODEL_REVISION
from .gate0 import SlotPins, check_cfg_bytes
from .runtime import LOAD_ERROR, RuntimeFailure, TensorField, TensorSchema

SAE_LENS_VERSION = "6.46.0"


NATIVE_ABI_ID = "sae-lens-6.46.0-native-forge-f32-v1"


INFERENCE_ARCHITECTURES = ("standard", "gated", "jumprelu", "topk")


_ROOT_FIELDS = frozenset(
    {
        "architecture",
        "d_in",
        "d_sae",
        "dtype",
        "device",
        "apply_b_dec_to_input",
        "normalize_activations",
        "reshape_activations",
        "metadata",
    }
)


_METADATA_FIELDS = frozenset(
    {
        "sae_lens_version",
        "sae_lens_training_version",
        "model_name",
        "model_class_name",
        "model_from_pretrained_kwargs",
        "hook_name",
        "hook_head_index",
        "context_size",
        "prepend_bos",
        "seqpos_slice",
        "exclude_special_tokens",
        "sequence_separator_token",
        "disable_concat_sequences",
        "dataset_path",
        "training_architecture",
        "training_architecture_details",
    }
)


_TRAINING_FIELDS = {
    "standard": {"l1_coefficient", "lp_norm", "l1_warm_up_steps"},
    "gated": {"l1_coefficient", "l1_warm_up_steps"},
    "jumprelu": {
        "jumprelu_init_threshold",
        "jumprelu_bandwidth",
        "jumprelu_sparsity_loss_mode",
        "l0_coefficient",
        "l0_warm_up_steps",
        "pre_act_loss_coefficient",
        "jumprelu_tanh_scale",
    },
    "topk": {"use_sparse_activations", "aux_loss_coefficient"},
    "batchtopk": {
        "use_sparse_activations",
        "aux_loss_coefficient",
        "k",
        "rescale_acts_by_decoder_norm",
        "topk_threshold_lr",
    },
    "matryoshka_batchtopk": {
        "use_sparse_activations",
        "aux_loss_coefficient",
        "k",
        "rescale_acts_by_decoder_norm",
        "topk_threshold_lr",
        "matryoshka_widths",
        "use_matryoshka_aux_loss",
    },
}


def _reject(detail: str) -> None:
    raise RuntimeFailure(LOAD_ERROR, detail)


class NativeCfgContentError(RuntimeFailure):
    """Only a validated submission's closed cfg is incompatible, not the runtime or loader."""

    def __init__(self, detail: str) -> None:
        super().__init__(LOAD_ERROR, detail)


def validate_native_cfg(cfg: Mapping[str, Any], pins: SlotPins) -> None:
    """Validate the complete saved inference cfg, including non-executable provenance."""

    try:
        _validate_native_cfg(cfg, pins)
    except RuntimeFailure as exc:
        raise NativeCfgContentError(str(exc)) from exc


def _validate_native_cfg(cfg: Mapping[str, Any], pins: SlotPins) -> None:

    architecture = cfg.get("architecture")
    if type(architecture) is not str or architecture not in INFERENCE_ARCHITECTURES:
        _reject("native inference architecture is not admitted")
    fields = _ROOT_FIELDS | (
        {"k", "rescale_acts_by_decoder_norm"} if architecture == "topk" else set()
    )
    if set(cfg) != fields:
        _reject("native inference cfg keys are not closed")
    # Keep Gate 0's signed pins and JSON/type/size checks as the single common-field authority.
    check_cfg_bytes(
        json.dumps(dict(cfg), allow_nan=False, separators=(",", ":")).encode("utf-8"), pins
    )
    if (
        cfg["device"] != "cuda"
        or cfg["dtype"] != "float32"
        or cfg["reshape_activations"] != "none"
        or cfg["normalize_activations"] != "none"
        or cfg["apply_b_dec_to_input"] is not False
        or (architecture == "topk" and cfg["rescale_acts_by_decoder_norm"] is not True)
    ):
        _reject("native inference semantics differ from the reviewed forge profile")
    metadata = cfg["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_FIELDS:
        _reject("native inference metadata keys are not closed")
    exact = {
        "sae_lens_version": SAE_LENS_VERSION,
        "sae_lens_training_version": SAE_LENS_VERSION,
        "model_class_name": "HookedTransformer",
        "model_from_pretrained_kwargs": {"revision": MODEL_REVISION, "dtype": MODEL_DTYPE},
        "hook_head_index": None,
        "prepend_bos": True,
        "seqpos_slice": [None],
        "exclude_special_tokens": True,
        "sequence_separator_token": "bos",
        "disable_concat_sequences": False,
    }
    # Exact Python types also matter: False and 0 must not compare as the same signed value.
    if any(
        type(metadata[key]) is not type(value) or metadata[key] != value
        for key, value in exact.items()
    ):
        _reject("native inference metadata semantics differ from the reviewed profile")
    provenance = metadata["dataset_path"]
    if (
        type(provenance) is not str
        or not provenance
        or len(provenance) > 4096
        or "\0" in provenance
    ):
        _reject("dataset provenance must be bounded text, never a runtime input")
    training = metadata["training_architecture"]
    if type(training) is not str or training not in _TRAINING_FIELDS:
        _reject("native training provenance architecture is unknown")
    expected_architecture = (
        "jumprelu" if training in {"batchtopk", "matryoshka_batchtopk"} else training
    )
    if architecture != expected_architecture:
        _reject("native training and inference architectures do not match")
    details = metadata["training_architecture_details"]
    if not isinstance(details, dict) or set(details) != _TRAINING_FIELDS[training] | {
        "decoder_init_norm"
    }:
        _reject("native training provenance detail keys are not closed")
    # These are provenance, not a newly imposed recipe/score policy. Check serialization types
    # only; thresholds, learning rates, and accepted recipes belong to their existing authority.
    for key, value in details.items():
        if key in {
            "use_sparse_activations",
            "rescale_acts_by_decoder_norm",
            "use_matryoshka_aux_loss",
        }:
            if type(value) is not bool:
                _reject("native training boolean detail has the wrong type")
        elif key in {"l1_warm_up_steps", "l0_warm_up_steps"}:
            if type(value) is not int:
                _reject("native training integer detail has the wrong type")
        elif key == "matryoshka_widths":
            if type(value) is not list or not value or any(type(item) is not int for item in value):
                _reject("native training widths have the wrong type")
        elif key == "jumprelu_sparsity_loss_mode":
            if type(value) is not str or value not in {"step", "tanh"}:
                _reject("native JumpReLU sparsity mode is unknown")
        elif key in {"decoder_init_norm", "pre_act_loss_coefficient"} and value is None:
            continue
        elif type(value) not in {int, float} or (type(value) is float and not math.isfinite(value)):
            _reject("native training numeric detail is not finite")


def native_tensor_schemas() -> Mapping[str, TensorSchema]:
    """The four inference classes' exact F32 tensor contracts, without importing ML libraries."""

    f32 = frozenset({"F32"})
    common = {
        "W_enc": TensorField(("d_in", "d_sae"), f32),
        "W_dec": TensorField(("d_sae", "d_in"), f32),
        "b_dec": TensorField(("d_in",), f32),
    }
    vector = TensorField(("d_sae",), f32)
    result: dict[str, TensorSchema] = {}
    for architecture in INFERENCE_ARCHITECTURES:
        fields = dict(common)
        if architecture == "gated":
            fields.update(dict.fromkeys(("b_gate", "b_mag", "r_mag"), vector))
        else:
            fields["b_enc"] = vector
        if architecture == "jumprelu":
            fields["threshold"] = vector
        result[architecture] = TensorSchema(
            architecture, fields, f"{NATIVE_ABI_ID}-{architecture}", validate_native_cfg
        )
    return MappingProxyType(result)
