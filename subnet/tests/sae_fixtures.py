"""Tiny local ABI fixtures; these do not represent a scoreable SAE."""

from subnet.lanes.interp_sae import stage
from subnet.lanes.interp_sae.assets import MODEL_DTYPE, MODEL_REVISION

MODEL_REPOSITORY = "allenai/OLMo-2-1124-7B"
_SLOT_ID = "olmo2-7b-l16"
_REVISION = "7df9a82518afdecae4e8c026b27adccc8c1f0032"


def _policy(**overrides) -> dict:
    policy = dict.fromkeys(stage._S7_SUFFIXES)
    policy.update(
        {
            "model": "allenai/OLMo-2-1124-7B",
            "revision": _REVISION,
            "runtime": "transformer_lens",
            "hook_id": "blocks.16.hook_resid_post",
            "layer": 16,
            "hook_point": "resid_post",
            "d_in": 4096,
            "d_sae": 32768,
            "model_dtype": "bfloat16",
            "sae_dtype": "float32",
            "model_load_processing": "none",
            "normalize_activations": "none",
            "reshape_activations": "none",
            "apply_b_dec_to_input": False,
            "rescale_acts_by_decoder_norm": True,
            "prepend_bos": True,
            "context_size": 128,
            "exclude_special_tokens": True,
            "seqpos_slice_json": "[null]",
            "topk_k_min": 40,
            "topk_k_max": 120,
            "draw_chars_per_family": 12,
            "draw_chunk_chars": 4,
            "artifact_max_bytes": 4 * 1024 * 1024 * 1024,
            "slot_weight": "1.0",
            "cadence_blocks": 2400,
            "phase_offset_blocks": 0,
            "status": "shadow",
            "tiebreak": "incumbent_holds",
            "random_floor": "0.26",
            "random_floor_delta": "0.10",
            "delta_cat_probe": "0.04",
            "delta_cat_abs": "0.02",
            "transformer_lens_version": "2.16.1",
        }
    )
    policy.update(overrides)
    return policy


def _selected_policy(**overrides) -> dict:
    return {"_slot_id": _SLOT_ID, **_policy(**overrides)}


def _metadata():
    return {
        "sae_lens_version": "6.46.0",
        "sae_lens_training_version": "6.46.0",
        "model_name": MODEL_REPOSITORY,
        "model_class_name": "HookedTransformer",
        "model_from_pretrained_kwargs": {"revision": MODEL_REVISION, "dtype": MODEL_DTYPE},
        "hook_name": "blocks.16.hook_resid_post",
        "hook_head_index": None,
        "context_size": 2048,
        "prepend_bos": True,
        "seqpos_slice": [None],
        "exclude_special_tokens": True,
        "sequence_separator_token": "bos",
        "disable_concat_sequences": False,
        "dataset_path": "unit-serialization-provenance-only",
    }


def _cfg():
    metadata = _metadata()
    metadata.update(
        training_architecture="standard",
        training_architecture_details={
            "decoder_init_norm": None,
            "l1_coefficient": 1.0,
            "lp_norm": 1.0,
            "l1_warm_up_steps": 0,
        },
    )
    return {
        "architecture": "standard",
        "d_in": 4,
        "d_sae": 8,
        "dtype": "float32",
        "device": "cuda",
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "metadata": metadata,
    }
