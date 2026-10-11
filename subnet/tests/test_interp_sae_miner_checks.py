"""Local check reports preserve mismatches and use the real bounded tensor parser."""

import json
import struct

import pytest

from subnet.lanes.interp_sae.miner_checks import check_bundle, window_preview
from .sae_fixtures import _selected_policy
from .sae_fixtures import _cfg

pytestmark = pytest.mark.miner


def bundle(tmp_path, *, nonfinite=False):
    root = tmp_path / "bundle"
    root.mkdir()
    cfg = _cfg()
    (root / "cfg.json").write_text(json.dumps(cfg))
    offset = 0
    header = {}
    payload = b""
    for name, shape in {"W_enc": [4, 8], "W_dec": [8, 4], "b_enc": [8], "b_dec": [4]}.items():
        count = 1
        for size in shape:
            count *= size
        data = struct.pack("<f", float("nan") if nonfinite else 0.5) * count
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload += data
        offset += len(data)
    encoded = json.dumps(header).encode()
    (root / "sae_weights.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    return root


def policy():
    return _selected_policy(d_in=4, d_sae=8, context_size=2048, topk_k_min=1, topk_k_max=8)


def test_real_tensor_inspection_and_all_metadata_mismatches(tmp_path):
    root = bundle(tmp_path)
    report = check_bundle(root, policy())
    assert report["compatible"], report
    assert report["tensor_status"] == "pass"
    wrong = policy() | {"d_in": 5, "d_sae": 9}
    report = check_bundle(root, wrong)
    assert [r["field"] for r in report["checks"] if r["status"] == "fail"] == ["d_in", "d_sae"]
    assert report["tensor_status"] == "not_run"


def test_training_metadata_is_informational_not_a_compatibility_failure(tmp_path):
    """ADR-0229 D1: a differing training recipe is reported, never marked incompatible."""

    root = bundle(tmp_path)
    other_recipe = policy() | {"context_size": 128, "prepend_bos": False}
    report = check_bundle(root, other_recipe)
    assert report["compatible"], report
    info = {r["field"]: r for r in report["checks"] if r["status"] == "info"}
    assert set(info) == {
        "metadata.prepend_bos",
        "metadata.context_size",
        "metadata.exclude_special_tokens",
    }
    assert info["metadata.context_size"]["observed"] == 2048
    assert info["metadata.context_size"]["expected"] == 128
    assert not any(r["status"] == "fail" for r in report["checks"])


def test_nonfinite_bytes_cannot_pass(tmp_path):
    report = check_bundle(bundle(tmp_path, nonfinite=True), policy())
    assert not report["compatible"]
    assert report["error"]["code"] == "nonfinite_weights"


def test_schedule_boundaries_never_select_expired_epoch():
    p = policy()
    assert window_preview(p, head=2200)["slot_epoch"] == 1
    assert window_preview(p, head=2299)["status"] == "open"
    assert window_preview(p, head=2300)["slot_epoch"] == 2
    assert window_preview(p, head=2300, epoch=1)["status"] == "expired"
    assert window_preview(p, head=0)["commit_start"] == 2200
