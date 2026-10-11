"""The standalone distribution contains only working miner dependency modules."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import pkgutil

import pytest
import subnet

pytestmark = pytest.mark.miner


def test_every_shipped_module_imports_from_the_standalone_tree():
    root = Path(subnet.__file__).resolve().parent
    modules = list(pkgutil.walk_packages(subnet.__path__, "subnet."))
    assert modules
    for info in modules:
        module = importlib.import_module(info.name)
        assert root in Path(module.__file__).resolve().parents, info.name
    for absent in ("validator", "runtime_maintenance"):
        assert not (root / absent).exists()
    from subnet.lanes.interp_sae.referee import runtime

    assert not hasattr(runtime, "evaluate_draw")
    assert not hasattr(runtime, "run_battery_profile")


def test_no_private_service_or_validator_import_is_shipped():
    root = Path(subnet.__file__).resolve().parent
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith(
                    ("subnet.validator", "central_api", "sidecars", "product")
                ), path


def test_wallet_view_keeps_coldkey_out_and_cannot_overwrite(tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts/prepare_wallet_view.py"
    spec = importlib.util.spec_from_file_location("wallet_view_helper", script)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source = tmp_path / "source"
    wallet = source / "miner"
    (wallet / "hotkeys").mkdir(parents=True)
    (wallet / "coldkeypub.txt").write_text("public coldkey document")
    (wallet / "coldkey").write_text("private coldkey must never be copied")
    (wallet / "hotkeys/default").write_text("selected hotkey fixture")
    from subnet.shared.ss58 import ss58_encode

    output = tmp_path / "runtime"
    result = helper.prepare_view(source, output, "miner", "default", ss58_encode(b"a" * 32))
    assert not (result / "coldkey").exists()
    assert (result / "hotkeys/default").read_text() == "selected hotkey fixture"
    with pytest.raises(FileExistsError):
        helper.prepare_view(source, output, "miner", "default", ss58_encode(b"a" * 32))
