"""Reviewed chain scope changes preserve the tariff and refuse unknown runtimes."""

import pytest

from subnet.shared.sn37_economics import GENESIS_HASH, TARIFF_VERSION, selected_tariff

pytestmark = pytest.mark.shared


@pytest.mark.parametrize("runtime", ["475", "477"])
def test_reviewed_runtime_selects_the_same_retention_tariff(runtime):
    scope = {
        "AURELIUS_ENV": "mainnet",
        "NETUID": "37",
        "AURELIUS_RUNTIME_SPEC_PIN": runtime,
        "AURELIUS_FINALITY_EXPECTED_GENESIS_HASH": GENESIS_HASH,
        "AURELIUS_ECONOMIC_POLICY": TARIFF_VERSION,
        "AURELIUS_RETENTION_PURCHASE_TARIFF": TARIFF_VERSION,
    }
    assert selected_tariff(scope) == TARIFF_VERSION
    scope["AURELIUS_RUNTIME_SPEC_PIN"] = "478"
    with pytest.raises(ValueError, match="exact mainnet retention scope"):
        selected_tariff(scope)
    scope["AURELIUS_RUNTIME_SPEC_PIN"] = runtime
    scope["NETUID"] = "455"
    with pytest.raises(ValueError, match="exact mainnet retention scope"):
        selected_tariff(scope)
