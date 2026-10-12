"""ADR-0365 mainnet launch identity; selection is not funding activation."""

from __future__ import annotations

from collections.abc import Mapping

TARIFF_VERSION = "sn37-retention-1wt-sae10-v1"
GENESIS_HASH = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
NETUID = 37
CHARGE_TUPLE = ("interp_sae", "gpu_sae_verdict", 2)
CHARGE_WORK_TOKENS = 10


def selected_tariff(environment: Mapping[str, str]) -> str | None:
    selected = environment.get("AURELIUS_ECONOMIC_POLICY")
    if selected in (None, ""):
        return None
    if selected != TARIFF_VERSION and environment.get("AURELIUS_ENV") != "mainnet":
        return None
    if not (
        selected == TARIFF_VERSION
        and environment.get("AURELIUS_ENV") == "mainnet"
        and environment.get("NETUID") == "37"
        and environment.get("AURELIUS_RUNTIME_SPEC_PIN") in {"475", "477"}
        and environment.get("AURELIUS_FINALITY_EXPECTED_GENESIS_HASH") == GENESIS_HASH
        and environment.get("AURELIUS_RETENTION_PURCHASE_TARIFF") == TARIFF_VERSION
        and all(
            environment.get(key) in (None, "")
            for key in (
                "TESTNET_SYNTHETIC_DEPOSIT_CHAIN_ID",
                "PRODUCTION_COST_MODEL_VERSION",
                "AURELIUS_SR25519",
                "AURELIUS_TESTLAB_FAULTS",
            )
        )
    ):
        raise ValueError("SN37 economic policy requires its exact mainnet retention scope")
    return selected
