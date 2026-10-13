"""Automatic SN37 trust bootstrap preserves independent signature verification."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from subnet.miner import cli, config_schema, economics_client, observation_client
from subnet.miner.journal import AttemptJournal, identity_for
from subnet.miner.serve import WorkIdMinter, validate_miner_startup_posture
from subnet.miner.trust import SN37_CONFIG_SIGNING_PUBKEY, config_signing_pubkey
from subnet.lanes.interp_sae import miner_cli as sae_cli
from subnet.shared.config_snapshot import SNAPSHOT_SIGNATURE_DOMAIN, canonical_hash

pytestmark = pytest.mark.miner

# Public config fetched 2026-10-11. The bundled pin comes independently from the deployed
# SN37 miner's public-key configuration, not a key supplied inside this snapshot.
PUBLIC_SNAPSHOT = Path(__file__).parent / "fixtures" / "sn37_config_v24.json"


@pytest.fixture(autouse=True)
def no_key_override(monkeypatch):
    monkeypatch.delenv("CONFIG_SIGNING_PUBKEY", raising=False)


def test_published_sn37_snapshot_verifies_without_manual_key(monkeypatch):
    monkeypatch.setenv("AURELIUS_CONFIG_SNAPSHOT", str(PUBLIC_SNAPSHOT))
    assert config_schema.config_source() == (
        "file",
        str(PUBLIC_SNAPSHOT),
        SN37_CONFIG_SIGNING_PUBKEY,
    )
    entries = config_schema.load_published_entries()
    assert "config_schema.interp_sae" in entries
    assert entries == json.loads(PUBLIC_SNAPSHOT.read_text())["body"]["entries"]


def test_production_startup_accepts_bundled_key():
    validate_miner_startup_posture(
        {
            "CENTRAL_API_URL": "https://central.aureliusaligned.ai",
            "AURELIUS_ENV": "mainnet",
            "AURELIUS_TRANSPORT": "btauth",
            "AURELIUS_IDENTITY_SOURCE": "bittensor",
            "AURELIUS_CLOCK": "real",
            "AURELIUS_REQUIRE_HOTKEY_ONLY_WALLET": "true",
            "AURELIUS_AXON_ENDPOINT_PROFILE": "public",
            "AURELIUS_BITTENSOR_NONCE_DB": "/state/nonces.sqlite3",
            "AURELIUS_AXON_PUBLISH_STATE_DIR": "/state/axon",
            "AURELIUS_AXON_EXPECTED_CHAIN_NAME": "Bittensor",
            "AURELIUS_AXON_EXPECTED_GENESIS_HASH": "0x" + "ab" * 32,
            "AURELIUS_RUNTIME_SPEC_PIN": "475",
            "CHAIN_ID": "bittensor-mainnet:reviewed-genesis",
            "SUBSTRATE_URL": "wss://archive.chain.opentensor.ai:443",
        }
    )


@pytest.mark.parametrize("bad", ["", "ab", "g" * 64, " " + "ab" * 32, "02" + "ab" * 32])
def test_invalid_override_never_falls_back_to_shipped_key(bad):
    with pytest.raises(ValueError, match="CONFIG_SIGNING_PUBKEY"):
        config_signing_pubkey({"CONFIG_SIGNING_PUBKEY": bad})
    with pytest.raises(config_schema.SchemaUnavailable, match="CONFIG_SIGNING_PUBKEY"):
        config_schema.config_source({"CONFIG_SIGNING_PUBKEY": bad})


def test_wrong_but_well_formed_override_refuses_production_snapshot(monkeypatch):
    monkeypatch.setenv("CONFIG_SIGNING_PUBKEY", "00" * 32)
    monkeypatch.setenv("AURELIUS_CONFIG_SNAPSHOT", str(PUBLIC_SNAPSHOT))
    with pytest.raises(config_schema.SchemaUnavailable, match="snapshot unavailable/invalid"):
        config_schema.load_published_entries()


def test_remote_replacement_key_cannot_authorize_forged_snapshot(monkeypatch, tmp_path):
    doc = copy.deepcopy(json.loads(PUBLIC_SNAPSHOT.read_text()))
    # An attacker can re-hash the body and sign it with their own key, but cannot replace
    # the release trust anchor by adding that public key to the downloaded document.
    private = Ed25519PrivateKey.from_private_bytes(b"t" * 32)
    attacker_key = private.public_key().public_bytes_raw().hex()
    doc["body"]["entries"]["window_blocks"] += 1
    doc["canonical_hash"] = canonical_hash(doc["body"])
    preimage = SNAPSHOT_SIGNATURE_DOMAIN + f"{doc['canonical_hash']}|{doc['version']}".encode()
    doc["signature_hex"] = private.sign(preimage).hex()
    doc["public_key"] = attacker_key
    doc["signer_key_id"] = "attacker"
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv("AURELIUS_CONFIG_SNAPSHOT", str(path))
    with pytest.raises(config_schema.SchemaUnavailable, match="snapshot unavailable/invalid"):
        config_schema.load_published_entries()
    # Selecting another deployment's key remains an explicit local decision.
    monkeypatch.setenv("CONFIG_SIGNING_PUBKEY", attacker_key)
    assert config_schema.load_published_entries() == doc["body"]["entries"]


def test_tampered_production_snapshot_still_refused(monkeypatch, tmp_path):
    doc = json.loads(PUBLIC_SNAPSHOT.read_text())
    doc["body"]["entries"]["window_blocks"] += 1
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv("AURELIUS_CONFIG_SNAPSHOT", str(path))
    with pytest.raises(config_schema.SchemaUnavailable, match="snapshot unavailable/invalid"):
        config_schema.load_published_entries()


@pytest.mark.parametrize("override", [None, "ab" * 32])
def test_economics_uses_resolved_key(monkeypatch, override):
    if override is not None:
        monkeypatch.setenv("CONFIG_SIGNING_PUBKEY", override)
    seen = []
    monkeypatch.setattr(economics_client, "read_economics", lambda *a, **kw: seen.append(kw) or {})
    monkeypatch.setattr("subnet.miner.serve.minter_from_env", lambda *a: WorkIdMinter("fixture"))
    assert cli.main(["economics", "--netuid", "37"]) == 0
    assert seen[0]["config_pubkey"] == (override or SN37_CONFIG_SIGNING_PUBKEY)


@pytest.mark.parametrize("command", ["observe", "sae-accounting", "sae-progress"])
@pytest.mark.parametrize("override", [None, "ab" * 32])
def test_observation_commands_use_resolved_key(monkeypatch, tmp_path, command, override):
    if override is not None:
        monkeypatch.setenv("CONFIG_SIGNING_PUBKEY", override)
    minter = WorkIdMinter("fixture")
    monkeypatch.setattr("subnet.miner.serve.minter_from_env", lambda *a: minter)
    item = {
        "task_type": "interp_sae",
        "schema_version": 2,
        "config": {"payload": "a"},
        "eligible_fanout_window": 7,
    }
    journal = tmp_path / "attempts.sqlite3"
    AttemptJournal(journal).record(identity_for(item, minter.hotkey_ss58), "prepared", "local")
    seen = []

    def read(*args, **kwargs):
        seen.append(kwargs)
        return {"charge": "recorded", "refund": "not_observed", "observed_at": 1, "validators": []}

    monkeypatch.setattr(observation_client, "observe", read)
    monkeypatch.setattr(observation_client, "progress", read)
    common = ["--journal", str(journal), "--central", "https://central.example", "--netuid", "37"]
    if command == "observe":
        assert cli.main(["observe", *common]) == 0
    else:
        option = (
            ["--accounting"]
            if command == "sae-accounting"
            else ["--validator", "evaluator=https://evaluator.example"]
        )
        assert sae_cli.main(["status", *common, *option]) == 0
    assert seen
    assert all(row["config_pubkey"] == (override or SN37_CONFIG_SIGNING_PUBKEY) for row in seen)


def test_invalid_override_refused_by_economics_and_sae(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CONFIG_SIGNING_PUBKEY", "")
    with pytest.raises(SystemExit) as exc:
        cli.main(["economics", "--netuid", "37"])
    assert exc.value.code == 2
    assert (
        sae_cli.main(
            [
                "status",
                "--accounting",
                "--netuid",
                "37",
                "--central",
                "https://central.example",
                "--journal",
                str(tmp_path / "attempts.sqlite3"),
            ]
        )
        == 2
    )
    assert "CONFIG_SIGNING_PUBKEY" in capsys.readouterr().out
