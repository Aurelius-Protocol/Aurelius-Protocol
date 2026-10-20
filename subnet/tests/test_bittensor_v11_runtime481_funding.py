"""Funding authority checked against attributed runtime481 metadata and two codecs."""

import hashlib
import json
from pathlib import Path

import pytest
from scalecodec.base import RuntimeConfigurationObject, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset

from subnet.miner.funding import (
    FundingRefused,
    RUNTIME_PROFILES,
    transfer_call,
    transfer_event_index,
)
from subnet.miner.funding_client import TREASURY
from subnet.operator.prepared_extrinsic import BITTENSOR_V11_SIGNED_EXTENSIONS
from subnet.shared.ss58 import ss58_encode
from subnet.shared.sn37_economics import GENESIS_HASH

pytestmark = pytest.mark.shared
FIXTURE = Path(__file__).parent / "fixtures/runtime481"


@pytest.fixture(scope="module")
def codecs():
    import bittensor as bt
    from bittensor._transport.codec import RuntimeCodec

    raw = (FIXTURE / "metadata.scale").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RUNTIME_PROFILES[481][0]
    assert bt.__version__ == "11.0.0"
    runtime = RuntimeConfigurationObject(ss58_format=42)
    runtime.update_type_registry(load_type_registry_preset(name="core"))
    metadata = runtime.create_scale_object("MetadataVersioned", data=ScaleBytes(raw))
    metadata.decode()
    runtime.add_portable_registry(metadata)
    native = RuntimeCodec(raw, spec_version=481, transaction_version=1, spec_name="node-subtensor")
    return runtime, metadata, native


def test_runtime481_funding_signed_extensions_remain_exact(codecs):
    _, metadata, _ = codecs
    assert (
        tuple(e["identifier"] for e in metadata.value[1]["V14"]["extrinsic"]["signed_extensions"])
        == BITTENSOR_V11_SIGNED_EXTENSIONS
    )


@pytest.mark.parametrize("alpha", [1, 10_000_000_000, 2**64 - 2])
def test_runtime481_transfer_matches_independent_and_sdk_codecs(codecs, alpha):
    runtime, metadata, native = codecs
    hotkey = ss58_encode(bytes([2]) * 32)
    args = {
        "destination_coldkey": TREASURY,
        "hotkey": hotkey,
        "origin_netuid": 37,
        "destination_netuid": 37,
        "alpha_amount": alpha,
    }
    expected = transfer_call(hotkey, alpha)
    encoder = runtime.create_scale_object("Call", metadata=metadata)
    encoded = encoder.encode(
        {"call_module": "SubtensorModule", "call_function": "transfer_stake", "call_args": args}
    )
    assert bytes(encoded.data) == expected
    decoder = runtime.create_scale_object("Call", metadata=metadata, data=ScaleBytes(expected))
    decoder.decode(check_remaining=True)
    assert decoder.value["call_index"] == "0x0756"
    assert {item["name"]: item["value"] for item in decoder.value["call_args"]} == args
    call = native.compose_call("SubtensorModule", "transfer_stake", args)
    assert call.data == expected
    assert native.decode_call(expected) == decoder.value


def test_runtime481_native_mortal_signing_payload_matches_reviewed_layout(codecs):
    _, _, native = codecs
    call = transfer_call(ss58_encode(bytes([2]) * 32), 10_000_000_000)
    anchor = "0x" + "cd" * 32
    params = dict(
        era={"period": 64, "current": 100},
        nonce=7,
        tip=0,
        tip_asset_id=None,
        genesis_hash=GENESIS_HASH,
        era_block_hash=anchor,
        metadata_hash=None,
    )
    parts = native.signature_payload_parts(call, **params)
    expected = (
        call,
        bytes.fromhex("45021c0000"),
        (481).to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + bytes.fromhex(GENESIS_HASH[2:])
        + bytes.fromhex(anchor[2:])
        + b"\x00",
    )
    assert parts == expected
    assert native.signature_payload(call, **params) == b"".join(expected)


def test_recorded_runtime481_foreign_transfer_is_not_a_sn37_funding_success(codecs):
    _, _, native = codecs
    recorded = json.loads((FIXTURE / "native-transfer.json").read_text())
    assert recorded["execution_code_hash"] == RUNTIME_PROFILES[481][1]
    decoded = native.decode_extrinsic(recorded["extrinsic_hex"])
    assert decoded["call"]["call_module"] == "SubtensorModule"
    assert decoded["call"]["call_function"] == "transfer_stake"
    args = {item["name"]: item["value"] for item in decoded["call"]["call_args"]}
    assert args["origin_netuid"] == args["destination_netuid"] == 93
    assert args["alpha_amount"] == 2_751_908_400
    assert args["destination_coldkey"] != TREASURY
    events = native.decode(
        native.storage_entry("System", "Events").value_type,
        bytes.fromhex(recorded["events_hex"][2:]),
    )
    prepared = dict(
        coldkey=decoded["address"],
        hotkey=args["hotkey"],
        alpha_rao=args["alpha_amount"],
        fee_cap_tao_rao=10_000_000,
    )
    with pytest.raises(FundingRefused):
        transfer_event_index(events, prepared, recorded["coordinates"]["extrinsic_index"])
