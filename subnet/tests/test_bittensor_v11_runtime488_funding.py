"""Runtime488 funding pins and signing bytes proven against two independent codecs."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scalecodec.base import RuntimeConfigurationObject, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset

from subnet.miner.funding import FundingRefused, RUNTIME_PROFILES, chain_scope, transfer_call
from subnet.miner.funding_client import TREASURY
from subnet.operator.prepared_extrinsic import BITTENSOR_V11_SIGNED_EXTENSIONS
from subnet.shared.sn37_economics import GENESIS_HASH
from subnet.shared.ss58 import ss58_encode

pytestmark = pytest.mark.shared
FIXTURE = Path(__file__).parent / "fixtures/runtime488"


@pytest.fixture(scope="module")
def codecs():
    import bittensor as bt
    from bittensor._transport.codec import RuntimeCodec

    raw = (FIXTURE / "metadata.scale").read_bytes()
    assert bt.__version__ == "11.0.0"
    assert hashlib.sha256(raw).hexdigest() == RUNTIME_PROFILES[488][0]
    capture = json.loads((FIXTURE / "capture.json").read_text())
    assert capture["metadata_sha256"] == RUNTIME_PROFILES[488][0]
    assert capture["code_storage_hash"] == RUNTIME_PROFILES[488][1]
    assert len(capture["providers"]) == 2
    runtime = RuntimeConfigurationObject(ss58_format=42)
    runtime.update_type_registry(load_type_registry_preset(name="core"))
    metadata = runtime.create_scale_object("MetadataVersioned", data=ScaleBytes(raw))
    metadata.decode()
    runtime.add_portable_registry(metadata)
    native = RuntimeCodec(raw, spec_version=488, transaction_version=1, spec_name="node-subtensor")
    return runtime, metadata, native


def test_runtime488_funding_signed_extensions_remain_exact(codecs):
    _, metadata, _ = codecs
    assert (
        tuple(e["identifier"] for e in metadata.value[1]["V14"]["extrinsic"]["signed_extensions"])
        == BITTENSOR_V11_SIGNED_EXTENSIONS
    )


@pytest.mark.parametrize("alpha", [1, 10_000_000_000, 2**64 - 2])
def test_runtime488_transfer_matches_independent_and_sdk_codecs(codecs, alpha):
    runtime, metadata, native = codecs
    args = {
        "destination_coldkey": TREASURY,
        "hotkey": ss58_encode(bytes([2]) * 32),
        "origin_netuid": 37,
        "destination_netuid": 37,
        "alpha_amount": alpha,
    }
    expected = transfer_call(args["hotkey"], alpha)
    encoder = runtime.create_scale_object("Call", metadata=metadata)
    encoded = encoder.encode(
        {"call_module": "SubtensorModule", "call_function": "transfer_stake", "call_args": args}
    )
    assert bytes(encoded.data) == expected
    decoder = runtime.create_scale_object("Call", metadata=metadata, data=ScaleBytes(expected))
    decoder.decode(check_remaining=True)
    assert decoder.value["call_index"] == "0x0756"
    assert {item["name"]: item["value"] for item in decoder.value["call_args"]} == args
    assert native.compose_call("SubtensorModule", "transfer_stake", args).data == expected
    assert native.decode_call(expected) == decoder.value


def test_runtime488_native_mortal_signing_payload_matches_reviewed_layout(codecs):
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
        (488).to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + bytes.fromhex(GENESIS_HASH[2:])
        + bytes.fromhex(anchor[2:])
        + b"\x00",
    )
    assert parts == expected
    assert native.signature_payload(call, **params) == b"".join(expected)


@pytest.mark.parametrize("fault", [None, "code", "metadata", "transaction", "runtime", "genesis"])
def test_runtime488_scope_requires_exact_reviewed_identity(fault):
    capture = json.loads((FIXTURE / "capture.json").read_text())
    snapshot = SimpleNamespace(
        genesis_hash="0x" + "00" * 32 if fault == "genesis" else GENESIS_HASH,
        runtime_spec_version=489 if fault == "runtime" else 488,
        transaction_version=2 if fault == "transaction" else 1,
        chain="Bittensor",
        head_hash=capture["block_hash"],
    )

    def read(method, args):
        assert args[-1] == capture["block_hash"]
        if method == "state_getStorageHash":
            return "0x" + "00" * 32 if fault == "code" else capture["code_storage_hash"]
        assert method == "state_getMetadata"
        return (
            "0x00"
            if fault == "metadata"
            else "0x" + (FIXTURE / "metadata.scale").read_bytes().hex()
        )

    rpc = SimpleNamespace(snapshot=lambda: snapshot, call=read)
    if fault is None:
        assert chain_scope(rpc) is snapshot
    else:
        with pytest.raises(FundingRefused):
            chain_scope(rpc)
