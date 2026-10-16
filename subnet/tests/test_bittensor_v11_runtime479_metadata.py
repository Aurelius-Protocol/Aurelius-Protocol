"""Independent SCALE review of the attributed runtime479 ServeAxon call."""

from __future__ import annotations

import hashlib
import ipaddress
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scalecodec.base import RuntimeConfigurationObject, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset

from subnet.miner import axon_serve

pytestmark = pytest.mark.shared
FIXTURE = Path(__file__).parent / "fixtures/runtime479"
SHA256 = "075ee9614fee27fb83a44bda737f6803bd3f6642d57ab492321aa47dad359080"
EXTENSIONS = (
    "CheckNonZeroSender",
    "CheckSpecVersion",
    "CheckTxVersion",
    "CheckGenesis",
    "CheckMortality",
    "CheckNonce",
    "CheckWeight",
    "ChargeTransactionPayment",
    "SudoTransactionExtension",
    "CheckShieldedTxValidity",
    "SubtensorTransactionExtension",
    "DrandPriority",
    "CheckMetadataHash",
)


def test_runtime479_metadata_attribution():
    raw = (FIXTURE / "metadata.scale").read_bytes()
    capture = json.loads((FIXTURE / "capture.json").read_text())
    assert len(raw) == capture["metadata_bytes"] == 359018
    assert hashlib.sha256(raw).hexdigest() == capture["metadata_sha256"] == SHA256
    assert (
        capture["code_sha256"] == "d0ea4955e6b4366d8561038307d5e47572c1ad67664638402c8253cde26a84be"
    )
    assert (
        capture["source_attribution"]
        == "configured-finalized-chain-and-two-independent-public-rpc-digest-match"
    )
    assert raw[:5] == b"meta\x0e"
    assert capture["height"] == 9291893
    assert (
        capture["block_hash"]
        == "0xdacc1983efedd6bc946266bbed222273d79bd4eb60e0da452f49ae7d1c6e7f21"
    )
    assert capture["runtime"]["specVersion"] == capture["parent_runtime"]["specVersion"] == 479
    assert capture["runtime"]["transactionVersion"] == 1


@pytest.fixture(scope="module")
def independent():
    runtime = RuntimeConfigurationObject(ss58_format=42)
    runtime.update_type_registry(load_type_registry_preset(name="core"))
    metadata = runtime.create_scale_object(
        "MetadataVersioned", data=ScaleBytes((FIXTURE / "metadata.scale").read_bytes())
    )
    metadata.decode()
    runtime.add_portable_registry(metadata)
    return runtime, metadata, metadata.value[1]["V14"]


def test_runtime479_signed_extensions(independent):
    _, _, document = independent
    assert (
        tuple(item["identifier"] for item in document["extrinsic"]["signed_extensions"])
        == EXTENSIONS
    )


def test_runtime479_serve_axon_wire_schema(independent):
    _, _, document = independent
    types = {item["id"]: item["type"] for item in document["types"]["types"]}
    pallet = next(p for p in document["pallets"] if p["name"] == "SubtensorModule")
    variants = types[pallet["calls"]["ty"]]["def"]["variant"]["variants"]
    call = next(v for v in variants if v["name"] == "serve_axon")
    assert bytes([pallet["index"], call["index"]]).hex() == "0704"
    assert tuple(f["name"] for f in call["fields"]) == axon_serve._SERVE_AXON_CALL_ARG_ORDER

    def primitive(identifier):
        definition = types[identifier]["def"]
        if "primitive" in definition:
            return definition["primitive"]
        fields = definition["composite"]["fields"]
        assert len(fields) == 1 and fields[0]["name"] is None
        return primitive(fields[0]["type"])

    assert tuple(primitive(f["type"]) for f in call["fields"]) == (
        "u16",
        "u32",
        "u128",
        "u16",
        "u8",
        "u8",
        "u8",
        "u8",
    )


@pytest.mark.parametrize("port", [8091, 30137])
def test_runtime479_sn37_endpoints_match_independent_and_native_codecs(independent, port):
    runtime, metadata, _ = independent
    ip = int(ipaddress.ip_address("198.51.100.37"))
    args = dict(
        netuid=37,
        version=1,
        ip=ip,
        port=port,
        ip_type=4,
        protocol=4,
        placeholder1=0,
        placeholder2=0,
    )
    call = runtime.create_scale_object("Call", metadata=metadata)
    encoded = call.encode(
        dict(call_module="SubtensorModule", call_function="serve_axon", call_args=args)
    )
    expected = (
        b"\x07\x04"
        + (37).to_bytes(2, "little")
        + (1).to_bytes(4, "little")
        + ip.to_bytes(16, "little")
        + port.to_bytes(2, "little")
        + bytes([4, 4, 0, 0])
    )
    assert bytes(encoded.data) == expected
    decoded = runtime.create_scale_object("Call", data=ScaleBytes(expected), metadata=metadata)
    decoded.decode()
    assert decoded.value["call_index"] == axon_serve._SERVE_AXON_CALL_INDEX
    assert {a["name"]: a["value"] for a in decoded.value["call_args"]} == args
    import bittensor as bt
    from bittensor._transport.codec import RuntimeCodec

    assert bt.__version__ == "11.0.0"
    native = RuntimeCodec(
        (FIXTURE / "metadata.scale").read_bytes(),
        spec_version=479,
        transaction_version=1,
        spec_name="node-subtensor",
    )
    call = native.compose_call("SubtensorModule", "serve_axon", args)
    assert call.data == expected
    assert native.decode_call(call.data) == decoded.value
    data, version, call_hash = axon_serve._validate_planned_serve_axon_call(
        SimpleNamespace(decode_scale=lambda type_name, data: native.decode_call(data)),
        call,
        runtime_pin=479,
        expected_call_args=args,
    )
    assert data == expected and version == 479
    assert call_hash == "0x" + hashlib.blake2b(expected, digest_size=32).hexdigest()


def test_runtime479_serve_axon_explicitly_reviewed():
    assert 479 in axon_serve._REVIEWED_SERVE_AXON_RUNTIME_SPECS
