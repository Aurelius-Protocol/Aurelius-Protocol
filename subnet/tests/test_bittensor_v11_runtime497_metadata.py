"""Independent SCALE review of the attributed runtime497 ServeAxon call.

Runtime497 is the first reviewed runtime at transaction version 2. What the miner signs at 497
is compared with runtime479, the last runtime reviewed before it: the signed-extension list,
the extrinsic version and the ServeAxon wire schema, then the exact endpoint bytes through an
independent codec and the SDK's.
"""

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
from subnet.miner.prepared_extrinsic import BITTENSOR_V11_SIGNED_EXTENSIONS

pytestmark = pytest.mark.shared
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "runtime497"
METADATA_SHA256 = "d54cab26911b8c2d840b5bc0534945e43d43cf7be23266d5100f9c0c64ecdd4e"
CODE_SHA256 = "a335ac30eeadb3829486179a3162387d167984efff4ae3ee787de1b4e9da1e21"
CODE_STORAGE_HASH = "0x1ef826aebed0d77300b7da09d4e3b23ad507317d72c8c01d827537643dbf2c15"
BLOCK_HASH = "0x99057c78d3da30f9dcf71dd5ff059355c7afa828054402c97a9fb60c30ed9635"


def decoded(path):
    """The metadata at `path`, decoded by scalecodec with no help from the miner."""
    runtime = RuntimeConfigurationObject(ss58_format=42)
    runtime.update_type_registry(load_type_registry_preset(name="core"))
    metadata = runtime.create_scale_object("MetadataVersioned", data=ScaleBytes(path.read_bytes()))
    metadata.decode()
    runtime.add_portable_registry(metadata)
    return runtime, metadata, metadata.value[1]["V14"]


def test_runtime497_metadata_attribution():
    raw = (FIXTURE / "metadata.scale").read_bytes()
    capture = json.loads((FIXTURE / "capture.json").read_text())
    assert len(raw) == capture["metadata_bytes"] == 409264
    assert hashlib.sha256(raw).hexdigest() == capture["metadata_sha256"] == METADATA_SHA256
    assert capture["code_sha256"] == CODE_SHA256 and capture["code_length"] == 2958775
    assert capture["code_storage_hash"] == CODE_STORAGE_HASH
    assert (
        capture["source_attribution"]
        == "configured-finalized-chain-and-two-independent-public-rpc-digest-match"
    )
    assert raw[:5] == b"meta\x0e"
    assert capture["height"] == capture["block_number"] == 9602450
    assert capture["block_hash"] == BLOCK_HASH
    assert capture["runtime"]["specVersion"] == capture["parent_runtime"]["specVersion"] == 497
    assert capture["runtime"]["transactionVersion"] == 2
    assert capture["transaction_version"] == 2
    # Two providers, each having read the same finalized block and agreed on every digest.
    assert len(capture["providers"]) == 2
    for provider in capture["providers"]:
        assert provider["captured_block_hash"] == BLOCK_HASH
        assert provider["runtime"]["specVersion"] == 497
        assert provider["runtime"]["transactionVersion"] == 2
        assert provider["metadata_sha256"] == METADATA_SHA256
        assert provider["code_sha256"] == CODE_SHA256
        assert provider["code_storage_hash"] == CODE_STORAGE_HASH


@pytest.fixture(scope="module")
def independent():
    return decoded(FIXTURE / "metadata.scale")


def test_runtime497_signing_extensions_are_unchanged_since_479(independent):
    _, _, document = independent
    identifiers = tuple(item["identifier"] for item in document["extrinsic"]["signed_extensions"])
    assert identifiers == BITTENSOR_V11_SIGNED_EXTENSIONS
    earlier = decoded(FIXTURES / "runtime479/metadata.scale")[2]
    assert identifiers == tuple(
        item["identifier"] for item in earlier["extrinsic"]["signed_extensions"]
    )
    assert document["extrinsic"]["version"] == earlier["extrinsic"]["version"] == 4


def test_runtime497_serve_axon_wire_schema(independent):
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
def test_runtime497_sn37_endpoints_match_independent_and_native_codecs(independent, port):
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
    decoded_call = runtime.create_scale_object("Call", data=ScaleBytes(expected), metadata=metadata)
    decoded_call.decode()
    assert decoded_call.value["call_index"] == axon_serve._SERVE_AXON_CALL_INDEX
    assert {a["name"]: a["value"] for a in decoded_call.value["call_args"]} == args
    import bittensor as bt
    from bittensor._transport.codec import RuntimeCodec

    assert bt.__version__ == "11.0.0"
    native = RuntimeCodec(
        (FIXTURE / "metadata.scale").read_bytes(),
        spec_version=497,
        transaction_version=2,
        spec_name="node-subtensor",
    )
    call = native.compose_call("SubtensorModule", "serve_axon", args)
    assert call.data == expected
    assert native.decode_call(call.data) == decoded_call.value
    data, version, call_hash = axon_serve._validate_planned_serve_axon_call(
        SimpleNamespace(decode_scale=lambda type_name, data: native.decode_call(data)),
        call,
        runtime_pin=497,
        expected_call_args=args,
    )
    assert data == expected and version == 497
    assert call_hash == "0x" + hashlib.blake2b(expected, digest_size=32).hexdigest()


def test_runtime497_serve_axon_explicitly_reviewed():
    assert 497 in axon_serve._REVIEWED_SERVE_AXON_RUNTIME_SPECS
