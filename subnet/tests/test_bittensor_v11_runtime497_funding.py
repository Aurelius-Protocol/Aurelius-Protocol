"""Runtime497 funding pins and signing bytes proven against two independent codecs.

Runtime497 is the first reviewed runtime at transaction version 2. The version the coldkey
signs is the one reviewed for that runtime: never a literal, and never the server's claim.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scalecodec.base import RuntimeConfigurationObject, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset

from subnet.miner import axon_serve
from subnet.miner.funding import FundingRefused, chain_scope, transfer_call
from subnet.miner.funding_client import TREASURY
from subnet.miner.funding_runtime import RUNTIME_PROFILES, RUNTIME_TRANSACTION_VERSIONS
from subnet.miner.prepared_extrinsic import BITTENSOR_V11_SIGNED_EXTENSIONS
from subnet.shared.sn37_economics import GENESIS_HASH
from subnet.shared.ss58 import ss58_encode

pytestmark = pytest.mark.shared
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "runtime497"


@pytest.fixture(scope="module")
def codecs():
    import bittensor as bt
    from bittensor._transport.codec import RuntimeCodec

    raw = (FIXTURE / "metadata.scale").read_bytes()
    assert bt.__version__ == "11.0.0"
    assert hashlib.sha256(raw).hexdigest() == RUNTIME_PROFILES[497][0]
    capture = json.loads((FIXTURE / "capture.json").read_text())
    assert capture["metadata_sha256"] == RUNTIME_PROFILES[497][0]
    assert capture["code_storage_hash"] == RUNTIME_PROFILES[497][1]
    runtime = RuntimeConfigurationObject(ss58_format=42)
    runtime.update_type_registry(load_type_registry_preset(name="core"))
    metadata = runtime.create_scale_object("MetadataVersioned", data=ScaleBytes(raw))
    metadata.decode()
    runtime.add_portable_registry(metadata)
    native = RuntimeCodec(raw, spec_version=497, transaction_version=2, spec_name="node-subtensor")
    return runtime, metadata, native


def test_runtime497_funding_signed_extensions_remain_exact(codecs):
    _, metadata, _ = codecs
    assert (
        tuple(e["identifier"] for e in metadata.value[1]["V14"]["extrinsic"]["signed_extensions"])
        == BITTENSOR_V11_SIGNED_EXTENSIONS
    )


@pytest.mark.parametrize("alpha", [1, 10_000_000_000, 2**64 - 2])
def test_runtime497_transfer_matches_independent_and_sdk_codecs(codecs, alpha):
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


def signing_parts(native):
    """The SDK's mortal payload for one transfer: its parts, the whole, and the input."""
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
    return (
        native.signature_payload_parts(call, **params),
        native.signature_payload(call, **params),
        call,
        anchor,
    )


def signed_data(anchor, transaction_version):
    """What the chain's signed extensions add after the call and the era."""
    return (
        (497).to_bytes(4, "little")
        + transaction_version.to_bytes(4, "little")
        + bytes.fromhex(GENESIS_HASH[2:])
        + bytes.fromhex(anchor[2:])
        + b"\x00"
    )


def test_runtime497_native_mortal_signing_payload_matches_reviewed_layout(codecs):
    _, _, native = codecs
    assert RUNTIME_TRANSACTION_VERSIONS[497] == 2
    parts, payload, call, anchor = signing_parts(native)
    expected = (call, bytes.fromhex("45021c0000"), signed_data(anchor, 2))
    assert parts == expected and payload == b"".join(expected)


def test_runtime497_signing_payload_is_not_the_transaction_version_1_layout(codecs):
    """A coldkey signature over the previous version's payload was over different bytes."""
    _, _, native = codecs
    parts, payload, _, anchor = signing_parts(native)
    previous = signed_data(anchor, 1)
    assert previous[4:8] == b"\x01\x00\x00\x00" and len(previous) == len(parts[2])
    # Little-endian 1 and 2 differ in one byte: the low byte of the transaction version.
    assert [left != right for left, right in zip(previous, parts[2])] == [
        index == 4 for index in range(len(previous))
    ]
    assert payload != parts[0] + parts[1] + previous


@pytest.mark.parametrize("fault", [None, "code", "metadata", "transaction", "runtime", "genesis"])
def test_runtime497_scope_requires_exact_reviewed_identity(fault):
    capture = json.loads((FIXTURE / "capture.json").read_text())
    snapshot = SimpleNamespace(
        genesis_hash="0x" + "00" * 32 if fault == "genesis" else GENESIS_HASH,
        runtime_spec_version=496 if fault == "runtime" else 497,
        transaction_version=1 if fault == "transaction" else 2,
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


def test_each_reviewed_funding_runtime_names_its_transaction_version():
    assert set(RUNTIME_TRANSACTION_VERSIONS) == set(RUNTIME_PROFILES)
    assert RUNTIME_TRANSACTION_VERSIONS[497] == 2


@pytest.mark.parametrize("spec", [479, 481, 488, 497])
def test_each_capture_records_the_transaction_version_its_runtime_was_reviewed_at(spec):
    capture = json.loads((FIXTURES / f"runtime{spec}/capture.json").read_text())
    observed = capture.get("runtime") or {}
    assert observed.get("specVersion", capture.get("runtime_spec_version")) == spec
    assert (
        observed.get("transactionVersion", capture.get("transaction_version"))
        == RUNTIME_TRANSACTION_VERSIONS[spec]
    )
    assert capture["metadata_sha256"] == RUNTIME_PROFILES[spec][0]


def test_example_configuration_pins_a_reviewed_runtime_and_its_transaction_version():
    names = {"AURELIUS_RUNTIME_SPEC_PIN", "AURELIUS_RUNTIME_TRANSACTION_VERSION_PIN"}
    pins = {}
    for line in (FIXTURES.parents[2] / ".env.example").read_text().splitlines():
        name, separator, value = line.partition("=")
        if separator and name in names:
            pins[name] = value
    spec = int(pins["AURELIUS_RUNTIME_SPEC_PIN"])
    assert spec in axon_serve._REVIEWED_SERVE_AXON_RUNTIME_SPECS and spec in RUNTIME_PROFILES
    pinned_transaction_version = int(pins["AURELIUS_RUNTIME_TRANSACTION_VERSION_PIN"])
    assert pinned_transaction_version == RUNTIME_TRANSACTION_VERSIONS[spec]
