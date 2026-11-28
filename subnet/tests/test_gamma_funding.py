"""Gamma funding operation: exact bytes, one-shot journal, refusals, recovery — strict fakes.

Mirrors ``test_miner_funding``: no socket, no SDK, no real wallet. The independent
scalecodec block pins the hand-composed SCALE against the captured runtime-490 metadata
(``fixtures/runtime490`` — the real bittensor-mainnet capture of G3) and against a real
mainnet ``spend_gamma`` extrinsic. Those tests skip when scalecodec is unavailable; every
operation/journal/codec test runs on the default dependency set.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import struct

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import subnet.miner.gamma_funding as gf
from subnet.miner.funding_client import CHAIN_ID
from subnet.miner.gamma_funding import (
    GAMMA_AUTH_DOMAIN,
    GammaChain,
    GammaFundingJournal,
    GammaRefused,
    SingleShotSubmitter,
    compose_signed_extrinsic,
    decode_signed_extrinsic,
    decode_spend_gamma_call,
    era_bounds,
    extrinsic_tx_hash,
    gamma_check,
    gamma_check_account,
    gamma_execute,
    gamma_mint_setup,
    gamma_observe,
    gamma_prepare,
    gamma_status,
    mint_gamma_call,
    mortal_era_bytes,
    scale_compact_uint,
    spend_gamma_call,
    twox128,
    unsigned_extrinsic_dict,
    xxh64,
)
from subnet.miner.prepared_extrinsic import BITTENSOR_V11_SIGNED_EXTENSIONS
from subnet.shared import gamma_funding_receipt, gamma_intent, gamma_policy
from subnet.shared.config_snapshot import SNAPSHOT_SIGNATURE_DOMAIN, canonical_hash
from subnet.shared.signing import keypair_from_seed, sign, verify
from subnet.shared.sn37_economics import GENESIS_HASH
from subnet.shared.ss58 import ss58_decode, ss58_encode

pytestmark = pytest.mark.miner

META = b"gamma unit metadata runtime490"
CODE = b"gamma unit runtime code 490"
# The funding coldkey is a deterministic sr25519 pair so the fake wallet, the fake server
# identity and the deposit storage all describe the same account.
COLDKEY_PAIR = keypair_from_seed(b"c" * 32)
COLDKEY = ss58_encode(COLDKEY_PAIR[0])
HOTKEY = ss58_encode(bytes([5]) * 32)
GAMMA_HOTKEY = ss58_encode(bytes([9]) * 32)
OTHER_COLDKEY = ss58_encode(bytes([7]) * 32)
ATOMS = 1_000_000_000
CONFIG_KEY = Ed25519PrivateKey.from_private_bytes(bytes([47]) * 32)
QUOTE_KEY = Ed25519PrivateKey.from_private_bytes(bytes([31]) * 32)
READ_METHODS = {
    "chain_getFinalizedHead",
    "chain_getHeader",
    "state_getRuntimeVersion",
    "system_chain",
    "chain_getBlockHash",
    "system_accountNextIndex",
    "state_getMetadata",
    "state_getStorage",
    "payment_queryInfo",
    "chain_getBlock",
}

# A real mainnet spend_gamma extrinsic (runtime 490, block 9478505, extrinsic 10). Public
# chain data, no wallet material.
REAL_SPEND_HEX = (
    "0xd502840098b62654bf786866f188d5a2ba6b21a64904726c92e566c0a2dd60a5694dc26e016e039072"
    "4942b0dddfa9bf966e5bdf8c1cec11f65b062875e1eb23c20cfb553348202b46461c82f6419b3823c185d5"
    "6be04413194680be1ac736e3dcaed7b48144004c000007a4350004feda65b42b11beeda0c958b32718289"
    "82230057e39628582c9e3bea44d93bf5800ca9a3b00000000dc703fcc934bfdae1b2090f3cb4e12218795"
    "a1b3b88405a3750fbd1c854cdce4"
)
REAL_SPEND = {
    "signer": "98b62654bf786866f188d5a2ba6b21a64904726c92e566c0a2dd60a5694dc26e",
    "era": "4400",
    "nonce": 19,
    "netuid": 53,
    "hotkey": "feda65b42b11beeda0c958b3271828982230057e39628582c9e3bea44d93bf58",
    "amount": 1_000_000_000,
    "reference": "dc703fcc934bfdae1b2090f3cb4e12218795a1b3b88405a3750fbd1c854cdce4",
}
REFERENCE = "0x" + "11" * 32


# --- pure SCALE primitives ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "data,expected",
    [
        (b"", 0xEF46DB3751D8E999),
        (b"a", 0xD24EC4F1A98C6E5B),
        (b"abc", 0x44BC2CF5AD770999),
    ],
)
def test_xxh64_reproduces_the_canonical_vectors(data, expected):
    assert xxh64(data) == expected


def test_twox128_matches_the_real_chain_storage_prefixes():
    # The twox128("System") prefix and the exact SubtensorModule GammaEnabled storage key,
    # as observed on bittensor mainnet.
    assert twox128("System").hex() == "26aa394eea5630e07c48ae0c9558cef7"
    assert (
        twox128("SubtensorModule") + twox128("GammaEnabled")
    ).hex() == "658faa385070e074c85bf6b568cf0555c632b24025fe3007204114bed21cf60c"


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, b"\x00"),
        (1, b"\x04"),
        (63, b"\xfc"),
        (64, b"\x01\x01"),
        (16383, b"\xfd\xff"),
        (16384, b"\x02\x00\x01\x00"),
        (1 << 32, b"\x07\x00\x00\x00\x00\x01"),
    ],
)
def test_scale_compact_round_trips(value, expected):
    encoded = scale_compact_uint(value)
    assert encoded == expected
    decoded, offset = gf.scale_compact_decode(encoded + b"\xff", 0)
    assert decoded == value and offset == len(encoded)


@pytest.mark.parametrize("raw", [b"\x01\x00", b"\x02\x01\x00", b"\x03\x00\x00\x00\x00"])
def test_non_canonical_compact_decodings_refuse(raw):
    with pytest.raises(GammaRefused):
        gf.scale_compact_decode(raw, 0)


def test_mortal_era_encoding_matches_the_audited_vector():
    assert mortal_era_bytes(100, 64) == bytes.fromhex("4502")
    assert era_bounds(1000, 64) == (1000, 1001, 1063)
    assert mortal_era_bytes(1000, 64) == bytes.fromhex("8502")
    with pytest.raises(GammaRefused):
        mortal_era_bytes(1000, 63)


def test_spend_call_is_exactly_one_single_entry_burn():
    call = spend_gamma_call(37, GAMMA_HOTKEY, ATOMS, REFERENCE)
    assert call[:2] == bytes.fromhex("07a4")
    assert call[2:4] == struct.pack("<H", 37)
    assert call[4] == 0x04  # compact(1): exactly one entry
    assert call[5:37] == ss58_decode(GAMMA_HOTKEY)
    assert struct.unpack("<Q", call[37:45])[0] == ATOMS
    assert call[45:] == bytes([0x11]) * 32
    assert len(call) == 77
    expected = {
        "netuid": 37,
        "deposit_hotkey": GAMMA_HOTKEY,
        "amount_atoms": ATOMS,
        "reference_hex": REFERENCE,
    }
    assert decode_spend_gamma_call(call, expected)["deposit_hotkey"] == GAMMA_HOTKEY
    for bad in (
        call[:76],
        b"\x07\xa5" + call[2:],
        call[:45] + struct.pack("<Q", ATOMS - 1) + call[53:],
        call[:37] + ss58_decode(HOTKEY) + call[69:],
    ):
        with pytest.raises(GammaRefused):
            decode_spend_gamma_call(bad, expected)


def test_mint_call_always_names_the_own_coldkey_spender():
    call = mint_gamma_call(
        37, GAMMA_HOTKEY, 10**9, 100_000_000, spender=COLDKEY, daily_spend_cap_atoms=None
    )
    assert call[:2] == bytes.fromhex("079e")
    assert call[2:4] == struct.pack("<H", 37)
    assert call[4:36] == ss58_decode(GAMMA_HOTKEY)
    assert struct.unpack("<Q", call[36:44])[0] == 10**9
    assert struct.unpack("<Q", call[44:52])[0] == 100_000_000
    assert call[52] == 0x01 and call[53:85] == ss58_decode(COLDKEY)  # Some(spender), never None
    assert call[85:] == b"\x00"
    capped = mint_gamma_call(
        37, GAMMA_HOTKEY, 10**9, 1, spender=COLDKEY, daily_spend_cap_atoms=5_000_000_000
    )
    assert capped[85:] == b"\x01" + struct.pack("<Q", 5_000_000_000)


def test_unsigned_dict_passes_the_audited_validator():
    call = spend_gamma_call(37, GAMMA_HOTKEY, ATOMS, REFERENCE)
    unsigned = unsigned_extrinsic_dict(
        call=call,
        address=COLDKEY,
        era_current=1000,
        era_period=64,
        nonce=7,
        genesis_hash=GENESIS_HASH,
        era_block_hash="0x" + "cd" * 32,
        spec_version=490,
        transaction_version=1,
    )
    assert set(unsigned) == {
        "call_data",
        "address",
        "public_key",
        "crypto_type",
        "era",
        "nonce",
        "tip",
        "tip_asset_id",
        "genesis_hash",
        "era_block_hash",
        "spec_version",
        "transaction_version",
        "metadata_hash",
        "payload",
        "included_in_extrinsic",
        "included_in_signed_data",
        "payload_json",
    }
    authority = gf._validated_authority(unsigned, expected_call=call, expected_period=64)
    assert authority.nonce == 7 and authority.era_birth == 1000 and authority.tip == 0
    assert authority.payload == call + bytes.fromhex(
        unsigned["included_in_extrinsic"][2:]
    ) + bytes.fromhex(unsigned["included_in_signed_data"][2:])


def test_composed_extrinsic_round_trips_and_matches_the_real_envelope():
    pair = keypair_from_seed(b"f" * 32)
    call = spend_gamma_call(37, GAMMA_HOTKEY, ATOMS, REFERENCE)
    extra = mortal_era_bytes(1000, 64) + scale_compact_uint(7) + scale_compact_uint(0) + b"\x00"
    payload = call + extra + b"x" * 105
    signature = sign(payload, pair)
    extrinsic = compose_signed_extrinsic(
        call=call, extra=extra, signer_public_key=pair[0], signature=signature
    )
    decoded = decode_signed_extrinsic(extrinsic, expected_call=call)
    assert decoded["signature"] == signature and decoded["nonce"] == 7
    assert decoded["era"] == mortal_era_bytes(1000, 64).hex()
    assert decoded["tip"] == 0 and decoded["signer_public_key"] == pair[0]
    assert (
        extrinsic_tx_hash(extrinsic)
        == "0x" + hashlib.blake2b(extrinsic, digest_size=32).hexdigest()
    )
    # A real mainnet spend decodes field-for-field with the same strict parser.
    real_call = spend_gamma_call(
        53,
        ss58_encode(bytes.fromhex(REAL_SPEND["hotkey"])),
        REAL_SPEND["amount"],
        "0x" + REAL_SPEND["reference"],
    )
    real = decode_signed_extrinsic(bytes.fromhex(REAL_SPEND_HEX[2:]), expected_call=real_call)
    assert real["era"] == REAL_SPEND["era"] and real["nonce"] == REAL_SPEND["nonce"]
    assert real["signer_public_key"].hex() == REAL_SPEND["signer"]
    # Any drift between composed and expected bytes refuses at decode-back.
    with pytest.raises(GammaRefused):
        decode_signed_extrinsic(bytes.fromhex(REAL_SPEND_HEX[2:]), expected_call=call)
    with pytest.raises(GammaRefused):
        decode_signed_extrinsic(bytes.fromhex(REAL_SPEND_HEX[2:])[:-1], expected_call=real_call)


# --- journal -----------------------------------------------------------------------------------


def test_journal_claims_are_atomic_immutable_and_secret_free(tmp_path):
    journal = GammaFundingJournal(tmp_path / "gamma")
    assert journal.claim("prepared", {"status": "PREPARED", "nonce": 4}) is True
    assert journal.claim("prepared", {"status": "PREPARED", "nonce": 4}) is False
    with pytest.raises(GammaRefused):
        journal.claim("prepared", {"status": "PREPARED", "nonce": 5})
    assert (tmp_path / "gamma" / "prepared.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(GammaRefused):
        journal.read("not-a-record")
    document = (tmp_path / "gamma" / "prepared.json").read_text()
    assert "mnemonic" not in document and "seed" not in document


def test_journal_refuses_symlinks(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(GammaRefused):
        GammaFundingJournal(link)
    journal = GammaFundingJournal(target)
    (target / "prepared.json").symlink_to(tmp_path / "missing")
    with pytest.raises(GammaRefused):
        journal.read("prepared")


# --- strict fakes ------------------------------------------------------------------------------


def purchase_policy(*, lifetime=300):
    body = {
        "schema": gamma_policy.SCHEMA,
        "service_id": "gamma-aurelius-mainnet",
        "chain_id": CHAIN_ID,
        "genesis_hash": GENESIS_HASH,
        "netuid": 37,
        "policy_version": "gpp-2026-11",
        "packs": {"starter": {"gamma_atoms": str(ATOMS), "work_tokens": "1"}},
        "activation_block": 1,
        "quote_lifetime_blocks": lifetime,
        "max_packs_per_intent": 1,
        "per_account_packs_per_day": 24,
        "global_packs_per_day": 10_000,
        "payer_mode": gamma_policy.PAYER_MODE,
        "runtime_profile": "gamma-mainnet-490",
        "governing_config_hash": "0x" + "cd" * 32,
        "signing_key_id": "operator-receipt-v1",
    }
    policy = gamma_policy.from_map(body)
    preimage = (
        b"aurelius/gamma_purchase_policy/v1\n" + policy.digest()[2:].encode() + b"|gpp-2026-11"
    )
    return policy, {
        "body": body,
        "signature_hex": CONFIG_KEY.sign(preimage).hex(),
        "policy_digest": policy.digest(),
    }


def deposit_value(owner, spender, balance, cap=None, available=None, bucket_block=None):
    body = ss58_decode(owner) + ss58_decode(spender) + struct.pack("<Q", balance)
    if cap is None:
        body += b"\x00"
    else:
        body += (
            b"\x01"
            + struct.pack("<Q", cap)
            + struct.pack("<Q", available)
            + struct.pack("<Q", bucket_block)
        )
    return "0x" + body.hex()


def account_value(nonce, free):
    return (
        "0x"
        + struct.pack("<IIII", nonce, 0, 1, 0).hex()
        + struct.pack("<QQQ", free, 0, 0).hex()
        + (0).to_bytes(16, "little").hex()
    )


class FakeRpc:
    def __init__(self, *, head=1000, spec=490):
        self.head = head
        self.spec = spec
        self.storage = {}
        self.methods = []
        self.blocks = {}

    def snapshot(self):
        return SimpleNamespace(
            head_hash="0x" + "cd" * 32,
            block_number=self.head,
            genesis_hash=GENESIS_HASH,
            chain="Bittensor",
            runtime_spec_version=self.spec,
            transaction_version=1,
        )

    def call(self, method, params):
        self.methods.append(method)
        if method == "state_getMetadata":
            return "0x" + META.hex()
        if method == "state_getStorage":
            if params[0] == "0x3a636f6465":
                return "0x" + CODE.hex()
            return self.storage.get(params[0])
        if method == "chain_getBlockHash":
            return self.blocks.get(params[0], "0x" + "0a" * 32)
        if method == "chain_getBlock":
            # keyed by block HASH, exactly like the node RPC
            return self.blocks.get(
                ("block", params[0]), {"block": {"extrinsics": [], "header": {}}}
            )
        if method == "system_accountNextIndex":
            return self.storage.get(("next-index", params[0]), self.storage[("nonce", params[0])])
        if method == "payment_queryInfo":
            return {"partialFee": self.storage.get("fee", 1_000), "weight": {}, "class": "normal"}
        raise AssertionError(method)


class FakeServer:
    def __init__(self, *, gamma_reason="available", quote_key=QUOTE_KEY):
        self.gamma_reason = gamma_reason
        self.quote_key = quote_key
        self.quote_calls = []
        self.config_hash = None
        self.policy = purchase_policy()[0]

    def snapshot_document(self):
        body = {"config_version": 1, "entries": {}, "null_keys_present": False}
        self.config_hash = canonical_hash(body)
        return {
            "version": 1,
            "body": body,
            "canonical_hash": self.config_hash,
            "signature_hex": CONFIG_KEY.sign(
                SNAPSHOT_SIGNATURE_DOMAIN + f"{self.config_hash}|1".encode()
            ).hex(),
        }

    def terms_response(self):
        if self.config_hash is None:
            self.snapshot_document()
        _, envelope = purchase_policy()
        if self.gamma_reason != "available":
            gamma = {"available": False, "reason": self.gamma_reason}
        else:
            gamma = {"available": True, "reason": "available", "policy": envelope}
        return {
            "schema_version": 1,
            "identity": {"coldkey": COLDKEY, "hotkey": HOTKEY, "netuid": 37},
            "config_hash": self.config_hash,
            "finalized_block": 1000,
            "observed_at": 123,
            "terms": {"status": "unavailable", "reason": "mainnet_funding_not_selected"},
            "account": {
                "balance": {
                    "available_real": 0,
                    "available_synthetic": 0,
                    "deposited_lifetime": 0,
                    "consumed_lifetime": 0,
                    "synthetic_granted_lifetime": 0,
                    "synthetic_consumed_lifetime": 0,
                },
                "receipts": {
                    "real": [],
                    "synthetic": [],
                    "more_real": False,
                    "more_synthetic": False,
                    "limit_per_ledger": 25,
                },
            },
            "funding_methods": {
                "terms_version": 2,
                "legacy": {"available": False},
                "gamma": gamma,
            },
        }

    def quote(self, *, pack_id="starter", **overrides):
        pack = self.policy.pack(pack_id)
        assigns = {
            "intent_id": "gfi-" + "1" * 16,
            "beneficiary_coldkey": COLDKEY,
            "beneficiary_hotkey": HOTKEY,
            "funding_coldkey": COLDKEY,
            "gamma_api_hotkey": GAMMA_HOTKEY,
            "authorized_spender": COLDKEY,
            "gamma_atoms": str(pack["gamma_atoms"]),
            "work_tokens": str(pack["work_tokens"]),
            "config_hash": "0x" + self.config_hash,
            "issued_at_finalized_block": 1000,
            "valid_from_block": 1000,
            "valid_until_block": 1000 + self.policy.quote_lifetime_blocks,
            "server_nonce": "0x" + "22" * 32,
            "client_request_id": "req-flow",
        }
        assigns.update(overrides)
        intent = gamma_intent.build(self.policy, pack_id, **assigns)
        return {
            "schema_version": 1,
            "intent_id": intent.intent_id,
            "reference": intent.reference_hex,
            "valid_from_block": intent.body["valid_from_block"],
            "valid_until_block": intent.body["valid_until_block"],
            "envelope": {
                "body": intent.body,
                "signature_hex": self.quote_key.sign(intent.signature_payload()).hex(),
                "key_id": "operator-receipt-v1",
            },
        }

    def fetch(self, url, body=None):
        if body is None:
            return self.snapshot_document()
        if url.endswith("/api/miner/funding"):
            return self.terms_response()
        if url.endswith("/api/miner/funding/gamma/intents"):
            self.quote_calls.append(body)
            return self.quote()
        raise AssertionError(url)


class FakeWallet:
    def __init__(self, pair):
        self.pair = pair
        self.coldkeypub = SimpleNamespace(ss58_address=ss58_encode(pair[0]))
        self.unlocks = 0
        self.messages = []

    @property
    def coldkey(self):
        self.unlocks += 1
        wallet = self

        class Signer:
            ss58_address = wallet.coldkeypub.ss58_address

            @staticmethod
            def sign(message):
                wallet.messages.append(message)
                return sign(message, wallet.pair)

        return Signer()


def deposit_storage_key():
    return gf._u16_blake_double_storage_key("SubtensorModule", "GammaDeposits", 37, GAMMA_HOTKEY)


def account_storage_key():
    return (
        "0x"
        + (
            twox128("System") + twox128("Account") + gf.blake2_128_concat(ss58_decode(COLDKEY))
        ).hex()
    )


def flow(tmp_path, monkeypatch, *, deposit_kwargs=None):
    monkeypatch.setattr(
        gf,
        "GAMMA_PROFILES",
        {490: (hashlib.sha256(META).hexdigest(), hashlib.sha256(CODE).hexdigest())},
    )
    server = FakeServer()
    rpc = FakeRpc()
    deposit = {"owner": COLDKEY, "spender": COLDKEY, "balance": ATOMS * 10}
    deposit.update(deposit_kwargs or {})
    rpc.storage[deposit_storage_key()] = deposit_value(**deposit)
    rpc.storage[account_storage_key()] = account_value(7, 10**12)
    rpc.storage[("nonce", COLDKEY)] = 7
    rpc.storage[("next-index", COLDKEY)] = 7
    rpc.storage[
        gf._u16_blake_double_storage_key("SubtensorModule", "ColdkeyGammaHotkeys", 37, COLDKEY)
    ] = "0x" + (b"\x04" + ss58_decode(GAMMA_HOTKEY)).hex()
    rpc.storage[gf._u16_identity_storage_key("SubtensorModule", "SubnetGammaEnabled", 37)] = "0x01"
    rpc.storage[gf._plain_storage_key("SubtensorModule", "GammaEnabled")] = "0x01"
    rpc.storage[gf._u16_identity_storage_key("SubtensorModule", "SubnetGammaPrice", 37)] = (
        "0x" + (47_971_739_919_180_758).to_bytes(16, "little").hex()
    )
    rpc.storage[gf._plain_storage_key("SubtensorModule", "TaoUsdPrice")] = (
        "0x" + struct.pack("<QQQ", 209410090118, 209410090118, 990).hex()
    )
    wallet = FakeWallet(COLDKEY_PAIR)
    args = dict(
        chain=GammaChain(rpc),
        journal=GammaFundingJournal(tmp_path / "intent"),
        pack_id="starter",
        gamma_api_hotkey=GAMMA_HOTKEY,
        fee_cap_rao=10_000_000,
        era_period=64,
        fetch=server.fetch,
        base_url="https://example.test",
        client_request_id="req-flow",
        now=1000,
    )
    return SimpleNamespace(
        rpc=rpc,
        server=server,
        wallet=wallet,
        args=args,
        journal=args["journal"],
        chain=args["chain"],
        policy=server.policy,
    )


def hotkey_auth_signer(pair):
    def signer(message):
        assert message.startswith(GAMMA_AUTH_DOMAIN + b"\x00")
        return sign(message, pair)

    return signer


def validated_terms(case):
    return {
        "legacy": case.server.terms_response(),
        "funding_methods": gf._validate_funding_methods(
            case.server.terms_response()["funding_methods"],
            CONFIG_KEY.public_key().public_bytes_raw().hex(),
        ),
        "terms_version": 2,
    }


def run_prepare(case, **overrides):
    return gamma_prepare(
        wallet=case.wallet,
        terms=validated_terms(case),
        gamma_auth_signer=hotkey_auth_signer(keypair_from_seed(b"h" * 32)),
        quote_pubkey=QUOTE_KEY.public_key().public_bytes_raw().hex(),
        **{**case.args, **overrides},
    )


def prepared_case(tmp_path, monkeypatch):
    case = flow(tmp_path, monkeypatch)
    run_prepare(case)
    case.rpc.head = 1001  # the chain moves one block before the observation scan
    return case


def include_extrinsic(case, height=1001):
    prepared = case.journal.read("prepared")
    block_hash = "0x" + "0b" * 32
    case.rpc.blocks[height] = block_hash
    case.rpc.blocks[("block", block_hash)] = {
        "block": {"extrinsics": [prepared["extrinsic_hex"]], "header": {"number": hex(height)}}
    }
    return prepared


# --- prepare -----------------------------------------------------------------------------------


def test_prepare_signs_once_and_persists_hash_bound_bytes(tmp_path, monkeypatch):
    case = flow(tmp_path, monkeypatch)
    result = run_prepare(case)
    assert result["status"] == "PREPARED" and result["executed"] is False
    assert result["gamma_atoms"] == ATOMS and result["work_tokens"] == 1
    assert result["planned_fee_tao_rao"] == 1_000
    prepared = case.journal.read("prepared")
    assert prepared["tx_hash"] == extrinsic_tx_hash(bytes.fromhex(prepared["extrinsic_hex"][2:]))
    assert prepared["era"]["scan_end"] - prepared["era"]["birth"] + 1 == 64
    assert prepared["era"]["scan_end"] <= prepared["valid_until_block"]
    assert case.wallet.unlocks == 1  # one unlock: the binding proof and the payload sign
    assert len(case.wallet.messages) == 2
    assert case.wallet.messages[0].startswith(b"aurelius/gamma_funding_binding/v1\n")
    assert case.wallet.messages[1] == bytes.fromhex(prepared["unsigned"]["payload"][2:])
    decoded = decode_signed_extrinsic(
        bytes.fromhex(prepared["extrinsic_hex"][2:]),
        expected_call=spend_gamma_call(37, GAMMA_HOTKEY, ATOMS, prepared["reference_hex"]),
    )
    assert decoded["nonce"] == 7 and decoded["tip"] == 0
    assert verify(
        decoded["signature"],
        bytes.fromhex(prepared["unsigned"]["payload"][2:]),
        ss58_decode(COLDKEY),
    )
    # No journal secret material and no broadcast on the prepare path.
    assert "seed" not in json.dumps(prepared)
    assert "author_submitExtrinsic" not in case.rpc.methods


def test_prepare_is_idempotent_and_never_resigns(tmp_path, monkeypatch):
    case = flow(tmp_path, monkeypatch)
    first = run_prepare(case)
    before = case.wallet.messages[:]
    second = run_prepare(case)
    assert second["tx_hash"] == first["tx_hash"]
    assert case.wallet.messages == before
    with pytest.raises(GammaRefused, match="retained state"):
        run_prepare(case, pack_id="other-pack")


def test_a_crash_between_quote_and_prepared_reuses_the_retained_request_scope(
    tmp_path, monkeypatch
):
    case = flow(tmp_path, monkeypatch)
    original = case.journal.claim

    def interrupted_claim(name, value):
        if name == "prepared":
            raise OSError("process stopped before the prepared record was durable")
        return original(name, value)

    monkeypatch.setattr(case.journal, "claim", interrupted_claim)
    with pytest.raises(OSError):
        run_prepare(case)
    # The quote is retained; nothing was prepared and nothing was ever broadcast.
    assert case.journal.read("quote")["status"] == "QUOTE_VERIFIED"
    assert case.journal.read("prepared") is None
    assert "author_submitExtrinsic" not in case.rpc.methods
    monkeypatch.setattr(case.journal, "claim", original)
    result = run_prepare(case, client_request_id=None)
    assert result["status"] == "PREPARED"
    # The retry replays the SAME server idempotency key — never a replacement intent.
    assert [call["request"]["client_request_id"] for call in case.server.quote_calls] == [
        "req-flow",
        "req-flow",
    ]
    with pytest.raises(GammaRefused, match="retained state"):
        run_prepare(case, pack_id="other-pack")


def test_a_changed_scope_after_a_retained_quote_refuses(tmp_path, monkeypatch):
    case = flow(tmp_path, monkeypatch)
    original = case.journal.claim

    def interrupted_claim(name, value):
        if name == "prepared":
            raise OSError("stopped")
        return original(name, value)

    monkeypatch.setattr(case.journal, "claim", interrupted_claim)
    with pytest.raises(OSError):
        run_prepare(case)
    monkeypatch.setattr(case.journal, "claim", original)
    with pytest.raises(GammaRefused, match="scope changed"):
        run_prepare(case, pack_id="other-pack")


@pytest.mark.parametrize(
    "fault",
    [
        "runtime",
        "spender",
        "owner",
        "no_deposit",
        "balance",
        "cap",
        "fee",
        "era",
        "nonce_pending",
        "free_tao",
        "gamma_unavailable",
        "no_policy",
        "quote_signature",
        "quote_opens_after_inclusion",
    ],
)
def test_every_prepare_mismatch_refuses_before_signing_or_broadcast(tmp_path, monkeypatch, fault):
    case = flow(tmp_path, monkeypatch)
    if fault == "runtime":
        case.rpc.spec = 491
    elif fault == "spender":
        case.rpc.storage[deposit_storage_key()] = deposit_value(COLDKEY, OTHER_COLDKEY, ATOMS * 10)
    elif fault == "owner":
        case.rpc.storage[deposit_storage_key()] = deposit_value(OTHER_COLDKEY, COLDKEY, ATOMS * 10)
    elif fault == "no_deposit":
        del case.rpc.storage[deposit_storage_key()]
    elif fault == "balance":
        case.rpc.storage[deposit_storage_key()] = deposit_value(COLDKEY, COLDKEY, ATOMS - 1)
    elif fault == "cap":
        case.rpc.storage[deposit_storage_key()] = deposit_value(
            COLDKEY, COLDKEY, ATOMS * 10, cap=ATOMS - 1, available=0, bucket_block=1000
        )
    elif fault == "fee":
        case.rpc.storage["fee"] = 10_000_001
    elif fault == "era":
        case.args["era_period"] = 512
    elif fault == "nonce_pending":
        case.rpc.storage[("next-index", COLDKEY)] = 8
    elif fault == "free_tao":
        case.rpc.storage[account_storage_key()] = account_value(7, 5)
    elif fault == "gamma_unavailable":
        case.server.gamma_reason = "purchase_policy_unselected"
    elif fault == "no_policy":
        original_terms = case.server.terms_response

        def stripped_terms():
            response = original_terms()
            response["funding_methods"]["gamma"].pop("policy")
            return response

        case.server.terms_response = stripped_terms
    elif fault == "quote_signature":
        case.server.quote_key = Ed25519PrivateKey.from_private_bytes(bytes([50]) * 32)
    elif fault == "quote_opens_after_inclusion":
        original_quote = case.server.quote

        def future_quote():
            return original_quote(
                issued_at_finalized_block=1002,
                valid_from_block=1002,
                valid_until_block=1002 + case.server.policy.quote_lifetime_blocks,
            )

        case.server.quote = future_quote
    refusal = (
        "quote validity opens after the earliest possible inclusion block"
        if fault == "quote_opens_after_inclusion"
        else None
    )
    with pytest.raises(GammaRefused, match=refusal):
        run_prepare(case)
    # Every chain-state refusal happens BEFORE the wallet unlocks, a quote slot is consumed
    # or any submission transport exists.
    assert case.journal.read("prepared") is None
    assert "author_submitExtrinsic" not in case.rpc.methods
    if fault in {
        "runtime",
        "spender",
        "owner",
        "no_deposit",
        "balance",
        "cap",
        "fee",
        "era",
        "nonce_pending",
        "free_tao",
        "gamma_unavailable",
        "no_policy",
    }:
        assert case.wallet.unlocks == 0
        assert case.server.quote_calls == []
    # No spend payload was ever signed on any refusal path (binding-only signatures at most).
    assert all(not message.startswith(b"\x07\xa4") for message in case.wallet.messages)


def test_quote_opening_at_the_first_possible_inclusion_block_is_accepted(tmp_path, monkeypatch):
    case = flow(tmp_path, monkeypatch)
    original_quote = case.server.quote
    case.server.quote = lambda: original_quote(
        issued_at_finalized_block=1001,
        valid_from_block=1001,
        valid_until_block=1001 + case.server.policy.quote_lifetime_blocks,
    )
    assert run_prepare(case)["status"] == "PREPARED"
    prepared = case.journal.read("prepared")
    assert prepared["era"]["scan_start"] == 1001
    assert prepared["quote"]["valid_from_block"] == 1001
    assert "author_submitExtrinsic" not in case.rpc.methods


# --- execute + recovery ------------------------------------------------------------------------


def test_execute_submits_once_with_the_attempt_durable_first(tmp_path, monkeypatch):
    case = prepared_case(tmp_path, monkeypatch)
    prepared = include_extrinsic(case)
    order = []

    def submit(hex_payload):
        order.append("submit")
        # The attempt must already be durable BEFORE the network call returns processing.
        submission = case.journal.read("submission")
        assert submission["status"] == "SUBMISSION_ATTEMPTED"
        assert submission["prepared_sha256"] == gf._canonical_sha256(case.journal.read("prepared"))
        assert hex_payload == prepared["extrinsic_hex"]
        return prepared["tx_hash"]

    result = gamma_execute(chain=case.chain, journal=case.journal, submit=submit)
    assert result["status"] == "OBSERVED_ON_CHAIN"
    assert result["block_number"] == 1001 and result["extrinsic_index"] == 0
    assert result["spend_event"] == "events_unavailable"
    assert order == ["submit"]
    assert case.journal.read("observed")["status"] == "OBSERVED_ON_CHAIN"
    # The wallet was never touched on the execute path (no object is even accepted).
    assert case.wallet.unlocks == 1


def test_execute_before_the_head_advances_is_pending_not_a_refusal(tmp_path, monkeypatch):
    """The EMPTY observation window right after a successful broadcast (PR review
    75f1e28b5347): prepare ran at head 1000, so the era's scan_start is 1001 — and an
    --execute against an unchanged finalized head observes nothing yet. That is a pending
    OUTCOME_UNKNOWN (the journal keeps SUBMISSION_ATTEMPTED; the operator reruns), never
    a typed 'scan range out of bounds' refusal misreporting a successful broadcast as an
    operator error; once the chain advances one block the observation proceeds normally."""
    case = flow(tmp_path, monkeypatch)
    prepared = run_prepare(case)
    assert prepared["era"]["scan_start"] == 1001  # prepare head 1000 + 1

    result = gamma_execute(
        chain=case.chain, journal=case.journal, submit=lambda _: prepared["tx_hash"]
    )
    assert result["status"] == "OUTCOME_UNKNOWN" and result["executed"] is False
    assert case.journal.read("submission")["status"] == "SUBMISSION_ATTEMPTED"
    assert case.journal.read("observed") is None

    # one block advances: the window opens and the normal observation takes over
    case.rpc.head = 1001
    include_extrinsic(case)
    observed = gamma_observe(chain=case.chain, journal=case.journal)
    assert observed["status"] == "OBSERVED_ON_CHAIN"
    assert observed["block_number"] == 1001 and observed["extrinsic_index"] == 0


def test_a_matched_gamma_spent_event_distinguishes_payment_from_skip(tmp_path, monkeypatch):
    case = prepared_case(tmp_path, monkeypatch)
    prepared = include_extrinsic(case)
    spent = (
        b"\x07\xa5"
        + struct.pack("<H", 37)
        + ss58_decode(COLDKEY)
        + ss58_decode(GAMMA_HOTKEY)
        + ss58_decode(COLDKEY)
        + struct.pack("<Q", ATOMS)
        + bytes.fromhex(prepared["reference_hex"][2:])
    )
    skipped = (
        b"\x07\xa6"
        + struct.pack("<H", 37)
        + ss58_decode(GAMMA_HOTKEY)
        + ss58_decode(COLDKEY)
        + struct.pack("<Q", ATOMS)
        + bytes.fromhex(prepared["reference_hex"][2:])
        + b"\x03\x01\x00\x00"
    )
    events_key = gf._plain_storage_key("System", "Events")
    case.rpc.storage[events_key] = "0x" + (b"\x01" * 3 + spent + b"\x02" * 3).hex()
    result = gamma_execute(
        chain=case.chain, journal=case.journal, submit=lambda _: prepared["tx_hash"]
    )
    assert result["spend_event"] == "gamma_spent"
    case.rpc.storage[events_key] = "0x" + (b"\x01" + skipped).hex()
    case.journal.directory.joinpath("observed.json").unlink()
    assert (
        gamma_observe(chain=case.chain, journal=case.journal)["spend_event"]
        == "gamma_spend_skipped"
    )
    case.rpc.storage[events_key] = "0x" + (b"\x00" * 4).hex()
    case.journal.directory.joinpath("observed.json").unlink()
    assert (
        gamma_observe(chain=case.chain, journal=case.journal)["spend_event"]
        == "not_proven_in_events"
    )


def test_lost_response_and_restart_never_resign_or_resubmit(tmp_path, monkeypatch):
    case = prepared_case(tmp_path, monkeypatch)
    calls = []

    def dropping_submit(hex_payload):
        calls.append(hex_payload)
        raise OSError("connection dropped after broadcast")

    first = gamma_execute(chain=case.chain, journal=case.journal, submit=dropping_submit)
    assert first["status"] == "OUTCOME_UNKNOWN" and first["executed"] is True
    assert case.journal.read("submission")["status"] == "SUBMISSION_ATTEMPTED"
    # Restart: a fresh execute takes the observe-only path and never submits again.
    second = gamma_execute(chain=case.chain, journal=case.journal, submit=dropping_submit)
    assert second["status"] == "OUTCOME_UNKNOWN" and second["executed"] is False
    assert len(calls) == 1
    assert case.wallet.unlocks == 1  # unchanged since prepare


def test_an_rpc_timeout_leaves_the_same_unknown_state(tmp_path, monkeypatch):
    case = prepared_case(tmp_path, monkeypatch)

    def timing_out_submit(hex_payload):
        raise TimeoutError("rpc timeout")

    result = gamma_execute(chain=case.chain, journal=case.journal, submit=timing_out_submit)
    assert result["status"] == "OUTCOME_UNKNOWN"
    assert "never re-sign" in result["note"]


def test_complete_finalized_era_coverage_is_the_only_non_inclusion_proof(tmp_path, monkeypatch):
    case = prepared_case(tmp_path, monkeypatch)
    prepared = case.journal.read("prepared")
    gamma_execute(chain=case.chain, journal=case.journal, submit=lambda _: prepared["tx_hash"])
    case.rpc.head = 1100  # beyond scan_end 1063: complete finalized coverage, no inclusion
    result = gamma_status(chain=case.chain, journal=case.journal)
    assert result["status"] == "NOT_ON_CHAIN_AFTER_ERA"
    assert "settlement" in result["note"]
    assert "author_submitExtrinsic" not in case.rpc.methods
    case.rpc.head = 1005  # inside the era: absence is NOT proof of non-payment
    assert gamma_status(chain=case.chain, journal=case.journal)["status"] == "OUTCOME_UNKNOWN"


def test_expired_era_or_unreviewed_runtime_never_broadcasts_retained_bytes(tmp_path, monkeypatch):
    case = prepared_case(tmp_path, monkeypatch)
    case.rpc.head = 2000
    with pytest.raises(GammaRefused, match="mortal era expired"):
        gamma_execute(chain=case.chain, journal=case.journal, submit=lambda _: "0x" + "00")
    case.rpc.spec = 491
    case.rpc.head = 1001
    with pytest.raises(GammaRefused, match="unreviewed gamma runtime"):
        gamma_execute(chain=case.chain, journal=case.journal, submit=lambda _: "0x" + "00")
    assert case.journal.read("submission") is None


@pytest.mark.parametrize("fault", ["extrinsic_hex", "signature", "submission_hash"])
def test_tampered_journal_records_refuse_with_typed_errors(tmp_path, monkeypatch, fault):
    case = prepared_case(tmp_path, monkeypatch)
    prepared = case.journal.read("prepared")
    if fault != "submission_hash":
        if fault == "extrinsic_hex":
            tampered = {**prepared, "extrinsic_hex": "0x" + "00" * 90}
        else:
            tampered = {**prepared, "signature_hex": "00" * 64}
        (case.journal.directory / "prepared.json").write_text(
            json.dumps(tampered, sort_keys=True, separators=(",", ":"))
        )
        with pytest.raises(GammaRefused, match="differ"):
            gamma_execute(chain=case.chain, journal=case.journal, submit=lambda _: "0x" + "00")
        return
    gamma_execute(
        chain=case.chain,
        journal=case.journal,
        submit=lambda _: prepared["tx_hash"],
    )
    submission = case.journal.read("submission")
    (case.journal.directory / "submission.json").write_text(
        json.dumps(
            {**submission, "prepared_sha256": "0" * 64}, sort_keys=True, separators=(",", ":")
        )
    )
    with pytest.raises(GammaRefused, match="tampered"):
        gamma_observe(chain=case.chain, journal=case.journal)


def test_submission_transport_pins_genesis_and_returns_one_hash(tmp_path, monkeypatch):
    case = prepared_case(tmp_path, monkeypatch)
    prepared = case.journal.read("prepared")

    def fake_connect(endpoint, timeout=None):
        assert endpoint == "ws://node.example:9944"
        sent = []

        class Socket:
            def send(self, payload):
                sent.append(json.loads(payload)["method"])

            def recv(self):
                if sent == ["chain_getBlockHash"]:
                    return json.dumps({"jsonrpc": "2.0", "id": 1, "result": GENESIS_HASH})
                return json.dumps({"jsonrpc": "2.0", "id": 2, "result": prepared["tx_hash"]})

            def close(self):
                pass

        socket = Socket()
        socket.sent_methods = sent
        return socket

    monkeypatch.setattr("websocket.create_connection", fake_connect)
    submitter = SingleShotSubmitter("ws://node.example:9944")
    assert submitter(prepared["extrinsic_hex"]) == prepared["tx_hash"]
    bad = SingleShotSubmitter("ws://node.example:9944", expected_genesis="0x" + "00" * 32)
    with pytest.raises(GammaRefused, match="genesis"):
        bad(prepared["extrinsic_hex"])


# --- status + check: reads never sign -----------------------------------------------------------


def receipt_for(prepared):
    return {
        "schema": gamma_funding_receipt.SCHEMA,
        "funding_method": "gamma",
        "intent_id": prepared["intent_id"],
        "reference": prepared["reference_hex"],
        "pack_id": prepared["pack_id"],
        "purchase_policy_version": prepared["purchase_policy_version"],
        "purchase_policy_digest": prepared["purchase_policy_digest"],
        "config_hash": "0x" + prepared["config_hash"],
        "beneficiary_coldkey": COLDKEY,
        "beneficiary_hotkey": HOTKEY,
        "chain_id": CHAIN_ID,
        "genesis_hash": GENESIS_HASH,
        "netuid": 37,
        "finalized_block_hash": "0x" + "ee" * 32,
        "finalized_block_number": 1200,
        "extrinsic_index": 3,
        "event_index": 11,
        "tx_hash": "0x" + "ff" * 32,
        "funding_coldkey": COLDKEY,
        "gamma_api_hotkey": GAMMA_HOTKEY,
        "authorized_spender": COLDKEY,
        "gamma_atoms": str(prepared["gamma_atoms"]),
        "work_tokens": str(prepared["work_tokens"]),
        "deposit_key": gamma_funding_receipt.deposit_key(CHAIN_ID, "0x" + "ee" * 32, 3, 11),
        "ledger_transaction_id": 9,
        "verification_profile": "gamma-mainnet-490",
        "signing_key_id": "operator-receipt-v1",
    }


def test_status_reports_receipt_credit_only_from_a_verified_signature(tmp_path, monkeypatch):
    case = prepared_case(tmp_path, monkeypatch)
    prepared = case.journal.read("prepared")
    assert gamma_status(chain=case.chain, journal=case.journal) == {
        "status": "PREPARED_NOT_SUBMITTED",
        "executed": False,
        "tx_hash": prepared["tx_hash"],
        "note": "prepared and signed but never broadcast; only a fresh --execute submits",
    }
    gamma_execute(chain=case.chain, journal=case.journal, submit=lambda _: prepared["tx_hash"])
    terms = validated_terms(case)
    receipt_body = receipt_for(prepared)
    row = {
        "id": 1,
        "kind": "deposit",
        "amount": 1,
        "receipt": receipt_body,
        "signature_hex": QUOTE_KEY.sign(
            gamma_funding_receipt.signature_payload(receipt_body)
        ).hex(),
        "key_id": "operator-receipt-v1",
        "inserted_at": "2026-11-14T00:00:00Z",
    }
    terms["legacy"]["account"]["receipts"]["real"] = [row]
    result = gamma_status(
        chain=case.chain,
        journal=case.journal,
        terms=terms,
        quote_pubkey=QUOTE_KEY.public_key().public_bytes_raw().hex(),
    )
    assert result["status"] == "CENTRAL_CREDITED"
    assert result["receipt"]["intent_id"] == prepared["intent_id"]
    assert case.journal.read("credited")["receipt"]["ledger_transaction_id"] == 9
    # A forged row refuses instead of crediting.
    case.journal.directory.joinpath("credited.json").unlink()
    row["signature_hex"] = "00" * 64
    with pytest.raises(GammaRefused, match="receipt"):
        gamma_status(
            chain=case.chain,
            journal=case.journal,
            terms=terms,
            quote_pubkey=QUOTE_KEY.public_key().public_bytes_raw().hex(),
        )
    # The forged row was refused, so nothing was credited: a balance-only change never confirms.
    assert case.journal.read("credited") is None
    terms["legacy"]["account"]["receipts"]["real"] = []
    terms["legacy"]["account"]["balance"]["available_real"] += 10
    after = gamma_status(chain=case.chain, journal=case.journal, terms=terms)
    assert after["status"] == "OUTCOME_UNKNOWN"
    assert case.journal.read("credited") is None


def test_check_and_status_never_sign_or_touch_a_wallet(tmp_path, monkeypatch):
    case = flow(tmp_path, monkeypatch)
    terms = validated_terms(case)
    report = gamma_check(chain=case.chain, terms=terms)
    report = gamma_check_account(
        chain=case.chain,
        report=report,
        funding_coldkey=COLDKEY,
        gamma_api_hotkey=GAMMA_HOTKEY,
        pack_gamma_atoms=ATOMS,
        fee_cap_rao=10_000_000,
    )
    assert report["spend_readiness"]["ready"] is True
    assert report["mint_readiness"]["ready"] is True
    assert report["policy"]["packs"]["starter"] == {"gamma_atoms": ATOMS, "work_tokens": 1}
    assert set(report["spend_readiness"]["deposit"]) == {
        "owner_coldkey",
        "spender",
        "balance_atoms",
        "daily_spend_cap_atoms",
        "allowance_now_atoms",
    }
    assert set(case.rpc.methods) <= READ_METHODS
    assert case.wallet.unlocks == 0
    gamma_status(chain=case.chain, journal=case.journal)
    assert set(case.rpc.methods) <= READ_METHODS
    assert case.wallet.unlocks == 0


def test_check_reports_reasons_and_separates_mint_from_spend(tmp_path, monkeypatch):
    case = flow(
        tmp_path,
        monkeypatch,
        deposit_kwargs={
            "spender": OTHER_COLDKEY,
            "cap": ATOMS,
            "available": 0,
            "bucket_block": 1000,
        },
    )
    report = gamma_check(chain=case.chain, terms=validated_terms(case))
    report = gamma_check_account(
        chain=case.chain,
        report=report,
        funding_coldkey=COLDKEY,
        gamma_api_hotkey=GAMMA_HOTKEY,
        pack_gamma_atoms=ATOMS,
        fee_cap_rao=10_000_000,
    )
    assert report["spend_readiness"]["ready"] is False
    assert "spender_mismatch" in report["spend_readiness"]["reasons"]
    assert "daily_cap_insufficient" in report["spend_readiness"]["reasons"]
    assert report["mint_readiness"]["ready"] is False
    assert "set_gamma_deposit_spender" in report["mint_readiness"]["reason"]
    assert report["mint_readiness"]["current_deposit_spender"] == OTHER_COLDKEY


# --- mint setup --------------------------------------------------------------------------------


def test_mint_setup_is_explicit_and_never_silent(tmp_path, monkeypatch):
    case = flow(tmp_path, monkeypatch)
    result = gamma_mint_setup(
        chain=case.chain,
        journal=case.journal,
        wallet=case.wallet,
        gamma_api_hotkey=GAMMA_HOTKEY,
        tao="1",
        min_gamma_out="100000000",
        daily_spend_cap=None,
        execute=False,
        confirm=False,
        submit=None,
        era_period=64,
    )
    assert result["status"] == "MINT_CHECK" and result["executed"] is False
    assert result["mint_args"]["spender"] == COLDKEY
    assert result["mint_args"]["actual_current_spender"] == COLDKEY
    assert case.journal.read("mint_prepared") is None
    submitted = []

    def submit(hex_payload):
        submitted.append(hex_payload)
        return case.journal.read("mint_prepared")["tx_hash"]

    with pytest.raises(GammaRefused, match="confirm"):
        gamma_mint_setup(
            chain=case.chain,
            journal=case.journal,
            wallet=case.wallet,
            gamma_api_hotkey=GAMMA_HOTKEY,
            tao="1",
            min_gamma_out="100000000",
            daily_spend_cap=None,
            execute=True,
            confirm=False,
            submit=submit,
            era_period=64,
        )
    executed = gamma_mint_setup(
        chain=case.chain,
        journal=case.journal,
        wallet=case.wallet,
        gamma_api_hotkey=GAMMA_HOTKEY,
        tao="1",
        min_gamma_out="100000000",
        daily_spend_cap="5000000000",
        execute=True,
        confirm=True,
        submit=submit,
        era_period=64,
    )
    assert executed["status"] == "MINT_SUBMISSION_ATTEMPTED"
    assert len(submitted) == 1
    assert case.journal.read("mint_prepared")["spender"] == COLDKEY
    mint_call = bytes.fromhex(case.journal.read("mint_prepared")["unsigned"]["call_data"][2:])
    assert mint_call[52:85] == b"\x01" + ss58_decode(COLDKEY)  # Some(own coldkey) — never None
    with pytest.raises(GammaRefused, match="never repeat"):
        gamma_mint_setup(
            chain=case.chain,
            journal=case.journal,
            wallet=case.wallet,
            gamma_api_hotkey=GAMMA_HOTKEY,
            tao="1",
            min_gamma_out="100000000",
            daily_spend_cap=None,
            execute=True,
            confirm=True,
            submit=submit,
            era_period=64,
        )
    # The purchase path never touches the mint records and vice versa.
    assert case.journal.read("prepared") is None


def test_mint_refuses_an_existing_foreign_spender_deposit(tmp_path, monkeypatch):
    case = flow(tmp_path, monkeypatch, deposit_kwargs={"spender": OTHER_COLDKEY})
    with pytest.raises(GammaRefused, match="spender"):
        gamma_mint_setup(
            chain=case.chain,
            journal=case.journal,
            wallet=case.wallet,
            gamma_api_hotkey=GAMMA_HOTKEY,
            tao="1",
            min_gamma_out="1",
            daily_spend_cap=None,
            execute=True,
            confirm=True,
            submit=lambda _: "0x" + "00",
            era_period=64,
        )
    assert case.journal.read("mint_submission") is None


def test_spend_prepare_and_execute_never_invoke_mint_setup(tmp_path, monkeypatch):
    case = prepared_case(tmp_path, monkeypatch)
    gamma_execute(
        chain=case.chain,
        journal=case.journal,
        submit=lambda _: case.journal.read("prepared")["tx_hash"],
    )
    assert case.journal.read("mint_prepared") is None
    assert case.journal.read("mint_submission") is None


# --- CLI surface --------------------------------------------------------------------------------


@pytest.fixture
def terms_accepted(monkeypatch, tmp_path):
    """`fund` asks for the incentive terms before it touches either method. Accept them for the
    test and keep the acceptance record inside the test's temporary data directory."""
    monkeypatch.setenv("AURELIUS_MINER_DATA", str(tmp_path))
    monkeypatch.setenv("AURELIUS_ACCEPT_INCENTIVES", "yes")


def test_legacy_fund_dispatch_is_unchanged(monkeypatch, capsys, terms_accepted):
    captured = {}
    monkeypatch.setattr(
        "subnet.miner.funding.run",
        lambda **kwargs: captured.update(kwargs) or {"status": "preview"},
    )
    from subnet.miner import cli

    code = cli.main(
        [
            "fund",
            "--alpha",
            "1",
            "--wallet-name",
            "w",
            "--wallet-path",
            "/w",
            "--intent-dir",
            "/d",
            "--fee-cap-tao",
            "0.02",
        ]
    )
    assert code == 0
    assert captured == {
        "base_url": "https://central.aureliusaligned.ai",
        "endpoint": "wss://bittensor-finney.api.onfinality.io/public-ws",
        "wallet_name": "w",
        "wallet_path": "/w",
        "hotkey_name": "default",
        "alpha": "1",
        "fee_cap": "0.02",
        "execute": False,
        "directory": "/d",
    }
    assert json.loads(capsys.readouterr().out)["status"] == "preview"


def test_legacy_missing_arguments_still_fail_with_argparse_message(capsys):
    from subnet.miner import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["fund"])
    assert exc.value.code == 2
    assert "the following arguments are required: --alpha" in capsys.readouterr().err


def test_gamma_flags_are_rejected_on_the_legacy_method(capsys):
    from subnet.miner import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "fund",
                "--alpha",
                "1",
                "--wallet-name",
                "w",
                "--wallet-path",
                "/w",
                "--intent-dir",
                "/d",
                "--pack",
                "starter",
            ]
        )
    assert exc.value.code == 2
    assert "--pack require --method gamma" in capsys.readouterr().err


def test_gamma_check_is_the_wallet_free_default(monkeypatch, terms_accepted):
    captured = {}
    monkeypatch.setattr(
        "subnet.miner.gamma_funding.run",
        lambda **kwargs: captured.update(kwargs) or {"status": "gamma_check"},
    )
    from subnet.miner import cli

    assert cli.main(["fund", "--method", "gamma"]) == 0
    assert captured["action"] == "check" and captured["directory"] is None
    assert captured["execute"] is False and captured["era_period"] == 64


def test_gamma_actions_are_mutually_exclusive_and_gated(capsys):
    from subnet.miner import cli

    for argv, fragment in [
        (["fund", "--method", "gamma", "--prepare", "--status"], "mutually exclusive"),
        (["fund", "--method", "gamma", "--execute"], "requires --intent-dir"),
        (
            ["fund", "--method", "gamma", "--execute", "--intent-dir", "/d", "--check"],
            "never executes",
        ),
        (["fund", "--method", "gamma", "--mint-setup", "--mint-tao", "1"], "requires --intent-dir"),
    ]:
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2
        assert fragment in capsys.readouterr().err


def test_gamma_prepare_and_mint_flags_map_to_the_module(monkeypatch, terms_accepted):
    captured = {}
    monkeypatch.setattr(
        "subnet.miner.gamma_funding.run",
        lambda **kwargs: captured.update(kwargs) or {"status": "ok"},
    )
    from subnet.miner import cli

    cli.main(
        [
            "fund",
            "--method",
            "gamma",
            "--prepare",
            "--intent-dir",
            "/d",
            "--pack",
            "starter",
            "--gamma-hotkey",
            GAMMA_HOTKEY,
            "--wallet-name",
            "w",
            "--wallet-path",
            "/w",
            "--era-period-blocks",
            "128",
        ]
    )
    assert captured["action"] == "prepare"
    assert captured["pack"] == "starter" and captured["gamma_hotkey"] == GAMMA_HOTKEY
    assert captured["era_period"] == 128 and captured["confirm_mint"] is False
    captured.clear()
    cli.main(
        [
            "fund",
            "--method",
            "gamma",
            "--mint-setup",
            "--intent-dir",
            "/d",
            "--gamma-hotkey",
            GAMMA_HOTKEY,
            "--wallet-name",
            "w",
            "--wallet-path",
            "/w",
            "--execute",
            "--confirm-mint-burns-tao",
            "--mint-tao",
            "1",
            "--mint-min-gamma-out",
            "100000000",
            "--mint-daily-cap",
            "5000000000",
        ]
    )
    assert captured["action"] == "mint-setup" and captured["execute"] is True
    assert captured["confirm_mint"] is True
    assert captured["mint_tao"] == "1" and captured["mint_min_out"] == "100000000"
    assert captured["mint_cap"] == "5000000000"


@pytest.mark.parametrize(
    "argv",
    [
        ["fund", "--alpha", "1", "--wallet-name", "w", "--wallet-path", "/w", "--intent-dir", "/d"],
        ["fund", "--method", "gamma"],
        ["fund", "--method", "gamma", "--prepare", "--intent-dir", "/d", "--pack", "starter"],
        [
            "fund",
            "--method",
            "gamma",
            "--mint-setup",
            "--intent-dir",
            "/d",
            "--execute",
            "--confirm-mint-burns-tao",
            "--mint-tao",
            "1",
        ],
    ],
)
def test_fund_refuses_without_the_terms_agreement_whatever_the_method(
    monkeypatch, capsys, tmp_path, argv
):
    """Neither method reaches its module before the miner has agreed to the incentive terms:
    not the legacy purchase, and not a gamma check, prepare or mint."""
    import io

    monkeypatch.setenv("AURELIUS_MINER_DATA", str(tmp_path))
    monkeypatch.delenv("AURELIUS_ACCEPT_INCENTIVES", raising=False)
    # No terminal: consent is never assumed on behalf of the miner.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    reached = []
    monkeypatch.setattr("subnet.miner.funding.run", lambda **kwargs: reached.append("legacy"))
    monkeypatch.setattr("subnet.miner.gamma_funding.run", lambda **kwargs: reached.append("gamma"))
    from subnet.miner import cli

    assert cli.main(argv) == 2
    assert reached == []
    assert json.loads(capsys.readouterr().out)["status"] == "refused"


# --- the reviewed runtime table and the independent scalecodec proof -----------------------------

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_DIR = FIXTURES / "runtime490"
METADATA_SCALE = FIXTURE_DIR / "metadata.scale"
# bittensor mainnet: the chain, the runtime identity and the activation evidence of runtime 490.
CAPTURE = FIXTURE_DIR / "capture.json"


def test_every_reviewed_gamma_runtime_is_pinned_to_a_shipped_capture():
    """The static table never carries a runtime whose metadata is not shipped beside the tests.
    Each entry is re-derived from its capture, so a digest added or edited by hand fails here."""
    import subnet.miner.gamma_funding as module

    assert module.GAMMA_PROFILES, "no reviewed gamma runtime"
    for spec, (metadata_sha256, code_sha256) in module.GAMMA_PROFILES.items():
        directory = FIXTURES / f"runtime{spec}"
        metadata = directory / "metadata.scale"
        assert metadata.is_file(), f"runtime{spec}: ship its metadata capture"
        assert hashlib.sha256(metadata.read_bytes()).hexdigest() == metadata_sha256, spec
        capture = json.loads((directory / "capture.json").read_text())
        assert capture["runtime"]["specVersion"] == spec, spec
        assert capture["metadata_sha256"] == metadata_sha256, spec
        assert capture["code_sha256"] == code_sha256, spec


@pytest.fixture(scope="module")
def runtime490():
    pytest.importorskip("scalecodec")
    from scalecodec.base import RuntimeConfigurationObject, ScaleBytes
    from scalecodec.type_registry import load_type_registry_preset

    config = RuntimeConfigurationObject(ss58_format=42)
    config.update_type_registry(load_type_registry_preset(name="core"))
    metadata = config.create_scale_object(
        "MetadataVersioned", data=ScaleBytes(METADATA_SCALE.read_bytes())
    )
    metadata.decode()
    config.add_portable_registry(metadata)
    return config, metadata, metadata.value[1]["V14"]


@pytest.mark.skipif(not METADATA_SCALE.is_file(), reason="runtime490 capture not present")
def test_runtime490_metadata_pins_the_reviewed_identity(runtime490):
    import subnet.miner.gamma_funding as module

    digest = hashlib.sha256(METADATA_SCALE.read_bytes()).hexdigest()
    assert digest == module.GAMMA_PROFILES[490][0]
    capture = json.loads(CAPTURE.read_text())
    assert capture["metadata_sha256"] == digest
    assert capture["runtime"]["specVersion"] == 490
    assert capture["runtime"]["transactionVersion"] == 1
    assert capture["genesis"].lower() == GENESIS_HASH.lower()


@pytest.mark.skipif(not METADATA_SCALE.is_file(), reason="runtime490 capture not present")
def test_runtime490_signed_extensions_match_the_pinned_list(runtime490):
    _, _, document = runtime490
    assert tuple(
        item["identifier"] for item in document["extrinsic"]["signed_extensions"]
    ) == tuple(BITTENSOR_V11_SIGNED_EXTENSIONS)


@pytest.mark.skipif(not METADATA_SCALE.is_file(), reason="runtime490 capture not present")
def test_spend_call_bytes_encode_identically_from_the_pinned_metadata(runtime490):
    """Independent assembly from the pinned metadata's own arg types (sidecar-test pattern)."""
    config, _, document = runtime490
    types = {item["id"]: item["type"] for item in document["types"]["types"]}
    pallet = next(p for p in document["pallets"] if p["name"] == "SubtensorModule")
    variants = types[pallet["calls"]["ty"]]["def"]["variant"]["variants"]
    variant = next(v for v in variants if v["name"] == "spend_gamma")
    assert bytes([pallet["index"], variant["index"]]).hex() == "07a4"
    netuid_field, spends_field = variant["fields"]
    expected = (
        bytes([pallet["index"], variant["index"]])
        + bytes(config.create_scale_object(f"scale_info::{netuid_field['type']}").encode(37).data)
        + bytes(config.create_scale_object("Compact<u32>").encode(1).data)
        + ss58_decode(GAMMA_HOTKEY)
        + struct.pack("<Q", ATOMS)
        + bytes.fromhex(REFERENCE[2:])
    )
    assert expected == spend_gamma_call(37, GAMMA_HOTKEY, ATOMS, REFERENCE)


def _decoded_spend_entries(decoded_call):
    for arg in decoded_call.get("call_args") or []:
        if arg.get("name") == "spends":
            return arg["value"]
    return []


def _entry_fields(entry):
    if isinstance(entry, dict):
        return entry
    return dict(zip(("hotkey", "amount", "reference"), entry))


@pytest.mark.skipif(not METADATA_SCALE.is_file(), reason="runtime490 capture not present")
def test_composed_extrinsic_decodes_against_the_pinned_metadata(tmp_path, monkeypatch, runtime490):
    from scalecodec.base import ScaleBytes

    config, metadata, _ = runtime490
    case = flow(tmp_path, monkeypatch)
    run_prepare(case)
    prepared = case.journal.read("prepared")
    decoded = config.create_scale_object(
        "Extrinsic", data=ScaleBytes(prepared["extrinsic_hex"]), metadata=metadata
    )
    decoded.decode()
    value = decoded.value
    assert value["address"] == COLDKEY
    assert value["signature"]["Sr25519"].lower() == "0x" + prepared["signature_hex"]
    assert value["nonce"] == 7 and value["tip"] == 0
    assert list(value["era"]) == [64, 1000 % 64]
    fields = _entry_fields(_decoded_spend_entries(value["call"])[0])
    assert fields["hotkey"] == GAMMA_HOTKEY
    assert int(fields["amount"]) == ATOMS
    assert fields["reference"] == prepared["reference_hex"]


@pytest.mark.skipif(not METADATA_SCALE.is_file(), reason="runtime490 capture not present")
def test_the_real_mainnet_spend_decodes_identically_both_ways(runtime490):
    from scalecodec.base import ScaleBytes

    config, metadata, _ = runtime490
    decoded = config.create_scale_object(
        "Extrinsic", data=ScaleBytes(REAL_SPEND_HEX), metadata=metadata
    )
    decoded.decode()
    value = decoded.value
    assert value["call"]["call_index"] == "0x07a4"
    fields = _entry_fields(_decoded_spend_entries(value["call"])[0])
    assert int(fields["amount"]) == REAL_SPEND["amount"]
    assert fields["reference"] == "0x" + REAL_SPEND["reference"]
    strict = decode_signed_extrinsic(
        bytes.fromhex(REAL_SPEND_HEX[2:]),
        expected_call=spend_gamma_call(
            53,
            ss58_encode(bytes.fromhex(REAL_SPEND["hotkey"])),
            REAL_SPEND["amount"],
            "0x" + REAL_SPEND["reference"],
        ),
    )
    assert strict["nonce"] == value["nonce"]
    assert strict["era"] == REAL_SPEND["era"]


@pytest.mark.skipif(not METADATA_SCALE.is_file(), reason="runtime490 capture not present")
def test_storage_value_layouts_match_the_pinned_metadata(runtime490):
    """My hand-built storage layouts decode field-for-field through the pinned metadata."""
    from scalecodec.base import ScaleBytes

    config, metadata, _ = runtime490
    pallet = metadata.get_metadata_pallet("SubtensorModule")
    deposit_type = pallet.get_storage_function("GammaDeposits").get_value_type_string()
    capped = config.create_scale_object(
        deposit_type,
        data=ScaleBytes(
            deposit_value(
                COLDKEY,
                COLDKEY,
                ATOMS,
                cap=5_000_000_000,
                available=4_000_000_000,
                bucket_block=900,
            )
        ),
        metadata=metadata,
    )
    capped.decode()
    assert capped.value["coldkey"] == COLDKEY
    assert capped.value["spender"] == COLDKEY
    assert capped.value["balance"] == ATOMS
    assert capped.value["daily_spend_cap"] == {
        "cap": 5_000_000_000,
        "bucket": (4_000_000_000, 900),  # scalecodec decodes the bucket tuple as a tuple
    }
    uncapped = config.create_scale_object(
        deposit_type,
        data=ScaleBytes(deposit_value(COLDKEY, OTHER_COLDKEY, ATOMS * 2)),
        metadata=metadata,
    )
    uncapped.decode()
    assert uncapped.value["daily_spend_cap"] is None
    assert uncapped.value["spender"] == OTHER_COLDKEY
    # ...and my strict decoder returns exactly those facts from the same bytes.
    decoded = GammaChain.decode_gamma_deposit(
        deposit_value(
            COLDKEY, COLDKEY, ATOMS, cap=5_000_000_000, available=4_000_000_000, bucket_block=900
        )
    )
    assert decoded.owner_coldkey == COLDKEY and decoded.spender == COLDKEY
    assert decoded.balance_atoms == ATOMS
    assert decoded.cap_atoms == 5_000_000_000
    assert decoded.bucket_available == 4_000_000_000 and decoded.bucket_block == 900
    assert decoded.allowance_at(900 + 7200) == 5_000_000_000  # refilled to the cap
    account_type = metadata.get_metadata_pallet("System")
    account_type = account_type.get_storage_function("Account").get_value_type_string()
    raw = bytes.fromhex(account_value(7, 10**12)[2:])
    account = config.create_scale_object(
        account_type, data=ScaleBytes("0x" + raw.hex()), metadata=metadata
    )
    account.decode()
    assert account.value["nonce"] == 7
    assert account.value["data"]["free"] == 10**12
    assert len(raw) == 56  # the strict layout the miner decoder requires
    chain = GammaChain(
        SimpleNamespace(
            call=lambda method, params: "0x" + raw.hex() if method == "state_getStorage" else None
        )
    )
    assert chain.system_account("0x" + "cd" * 32, COLDKEY) == {"nonce": 7, "free_rao": 10**12}
