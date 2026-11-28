"""Gamma work-token funding operation — check/prepare/execute/status + explicit mint setup.

CONTRACT (capsule header)
- Extends the deployed no-resubmit discipline of :mod:`subnet.miner.funding` (gamma plan
  §11.2/§11.3, §14-G7). The payment is ONE direct ``spend_gamma`` call signed by the funding
  coldkey on a mortal era; WT credit belongs exclusively to Central and is confirmed only by a
  verified signed ``GammaFundingReceiptV1`` — a chain inclusion, SDK success or unrelated
  balance increase never proves credit.
- NO SIGNING ON READS: ``check``/``status`` never touch a wallet object at all (it is not even
  constructed on those paths). They use only the mining hotkey's read-domain signatures (the
  same class as the deployed economics read) plus read-only chain RPC. The G4 own-account
  quote/status routes additionally require a coldkey *binding* proof, so the client reaches
  those routes only from ``prepare`` (an executable-quote action); this module deliberately
  does not call the two-proof ``status`` route from ``status`` — the hard rule wins over the
  feature wish, and the server does not expose the settlement receipt there at G6 anyway.
- RAW COMPOSITION, NO SDK SUBMISSION: bittensor 11.0.0 has no gamma surface (G0 §d), so the
  spend call, the signed v4 extrinsic envelope and every storage key are composed/decoded by
  hand (SCALE primitives mirrored from :mod:`subnet.miner.prepared_extrinsic`, whose
  ``validate_unsigned_dict`` re-derives every extension byte before the payload is signed).
  Broadcast is one single-shot ``author_submitExtrinsic`` — never an SDK submit that could
  re-sign or auto-resubmit. The era is always mortal and its whole possible inclusion range
  must fit inside the quote validity.
- ONE-SHOT RECOVERY (§11.3): journal records QUOTE_VERIFIED -> PREPARED -> SUBMISSION_ATTEMPTED
  (fsynced BEFORE the network call) -> OBSERVED_ON_CHAIN | OUTCOME_UNKNOWN -> CENTRAL_CREDITED.
  A rerun after the attempt is observe-only: never re-sign, never a replacement intent, never
  an automatic re-spend. Absence from the mempool or a failed lookup is never proof of
  non-payment; only complete *finalized* coverage of the expired mortal era is. No wallet
  secrets, seeds or passwords ever enter a journal record, log line or error.
- ``spend_gamma`` protocol facts pinned from the reviewed runtime-490 source (G3
  runtime-source-review): the signing account MUST equal ``GammaDeposits(netuid, hotkey).spender``
  (not merely the owner), the spend tuple is (deposit hotkey, amount, 32-byte reference) and is
  a PURE BURN, ``GammaSpent``/``GammaSpendSkipped`` are pallet 7 events 0x07A5/0x07A6, a fully
  skipped batch is still a successful extrinsic, the chain does NOT deduplicate references, and
  the default mint spender is the SUBNET's service spender — so every mint composed here names
  the OWN coldkey as spender explicitly and never silently changes an existing one.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import time
from typing import Callable

from subnet.bittensor_v11 import FinalizedRpc, validate_subtensor_endpoint
from subnet.miner.funding import FundingRefused, amount_rao
from subnet.miner.funding_client import CHAIN_ID, read_funding
from subnet.miner.trust import SN37_RECEIPT_SIGNING_PUBKEY, config_signing_pubkey
from subnet.miner.prepared_extrinsic import (
    PreparedExtrinsicError,
    validate_era_chain_anchor,
    validate_unsigned_dict,
)
from subnet.shared import gamma_funding_receipt, gamma_intent, gamma_policy
from subnet.shared.canonicalization import canonical_bytes, parse_canonical_json
from subnet.shared.gamma_funding_receipt import deposit_key as gamma_deposit_key
from subnet.shared.signing import verify as verify_sr25519
from subnet.shared.sn37_economics import GENESIS_HASH
from subnet.shared.ss58 import ss58_decode, ss58_encode

NETUID = 37
TRANSACTION_VERSION = 1
# Reviewed gamma runtimes (ADR-0612 §10.1): ``spec -> (metadata_sha256, code_sha256)``, the
# runtime's exact execution identity. The binding is both raw SHA-256 digests, not the
# node-reported ``:code`` storage hash the funding profiles pin. A runtime enters this table
# only after review, with both digests; an unknown runtime refuses. The tests pin every entry
# to a shipped metadata capture under ``subnet/tests/fixtures/runtime<spec>/``.
GAMMA_PROFILES = {
    490: (
        "6c1f9314e97e88d9ec4ffbdc1e2f7335e86c2b41bf27e726bef5bd3fa5305401",
        "8835b086e8ac4257205f4057c894dc54efe8ce9087e8f2e81e2bd1f643eb8aa8",
    ),
}
MAX_U64 = 2**64 - 1
# The runtime debits GammaDeposits(netuid, entry.hotkey); one purchase is exactly one entry.
SPEND_GAMMA_CALL_INDEX = bytes.fromhex("07a4")  # SubtensorModule(7).spend_gamma(164)
MINT_GAMMA_CALL_INDEX = bytes.fromhex("079e")  # SubtensorModule(7).mint_gamma(158)
GAMMA_SPENT_EVENT_INDEX = bytes([7, 0xA5])
GAMMA_SPEND_SKIPPED_EVENT_INDEX = bytes([7, 0xA6])
# G3 runtime-source-review: continuous token bucket, refill cap/7200 blocks saturated at cap.
GAMMA_REFILL_BLOCKS = 7200
MAX_SCAN_BLOCKS = 65_536  # hard bound on any observe scan (mortal eras are <= 65536 blocks)

TERMS_V2_DOMAIN = b"aurelius/miner-funding-terms/v2"
GAMMA_AUTH_DOMAIN = b"aurelius/gamma_funding_auth/v1\n"
GAMMA_BINDING_DOMAIN = b"aurelius/gamma_funding_binding/v1\n"
GAMMA_POLICY_ADMISSION_DOMAIN = b"aurelius/gamma_purchase_policy/v1\n"
QUOTE_ROUTE = "/api/miner/funding/gamma/intents"
AUTH_TTL_SECONDS = 60

_JOURNAL_RECORDS = frozenset(
    {
        "quote",
        "prepared",
        "submission",
        "observed",
        "credited",
        "mint_prepared",
        "mint_submission",
    }
)


class GammaRefused(FundingRefused):
    """Typed gamma funding refusal; never a fallback, never a silent repair."""


def require(condition, reason):
    if not condition:
        raise GammaRefused(reason)


@contextmanager
def _typed_wire_refusals():
    """Surface the G1 wire modules' fail-closed kinds as one typed client refusal."""
    try:
        yield
    except (
        gamma_policy.GammaPolicyRefused,
        gamma_intent.GammaIntentRefused,
        gamma_funding_receipt.GammaFundingReceiptRefused,
    ) as exc:
        raise GammaRefused("gamma wire refusal: " + exc.kind) from exc


# --- pure SCALE primitives (mirrors of the audited prepared_extrinsic helpers) ----------------


def scale_compact_uint(value: int) -> bytes:
    require(type(value) is int and 0 <= value <= MAX_U64, "compact value exceeds u64")
    if value < 1 << 6:
        return bytes((value << 2,))
    if value < 1 << 14:
        return ((value << 2) | 0b01).to_bytes(2, "little")
    if value < 1 << 30:
        return ((value << 2) | 0b10).to_bytes(4, "little")
    encoded_bytes = max(4, (value.bit_length() + 7) // 8)
    return bytes((((encoded_bytes - 4) << 2) | 0b11,)) + value.to_bytes(encoded_bytes, "little")


def scale_compact_decode(raw: bytes, offset: int) -> tuple[int, int]:
    """Decode one canonical SCALE compact at ``raw[offset:]``; refuse non-canonical forms."""
    require(0 <= offset < len(raw), "compact prefix truncated")
    mode = raw[offset] & 0b11
    if mode == 0:
        return raw[offset] >> 2, offset + 1
    if mode == 1:
        require(offset + 2 <= len(raw), "compact prefix truncated")
        value = int.from_bytes(raw[offset : offset + 2], "little") >> 2
        require(value >= 1 << 6, "non-canonical compact encoding")
        return value, offset + 2
    if mode == 2:
        require(offset + 4 <= len(raw), "compact prefix truncated")
        value = int.from_bytes(raw[offset : offset + 4], "little") >> 2
        require(value >= 1 << 14, "non-canonical compact encoding")
        return value, offset + 4
    encoded_bytes = (raw[offset] >> 2) + 4
    require(
        1 <= encoded_bytes <= 8 and offset + 1 + encoded_bytes <= len(raw),
        "compact truncated",
    )
    value = int.from_bytes(raw[offset + 1 : offset + 1 + encoded_bytes], "little")
    require(value >= 1 << 30 and raw[offset + encoded_bytes] != 0, "non-canonical compact")
    return value, offset + 1 + encoded_bytes


def _require_era_period(period):
    require(
        type(period) is int and 4 <= period <= 65_536 and period & (period - 1) == 0,
        "era period must be a power of two in 4..65536",
    )


def mortal_era_bytes(current: int, period: int) -> bytes:
    """The 2-byte MortalEra SCALE encoding for a preparation head (mirrors the audited helper)."""
    _require_era_period(period)
    require(type(current) is int and 0 <= current < 2**32, "invalid era current")
    quantize = max(period >> 12, 1)
    phase = (current % period // quantize) * quantize
    exponent = period.bit_length() - 2
    encoded = min(15, max(1, exponent)) | ((phase // quantize) << 4)
    return encoded.to_bytes(2, "little")


def era_bounds(current: int, period: int) -> tuple[int, int, int]:
    """``(birth, scan_start, scan_end)``: the possible finalized inclusion range of the payload."""
    _require_era_period(period)
    require(type(current) is int and 0 <= current < 2**32, "invalid era current")
    quantize = max(period >> 12, 1)
    phase = (current % period // quantize) * quantize
    birth = current - ((current - phase) % period)
    return birth, current + 1, birth + period - 1


_XXH_PRIMES = (0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x85EBCA77C2B2AE63)
_XXH_P5 = 0x27D4EB2F165667C5
_XXH_MASK = (1 << 64) - 1


def _rotl64(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (64 - shift))) & _XXH_MASK


def _xxh64_round(acc: int, value: int) -> int:
    acc = (acc + value * _XXH_PRIMES[1]) & _XXH_MASK
    acc = _rotl64(acc, 31)
    return (acc * _XXH_PRIMES[0]) & _XXH_MASK


def _xxh64_merge(acc: int, value: int) -> int:
    value = _xxh64_round(0, value)
    return ((acc ^ value) * _XXH_PRIMES[0] + _XXH_PRIMES[3]) & _XXH_MASK


def xxh64(data: bytes, seed: int = 0) -> int:
    """Pure-python XXH64 (the twox hash) per the xxHash spec. Canonical vectors in tests."""
    length = len(data)
    offset = 0
    if length >= 32:
        v1 = (seed + _XXH_PRIMES[0] + _XXH_PRIMES[1]) & _XXH_MASK
        v2 = (seed + _XXH_PRIMES[1]) & _XXH_MASK
        v3 = seed & _XXH_MASK
        v4 = (seed - _XXH_PRIMES[0]) & _XXH_MASK
        while offset + 32 <= length:
            v1 = _xxh64_round(v1, int.from_bytes(data[offset : offset + 8], "little"))
            v2 = _xxh64_round(v2, int.from_bytes(data[offset + 8 : offset + 16], "little"))
            v3 = _xxh64_round(v3, int.from_bytes(data[offset + 16 : offset + 24], "little"))
            v4 = _xxh64_round(v4, int.from_bytes(data[offset + 24 : offset + 32], "little"))
            offset += 32
        acc = (_rotl64(v1, 1) + _rotl64(v2, 7) + _rotl64(v3, 12) + _rotl64(v4, 18)) & _XXH_MASK
        for value in (v1, v2, v3, v4):
            acc = _xxh64_merge(acc, value)
    else:
        acc = (seed + _XXH_P5) & _XXH_MASK
    acc = (acc + length) & _XXH_MASK
    while offset + 8 <= length:
        acc ^= _xxh64_round(0, int.from_bytes(data[offset : offset + 8], "little"))
        acc = (_rotl64(acc, 27) * _XXH_PRIMES[0] + _XXH_PRIMES[3]) & _XXH_MASK
        offset += 8
    if offset + 4 <= length:
        lane = int.from_bytes(data[offset : offset + 4], "little")
        acc ^= (lane * _XXH_PRIMES[0]) & _XXH_MASK
        acc = (_rotl64(acc, 23) * _XXH_PRIMES[1] + _XXH_PRIMES[2]) & _XXH_MASK
        offset += 4
    while offset < length:
        acc ^= (data[offset] * _XXH_P5) & _XXH_MASK
        acc = (_rotl64(acc, 11) * _XXH_PRIMES[0]) & _XXH_MASK
        offset += 1
    acc ^= acc >> 33
    acc = (acc * _XXH_PRIMES[1]) & _XXH_MASK
    acc ^= acc >> 29
    acc = (acc * _XXH_PRIMES[2]) & _XXH_MASK
    acc ^= acc >> 32
    return acc


def twox128(name: str) -> bytes:
    encoded = name.encode()
    return xxh64(encoded).to_bytes(8, "little") + xxh64(encoded, seed=1).to_bytes(8, "little")


def blake2_128_concat(key: bytes) -> bytes:
    return hashlib.blake2b(key, digest_size=16).digest() + key


def _plain_storage_key(pallet: str, item: str) -> str:
    return "0x" + (twox128(pallet) + twox128(item)).hex()


def _u16_identity_storage_key(pallet: str, item: str, netuid: int) -> str:
    return "0x" + (twox128(pallet) + twox128(item) + struct.pack("<H", netuid)).hex()


def _u16_blake_double_storage_key(pallet: str, item: str, netuid: int, account: str) -> str:
    raw = twox128(pallet) + twox128(item) + struct.pack("<H", netuid)
    return "0x" + (raw + blake2_128_concat(ss58_decode(account))).hex()


def _hex_value(value, *, label: str) -> bytes:
    require(
        type(value) is str and value.startswith("0x") and len(value) % 2 == 0,
        f"{label} storage value is not hex",
    )
    try:
        return bytes.fromhex(value[2:])
    except ValueError as exc:
        raise GammaRefused(f"{label} storage value is not hex") from exc


# --- spend/mint call composition + strict decode-back -----------------------------------------


def spend_gamma_call(netuid: int, deposit_hotkey: str, amount_atoms: int, reference_hex: str):
    """One exact single-entry ``spend_gamma`` call: 07a4, netuid u16, vec[(hotkey,u64,ref32)]."""
    require(type(netuid) is int and 1 <= netuid <= 65_535, "invalid netuid")
    require(type(amount_atoms) is int and 0 < amount_atoms <= MAX_U64, "invalid gamma amount")
    require(
        type(reference_hex) is str and re.fullmatch(r"0x[0-9a-f]{64}", reference_hex) is not None,
        "reference must be 0x + 32 bytes",
    )
    return (
        SPEND_GAMMA_CALL_INDEX
        + struct.pack("<H", netuid)
        + scale_compact_uint(1)
        + ss58_decode(deposit_hotkey)
        + struct.pack("<Q", amount_atoms)
        + bytes.fromhex(reference_hex[2:])
    )


def mint_gamma_call(
    netuid: int,
    deposit_hotkey: str,
    tao_rao: int,
    min_gamma_out_atoms: int,
    *,
    spender: str,
    daily_spend_cap_atoms: int | None,
):
    """One exact ``mint_gamma`` call with an EXPLICIT own-coldkey spender (never the default)."""
    require(type(netuid) is int and 1 <= netuid <= 65_535, "invalid netuid")
    require(type(tao_rao) is int and 0 < tao_rao <= MAX_U64, "invalid tao amount")
    require(
        type(min_gamma_out_atoms) is int and 0 <= min_gamma_out_atoms <= MAX_U64,
        "invalid minimum gamma out",
    )
    if daily_spend_cap_atoms is not None:
        require(
            type(daily_spend_cap_atoms) is int and 0 < daily_spend_cap_atoms <= MAX_U64,
            "invalid daily spend cap",
        )
    encoded_cap = (
        b"\x00"
        if daily_spend_cap_atoms is None
        else b"\x01" + struct.pack("<Q", daily_spend_cap_atoms)
    )
    return (
        MINT_GAMMA_CALL_INDEX
        + struct.pack("<H", netuid)
        + ss58_decode(deposit_hotkey)
        + struct.pack("<Q", tao_rao)
        + struct.pack("<Q", min_gamma_out_atoms)
        + b"\x01"
        + ss58_decode(spender)
        + encoded_cap
    )


def decode_spend_gamma_call(call: bytes, expected: dict) -> dict:
    """Decode composed spend bytes back and require equality with the expected authority.

    Never trust the composing objects alone: the exact call index, netuid, single-entry bound,
    deposit hotkey, amount and 32-byte reference must re-appear in the bytes.
    """
    require(type(call) is bytes and len(call) == 77, "spend call length differs")
    require(call[:2] == SPEND_GAMMA_CALL_INDEX, "spend call index differs")
    netuid = struct.unpack("<H", call[2:4])[0]
    entries, offset = scale_compact_decode(call, 4)
    require(entries == 1 and offset == 5, "spend call must be exactly one entry")
    hotkey = call[offset : offset + 32]
    amount = struct.unpack("<Q", call[offset + 32 : offset + 40])[0]
    reference = call[offset + 40 : offset + 72]
    require(offset + 72 == len(call), "spend call has trailing bytes")
    require(
        (netuid, hotkey, amount, reference)
        == (
            expected["netuid"],
            ss58_decode(expected["deposit_hotkey"]),
            expected["amount_atoms"],
            bytes.fromhex(expected["reference_hex"][2:]),
        ),
        "decoded spend call differs from the expected authority",
    )
    return {
        "netuid": netuid,
        "deposit_hotkey": ss58_encode(hotkey),
        "amount_atoms": amount,
        "reference_hex": "0x" + reference.hex(),
    }


def compose_signed_extrinsic(
    *, call: bytes, extra: bytes, signer_public_key: bytes, signature: bytes
) -> bytes:
    """Substrate v4 signed extrinsic: compact(len) || 0x84 || MultiAddress::Id || sig || extra || call.

    The envelope layout is pinned against real runtime-490 ``spend_gamma`` extrinsics
    (bittensor mainnet blocks 9478505 and 9478550) and re-proved by the tests in this
    repository, which decode the composed bytes with scalecodec against the pinned metadata.
    """
    require(len(signature) == 64 and len(signer_public_key) == 32, "invalid signature/signer")
    body = b"\x84\x00" + signer_public_key + b"\x01" + signature + extra + call
    return scale_compact_uint(len(body)) + body


def decode_signed_extrinsic(raw: bytes, *, expected_call: bytes) -> dict:
    """Strict decode-back of the composed extrinsic; any drift from the expectation refuses."""
    body_length, offset = scale_compact_decode(raw, 0)
    require(offset + body_length == len(raw), "extrinsic compact length differs from its bytes")
    body = raw[offset:]
    require(len(body) >= 2 + 32 + 1 + 64 + 2 + 1 + len(expected_call), "extrinsic truncated")
    require(body[0] == 0x84, "extrinsic is not signed version 4")
    require(body[1] == 0x00, "signer address is not MultiAddress::Id")
    cursor = 2
    signer = body[cursor : cursor + 32]
    cursor += 32
    require(body[cursor] == 0x01, "signature is not sr25519")
    cursor += 1
    signature = body[cursor : cursor + 64]
    require(len(signature) == 64, "signature truncated")
    cursor += 64
    era = body[cursor : cursor + 2]
    cursor += 2
    nonce, cursor = scale_compact_decode(body, cursor)
    tip, cursor = scale_compact_decode(body, cursor)
    require(body[cursor] == 0x00, "metadata-hash mode is not Disabled")
    cursor += 1
    require(cursor == len(body) - len(expected_call), "extrinsic extension length differs")
    require(body[cursor:] == expected_call, "decoded extrinsic call differs")
    return {
        "signer_public_key": signer,
        "signature": signature,
        "era": era.hex(),
        "nonce": nonce,
        "tip": tip,
        "call": body[cursor:],
    }


def extrinsic_tx_hash(raw: bytes) -> str:
    """Substrate tx hash of a v4 signed extrinsic: blake2b-256 over the exact bytes."""
    return "0x" + hashlib.blake2b(raw, digest_size=32).hexdigest()


def _signed_extensions() -> list[str]:
    from subnet.miner.prepared_extrinsic import BITTENSOR_V11_SIGNED_EXTENSIONS

    return list(BITTENSOR_V11_SIGNED_EXTENSIONS)


def unsigned_extrinsic_dict(
    *,
    call: bytes,
    address: str,
    era_current: int,
    era_period: int,
    nonce: int,
    genesis_hash: str,
    era_block_hash: str,
    spec_version: int,
    transaction_version: int,
) -> dict:
    """Build the SDK11 ``UnsignedExtrinsic.to_dict()`` shape by hand, then validate it.

    ``validate_unsigned_dict`` (the audited validator) re-derives every extension
    byte, the mortal era bounds and the signing payload from these fields — my construction is
    authority only after it survives that check. The result is exactly the 17 audited fields.
    """
    public_key = ss58_decode(address)
    era = mortal_era_bytes(era_current, era_period)
    included_extrinsic = era + scale_compact_uint(nonce) + scale_compact_uint(0) + b"\x00"
    included_signed = (
        struct.pack("<II", spec_version, transaction_version)
        + bytes.fromhex(genesis_hash[2:])
        + bytes.fromhex(era_block_hash[2:])
        + b"\x00"
    )
    full_payload = call + included_extrinsic + included_signed
    payload = (
        hashlib.blake2b(full_payload, digest_size=32).digest()
        if len(full_payload) > 256
        else full_payload
    )
    return {
        "call_data": "0x" + call.hex(),
        "address": address,
        "public_key": "0x" + public_key.hex(),
        "crypto_type": 1,
        "era": {"period": era_period, "current": era_current},
        "nonce": nonce,
        "tip": 0,
        "tip_asset_id": None,
        "genesis_hash": genesis_hash,
        "era_block_hash": era_block_hash,
        "spec_version": spec_version,
        "transaction_version": transaction_version,
        "metadata_hash": None,
        "payload": "0x" + payload.hex(),
        "included_in_extrinsic": "0x" + included_extrinsic.hex(),
        "included_in_signed_data": "0x" + included_signed.hex(),
        "payload_json": {
            "address": address,
            "blockHash": era_block_hash,
            "genesisHash": genesis_hash,
            "method": "0x" + call.hex(),
            "nonce": "0x" + nonce.to_bytes(4, "big").hex(),
            "specVersion": "0x" + spec_version.to_bytes(4, "big").hex(),
            "tip": "0x" + "00" * 16,
            "transactionVersion": "0x" + transaction_version.to_bytes(4, "big").hex(),
            "era": "0x" + era.hex(),
            "version": 4,
            "signedExtensions": _signed_extensions(),
            "mode": 0,
            "metadataHash": None,
        },
    }


def _validated_authority(unsigned: dict, *, expected_call: bytes, expected_period: int):
    """Run the audited validator and bind it to the exact composed call bytes.

    ``PreparedExtrinsicError`` is not a funding refusal; on the client it becomes one so the
    CLI surfaces a typed reason instead of a traceback.
    """
    try:
        return validate_unsigned_dict(
            unsigned,
            expected_address=unsigned["address"],
            expected_genesis_hash=GENESIS_HASH,
            expected_spec_version=unsigned["spec_version"],
            expected_transaction_version=TRANSACTION_VERSION,
            expected_period=expected_period,
            expected_tip=0,
            expected_call_data_sha256=hashlib.sha256(expected_call).hexdigest(),
        )
    except PreparedExtrinsicError as exc:
        raise GammaRefused("prepared extrinsic authority refused: " + str(exc)) from exc


def _validated_era_anchor(authority, era_block_hash: str) -> None:
    try:
        validate_era_chain_anchor(
            authority,
            current_finalized_block_number=authority.era_current,
            birth_block_hash=era_block_hash,
        )
    except PreparedExtrinsicError as exc:
        raise GammaRefused("prepared era anchor refused: " + str(exc)) from exc


# --- read-only chain facade --------------------------------------------------------------------


@dataclass(frozen=True)
class GammaDeposit:
    owner_coldkey: str
    spender: str
    balance_atoms: int
    cap_atoms: int | None
    bucket_available: int | None
    bucket_block: int | None

    def allowance_at(self, block: int) -> int | None:
        """Runtime token-bucket allowance (G3 review §8); None means uncapped."""
        if self.cap_atoms is None:
            return None
        available, bucket_block = self.bucket_available, self.bucket_block
        require(
            type(available) is int and type(bucket_block) is int and block >= bucket_block,
            "invalid cap bucket state",
        )
        grown = self.cap_atoms * (block - bucket_block) // GAMMA_REFILL_BLOCKS
        return min(available + grown, self.cap_atoms)


class GammaChain:
    """Wallet-free read-only chain access over the finalized RPC transport."""

    def __init__(self, rpc):
        self.rpc = rpc

    def snapshot(self):
        return self.rpc.snapshot()

    def runtime_identity(self, snapshot) -> dict:
        """Exact execution identity at the finalized head, against the compiled gamma profiles."""
        require(
            snapshot.genesis_hash == GENESIS_HASH
            and snapshot.chain == "Bittensor"
            and snapshot.transaction_version == TRANSACTION_VERSION,
            "unreviewed gamma chain identity",
        )
        profile = GAMMA_PROFILES.get(snapshot.runtime_spec_version)
        require(profile is not None, "unreviewed gamma runtime")
        metadata = self.rpc.call("state_getMetadata", [snapshot.head_hash])
        require(type(metadata) is str and metadata.startswith("0x"), "metadata read failed")
        metadata_digest = hashlib.sha256(bytes.fromhex(metadata[2:])).hexdigest()
        require(metadata_digest == profile[0], "gamma runtime metadata differs")
        raw_code = self.rpc.call("state_getStorage", ["0x3a636f6465", snapshot.head_hash])
        require(type(raw_code) is str and raw_code.startswith("0x"), "runtime code read failed")
        code_digest = hashlib.sha256(bytes.fromhex(raw_code[2:])).hexdigest()
        require(code_digest == profile[1], "gamma runtime code differs")
        return {
            "spec_version": snapshot.runtime_spec_version,
            "transaction_version": snapshot.transaction_version,
            "metadata_sha256": metadata_digest,
            "code_sha256": code_digest,
        }

    def gamma_enabled(self, at: str) -> bool:
        value = self.rpc.call(
            "state_getStorage", [_plain_storage_key("SubtensorModule", "GammaEnabled"), at]
        )
        return self._decode_bool(value, label="GammaEnabled")

    def subnet_gamma_enabled(self, at: str, netuid: int) -> bool:
        value = self.rpc.call(
            "state_getStorage",
            [
                _u16_identity_storage_key("SubtensorModule", "SubnetGammaEnabled", netuid),
                at,
            ],
        )
        return self._decode_bool(value, label="SubnetGammaEnabled")

    def gamma_deposit_key(self, netuid: int, gamma_api_hotkey: str) -> str:
        return _u16_blake_double_storage_key(
            "SubtensorModule", "GammaDeposits", netuid, gamma_api_hotkey
        )

    def gamma_deposit(self, at: str, netuid: int, gamma_api_hotkey: str) -> GammaDeposit | None:
        value = self.rpc.call(
            "state_getStorage", [self.gamma_deposit_key(netuid, gamma_api_hotkey), at]
        )
        return self.decode_gamma_deposit(value)

    def coldkey_gamma_hotkeys(self, at: str, netuid: int, coldkey: str) -> list[str]:
        key = _u16_blake_double_storage_key(
            "SubtensorModule", "ColdkeyGammaHotkeys", netuid, coldkey
        )
        value = self.rpc.call("state_getStorage", [key, at])
        if value is None:
            return []
        raw = _hex_value(value, label="ColdkeyGammaHotkeys")
        count, offset = scale_compact_decode(raw, 0)
        require(count <= 4096, "unreasonable gamma hotkey list")
        require(offset + 32 * count == len(raw), "gamma hotkey list length differs")
        return [ss58_encode(raw[offset + 32 * i : offset + 32 * (i + 1)]) for i in range(count)]

    def subnet_gamma_price_bits(self, at: str, netuid: int) -> int | None:
        """The chain's mark price for the subnet as raw U64F64 bits (a FACT, not a quote)."""
        value = self.rpc.call(
            "state_getStorage",
            [_u16_identity_storage_key("SubtensorModule", "SubnetGammaPrice", netuid), at],
        )
        if value is None:
            return None
        raw = _hex_value(value, label="SubnetGammaPrice")
        require(len(raw) == 16, "unexpected gamma price width")
        return int.from_bytes(raw, "little")

    def tao_usd_price(self, at: str) -> dict | None:
        """The oracle feed fact: minting fails when the feed is >300 blocks stale (G3 §6)."""
        value = self.rpc.call(
            "state_getStorage", [_plain_storage_key("SubtensorModule", "TaoUsdPrice"), at]
        )
        if value is None:
            return None
        raw = _hex_value(value, label="TaoUsdPrice")
        require(len(raw) == 24, "unexpected TaoUsdPrice layout")
        accepted, feed, fed_at = struct.unpack("<QQQ", raw)
        return {"accepted": accepted, "feed": feed, "fed_at": fed_at}

    def system_account(self, at: str, coldkey: str) -> dict:
        """``{nonce, free_rao}`` from ``System.Account`` (strict 56-byte AccountInfo layout)."""
        key = (
            "0x"
            + (
                twox128("System") + twox128("Account") + blake2_128_concat(ss58_decode(coldkey))
            ).hex()
        )
        value = self.rpc.call("state_getStorage", [key, at])
        require(value is not None, "system account not found")
        raw = _hex_value(value, label="System.Account")
        require(len(raw) == 56, "unreviewed system account layout")
        nonce = struct.unpack("<I", raw[0:4])[0]
        free = struct.unpack("<Q", raw[16:24])[0]
        return {"nonce": nonce, "free_rao": free}

    def account_next_index(self, coldkey: str) -> int:
        value = self.rpc.call("system_accountNextIndex", [coldkey])
        require(type(value) is int and value >= 0, "invalid account next index")
        return value

    def fee_estimate_rao(self, extrinsic_hex: str) -> int:
        info = self.rpc.call("payment_queryInfo", [extrinsic_hex])
        require(type(info) is dict and type(info.get("partialFee")) is int, "invalid fee info")
        require(info["partialFee"] >= 0, "negative fee estimate")
        return info["partialFee"]

    def block_hash(self, number: int) -> str:
        value = self.rpc.call("chain_getBlockHash", [number])
        require(type(value) is str and value.startswith("0x"), "invalid block hash read")
        return value

    def scan_for_extrinsic(self, extrinsic_hex: str, start: int, end: int) -> dict | None:
        """Read-only finalized scan for the EXACT composed bytes in ``[start, end]``."""
        require(end >= start and end - start + 1 <= MAX_SCAN_BLOCKS, "scan range out of bounds")
        target = extrinsic_hex.lower()
        for height in range(start, end + 1):
            block_hash = self.block_hash(height)
            block = self.rpc.call("chain_getBlock", [block_hash])
            require(type(block) is dict and type(block.get("block")) is dict, "malformed block")
            extrinsics = block["block"].get("extrinsics")
            require(type(extrinsics) is list, "malformed extrinsic list")
            for index, raw in enumerate(extrinsics):
                require(type(raw) is str and raw.startswith("0x"), "malformed raw extrinsic")
                if raw.lower() == target:
                    return {
                        "block_number": height,
                        "block_hash": block_hash.lower(),
                        "extrinsic_index": index,
                    }
        return None

    def block_events_raw(self, block_hash: str) -> bytes | None:
        value = self.rpc.call(
            "state_getStorage", [_plain_storage_key("System", "Events"), block_hash]
        )
        if value is None:
            return None
        return _hex_value(value, label="System.Events")

    @staticmethod
    def _decode_bool(value, *, label: str) -> bool:
        if value is None:
            return False  # both gamma toggles default to false when the slot is absent
        raw = _hex_value(value, label=label)
        if raw == b"\x00":
            return False
        if raw == b"\x01":
            return True
        raise GammaRefused(f"{label} value differs from a plain bool")

    @staticmethod
    def decode_gamma_deposit(value) -> GammaDeposit | None:
        """Strict ``{coldkey, spender, balance, Option<{cap,(available,block)}>}`` decode.

        The storage modifier is ``Optional`` (OptionQuery): an absent slot is a null RPC result
        and the stored bytes are the BARE struct — there is no encoded Option prefix. Proven
        against the pinned runtime-490 metadata by the scalecodec layout test.
        """
        if value is None:
            return None
        raw = _hex_value(value, label="GammaDeposits")
        require(len(raw) in (73, 97), "unreviewed GammaDeposits layout")
        owner, spender = raw[0:32], raw[32:64]
        balance = struct.unpack("<Q", raw[64:72])[0]
        cap_atoms = bucket_available = bucket_block = None
        if len(raw) == 97:
            require(raw[72] == 0x01, "unexpected daily cap option prefix")
            cap_atoms = struct.unpack("<Q", raw[73:81])[0]
            bucket_available = struct.unpack("<Q", raw[81:89])[0]
            bucket_block = struct.unpack("<Q", raw[89:97])[0]
        else:
            require(raw[72] == 0x00, "unexpected daily cap option prefix")
        return GammaDeposit(
            owner_coldkey=ss58_encode(owner),
            spender=ss58_encode(spender),
            balance_atoms=balance,
            cap_atoms=cap_atoms,
            bucket_available=bucket_available,
            bucket_block=bucket_block,
        )


# --- journal -----------------------------------------------------------------------------------


class GammaFundingJournal:
    """Immutable fsynced gamma records; one lock covers quote, prepare, submit and observe."""

    RECORDS = _JOURNAL_RECORDS

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.directory.lstat()
        require(
            stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid(),
            "gamma funding journal must be an owned real directory",
        )
        require(
            info.st_mode & 0o022 == 0, "gamma funding journal must not be writable by other users"
        )

    @contextmanager
    def lock(self):
        fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def read(self, name):
        require(name in self.RECORDS, "invalid gamma journal record")
        path = self.directory / (name + ".json")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise GammaRefused("unsafe gamma funding journal record") from exc
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and info.st_uid == os.geteuid()
                and stat.S_IMODE(info.st_mode) == 0o600
                and info.st_size <= 1_048_576,
                "unsafe gamma funding journal record",
            )
            return parse_canonical_json(stream.read())

    def claim(self, name, document):
        """Write once, fsync, then immutable: a different body refuses; the same replays."""
        require(name in self.RECORDS, "invalid gamma journal record")
        existing = self.read(name)
        if existing is not None:
            require(
                canonical_bytes(existing) == canonical_bytes(document),
                "gamma funding record changed; retained state requires observation",
            )
            return False
        path = self.directory / (name + ".json")
        encoded = json.dumps(
            document, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        require(len(encoded) <= 1_048_576, "oversized gamma funding record")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True


def _canonical_sha256(document) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


# --- server reads ------------------------------------------------------------------------------


def _ed25519_verify(payload: bytes, signature_hex, pubkey_hex: str) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex)).verify(
            bytes.fromhex(signature_hex), payload
        )
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def verify_policy_envelope(envelope, *, config_pubkey: str) -> gamma_policy.GammaPurchasePolicy:
    """Verify the signed policy discovery envelope against the shipped config trust root."""
    require(
        type(envelope) is dict and set(envelope) == {"body", "signature_hex", "policy_digest"},
        "invalid gamma policy discovery envelope",
    )
    with _typed_wire_refusals():
        policy = gamma_policy.from_map(envelope["body"])
    digest = policy.digest()
    require(digest == envelope["policy_digest"], "gamma policy digest differs")
    preimage = (
        GAMMA_POLICY_ADMISSION_DOMAIN + digest[2:].encode() + b"|" + policy.policy_version.encode()
    )
    require(
        _ed25519_verify(preimage, envelope["signature_hex"], config_pubkey),
        "gamma policy signature is not the shipped config authority",
    )
    require(
        policy.chain_id == CHAIN_ID
        and policy.genesis_hash == GENESIS_HASH
        and policy.netuid == NETUID,
        "gamma policy targets another chain or subnet",
    )
    return policy


def read_gamma_terms(
    base_url: str,
    *,
    hotkey: str,
    funding_terms_signer: Callable[[bytes], bytes],
    economics_signer: Callable[[bytes], bytes],
    config_pubkey: str,
    fetch,
    now: int | None = None,
) -> dict:
    """Terms read with the signed ``terms_version: 2`` opt-in; old servers fall back cleanly.

    The legacy part of the response is validated by the DEPLOYED strict parser
    (:func:`subnet.miner.funding_client.read_funding`) over the exact same bytes (with the
    additive ``funding_methods`` block stripped for that check); the gamma block is validated
    here and its policy envelope verified against the shipped config key. A server that does
    not speak v2 (401 on the extended body) is answered with the plain v1 read — new miners
    keep working against old servers exactly like the deployed client.
    """
    from subnet.miner.read_client import ReadUnavailable

    now = int(time.time()) if now is None else now
    snapshot = fetch(base_url.rstrip("/") + "/api/config")
    body = {
        "hotkey": hotkey,
        "netuid": NETUID,
        "config_hash": snapshot["canonical_hash"],
        "timestamp": now,
        "nonce": secrets.token_hex(32),
        "terms_version": 2,
    }
    signature = funding_terms_signer(TERMS_V2_DOMAIN + b"\x00" + canonical_bytes(body))
    require(type(signature) is bytes and len(signature) == 64, "invalid hotkey signature")
    try:
        raw = fetch(
            base_url.rstrip("/") + "/api/miner/funding",
            body={**body, "signature_hex": signature.hex()},
        )
    except ReadUnavailable as exc:
        if exc.status in (400, 401, 404, 422):
            legacy = read_funding(
                base_url,
                hotkey=hotkey,
                signer=economics_signer,
                config_pubkey=config_pubkey,
                fetch=fetch,
                now=now,
            )
            return {"legacy": legacy, "funding_methods": None, "terms_version": 1}
        raise
    require(type(raw) is dict, "invalid funding terms response")
    funding_methods = raw.get("funding_methods")
    stripped = {key: value for key, value in raw.items() if key != "funding_methods"}

    def validating_fetch(url, body=None):
        return snapshot if body is None else stripped

    legacy = read_funding(
        base_url,
        hotkey=hotkey,
        signer=economics_signer,
        config_pubkey=config_pubkey,
        fetch=validating_fetch,
        now=now,
    )
    return {
        "legacy": legacy,
        "funding_methods": _validate_funding_methods(funding_methods, config_pubkey),
        "terms_version": 2,
    }


def _validate_funding_methods(block, config_pubkey: str) -> dict | None:
    if block is None:
        return None
    require(
        type(block) is dict
        and set(block) == {"terms_version", "legacy", "gamma"}
        and block["terms_version"] == 2
        and type(block["legacy"]) is dict
        and set(block["legacy"]) == {"available"}
        and type(block["legacy"]["available"]) is bool
        and type(block["gamma"]) is dict
        and {"available", "reason"} <= set(block["gamma"]) <= {"available", "reason", "policy"}
        and type(block["gamma"]["available"]) is bool
        and type(block["gamma"]["reason"]) is str,
        "invalid funding_methods block",
    )
    gamma = dict(block["gamma"])
    if "policy" in gamma:
        policy = verify_policy_envelope(gamma["policy"], config_pubkey=config_pubkey)
        gamma["policy"] = {
            "envelope": gamma["policy"],
            "policy": policy.body,
            "packs": policy.packs(),
            "policy_digest": policy.digest(),
        }
    return {"terms_version": 2, "legacy": block["legacy"], "gamma": gamma}


def build_quote_auth(
    *,
    beneficiary_coldkey: str,
    beneficiary_hotkey: str,
    funding_coldkey: str,
    gamma_api_hotkey: str,
    pack_id: str,
    config_hash: str,
    client_request_id: str,
    gamma_auth_signer: Callable[[bytes], bytes],
    coldkey_signer: Callable[[bytes], bytes],
    now: int | None = None,
    valid_until_block: int | None = None,
) -> dict:
    """The exact G4 two-proof request authorization (quote purpose only)."""
    now = int(time.time()) if now is None else now
    binding = {
        "beneficiary_coldkey": beneficiary_coldkey,
        "beneficiary_hotkey": beneficiary_hotkey,
        "funding_coldkey": funding_coldkey,
        "gamma_api_hotkey": gamma_api_hotkey,
        "authorized_spender": funding_coldkey,
        "netuid": NETUID,
        "chain_id": CHAIN_ID,
        "genesis_hash": GENESIS_HASH,
    }
    if valid_until_block is not None:
        require(type(valid_until_block) is int and 0 <= valid_until_block < 2**32, "bad window")
        binding["valid_until_block"] = valid_until_block
    body = {
        "method": "quote",
        "route": QUOTE_ROUTE,
        "chain_id": CHAIN_ID,
        "genesis_hash": GENESIS_HASH,
        "netuid": NETUID,
        "beneficiary_coldkey": beneficiary_coldkey,
        "beneficiary_hotkey": beneficiary_hotkey,
        "gamma_api_hotkey": gamma_api_hotkey,
        "pack_id": pack_id,
        "config_hash": config_hash,
        "client_request_id": client_request_id,
        "nonce": secrets.token_hex(32),
        "issued_at": now,
        "expires_at": now + AUTH_TTL_SECONDS,
        "binding": binding,
    }
    auth_signature = gamma_auth_signer(GAMMA_AUTH_DOMAIN + b"\x00" + canonical_bytes(body))
    binding_signature = coldkey_signer(GAMMA_BINDING_DOMAIN + canonical_bytes(binding))
    require(
        type(auth_signature) is bytes and len(auth_signature) == 64,
        "invalid hotkey auth signature",
    )
    require(
        type(binding_signature) is bytes and len(binding_signature) == 64,
        "invalid coldkey binding signature",
    )
    return {
        **body,
        "auth_signature_hex": auth_signature.hex(),
        "binding_signature_hex": binding_signature.hex(),
    }


def _quote_request_body(auth: dict) -> dict:
    return {
        "beneficiary_coldkey": auth["beneficiary_coldkey"],
        "beneficiary_hotkey": auth["beneficiary_hotkey"],
        "pack_id": auth["pack_id"],
        "client_request_id": auth["client_request_id"],
    }


def request_gamma_quote(base_url: str, *, auth: dict, fetch) -> dict:
    """POST the two-proof quote request; typed server refusals stay typed on the client."""
    from subnet.miner.read_client import ReadUnavailable

    try:
        result = fetch(
            base_url.rstrip("/") + QUOTE_ROUTE,
            body={"auth": auth, "request": _quote_request_body(auth)},
        )
    except ReadUnavailable as exc:
        detail = getattr(exc, "detail", None)
        reason = detail.get("error") if type(detail) is dict else None
        raise GammaRefused(
            "server refused the quote: " + (str(reason) if reason else f"http_{exc.status}")
        ) from exc
    require(
        type(result) is dict
        and set(result)
        == {
            "schema_version",
            "intent_id",
            "reference",
            "valid_from_block",
            "valid_until_block",
            "envelope",
        },
        "invalid quote response",
    )
    return result


def verify_quote(
    quote_response: dict,
    *,
    policy: gamma_policy.GammaPurchasePolicy,
    beneficiary_coldkey: str,
    beneficiary_hotkey: str,
    funding_coldkey: str,
    gamma_api_hotkey: str,
    config_hash: str,
    pack_id: str,
    quote_pubkey: str = SN37_RECEIPT_SIGNING_PUBKEY,
) -> gamma_intent.GammaFundingIntent:
    """Verify every quote fact BEFORE it becomes signing authority (plan §11.2)."""
    require(quote_response["schema_version"] == 1, "unsupported quote schema")
    envelope = quote_response["envelope"]
    require(
        type(envelope) is dict and set(envelope) == {"body", "signature_hex", "key_id"},
        "invalid quote envelope",
    )
    with _typed_wire_refusals():
        intent = gamma_intent.from_map(envelope["body"])
        pack = policy.pack(pack_id)
    require(
        _ed25519_verify(intent.signature_payload(), envelope["signature_hex"], quote_pubkey),
        "quote signature is not the trusted authority",
    )
    require(envelope["key_id"] == policy.signing_key_id, "quote key differs from the policy")
    require(intent.intent_id == quote_response["intent_id"], "quote intent id differs")
    require(intent.reference_hex == quote_response["reference"], "quote reference differs")
    require(
        intent.body["valid_from_block"] == quote_response["valid_from_block"]
        and intent.body["valid_until_block"] == quote_response["valid_until_block"],
        "quote validity differs",
    )
    require(intent.body["pack_id"] == pack_id, "quote pack differs from the request")
    require(
        int(intent.body["gamma_atoms"]) == pack["gamma_atoms"]
        and int(intent.body["work_tokens"]) == pack["work_tokens"],
        "quote amounts differ from the policy pack",
    )
    require(
        intent.body["purchase_policy_version"] == policy.policy_version
        and intent.body["purchase_policy_digest"] == policy.digest(),
        "quote policy identity differs from the discovered policy",
    )
    require(
        intent.body["chain_id"] == CHAIN_ID
        and intent.body["genesis_hash"] == GENESIS_HASH
        and intent.body["netuid"] == NETUID,
        "quote targets another chain or subnet",
    )
    require(intent.body["config_hash"] == "0x" + config_hash, "quote config hash differs")
    require(
        intent.body["beneficiary_coldkey"] == beneficiary_coldkey == funding_coldkey
        and intent.body["beneficiary_hotkey"] == beneficiary_hotkey
        and intent.body["funding_coldkey"] == funding_coldkey
        and intent.body["gamma_api_hotkey"] == gamma_api_hotkey
        and intent.body["authorized_spender"] == funding_coldkey
        and intent.body["payer_mode"] == gamma_policy.PAYER_MODE,
        "quote beneficiary/binding differs from this wallet",
    )
    require(
        gamma_api_hotkey not in {beneficiary_hotkey, beneficiary_coldkey, funding_coldkey},
        "gamma api hotkey is not a distinct key",
    )
    return intent


def verify_gamma_receipt(
    envelope,
    *,
    intent: dict,
    quote_pubkey: str = SN37_RECEIPT_SIGNING_PUBKEY,
) -> dict:
    """Full-lineage verification of a signed GammaFundingReceiptV1 (the only credit proof)."""
    require(
        type(envelope) is dict and set(envelope) == {"body", "signature_hex", "key_id"},
        "invalid receipt envelope",
    )
    with _typed_wire_refusals():
        receipt = gamma_funding_receipt.from_map(envelope["body"])
        payload = receipt.signature_payload()
    require(
        gamma_funding_receipt.verify_signature(payload, envelope["signature_hex"], quote_pubkey),
        "receipt signature is not the trusted authority",
    )
    require(
        envelope["key_id"] == receipt.body["signing_key_id"], "receipt key id differs from body"
    )
    expected = {
        "intent_id": intent["intent_id"],
        "reference": intent["reference_hex"],
        "pack_id": intent["pack_id"],
        "purchase_policy_version": intent["purchase_policy_version"],
        "purchase_policy_digest": intent["purchase_policy_digest"],
        "config_hash": intent["config_hash"],
        "beneficiary_coldkey": intent["beneficiary_coldkey"],
        "beneficiary_hotkey": intent["beneficiary_hotkey"],
        "funding_coldkey": intent["funding_coldkey"],
        "gamma_api_hotkey": intent["gamma_api_hotkey"],
        "authorized_spender": intent["authorized_spender"],
        "chain_id": intent["chain_id"],
        "genesis_hash": intent["genesis_hash"],
        "netuid": intent["netuid"],
        "gamma_atoms": intent["gamma_atoms"],
        "work_tokens": intent["work_tokens"],
    }
    for field, value in expected.items():
        require(receipt.body[field] == value, "receipt lineage differs: " + field)
    require(
        receipt.deposit_key
        == gamma_deposit_key(
            receipt.body["chain_id"],
            receipt.body["finalized_block_hash"],
            receipt.body["extrinsic_index"],
            receipt.body["event_index"],
        ),
        "receipt deposit key differs",
    )
    return receipt.body


def find_receipt_envelope(terms_response: dict, *, intent: dict | None = None) -> dict | None:
    """Locate a gamma receipt row on the own-account read (None when the server ships none).

    Only a row whose ``receipt`` body IS a gamma funding receipt counts; a balance change or an
    alpha receipt row is never a gamma confirmation. When ``intent`` carries the journal's
    prepared intent (``intent_id`` + ``reference_hex``), rows for OTHER purchases under the
    same account are skipped: a genuine foreign receipt is never mistaken for this journal's
    confirmation (and so never a forged-lineage refusal) — scanning keeps going.
    """
    account = terms_response.get("account") if type(terms_response) is dict else None
    if type(account) is not dict:
        return None
    receipts = account.get("receipts")
    rows = receipts.get("real", []) if type(receipts) is dict else []
    for row in rows if type(rows) is list else []:
        if type(row) is not dict or type(row.get("receipt")) is not dict:
            continue
        if row["receipt"].get("schema") != gamma_funding_receipt.SCHEMA:
            continue
        if intent is not None and (
            row["receipt"].get("intent_id") != intent.get("intent_id")
            or row["receipt"].get("reference") != intent.get("reference_hex")
        ):
            continue
        return {
            "body": row["receipt"],
            "signature_hex": row["signature_hex"],
            "key_id": row["key_id"],
        }
    return None


# --- operations --------------------------------------------------------------------------------


def gamma_check(*, chain: GammaChain, terms: dict) -> dict:
    """Read-only readiness report. Mint readiness is reported SEPARATELY from spend readiness."""
    snapshot = chain.snapshot()
    identity = chain.runtime_identity(snapshot)
    at = snapshot.head_hash
    head = snapshot.block_number
    funding_methods = terms.get("funding_methods")
    gamma_entry = funding_methods["gamma"] if funding_methods else None
    policy_info = gamma_entry.get("policy") if gamma_entry else None
    enabled_global = chain.gamma_enabled(at)
    enabled_subnet = chain.subnet_gamma_enabled(at, NETUID)
    price_bits = chain.subnet_gamma_price_bits(at, NETUID)
    oracle = chain.tao_usd_price(at)
    oracle_fresh = oracle is not None and head - oracle["fed_at"] <= 300
    return {
        "status": "gamma_check",
        "executed": False,
        "chain": {
            "genesis_hash": snapshot.genesis_hash,
            "chain": snapshot.chain,
            "finalized_head": head,
            **identity,
        },
        "terms": {
            "terms_version": terms["terms_version"],
            "legacy_available": terms["legacy"]["terms"]["status"] == "available",
            "gamma_available": None if gamma_entry is None else gamma_entry["available"],
            "gamma_reason": None if gamma_entry is None else gamma_entry["reason"],
        },
        "policy": policy_info,
        "mint_readiness": {
            "gamma_enabled_global": enabled_global,
            "subnet_gamma_enabled": enabled_subnet,
            "mark_price_u64f64_bits": price_bits,
            "tao_usd_price": oracle,
            "oracle_fresh": oracle_fresh,
            "ready": bool(
                enabled_global and enabled_subnet and price_bits is not None and oracle_fresh
            ),
        },
    }


def gamma_check_account(
    *,
    chain: GammaChain,
    report: dict,
    funding_coldkey: str,
    gamma_api_hotkey: str,
    pack_gamma_atoms: int | None,
    fee_cap_rao: int,
) -> dict:
    """Deposit/account part of the readiness report (still read-only, still no wallet)."""
    snapshot = chain.snapshot()
    at = snapshot.head_hash
    deposit = chain.gamma_deposit(at, NETUID, gamma_api_hotkey)
    account = chain.system_account(at, funding_coldkey)
    next_index = chain.account_next_index(funding_coldkey)
    hotkeys = chain.coldkey_gamma_hotkeys(at, NETUID, funding_coldkey)
    allowance = None if deposit is None else deposit.allowance_at(snapshot.block_number)
    reasons = []
    if deposit is None:
        reasons.append("gamma_deposit_absent")
    else:
        if deposit.owner_coldkey != funding_coldkey:
            reasons.append("deposit_owner_mismatch")
        if deposit.spender != funding_coldkey:
            reasons.append("spender_mismatch")
        if pack_gamma_atoms is not None and deposit.balance_atoms < pack_gamma_atoms:
            reasons.append("insufficient_gamma_balance")
        if pack_gamma_atoms is not None:
            if allowance is not None and allowance < pack_gamma_atoms:
                reasons.append("daily_cap_insufficient")
    if account["free_rao"] < fee_cap_rao:
        reasons.append("insufficient_free_tao_for_fee")
    if next_index != account["nonce"]:
        reasons.append("pending_transactions_from_signer")
    if deposit is not None and gamma_api_hotkey not in hotkeys:
        reasons.append("gamma_hotkey_not_listed_for_coldkey")
    report["spend_readiness"] = {
        "funding_coldkey": funding_coldkey,
        "gamma_api_hotkey": gamma_api_hotkey,
        "deposit": None
        if deposit is None
        else {
            "owner_coldkey": deposit.owner_coldkey,
            "spender": deposit.spender,
            "balance_atoms": deposit.balance_atoms,
            "daily_spend_cap_atoms": deposit.cap_atoms,
            "allowance_now_atoms": allowance,
        },
        "account": {"nonce": account["nonce"], "free_tao_rao": account["free_rao"]},
        "coldkey_gamma_hotkeys": hotkeys,
        "pack_gamma_atoms": pack_gamma_atoms,
        "fee_cap_tao_rao": fee_cap_rao,
        "ready": not reasons,
        "reasons": reasons,
    }
    report["mint_readiness"]["current_deposit_spender"] = (
        None if deposit is None else deposit.spender
    )
    if deposit is not None and deposit.spender != funding_coldkey:
        report["mint_readiness"]["ready"] = False
        report["mint_readiness"]["reason"] = (
            "existing deposit names spender "
            + deposit.spender
            + "; an explicit set_gamma_deposit_spender by the deposit owner is required "
            "before self-spend (this tool never changes a spender silently)"
        )
    return report


def gamma_prepare(
    *,
    chain: GammaChain,
    journal: GammaFundingJournal,
    wallet,
    terms: dict,
    gamma_auth_signer: Callable[[bytes], bytes],
    pack_id: str,
    gamma_api_hotkey: str,
    fee_cap_rao: int,
    era_period: int,
    fetch,
    base_url: str,
    client_request_id: str | None = None,
    now: int | None = None,
    quote_pubkey: str = SN37_RECEIPT_SIGNING_PUBKEY,
) -> dict:
    """Obtain + verify the quote, compose and SIGN the one spend extrinsic, persist pre-broadcast."""
    require(
        type(fee_cap_rao) is int and 0 <= fee_cap_rao <= 100_000_000,
        "fee cap must be 0..0.1 TAO",
    )
    _require_era_period(era_period)
    now = int(time.time()) if now is None else now
    coldkey = wallet.coldkeypub.ss58_address
    legacy = terms["legacy"]
    require(legacy["identity"]["netuid"] == NETUID, "funding terms identity differs")
    beneficiary_coldkey = legacy["identity"]["coldkey"]
    require(beneficiary_coldkey == coldkey, "funding wallet must own the beneficiary coldkey")
    funding_methods = terms.get("funding_methods")
    require(funding_methods is not None, "server does not advertise gamma funding methods")
    gamma_entry = funding_methods["gamma"]
    require(gamma_entry["available"], "gamma funding unavailable: " + gamma_entry["reason"])
    policy_info = gamma_entry.get("policy")
    require(policy_info is not None, "no signed purchase policy in the discovery envelope")
    with _typed_wire_refusals():
        policy = gamma_policy.from_map(policy_info["policy"])

    with journal.lock():
        retained = journal.read("prepared")
        if retained is not None:
            require(
                retained["coldkey"] == coldkey
                and retained["gamma_api_hotkey"] == gamma_api_hotkey
                and retained["pack_id"] == pack_id,
                "gamma intent changed; retained state requires observation",
            )
            return _prepared_result(retained)

        retained_quote = journal.read("quote")
        if retained_quote is not None:
            require(
                retained_quote["pack_id"] == pack_id and retained_quote["coldkey"] == coldkey,
                "gamma quote scope changed; use a new journal directory",
            )
            client_request_id = retained_quote["client_request_id"]
        elif client_request_id is None:
            client_request_id = secrets.token_hex(16)

        snapshot = chain.snapshot()
        identity = chain.runtime_identity(snapshot)
        head = snapshot.block_number
        require(policy.active_at(head), "purchase policy not active at the finalized head")
        with _typed_wire_refusals():
            pack = policy.pack(pack_id)  # typed pack_unknown/pack_id_invalid refusal, pre-quote
        gamma_atoms = pack["gamma_atoms"]

        # EVERY check that does not need the quote runs BEFORE the wallet is unlocked or a
        # quote slot is consumed: a refusal here costs nothing and signs nothing.
        deposit = chain.gamma_deposit(snapshot.head_hash, NETUID, gamma_api_hotkey)
        require(deposit is not None, "gamma deposit does not exist")
        require(deposit.owner_coldkey == coldkey, "gamma deposit owner differs")
        require(
            deposit.spender == coldkey,
            "gamma deposit spender differs from the funding coldkey; the spend would be skipped",
        )
        require(deposit.balance_atoms >= gamma_atoms, "insufficient gamma balance for the pack")
        allowance = deposit.allowance_at(head)
        require(
            allowance is None or allowance >= gamma_atoms,
            "daily spend cap headroom is insufficient for the pack",
        )
        account = chain.system_account(snapshot.head_hash, coldkey)
        require(
            chain.account_next_index(coldkey) == account["nonce"],
            "pending transactions from this signer",
        )
        require(account["free_rao"] >= fee_cap_rao, "insufficient free TAO for the fee cap")
        require(
            policy.quote_lifetime_blocks >= era_period,
            "era period cannot fit inside this policy's quote lifetime",
        )
        birth, scan_start, scan_end = era_bounds(head, era_period)
        era_block_hash = chain.block_hash(birth)
        # Fee bound pre-check with a dummy-signed preview (identical length/weight; never
        # persisted, never signed by the wallet). The real bytes are re-checked post-sign.
        preview_call = spend_gamma_call(NETUID, gamma_api_hotkey, gamma_atoms, "0x" + "00" * 32)
        preview = compose_signed_extrinsic(
            call=preview_call,
            extra=mortal_era_bytes(head, era_period)
            + scale_compact_uint(account["nonce"])
            + scale_compact_uint(0)
            + b"\x00",
            signer_public_key=ss58_decode(coldkey),
            signature=b"\x00" * 64,
        )
        require(
            chain.fee_estimate_rao("0x" + preview.hex()) <= fee_cap_rao,
            "planned fee exceeds the fee cap",
        )

        # Unlock the coldkey exactly once: the same signer object proves the binding and
        # signs the spend payload.
        signer = wallet.coldkey
        require(signer.ss58_address == coldkey, "unlocked coldkey differs")

        auth = build_quote_auth(
            beneficiary_coldkey=beneficiary_coldkey,
            beneficiary_hotkey=legacy["identity"]["hotkey"],
            funding_coldkey=coldkey,
            gamma_api_hotkey=gamma_api_hotkey,
            pack_id=pack_id,
            config_hash=legacy["config_hash"],
            client_request_id=client_request_id,
            gamma_auth_signer=gamma_auth_signer,
            coldkey_signer=signer.sign,
            now=now,
        )
        quote = request_gamma_quote(base_url, auth=auth, fetch=fetch)
        intent = verify_quote(
            quote,
            policy=policy,
            beneficiary_coldkey=beneficiary_coldkey,
            beneficiary_hotkey=legacy["identity"]["hotkey"],
            funding_coldkey=coldkey,
            gamma_api_hotkey=gamma_api_hotkey,
            config_hash=legacy["config_hash"],
            pack_id=pack_id,
            quote_pubkey=quote_pubkey,
        )
        journal.claim(
            "quote",
            {
                "status": "QUOTE_VERIFIED",
                "quote": quote,
                "policy_digest": policy.digest(),
                "client_request_id": client_request_id,
                "pack_id": pack_id,
                "coldkey": coldkey,
            },
        )

        # Era plan: the WHOLE possible inclusion range must fit inside the quote validity.
        require(
            quote["valid_from_block"] <= scan_start,
            "quote validity opens after the earliest possible inclusion block",
        )
        require(
            quote["valid_until_block"] - head >= era_period,
            "era range does not fit inside the quote validity; use a shorter era period",
        )
        require(head <= quote["valid_until_block"], "quote already expired")

        call = spend_gamma_call(NETUID, gamma_api_hotkey, gamma_atoms, intent.reference_hex)
        decode_spend_gamma_call(
            call,
            {
                "netuid": NETUID,
                "deposit_hotkey": gamma_api_hotkey,
                "amount_atoms": gamma_atoms,
                "reference_hex": intent.reference_hex,
            },
        )
        unsigned = unsigned_extrinsic_dict(
            call=call,
            address=coldkey,
            era_current=head,
            era_period=era_period,
            nonce=account["nonce"],
            genesis_hash=GENESIS_HASH,
            era_block_hash=era_block_hash,
            spec_version=identity["spec_version"],
            transaction_version=TRANSACTION_VERSION,
        )
        authority = _validated_authority(unsigned, expected_call=call, expected_period=era_period)
        _validated_era_anchor(authority, era_block_hash)
        signature = signer.sign(authority.payload)
        require(type(signature) is bytes and len(signature) == 64, "invalid coldkey signature")
        require(
            verify_sr25519(signature, authority.payload, ss58_decode(coldkey)),
            "gamma spend signature does not verify",
        )
        extrinsic = compose_signed_extrinsic(
            call=call,
            extra=bytes.fromhex(unsigned["included_in_extrinsic"][2:]),
            signer_public_key=ss58_decode(coldkey),
            signature=signature,
        )
        decoded = decode_signed_extrinsic(extrinsic, expected_call=call)
        require(
            decoded["signature"] == signature
            and decoded["nonce"] == account["nonce"]
            and decoded["era"] == mortal_era_bytes(head, era_period).hex()
            and decoded["tip"] == 0
            and decoded["signer_public_key"] == ss58_decode(coldkey),
            "decoded composed extrinsic differs from the prepared authority",
        )
        extrinsic_hex = "0x" + extrinsic.hex()
        tx_hash = extrinsic_tx_hash(extrinsic)
        planned_fee = chain.fee_estimate_rao(extrinsic_hex)
        require(planned_fee <= fee_cap_rao, "planned fee exceeds the fee cap")
        prepared = {
            "status": "PREPARED",
            "coldkey": coldkey,
            "beneficiary_hotkey": legacy["identity"]["hotkey"],
            "gamma_api_hotkey": gamma_api_hotkey,
            "pack_id": pack_id,
            "intent_id": intent.intent_id,
            "reference_hex": intent.reference_hex,
            "gamma_atoms": gamma_atoms,
            "work_tokens": int(intent.body["work_tokens"]),
            "purchase_policy_version": policy.policy_version,
            "purchase_policy_digest": policy.digest(),
            "config_hash": legacy["config_hash"],
            "quote": quote,
            "unsigned": unsigned,
            "signature_hex": signature.hex(),
            "extrinsic_hex": extrinsic_hex,
            "tx_hash": tx_hash,
            "nonce": account["nonce"],
            "era": {
                "period": era_period,
                "current": head,
                "birth": birth,
                "death": birth + era_period,
                "scan_start": scan_start,
                "scan_end": scan_end,
            },
            "runtime": identity,
            "valid_until_block": quote["valid_until_block"],
            "fee_cap_tao_rao": fee_cap_rao,
            "planned_fee_tao_rao": planned_fee,
            "free_tao_rao": account["free_rao"],
        }
        journal.claim("prepared", prepared)
        return _prepared_result(prepared)


def _prepared_result(prepared: dict) -> dict:
    return {
        "status": prepared["status"],
        "executed": False,
        "intent_id": prepared["intent_id"],
        "reference_hex": prepared["reference_hex"],
        "pack_id": prepared["pack_id"],
        "gamma_atoms": prepared["gamma_atoms"],
        "work_tokens": prepared["work_tokens"],
        "tx_hash": prepared["tx_hash"],
        "era": prepared["era"],
        "runtime": prepared["runtime"],
        "fee_cap_tao_rao": prepared["fee_cap_tao_rao"],
        "planned_fee_tao_rao": prepared["planned_fee_tao_rao"],
        "free_tao_rao": prepared["free_tao_rao"],
        "note": "prepared and signed; only a fresh explicit --execute submits once",
    }


def _verify_prepared_untampered(journal: GammaFundingJournal) -> dict:
    """Hash-bound, decode-back-verified reload of the prepared record; no wallet involved."""
    prepared = journal.read("prepared")
    require(prepared is not None, "no prepared gamma operation in this journal")
    submission = journal.read("submission")
    if submission is not None:
        require(
            submission["prepared_sha256"] == _canonical_sha256(prepared),
            "prepared gamma record was tampered with after the submission claim",
        )
    unsigned = prepared["unsigned"]
    call = bytes.fromhex(unsigned["call_data"][2:])
    decode_spend_gamma_call(
        call,
        {
            "netuid": NETUID,
            "deposit_hotkey": prepared["gamma_api_hotkey"],
            "amount_atoms": prepared["gamma_atoms"],
            "reference_hex": prepared["reference_hex"],
        },
    )
    authority = _validated_authority(
        unsigned, expected_call=call, expected_period=unsigned["era"]["period"]
    )
    signature = bytes.fromhex(prepared["signature_hex"])
    require(
        verify_sr25519(signature, authority.payload, ss58_decode(prepared["coldkey"])),
        "retained gamma signature differs",
    )
    extrinsic = compose_signed_extrinsic(
        call=call,
        extra=bytes.fromhex(unsigned["included_in_extrinsic"][2:]),
        signer_public_key=ss58_decode(prepared["coldkey"]),
        signature=signature,
    )
    require(
        "0x" + extrinsic.hex() == prepared["extrinsic_hex"]
        and extrinsic_tx_hash(extrinsic) == prepared["tx_hash"],
        "retained gamma extrinsic bytes differ from the recomposed authority",
    )
    decode_signed_extrinsic(extrinsic, expected_call=call)
    return prepared


def gamma_execute(
    *,
    chain: GammaChain,
    journal: GammaFundingJournal,
    submit: Callable[[str], str],
) -> dict:
    """Fresh-explicit only: verify untampered, claim SUBMISSION_ATTEMPTED (fsync), submit ONCE."""
    with journal.lock():
        prepared = _verify_prepared_untampered(journal)
        if journal.read("submission") is not None:
            return gamma_observe(chain=chain, journal=journal)
        snapshot = chain.snapshot()
        head = snapshot.block_number
        identity = chain.runtime_identity(snapshot)
        require(
            identity["spec_version"] == prepared["runtime"]["spec_version"]
            and identity["metadata_sha256"] == prepared["runtime"]["metadata_sha256"]
            and identity["code_sha256"] == prepared["runtime"]["code_sha256"],
            "runtime drifted since preparation; refusing to submit the retained bytes",
        )
        era = prepared["era"]
        require(head <= era["scan_end"], "mortal era expired before broadcast")
        require(head <= prepared["valid_until_block"], "quote validity ended before broadcast")
        # The uncertain window must always leave SUBMISSION_ATTEMPTED behind, never a state
        # that invites re-signing: the claim is durable BEFORE the network call.
        journal.claim(
            "submission",
            {
                "status": "SUBMISSION_ATTEMPTED",
                "prepared_sha256": _canonical_sha256(prepared),
                "tx_hash": prepared["tx_hash"],
                "submitted_at": int(time.time()),
            },
        )
        try:
            returned = submit(prepared["extrinsic_hex"])
            if isinstance(returned, str) and returned.startswith("0x"):
                require(
                    returned.lower() == prepared["tx_hash"],
                    "node returned a different tx hash for the submitted bytes",
                )
        except Exception as exc:  # noqa: BLE001 — ambiguity is a state, never a re-sign
            return {
                "status": "OUTCOME_UNKNOWN",
                "executed": True,
                "tx_hash": prepared["tx_hash"],
                "reason": "submission transport failed after the attempt: " + type(exc).__name__,
                "note": "rerun --execute or --status to observe; never re-sign or replace",
            }
        return gamma_observe(chain=chain, journal=journal)


def gamma_observe(*, chain: GammaChain, journal: GammaFundingJournal) -> dict:
    """Read-only observation of the one attempted submission; never signs, never resubmits."""
    prepared = _verify_prepared_untampered(journal)
    require(journal.read("submission") is not None, "no submission attempted yet")
    snapshot = chain.snapshot()
    head = snapshot.block_number
    era = prepared["era"]
    if head < era["scan_start"]:
        # The finalized head has not advanced past the prepare head (scan_start is
        # prepare_head + 1), so the observable window is EMPTY: nothing about inclusion
        # is knowable yet. That is a pending state — the journal keeps its
        # SUBMISSION_ATTEMPTED claim and the operator reruns after the chain advances —
        # never a typed refusal for a window that simply has not opened (the empty range
        # must not reach scan_for_extrinsic's bounds guard).
        return {
            "status": "OUTCOME_UNKNOWN",
            "executed": False,
            "tx_hash": prepared["tx_hash"],
            "reason": "finalized head has not reached the era scan window; nothing observable yet",
            "note": "observe_same_intent; rerun after the finalized head advances",
        }
    scan_end = min(head, era["scan_end"])
    match = chain.scan_for_extrinsic(prepared["extrinsic_hex"], era["scan_start"], scan_end)
    if match is None:
        if head >= era["scan_end"]:
            return {
                "status": "NOT_ON_CHAIN_AFTER_ERA",
                "executed": False,
                "tx_hash": prepared["tx_hash"],
                "note": (
                    "complete finalized coverage of the expired mortal era with no inclusion; "
                    "a NEW operation still additionally requires the server to report no "
                    "settlement for this intent"
                ),
            }
        return {
            "status": "OUTCOME_UNKNOWN",
            "executed": False,
            "tx_hash": prepared["tx_hash"],
            "reason": "submission unresolved inside the mortal era; absence is not proof",
            "note": "observe_same_intent; never automatically resubmit",
        }
    require(
        chain.block_hash(match["block_number"]) == match["block_hash"],
        "observed block hash is not canonical",
    )
    spend_event = _match_spend_event(chain.block_events_raw(match["block_hash"]), prepared)
    journal.claim(
        "observed",
        {
            "status": "OBSERVED_ON_CHAIN",
            "block_number": match["block_number"],
            "block_hash": match["block_hash"],
            "extrinsic_index": match["extrinsic_index"],
            "tx_hash": prepared["tx_hash"],
            "spend_event": spend_event,
        },
    )
    return {
        "status": "OBSERVED_ON_CHAIN",
        "executed": False,
        "block_number": match["block_number"],
        "block_hash": match["block_hash"],
        "extrinsic_index": match["extrinsic_index"],
        "tx_hash": prepared["tx_hash"],
        "spend_event": spend_event,
        "note": (
            "the independent observer settles the payment; CENTRAL_CREDITED is reported only "
            "from a verified signed receipt"
        ),
    }


def _match_spend_event(events: bytes | None, prepared: dict) -> str:
    """Distinguish GammaSpent from a fully-skipped batch (both are successful extrinsics)."""
    if events is None:
        return "events_unavailable"
    coldkey = ss58_decode(prepared["coldkey"])
    hotkey = ss58_decode(prepared["gamma_api_hotkey"])
    amount = struct.pack("<Q", prepared["gamma_atoms"])
    reference = bytes.fromhex(prepared["reference_hex"][2:])
    netuid = struct.pack("<H", NETUID)
    # GammaSpent{netuid, coldkey, hotkey, spender, amount, reference}
    spent = GAMMA_SPENT_EVENT_INDEX + netuid + coldkey + hotkey + coldkey + amount + reference
    # GammaSpendSkipped{netuid, hotkey, spender, amount, reference, reason..}
    skipped = GAMMA_SPEND_SKIPPED_EVENT_INDEX + netuid + hotkey + coldkey + amount + reference
    if spent in events:
        return "gamma_spent"
    if skipped in events:
        return "gamma_spend_skipped"
    return "not_proven_in_events"


def gamma_status(
    *,
    chain: GammaChain,
    journal: GammaFundingJournal,
    terms: dict | None = None,
    quote_pubkey: str = SN37_RECEIPT_SIGNING_PUBKEY,
) -> dict:
    """Read-only status; CENTRAL_CREDITED only from a fully verified signed receipt."""
    with journal.lock():
        prepared = journal.read("prepared")
        if prepared is None:
            return {
                "status": "NO_PREPARED_OPERATION",
                "note": "nothing prepared in this journal directory",
            }
        _verify_prepared_untampered(journal)
        credited = journal.read("credited")
        if credited is not None:
            return {
                "status": "CENTRAL_CREDITED",
                "executed": False,
                "receipt": credited["receipt"],
            }
        result = (
            gamma_observe(chain=chain, journal=journal)
            if journal.read("submission") is not None
            else {
                "status": "PREPARED_NOT_SUBMITTED",
                "executed": False,
                "tx_hash": prepared["tx_hash"],
                "note": "prepared and signed but never broadcast; only a fresh --execute submits",
            }
        )
        if terms is not None:
            intent = _receipt_intent(prepared)
            envelope = find_receipt_envelope(terms["legacy"], intent=intent)
            if envelope is not None:
                receipt = verify_gamma_receipt(envelope, intent=intent, quote_pubkey=quote_pubkey)
                journal.claim("credited", {"receipt": receipt})
                result["status"] = "CENTRAL_CREDITED"
                result["receipt"] = receipt
        return result


def _receipt_intent(prepared: dict) -> dict:
    return {
        "intent_id": prepared["intent_id"],
        "reference_hex": prepared["reference_hex"],
        "pack_id": prepared["pack_id"],
        "purchase_policy_version": prepared["purchase_policy_version"],
        "purchase_policy_digest": prepared["purchase_policy_digest"],
        "config_hash": "0x" + prepared["config_hash"],
        "beneficiary_coldkey": prepared["coldkey"],
        "beneficiary_hotkey": prepared["beneficiary_hotkey"],
        "funding_coldkey": prepared["coldkey"],
        "gamma_api_hotkey": prepared["gamma_api_hotkey"],
        "authorized_spender": prepared["coldkey"],
        "chain_id": CHAIN_ID,
        "genesis_hash": GENESIS_HASH,
        "netuid": NETUID,
        "gamma_atoms": str(prepared["gamma_atoms"]),
        "work_tokens": str(prepared["work_tokens"]),
    }


# --- explicit mint setup (never invoked by prepare/execute) ------------------------------------


def _positive_integer(value, label: str) -> int:
    require(
        type(value) is str and re.fullmatch(r"[1-9][0-9]{0,18}", value) is not None,
        label + " must be a positive integer atom string",
    )
    return int(value)


def gamma_mint_setup(
    *,
    chain: GammaChain,
    journal: GammaFundingJournal,
    wallet,
    gamma_api_hotkey: str,
    tao: str | None,
    min_gamma_out: str | None,
    daily_spend_cap: str | None,
    execute: bool,
    confirm: bool,
    submit: Callable[[str], str] | None,
    era_period: int,
) -> dict:
    """Explicit, separately confirmed mint/setup; never a silent burn or spender change.

    The mint ALWAYS names the own coldkey as spender — the chain default is the SUBNET's
    service spender, which this wallet could never spend (G3 §6). ``execute`` plus the
    separate ``confirm`` flag performs the one-shot mint; without both this is a read-only
    readiness report that shows the ACTUAL current spender.
    """
    _require_era_period(era_period)
    coldkey = wallet.coldkeypub.ss58_address
    snapshot = chain.snapshot()
    identity = chain.runtime_identity(snapshot)
    at = snapshot.head_hash
    head = snapshot.block_number
    enabled_global = chain.gamma_enabled(at)
    enabled_subnet = chain.subnet_gamma_enabled(at, NETUID)
    price_bits = chain.subnet_gamma_price_bits(at, NETUID)
    oracle = chain.tao_usd_price(at)
    deposit = chain.gamma_deposit(at, NETUID, gamma_api_hotkey)
    account = chain.system_account(at, coldkey)
    oracle_fresh = oracle is not None and head - oracle["fed_at"] <= 300
    mint_ready = bool(enabled_global and enabled_subnet and price_bits is not None and oracle_fresh)
    mint_reasons = []
    if not enabled_global:
        mint_reasons.append("gamma_not_enabled")
    if not enabled_subnet:
        mint_reasons.append("subnet_gamma_not_enabled")
    if price_bits is None:
        mint_reasons.append("no_mark_price")
    if oracle is None:
        mint_reasons.append("tao_usd_price_unavailable")
    elif not oracle_fresh:
        mint_reasons.append("tao_usd_price_stale")
    if deposit is not None:
        if deposit.owner_coldkey != coldkey:
            mint_reasons.append("existing_deposit_owned_by_another_coldkey")
            mint_ready = False
        if deposit.spender != coldkey:
            mint_reasons.append("existing_deposit_spender_is_not_this_coldkey")
            mint_ready = False
    if tao is not None:
        tao_rao = amount_rao(tao)
        require(min_gamma_out is not None, "mint requires --mint-min-gamma-out")
        min_out = _positive_integer(min_gamma_out, "mint-min-gamma-out")
        cap = (
            None
            if daily_spend_cap is None
            else _positive_integer(daily_spend_cap, "mint-daily-cap")
        )
    else:
        tao_rao = min_out = cap = None
    args = {
        "call": "SubtensorModule.mint_gamma",
        "netuid": NETUID,
        "gamma_api_hotkey": gamma_api_hotkey,
        "tao_rao": tao_rao,
        "min_gamma_out_atoms": min_out,
        "spender": coldkey,
        "daily_spend_cap_atoms": cap,
        "actual_current_spender": None if deposit is None else deposit.spender,
        "note": "exact gamma-out is chain-computed (AMM spot, mark price, oracle); never invented",
    }
    if not execute:
        return {
            "status": "MINT_CHECK",
            "executed": False,
            "chain": {"finalized_head": head, **identity},
            "mint_readiness": {
                "ready": mint_ready,
                "reasons": mint_reasons,
                "gamma_enabled_global": enabled_global,
                "subnet_gamma_enabled": enabled_subnet,
                "mark_price_u64f64_bits": price_bits,
                "tao_usd_price": oracle,
                "free_tao_rao": account["free_rao"],
                "existing_deposit": None
                if deposit is None
                else {
                    "owner_coldkey": deposit.owner_coldkey,
                    "spender": deposit.spender,
                    "balance_atoms": deposit.balance_atoms,
                },
            },
            "mint_args": args,
        }
    require(confirm, "mint execution requires the separate --confirm-mint-burns-tao flag")
    require(submit is not None, "mint execution requires a submission transport")
    require(tao_rao is not None, "mint execution requires --mint-tao")
    require(mint_ready, "mint not ready: " + ",".join(mint_reasons))
    with journal.lock():
        require(
            journal.read("mint_prepared") is None and journal.read("mint_submission") is None,
            "a mint operation already exists in this journal; observe it, never repeat it",
        )
        require(chain.account_next_index(coldkey) == account["nonce"], "pending transactions")
        require(account["free_rao"] >= tao_rao, "insufficient free TAO for the mint input")
        birth, scan_start, scan_end = era_bounds(head, era_period)
        call = mint_gamma_call(
            NETUID,
            gamma_api_hotkey,
            tao_rao,
            min_out,
            spender=coldkey,
            daily_spend_cap_atoms=cap,
        )
        unsigned = unsigned_extrinsic_dict(
            call=call,
            address=coldkey,
            era_current=head,
            era_period=era_period,
            nonce=account["nonce"],
            genesis_hash=GENESIS_HASH,
            era_block_hash=chain.block_hash(birth),
            spec_version=identity["spec_version"],
            transaction_version=TRANSACTION_VERSION,
        )
        authority = _validated_authority(unsigned, expected_call=call, expected_period=era_period)
        signer = wallet.coldkey
        require(signer.ss58_address == coldkey, "unlocked coldkey differs")
        signature = signer.sign(authority.payload)
        require(
            verify_sr25519(signature, authority.payload, ss58_decode(coldkey)),
            "mint signature does not verify",
        )
        extrinsic = compose_signed_extrinsic(
            call=call,
            extra=bytes.fromhex(unsigned["included_in_extrinsic"][2:]),
            signer_public_key=ss58_decode(coldkey),
            signature=signature,
        )
        decode_signed_extrinsic(extrinsic, expected_call=call)
        # The mint burns the TAO input AND pays its fee from the same free balance.
        planned_fee = chain.fee_estimate_rao("0x" + extrinsic.hex())
        require(
            account["free_rao"] >= tao_rao + planned_fee,
            "insufficient free TAO for the mint input plus its fee",
        )
        prepared = {
            "status": "MINT_PREPARED",
            "coldkey": coldkey,
            "gamma_api_hotkey": gamma_api_hotkey,
            "tao_rao": tao_rao,
            "min_gamma_out_atoms": min_out,
            "daily_spend_cap_atoms": cap,
            "spender": coldkey,
            "unsigned": unsigned,
            "signature_hex": signature.hex(),
            "extrinsic_hex": "0x" + extrinsic.hex(),
            "tx_hash": extrinsic_tx_hash(extrinsic),
            "nonce": account["nonce"],
            "era": {
                "period": era_period,
                "current": head,
                "birth": birth,
                "death": birth + era_period,
                "scan_start": scan_start,
                "scan_end": scan_end,
            },
            "runtime": identity,
        }
        journal.claim("mint_prepared", prepared)
        journal.claim(
            "mint_submission",
            {
                "status": "SUBMISSION_ATTEMPTED",
                "prepared_sha256": _canonical_sha256(prepared),
                "tx_hash": prepared["tx_hash"],
                "submitted_at": int(time.time()),
            },
        )
        try:
            returned = submit(prepared["extrinsic_hex"])
            if isinstance(returned, str) and returned.startswith("0x"):
                require(
                    returned.lower() == prepared["tx_hash"],
                    "node returned a different tx hash for the submitted mint bytes",
                )
        except Exception as exc:  # noqa: BLE001 — ambiguity is a state, never a re-sign
            return {
                "status": "MINT_OUTCOME_UNKNOWN",
                "executed": True,
                "tx_hash": prepared["tx_hash"],
                "reason": "mint submission transport failed after the attempt: "
                + type(exc).__name__,
                "note": "observe the deposit balance read-only; never re-sign or repeat the mint",
            }
        return {
            "status": "MINT_SUBMISSION_ATTEMPTED",
            "executed": True,
            "tx_hash": prepared["tx_hash"],
            "era": prepared["era"],
            "mint_args": args,
        }


# --- one-shot broadcast transport --------------------------------------------------------------


class SingleShotSubmitter:
    """Exactly one ``author_submitExtrinsic`` per call — no reconnect-retry, no re-sign.

    The genesis hash is pinned on the socket before any submission; a transport failure after
    the send surfaces as an exception while the journal already holds SUBMISSION_ATTEMPTED,
    so ambiguity can never invite a second signing.
    """

    def __init__(
        self, endpoint: str, *, expected_genesis: str = GENESIS_HASH, timeout: float = 10.0
    ):
        self.endpoint = validate_subtensor_endpoint(endpoint)
        self.expected_genesis = expected_genesis.lower()
        self.timeout = timeout

    def __call__(self, extrinsic_hex: str) -> str:
        from websocket import create_connection

        ws = None
        try:
            ws = create_connection(self.endpoint, timeout=self.timeout)
            ws.send(
                json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": "chain_getBlockHash", "params": [0]}
                )
            )
            identity = json.loads(ws.recv())
            if (
                type(identity) is not dict
                or str(identity.get("result", "")).lower() != self.expected_genesis
            ):
                raise GammaRefused("submission endpoint genesis differs")
            ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "author_submitExtrinsic",
                        "params": [extrinsic_hex],
                    },
                    separators=(",", ":"),
                )
            )
            response = json.loads(ws.recv())
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:  # noqa: BLE001 — cleanup never masks the submission outcome
                    pass
        error = response.get("error") if type(response) is dict else None
        if type(error) is dict:
            raise GammaRefused("author_submitExtrinsic refused: " + str(error.get("message")))
        result = response.get("result") if type(response) is dict else None
        require(type(result) is str and result.startswith("0x"), "submission returned no tx hash")
        return result


# --- CLI wiring --------------------------------------------------------------------------------


def run(
    *,
    action: str,
    base_url: str,
    endpoint: str,
    directory: str | None,
    pack: str | None = None,
    gamma_hotkey: str | None = None,
    intent: str | None = None,
    fee_cap: str = "0.01",
    era_period: int = 64,
    execute: bool = False,
    wallet_path: str | None = None,
    wallet_name: str | None = None,
    mint_tao: str | None = None,
    mint_min_out: str | None = None,
    mint_cap: str | None = None,
    confirm_mint: bool = False,
) -> dict:
    """Wire the CLI action to transports; reads stay wallet-free, execute never loads a wallet."""
    from subnet.miner.serve import minter_from_env

    minter = minter_from_env()
    rpc = FinalizedRpc(endpoint)
    chain = GammaChain(rpc)
    try:
        if action in {"check", "status"}:
            terms = read_gamma_terms(
                base_url,
                hotkey=minter.hotkey_ss58,
                funding_terms_signer=minter.sign_miner_funding_terms,
                economics_signer=minter.sign_economics,
                config_pubkey=config_signing_pubkey(),
                fetch=_default_fetch(),
            )
            if action == "status":
                return gamma_status(
                    chain=chain, journal=GammaFundingJournal(directory), terms=terms
                )
            report = gamma_check(chain=chain, terms=terms)
            if gamma_hotkey is not None:
                policy_info = report.get("policy")
                pack_atoms = None
                if pack and policy_info and pack in policy_info["packs"]:
                    pack_atoms = policy_info["packs"][pack]["gamma_atoms"]
                report = gamma_check_account(
                    chain=chain,
                    report=report,
                    funding_coldkey=terms["legacy"]["identity"]["coldkey"],
                    gamma_api_hotkey=gamma_hotkey,
                    pack_gamma_atoms=pack_atoms,
                    fee_cap_rao=amount_rao(fee_cap),
                )
            return report
        journal = GammaFundingJournal(directory)
        if action == "execute":
            # Structurally wallet-free: a rerun can only observe the retained submission.
            return gamma_execute(chain=chain, journal=journal, submit=SingleShotSubmitter(endpoint))
        require(gamma_hotkey is not None, action + " requires --gamma-hotkey")
        require(wallet_name is not None and wallet_path is not None, "wallet required")
        from subnet.bittensor_v11 import load_wallet, require_bittensor_v11

        bt = require_bittensor_v11()
        wallet = load_wallet(bt, name=wallet_name, hotkey="default", path=wallet_path)
        if action == "mint-setup":
            return gamma_mint_setup(
                chain=chain,
                journal=journal,
                wallet=wallet,
                gamma_api_hotkey=gamma_hotkey,
                tao=mint_tao,
                min_gamma_out=mint_min_out,
                daily_spend_cap=mint_cap,
                execute=execute,
                confirm=confirm_mint,
                submit=SingleShotSubmitter(endpoint) if execute else None,
                era_period=era_period,
            )
        require(pack is not None, "gamma prepare requires --pack")
        fetch = _default_fetch()
        terms = read_gamma_terms(
            base_url,
            hotkey=minter.hotkey_ss58,
            funding_terms_signer=minter.sign_miner_funding_terms,
            economics_signer=minter.sign_economics,
            config_pubkey=config_signing_pubkey(),
            fetch=fetch,
        )
        result = gamma_prepare(
            chain=chain,
            journal=journal,
            wallet=wallet,
            terms=terms,
            gamma_auth_signer=minter.sign_gamma_funding_auth,
            pack_id=pack,
            gamma_api_hotkey=gamma_hotkey,
            fee_cap_rao=amount_rao(fee_cap),
            era_period=era_period,
            fetch=fetch,
            base_url=base_url,
        )
        if intent is not None:
            require(result["intent_id"] == intent, "prepared intent differs from --intent")
        return result
    finally:
        rpc.close()


def _default_fetch():
    from subnet.miner.read_client import request_json

    return request_json
