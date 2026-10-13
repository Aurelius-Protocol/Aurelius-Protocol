# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Reusable authority records for Bittensor 11 prepared-extrinsic submission.

``Subtensor.execute`` intentionally hides plan, signing, and submission behind one call.  That is
convenient for ordinary applications but too ambiguous for a one-shot operator mutation: a crash
cannot prove whether a signature was produced or submitted.  SDK11's public offline flow exposes
the boundaries instead::

    plan.call -> prepare_call -> signer.sign(payload) -> submit_signature

This module validates and serializes the task-agnostic pieces of that flow.  Call-specific modules
must independently validate ``plan.call`` and decoded ``call_data``.  A durable signature is never
re-created.  A durable submission claim is ambiguous until a bounded scan of its exact mortal era
proves one finalized match or proves no match after expiry.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping
from subnet.operator.evidence_reader import canonical_json_bytes
from subnet.shared.signing import verify as verify_sr25519
from subnet.shared.ss58 import ss58_decode

HEX_32 = re.compile(r"^0x[0-9a-fA-F]{64}$")


HEX_BYTES = re.compile(r"^0x(?:[0-9a-fA-F]{2})+$")


PREPARED_SCHEMA_VERSION = 1


SIGNATURE_SCHEMA_VERSION = 1


SUBMISSION_SCHEMA_VERSION = 1


RESOLUTION_SCHEMA_VERSION = 1


MAX_PREPARED_ATTEMPTS = 2


BITTENSOR_V11_SIGNED_EXTENSIONS = (
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


class PreparedExtrinsicError(RuntimeError):
    """Prepared SDK evidence does not grant the claimed narrow authority."""


@dataclass(frozen=True)
class PreparedAuthority:
    prepared_sha256: str
    call_data_sha256: str
    call_data_blake2b_256: str
    payload_sha256: str
    address: str
    nonce: int
    era_current: int
    era_period: int
    era_birth: int
    scan_start_block: int
    scan_end_block: int
    genesis_hash: str
    era_block_hash: str
    spec_version: int
    transaction_version: int
    tip: int
    metadata_hash_enabled: bool
    public_key: bytes
    payload: bytes


_PREPARED_RESERVED = {
    "schema_version",
    "attempt",
    "status",
    "unsigned",
    "prepared_sha256",
    "call_data_sha256",
    "call_data_blake2b_256",
    "payload_sha256",
    "nonce",
    "era_current",
    "era_period",
    "era_birth",
    "scan_start_block",
    "scan_end_block",
    "tip",
    "metadata_hash_enabled",
    "genesis_hash",
    "era_block_hash",
    "runtime_spec_version",
    "transaction_version",
}


_SIGNATURE_RESERVED = {
    "schema_version",
    "attempt",
    "status",
    "prepared_sha256",
    "payload_sha256",
    "signature_hex",
    "signature_sha256",
    "scan_start_block",
    "scan_end_block",
    "nonce",
}


_SUBMISSION_RESERVED = {
    "schema_version",
    "attempt",
    "status",
    "prepared_sha256",
    "payload_sha256",
    "signature_sha256",
    "nonce",
    "scan_start_block",
    "scan_end_block",
    "next_fallible_operation",
}


_RESOLUTION_RESERVED = {
    "schema_version",
    "attempt",
    "status",
    "terminal_for_attempt",
    "next_attempt_authorized",
    "prepared_sha256",
    "payload_sha256",
    "signature_sha256",
    "nonce",
    "scan_start_block",
    "scan_end_block",
    "scan_complete_through_block",
    "scanned_block_count",
    "finalized_head",
    "finalized_block_number",
    "candidate_count",
    "candidates",
}


def _identity(value: Mapping[str, Any], *, reserved: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparedExtrinsicError("prepared evidence identity must be a mapping")
    identity = dict(value)
    if any(not isinstance(key, str) or not key for key in identity):
        raise PreparedExtrinsicError("prepared evidence identity keys must be non-empty strings")
    collision = set(identity) & reserved
    if collision:
        raise PreparedExtrinsicError(
            "prepared evidence identity collides with reserved fields: "
            + ", ".join(sorted(collision))
        )
    return identity


def _attempt(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_PREPARED_ATTEMPTS:
        raise PreparedExtrinsicError(f"prepared attempt number must be 1..{MAX_PREPARED_ATTEMPTS}")
    return value


def _hex_bytes(value: Any, *, label: str, exact_bytes: int | None = None) -> bytes:
    if not isinstance(value, str) or HEX_BYTES.fullmatch(value) is None or value != value.lower():
        raise PreparedExtrinsicError(f"{label} is not canonical 0x-prefixed byte hex")
    try:
        decoded = bytes.fromhex(value[2:])
    except ValueError:
        raise PreparedExtrinsicError(f"{label} is malformed hex") from None
    if exact_bytes is not None and len(decoded) != exact_bytes:
        raise PreparedExtrinsicError(f"{label} must contain exactly {exact_bytes} bytes")
    return decoded


def _uint(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PreparedExtrinsicError(f"{label} must be a non-negative integer")
    return value


def _scale_extrinsic_body_length(raw: bytes) -> int:
    """Decode and canonical-check the SCALE compact length prefix on a raw extrinsic."""

    if not raw:
        raise PreparedExtrinsicError("raw finalized extrinsic is empty")
    mode = raw[0] & 0b11
    if mode == 0:
        prefix_bytes = 1
        body_length = raw[0] >> 2
    elif mode == 1:
        prefix_bytes = 2
        if len(raw) < prefix_bytes:
            raise PreparedExtrinsicError("raw finalized extrinsic has a truncated compact length")
        body_length = int.from_bytes(raw[:prefix_bytes], "little") >> 2
        if body_length < 1 << 6:
            raise PreparedExtrinsicError("raw finalized extrinsic length is not canonical SCALE")
    elif mode == 2:
        prefix_bytes = 4
        if len(raw) < prefix_bytes:
            raise PreparedExtrinsicError("raw finalized extrinsic has a truncated compact length")
        body_length = int.from_bytes(raw[:prefix_bytes], "little") >> 2
        if body_length < 1 << 14:
            raise PreparedExtrinsicError("raw finalized extrinsic length is not canonical SCALE")
    else:
        encoded_bytes = (raw[0] >> 2) + 4
        prefix_bytes = encoded_bytes + 1
        if len(raw) < prefix_bytes or encoded_bytes > 8:
            raise PreparedExtrinsicError(
                "raw finalized extrinsic has an unsupported compact length"
            )
        encoded = raw[1:prefix_bytes]
        body_length = int.from_bytes(encoded, "little")
        if body_length < 1 << 30 or encoded[-1] == 0:
            raise PreparedExtrinsicError("raw finalized extrinsic length is not canonical SCALE")
    if prefix_bytes + body_length != len(raw):
        raise PreparedExtrinsicError(
            "raw finalized extrinsic compact length differs from its exact bytes"
        )
    return body_length


def _big_endian_hex(value: int, width: int) -> str:
    try:
        encoded = value.to_bytes(width, "big")
    except OverflowError as exc:
        raise PreparedExtrinsicError("prepared integer exceeds its SDK payload width") from exc
    return "0x" + encoded.hex()


def _little_endian_uint(value: int, width: int, *, label: str) -> bytes:
    try:
        return value.to_bytes(width, "little")
    except OverflowError as exc:
        raise PreparedExtrinsicError(f"{label} exceeds its runtime width") from exc


def _scale_compact_uint(value: int, *, label: str) -> bytes:
    value = _uint(value, label=label)
    if value < 1 << 6:
        return bytes((value << 2,))
    if value < 1 << 14:
        return ((value << 2) | 0b01).to_bytes(2, "little")
    if value < 1 << 30:
        return ((value << 2) | 0b10).to_bytes(4, "little")
    encoded_bytes = max(4, (value.bit_length() + 7) // 8)
    if encoded_bytes > 67:
        raise PreparedExtrinsicError(f"{label} is too large for SCALE compact encoding")
    return bytes((((encoded_bytes - 4) << 2) | 0b11,)) + value.to_bytes(encoded_bytes, "little")


def _era_bounds(current: int, period: int) -> tuple[int, int, int]:
    # Substrate mortal-era phase is quantized when period > 4096.  We compile operator mutations
    # to 64 blocks, but retain the exact standard calculation so this reusable validator fails
    # predictably if a later reviewed profile chooses another valid period.
    if period < 4 or period > 65_536 or period & (period - 1):
        raise PreparedExtrinsicError("prepared mortal era period is not a power of two in 4..65536")
    quantize = max(period >> 12, 1)
    phase = (current % period // quantize) * quantize
    # MortalEra stores ``phase`` relative to a rolling period window. Its birth at the
    # preparation head is the latest block with that phase, not the nearest period boundary.
    # For the compiled 64-block profile quantize is one, so birth is exactly ``current``.
    birth = current - ((current - phase) % period)
    # Preparation occurs at a finalized current block. A submission cannot already be in that
    # finalized block, and the era dies at birth+period (exclusive).
    return birth, current + 1, birth + period - 1


def _era_phase(current: int, period: int) -> int:
    quantize = max(period >> 12, 1)
    return (current % period // quantize) * quantize


def _encoded_mortal_era(current: int, period: int) -> str:
    quantize = max(period >> 12, 1)
    phase = _era_phase(current, period)
    exponent = period.bit_length() - 2
    encoded = min(15, max(1, exponent)) | ((phase // quantize) << 4)
    return "0x" + encoded.to_bytes(2, "little").hex()


def validate_unsigned_dict(
    unsigned: Mapping[str, Any],
    *,
    expected_address: str,
    expected_genesis_hash: str,
    expected_spec_version: int,
    expected_transaction_version: int,
    expected_period: int,
    expected_tip: int = 0,
    expected_call_data_sha256: str | None = None,
) -> PreparedAuthority:
    """Validate one ``UnsignedExtrinsic.to_dict()`` without trusting SDK internals."""

    fields = {
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
        "payload_json",
        "included_in_extrinsic",
        "included_in_signed_data",
    }
    if not isinstance(unsigned, Mapping) or set(unsigned) != fields:
        raise PreparedExtrinsicError("prepared unsigned extrinsic has an unaudited field set")
    call_data = _hex_bytes(unsigned["call_data"], label="prepared call_data")
    public_key = _hex_bytes(unsigned["public_key"], label="prepared public key", exact_bytes=32)
    payload = _hex_bytes(unsigned["payload"], label="prepared signing payload")
    included_extrinsic = _hex_bytes(
        unsigned["included_in_extrinsic"], label="prepared extrinsic extensions"
    )
    included_signed = _hex_bytes(
        unsigned["included_in_signed_data"], label="prepared signed extensions"
    )
    address = unsigned["address"]
    if not isinstance(address, str) or address != expected_address:
        raise PreparedExtrinsicError("prepared signer address differs from operator authority")
    try:
        address_public_key = ss58_decode(address)
    except ValueError as exc:
        raise PreparedExtrinsicError("prepared signer address is not valid SS58") from exc
    if public_key != address_public_key:
        raise PreparedExtrinsicError("prepared public key differs from its SS58 signer")
    if unsigned["crypto_type"] != 1:
        raise PreparedExtrinsicError("prepared signer is not sr25519")
    era = unsigned["era"]
    if not isinstance(era, dict) or set(era) != {"period", "current"}:
        raise PreparedExtrinsicError("prepared extrinsic is not one explicit mortal era")
    period = _uint(era["period"], label="prepared era period")
    current = _uint(era["current"], label="prepared era current")
    if period != expected_period:
        raise PreparedExtrinsicError("prepared mortal era differs from the compiled period")
    birth, scan_start, scan_end = _era_bounds(current, period)
    if scan_start > scan_end:
        raise PreparedExtrinsicError("prepared mortal era has already expired at preparation")
    nonce = _uint(unsigned["nonce"], label="prepared account nonce")
    tip = _uint(unsigned["tip"], label="prepared transaction tip")
    spec_version = _uint(unsigned["spec_version"], label="prepared runtime spec version")
    transaction_version = _uint(
        unsigned["transaction_version"], label="prepared transaction version"
    )
    if tip != expected_tip or unsigned["tip_asset_id"] is not None:
        raise PreparedExtrinsicError("prepared extrinsic carries an unauthorized tip or asset")
    raw_genesis = unsigned["genesis_hash"]
    raw_era_hash = unsigned["era_block_hash"]
    genesis = str(raw_genesis).lower()
    era_hash = str(raw_era_hash).lower()
    if (
        raw_genesis != genesis
        or raw_era_hash != era_hash
        or HEX_32.fullmatch(genesis) is None
        or genesis != expected_genesis_hash.lower()
        or HEX_32.fullmatch(era_hash) is None
        or spec_version != expected_spec_version
        or transaction_version != expected_transaction_version
    ):
        raise PreparedExtrinsicError("prepared extrinsic has the wrong chain/runtime identity")
    if unsigned["metadata_hash"] is not None:
        raise PreparedExtrinsicError(
            "prepared metadata-hash signing is outside the reviewed SDK11 operator profile"
        )
    # For payloads up to 256 bytes SDK signs the concatenation. Longer payloads are a 32-byte
    # blake2b digest; checking both cases proves no serialized component drifted after prepare.
    full_payload = call_data + included_extrinsic + included_signed
    expected_payload = (
        hashlib.blake2b(full_payload, digest_size=32).digest()
        if len(full_payload) > 256
        else full_payload
    )
    if payload != expected_payload:
        raise PreparedExtrinsicError("prepared signing payload differs from its serialized parts")
    payload_json = unsigned["payload_json"]
    payload_fields = {
        "address",
        "blockHash",
        "era",
        "genesisHash",
        "metadataHash",
        "method",
        "mode",
        "nonce",
        "signedExtensions",
        "specVersion",
        "tip",
        "transactionVersion",
        "version",
    }
    if not isinstance(payload_json, dict) or set(payload_json) != payload_fields:
        raise PreparedExtrinsicError("prepared Polkadot signer payload is not an object")
    required_payload = {
        "address": address,
        "blockHash": era_hash,
        "genesisHash": genesis,
        "method": unsigned["call_data"].lower(),
        "nonce": _big_endian_hex(nonce, 4),
        "specVersion": _big_endian_hex(spec_version, 4),
        "tip": _big_endian_hex(tip, 16),
        "transactionVersion": _big_endian_hex(transaction_version, 4),
        "era": _encoded_mortal_era(current, period),
    }
    if any(payload_json.get(key) != value for key, value in required_payload.items()):
        raise PreparedExtrinsicError("prepared Polkadot signer payload differs from authority")
    if payload_json.get("version") != 4:
        raise PreparedExtrinsicError(
            "prepared Polkadot signer payload is not the reviewed signed extrinsic version"
        )
    if payload_json.get("signedExtensions") != list(BITTENSOR_V11_SIGNED_EXTENSIONS):
        raise PreparedExtrinsicError("prepared signed-extension order differs from SDK11 authority")
    if payload_json.get("mode") != 0 or payload_json.get("metadataHash") is not None:
        raise PreparedExtrinsicError("prepared payload unexpectedly enables metadata hashing")
    expected_extrinsic_extensions = (
        bytes.fromhex(_encoded_mortal_era(current, period)[2:])
        + _scale_compact_uint(nonce, label="prepared account nonce")
        + _scale_compact_uint(tip, label="prepared transaction tip")
        + b"\x00"  # CheckMetadataHash::Disabled
    )
    expected_signed_extensions = (
        _little_endian_uint(spec_version, 4, label="prepared runtime spec version")
        + _little_endian_uint(
            transaction_version,
            4,
            label="prepared transaction version",
        )
        + bytes.fromhex(genesis[2:])
        + bytes.fromhex(era_hash[2:])
        + b"\x00"  # No metadata hash in the signed additional data.
    )
    if (
        included_extrinsic != expected_extrinsic_extensions
        or included_signed != expected_signed_extensions
    ):
        raise PreparedExtrinsicError(
            "prepared signed-extension bytes differ from nonce/era/tip/chain authority"
        )
    call_hash = hashlib.sha256(call_data).hexdigest()
    if expected_call_data_sha256 is not None and call_hash != expected_call_data_sha256:
        raise PreparedExtrinsicError("prepared call_data differs from the validated plan call")
    canonical_unsigned = canonical_json_bytes(dict(unsigned))
    return PreparedAuthority(
        prepared_sha256=hashlib.sha256(canonical_unsigned).hexdigest(),
        call_data_sha256=call_hash,
        call_data_blake2b_256="0x" + hashlib.blake2b(call_data, digest_size=32).hexdigest(),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        address=address,
        nonce=nonce,
        era_current=current,
        era_period=period,
        era_birth=birth,
        scan_start_block=scan_start,
        scan_end_block=scan_end,
        genesis_hash=genesis,
        era_block_hash=era_hash,
        spec_version=spec_version,
        transaction_version=transaction_version,
        tip=tip,
        metadata_hash_enabled=unsigned["metadata_hash"] is not None,
        public_key=public_key,
        payload=payload,
    )


def validate_era_chain_anchor(
    authority: PreparedAuthority,
    *,
    current_finalized_block_number: int,
    birth_block_hash: str,
) -> None:
    """Bind the prepared era to independent canonical finalized block-number/birth reads.

    SDK11 anchors a mortal payload at its era *birth* hash. ``era.current`` is the finalized head
    number observed during preparation; the caller separately records the canonical hash returned
    by ``chain_getBlockHash(era.current)``. It is not accepted here as faux authority because the
    unsigned payload does not contain that current-block hash. Birth hash *is* signed and therefore
    is validated here.
    """

    current_number = _uint(
        current_finalized_block_number,
        label="prepared current finalized block number",
    )
    birth_hash = str(birth_block_hash).lower()
    if (
        current_number != authority.era_current
        or HEX_32.fullmatch(birth_hash) is None
        or birth_hash != authority.era_block_hash
    ):
        raise PreparedExtrinsicError("prepared mortal era is not bound to canonical chain anchors")


def prepared_evidence(
    unsigned: Mapping[str, Any],
    authority: PreparedAuthority,
    *,
    identity: Mapping[str, Any],
    attempt: int,
) -> dict[str, Any]:
    attempt = _attempt(attempt)
    identity_value = _identity(identity, reserved=_PREPARED_RESERVED)
    return {
        "schema_version": PREPARED_SCHEMA_VERSION,
        **identity_value,
        "attempt": attempt,
        "status": "prepared_unsigned_before_signing",
        "unsigned": dict(unsigned),
        "prepared_sha256": authority.prepared_sha256,
        "call_data_sha256": authority.call_data_sha256,
        "call_data_blake2b_256": authority.call_data_blake2b_256,
        "payload_sha256": authority.payload_sha256,
        "nonce": authority.nonce,
        "era_current": authority.era_current,
        "era_period": authority.era_period,
        "era_birth": authority.era_birth,
        "scan_start_block": authority.scan_start_block,
        "scan_end_block": authority.scan_end_block,
        "tip": authority.tip,
        "metadata_hash_enabled": authority.metadata_hash_enabled,
        "genesis_hash": authority.genesis_hash,
        "era_block_hash": authority.era_block_hash,
        "runtime_spec_version": authority.spec_version,
        "transaction_version": authority.transaction_version,
    }


def validate_prepared_evidence(
    value: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    attempt: int,
    expected_address: str,
    expected_genesis_hash: str,
    expected_spec_version: int,
    expected_transaction_version: int,
    expected_period: int,
    expected_tip: int = 0,
    expected_call_data_sha256: str | None = None,
) -> tuple[dict[str, Any], PreparedAuthority]:
    """Re-prove a durable prepared document before signing, submitting, or recovery."""

    if not isinstance(value, Mapping):
        raise PreparedExtrinsicError("durable prepared evidence must be a mapping")
    attempt = _attempt(attempt)
    identity_value = _identity(identity, reserved=_PREPARED_RESERVED)
    unsigned = value.get("unsigned")
    if not isinstance(unsigned, dict):
        raise PreparedExtrinsicError("durable prepared evidence lacks its unsigned SDK document")
    authority = validate_unsigned_dict(
        unsigned,
        expected_address=expected_address,
        expected_genesis_hash=expected_genesis_hash,
        expected_spec_version=expected_spec_version,
        expected_transaction_version=expected_transaction_version,
        expected_period=expected_period,
        expected_tip=expected_tip,
        expected_call_data_sha256=expected_call_data_sha256,
    )
    expected = prepared_evidence(
        unsigned,
        authority,
        identity=identity_value,
        attempt=attempt,
    )
    if dict(value) != expected:
        raise PreparedExtrinsicError(
            "durable prepared evidence differs from its exact unsigned payload"
        )
    return unsigned, authority


def reconstruct_unsigned(bt: Any, unsigned: Mapping[str, Any]) -> Any:
    """Use SDK11's public round-trip and refuse a type or byte-shape drift."""

    cls = getattr(bt, "UnsignedExtrinsic", None)
    if cls is None or not callable(getattr(cls, "from_dict", None)):
        raise PreparedExtrinsicError("Bittensor SDK lacks the public UnsignedExtrinsic contract")
    try:
        restored = cls.from_dict(dict(unsigned))
        roundtrip = restored.to_dict()
    except Exception as exc:  # noqa: BLE001 - no private/manual reconstruction fallback is safe
        raise PreparedExtrinsicError(
            "Bittensor could not reconstruct the prepared extrinsic"
        ) from exc
    if type(restored) is not cls or roundtrip != dict(unsigned):
        raise PreparedExtrinsicError(
            "Bittensor UnsignedExtrinsic round-trip changed authority bytes"
        )
    return restored


def signature_evidence(
    signature: bytes | str,
    authority: PreparedAuthority,
    *,
    identity: Mapping[str, Any],
    attempt: int,
) -> dict[str, Any]:
    attempt = _attempt(attempt)
    identity_value = _identity(identity, reserved=_SIGNATURE_RESERVED)
    if isinstance(signature, str):
        raw = _hex_bytes(signature, label="prepared signature")
    elif isinstance(signature, bytes):
        raw = signature
    else:
        raise PreparedExtrinsicError("prepared signature is not bytes or 0x hex")
    if len(raw) != 64:
        raise PreparedExtrinsicError("prepared sr25519 signature must contain exactly 64 bytes")
    if not verify_sr25519(raw, authority.payload, authority.public_key):
        raise PreparedExtrinsicError("prepared sr25519 signature does not verify over the payload")
    return {
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        **identity_value,
        "attempt": attempt,
        "status": "signature_persisted_before_submit_claim",
        "prepared_sha256": authority.prepared_sha256,
        "payload_sha256": authority.payload_sha256,
        "signature_hex": "0x" + raw.hex(),
        "signature_sha256": hashlib.sha256(raw).hexdigest(),
        "scan_start_block": authority.scan_start_block,
        "scan_end_block": authority.scan_end_block,
        "nonce": authority.nonce,
    }


def validate_signature_evidence(
    value: Mapping[str, Any],
    authority: PreparedAuthority,
    *,
    identity: Mapping[str, Any],
    attempt: int,
) -> bytes:
    if not isinstance(value, Mapping):
        raise PreparedExtrinsicError("durable signature evidence must be a mapping")
    attempt = _attempt(attempt)
    identity_value = _identity(identity, reserved=_SIGNATURE_RESERVED)
    expected_keys = {
        "schema_version",
        *identity_value.keys(),
        "attempt",
        "status",
        "prepared_sha256",
        "payload_sha256",
        "signature_hex",
        "signature_sha256",
        "scan_start_block",
        "scan_end_block",
        "nonce",
    }
    raw = _hex_bytes(value.get("signature_hex"), label="durable prepared signature", exact_bytes=64)
    if (
        set(value) != expected_keys
        or value.get("schema_version") != SIGNATURE_SCHEMA_VERSION
        or any(value.get(key) != expected for key, expected in identity_value.items())
        or value.get("attempt") != attempt
        or value.get("status") != "signature_persisted_before_submit_claim"
        or value.get("prepared_sha256") != authority.prepared_sha256
        or value.get("payload_sha256") != authority.payload_sha256
        or value.get("signature_sha256") != hashlib.sha256(raw).hexdigest()
        or value.get("scan_start_block") != authority.scan_start_block
        or value.get("scan_end_block") != authority.scan_end_block
        or value.get("nonce") != authority.nonce
        or not verify_sr25519(raw, authority.payload, authority.public_key)
    ):
        raise PreparedExtrinsicError("durable signature differs from its exact prepared payload")
    return raw


def submission_claim_evidence(
    authority: PreparedAuthority,
    *,
    identity: Mapping[str, Any],
    attempt: int,
    signature_sha256: str,
) -> dict[str, Any]:
    attempt = _attempt(attempt)
    identity_value = _identity(identity, reserved=_SUBMISSION_RESERVED)
    if re.fullmatch(r"[0-9a-f]{64}", signature_sha256) is None:
        raise PreparedExtrinsicError("submission claim signature digest is malformed")
    return {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        **identity_value,
        "attempt": attempt,
        "status": "submit_signature_next_fallible_operation",
        "prepared_sha256": authority.prepared_sha256,
        "payload_sha256": authority.payload_sha256,
        "signature_sha256": signature_sha256,
        "nonce": authority.nonce,
        "scan_start_block": authority.scan_start_block,
        "scan_end_block": authority.scan_end_block,
        "next_fallible_operation": "bittensor.Subtensor.submit_signature",
    }


def validate_submission_claim_evidence(
    value: Mapping[str, Any],
    authority: PreparedAuthority,
    *,
    identity: Mapping[str, Any],
    attempt: int,
    signature_sha256: str,
) -> None:
    if not isinstance(value, Mapping):
        raise PreparedExtrinsicError("durable submission claim must be a mapping")
    attempt = _attempt(attempt)
    identity_value = _identity(identity, reserved=_SUBMISSION_RESERVED)
    expected = submission_claim_evidence(
        authority,
        identity=identity_value,
        attempt=attempt,
        signature_sha256=signature_sha256,
    )
    if dict(value) != expected:
        raise PreparedExtrinsicError(
            "durable submission claim differs from its prepared payload and signature"
        )


def decoded_scan_candidate(
    decoded: Mapping[str, Any],
    raw_extrinsic: str,
    *,
    block_hash: str,
    block_number: int,
    extrinsic_index: int,
    authority: PreparedAuthority,
    signature: bytes,
) -> dict[str, Any] | None:
    """Match the exact SDK runtime-432 decoded signed-extrinsic authority.

    The caller must decode *every* raw extrinsic in the mortal window and fail closed on a decode
    error before using an empty candidate list as a no-match proof. Unrelated signed extrinsics,
    including a concurrent identical call from the same account with a different signature, return
    ``None``. Once the exact durable signature is encountered, every nonce/era/tip/call field must
    also match or the scan is contradictory and fails closed.
    """

    if not isinstance(decoded, Mapping):
        raise PreparedExtrinsicError("SDK decoded extrinsic is not an object")
    raw = _hex_bytes(raw_extrinsic, label="raw finalized extrinsic")
    body_length = _scale_extrinsic_body_length(raw)
    expected_raw_hash = "0x" + hashlib.blake2b(raw, digest_size=32).hexdigest()
    decoded_hash = str(decoded.get("extrinsic_hash", "")).lower()
    if decoded_hash != expected_raw_hash:
        raise PreparedExtrinsicError("SDK decoded extrinsic hash differs from canonical raw bytes")
    # Unsigned inherents have no signer/signature/era/nonce fields and cannot be our submission.
    if "address" not in decoded and "signature" not in decoded:
        return None
    expected_fields = {
        "extrinsic_hash",
        "extrinsic_length",
        "address",
        "signature",
        "era",
        "nonce",
        "tip",
        "mode",
        "call",
    }
    if set(decoded) != expected_fields:
        raise PreparedExtrinsicError("SDK decoded signed extrinsic has an unaudited field set")
    if _uint(decoded.get("extrinsic_length"), label="decoded extrinsic length") != body_length:
        raise PreparedExtrinsicError(
            "SDK decoded extrinsic length differs from canonical raw SCALE bytes"
        )
    decoded_signature = decoded.get("signature")
    if not isinstance(decoded_signature, Mapping) or set(decoded_signature) != {"Sr25519"}:
        # A different crypto scheme is unrelated unless it claims our exact signer, in which case
        # the runtime decode contradicts the prepared sr25519 authority.
        if decoded.get("address") == authority.address:
            raise PreparedExtrinsicError("prepared signer appeared with a non-sr25519 signature")
        return None
    raw_signature = _hex_bytes(
        decoded_signature["Sr25519"],
        label="decoded sr25519 signature",
        exact_bytes=64,
    )
    if raw_signature != signature:
        return None
    call = decoded.get("call")
    if not isinstance(call, Mapping):
        raise PreparedExtrinsicError("exact prepared signature has no decoded call")
    call_hash = str(call.get("call_hash", "")).lower()
    era = decoded.get("era")
    if not isinstance(era, (tuple, list)) or len(era) != 2:
        raise PreparedExtrinsicError("exact prepared signature has no decoded mortal era")
    era_period = _uint(era[0], label="decoded mortal era period")
    era_phase = _uint(era[1], label="decoded mortal era phase")
    mode = decoded.get("mode")
    expected_mode = {"mode": "Enabled" if authority.metadata_hash_enabled else "Disabled"}
    if (
        decoded.get("address") != authority.address
        or _uint(decoded.get("nonce"), label="decoded account nonce") != authority.nonce
        or era_period != authority.era_period
        or era_phase != _era_phase(authority.era_current, authority.era_period)
        or _uint(decoded.get("tip"), label="decoded transaction tip") != authority.tip
        or mode != expected_mode
        or call_hash != authority.call_data_blake2b_256
    ):
        raise PreparedExtrinsicError(
            "exact prepared signature differs from signer/nonce/era/tip/call authority"
        )
    normalized_block_hash = str(block_hash).lower()
    if HEX_32.fullmatch(normalized_block_hash) is None:
        raise PreparedExtrinsicError("prepared-era candidate block hash is malformed")
    block_number = _uint(block_number, label="candidate block number")
    extrinsic_index = _uint(extrinsic_index, label="candidate extrinsic index")
    return {
        "block_hash": normalized_block_hash,
        "block_number": block_number,
        "extrinsic_hash": expected_raw_hash,
        "extrinsic_index": extrinsic_index,
        "signer": authority.address,
        "signature_sha256": hashlib.sha256(signature).hexdigest(),
        "nonce": authority.nonce,
        "era_period": authority.era_period,
        "era_phase": _era_phase(authority.era_current, authority.era_period),
        "call_data_sha256": authority.call_data_sha256,
        "call_data_blake2b_256": authority.call_data_blake2b_256,
    }


def _scan_candidates(
    value: Any,
    authority: PreparedAuthority,
    *,
    signature_sha256: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise PreparedExtrinsicError("prepared-era scan candidates must be an array")
    normalized: list[dict[str, Any]] = []
    coordinates: set[tuple[int, int]] = set()
    extrinsic_hashes: set[str] = set()
    for candidate in value:
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "block_hash",
            "block_number",
            "extrinsic_hash",
            "extrinsic_index",
            "signer",
            "signature_sha256",
            "nonce",
            "era_period",
            "era_phase",
            "call_data_sha256",
            "call_data_blake2b_256",
        }:
            raise PreparedExtrinsicError("prepared-era scan candidate has an unaudited shape")
        block_hash = str(candidate["block_hash"]).lower()
        extrinsic_hash = str(candidate["extrinsic_hash"]).lower()
        block_number = _uint(candidate["block_number"], label="candidate block number")
        extrinsic_index = _uint(candidate["extrinsic_index"], label="candidate extrinsic index")
        nonce = _uint(candidate["nonce"], label="candidate nonce")
        era_period = _uint(candidate["era_period"], label="candidate era period")
        era_phase = _uint(candidate["era_phase"], label="candidate era phase")
        expected_phase = _era_phase(authority.era_current, authority.era_period)
        if HEX_32.fullmatch(block_hash) is None or HEX_32.fullmatch(extrinsic_hash) is None:
            raise PreparedExtrinsicError("prepared-era scan candidate has a malformed chain hash")
        if (
            candidate["signer"] != authority.address
            or candidate["signature_sha256"] != signature_sha256
            or nonce != authority.nonce
            or era_period != authority.era_period
            or era_phase != expected_phase
            or candidate["call_data_sha256"] != authority.call_data_sha256
            or candidate["call_data_blake2b_256"] != authority.call_data_blake2b_256
        ):
            raise PreparedExtrinsicError(
                "prepared-era candidate differs from exact signer/signature/nonce/era/call authority"
            )
        if not authority.scan_start_block <= block_number <= authority.scan_end_block:
            raise PreparedExtrinsicError("prepared-era scan candidate is outside the mortal era")
        coordinate = (block_number, extrinsic_index)
        if coordinate in coordinates or extrinsic_hash in extrinsic_hashes:
            raise PreparedExtrinsicError("prepared-era scan candidate is duplicated")
        coordinates.add(coordinate)
        extrinsic_hashes.add(extrinsic_hash)
        normalized.append(
            {
                "block_hash": block_hash,
                "block_number": block_number,
                "extrinsic_hash": extrinsic_hash,
                "extrinsic_index": extrinsic_index,
                "signer": authority.address,
                "signature_sha256": signature_sha256,
                "nonce": nonce,
                "era_period": era_period,
                "era_phase": era_phase,
                "call_data_sha256": authority.call_data_sha256,
                "call_data_blake2b_256": authority.call_data_blake2b_256,
            }
        )
    ordered = sorted(
        normalized,
        key=lambda item: (item["block_number"], item["extrinsic_index"], item["extrinsic_hash"]),
    )
    if normalized != ordered:
        raise PreparedExtrinsicError("prepared-era scan candidates are not canonically ordered")
    return normalized


def scan_resolution_evidence(
    authority: PreparedAuthority,
    *,
    identity: Mapping[str, Any],
    attempt: int,
    signature_sha256: str,
    finalized_head: str,
    finalized_block_number: int,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record the terminal result of scanning every finalized block in one mortal era.

    Candidate selection is intentionally call-specific: its producer must require the exact
    signer, signature and decoded call/event authority.  This shared record proves the scan was
    bounded to the prepared era and encodes the only condition that may authorize attempt two.
    """

    attempt = _attempt(attempt)
    identity_value = _identity(identity, reserved=_RESOLUTION_RESERVED)
    if re.fullmatch(r"[0-9a-f]{64}", signature_sha256) is None:
        raise PreparedExtrinsicError("prepared-era scan signature digest is malformed")
    head = str(finalized_head).lower()
    head_number = _uint(finalized_block_number, label="prepared-era finalized block number")
    if HEX_32.fullmatch(head) is None or head_number <= authority.scan_end_block:
        raise PreparedExtrinsicError(
            "prepared-era scan is not complete beyond the mortal era's final block"
        )
    normalized = _scan_candidates(
        candidates,
        authority,
        signature_sha256=signature_sha256,
    )
    if not normalized:
        status = "expired_exact_no_match"
        next_attempt = attempt < MAX_PREPARED_ATTEMPTS
    elif len(normalized) == 1:
        status = "finalized_exact_match"
        next_attempt = False
    else:
        status = "multiple_exact_matches_terminal"
        next_attempt = False
    return {
        "schema_version": RESOLUTION_SCHEMA_VERSION,
        **identity_value,
        "attempt": attempt,
        "status": status,
        "terminal_for_attempt": True,
        "next_attempt_authorized": next_attempt,
        "prepared_sha256": authority.prepared_sha256,
        "payload_sha256": authority.payload_sha256,
        "signature_sha256": signature_sha256,
        "nonce": authority.nonce,
        "scan_start_block": authority.scan_start_block,
        "scan_end_block": authority.scan_end_block,
        "scan_complete_through_block": authority.scan_end_block,
        "scanned_block_count": authority.scan_end_block - authority.scan_start_block + 1,
        "finalized_head": head,
        "finalized_block_number": head_number,
        "candidate_count": len(normalized),
        "candidates": normalized,
    }


def validate_scan_resolution_evidence(
    value: Mapping[str, Any],
    authority: PreparedAuthority,
    *,
    identity: Mapping[str, Any],
    attempt: int,
    signature_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparedExtrinsicError("durable prepared-era scan evidence must be a mapping")
    expected = scan_resolution_evidence(
        authority,
        identity=identity,
        attempt=attempt,
        signature_sha256=signature_sha256,
        finalized_head=str(value.get("finalized_head", "")),
        finalized_block_number=value.get("finalized_block_number"),
        candidates=value.get("candidates"),
    )
    if dict(value) != expected:
        raise PreparedExtrinsicError("durable prepared-era scan differs from exact authority")
    return expected


def validate_next_attempt_authority(
    prior_resolution: Mapping[str, Any],
    prior_authority: PreparedAuthority,
    *,
    identity: Mapping[str, Any],
    prior_attempt: int,
    signature_sha256: str,
    next_attempt: int,
) -> None:
    """Permit the sole recovery reprepare only after an exact expired zero-match scan."""

    resolution = validate_scan_resolution_evidence(
        prior_resolution,
        prior_authority,
        identity=identity,
        attempt=prior_attempt,
        signature_sha256=signature_sha256,
    )
    if (
        prior_attempt != 1
        or next_attempt != 2
        or resolution["status"] != "expired_exact_no_match"
        or resolution["candidate_count"] != 0
        or resolution["next_attempt_authorized"] is not True
    ):
        raise PreparedExtrinsicError(
            "a second prepared attempt requires attempt one's exact expired zero-match scan"
        )


def validate_reprepared_attempt(
    prior_authority: PreparedAuthority,
    next_authority: PreparedAuthority,
) -> None:
    """Prove attempt two is the same mutation freshly prepared after attempt one's expiry."""

    if (
        next_authority.address != prior_authority.address
        or next_authority.call_data_sha256 != prior_authority.call_data_sha256
        or next_authority.genesis_hash != prior_authority.genesis_hash
        or next_authority.spec_version != prior_authority.spec_version
        or next_authority.transaction_version != prior_authority.transaction_version
        or next_authority.era_period != prior_authority.era_period
        or next_authority.tip != prior_authority.tip
    ):
        raise PreparedExtrinsicError(
            "second prepared attempt differs from the original signer/call/chain authority"
        )
    if (
        next_authority.era_current <= prior_authority.scan_end_block
        or next_authority.scan_start_block <= prior_authority.scan_end_block
        or next_authority.prepared_sha256 == prior_authority.prepared_sha256
        or next_authority.payload_sha256 == prior_authority.payload_sha256
    ):
        raise PreparedExtrinsicError(
            "second prepared attempt is not fresh beyond the original mortal era"
        )
