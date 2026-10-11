"""Authenticated SN37 funding terms. Server data cannot replace shipped trust roots."""

import re
import secrets
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from subnet.miner.read_client import ReadUnavailable, request_json
from subnet.miner.trust import SN37_RECEIPT_SIGNING_PUBKEY
from subnet.shared.canonicalization import canonical_bytes
from subnet.shared.config_snapshot import SnapshotInvalid, verify_snapshot
from subnet.shared.miner_economics import signed_bytes
from subnet.shared.price_policy import resolve, shipped_catalog, validate_entries
from subnet.shared.sn37_economics import GENESIS_HASH, TARIFF_VERSION
from subnet.shared.ss58 import ss58_decode

TREASURY = "5Gx14QffqwC8wNHv4wUvfCfE2zAUYDNvF9Z7LjNY81WQx7iL"
CHAIN_ID = "bittensor-mainnet:" + GENESIS_HASH[2:]


def read_funding(base_url, *, hotkey, signer, config_pubkey, fetch=request_json, now=None):
    now = int(time.time()) if now is None else now
    snapshot = fetch(base_url.rstrip("/") + "/api/config")
    try:
        entries = validate_entries(verify_snapshot(snapshot, config_pubkey)["entries"])
    except (SnapshotInvalid, ValueError, KeyError, TypeError) as exc:
        raise ReadUnavailable("invalid_funding_config") from exc
    body = dict(
        hotkey=hotkey,
        netuid=37,
        config_hash=snapshot["canonical_hash"],
        timestamp=now,
        nonce=secrets.token_hex(32),
    )
    signature = signer(signed_bytes(body))
    if type(signature) is not bytes or len(signature) != 64:
        raise ReadUnavailable("invalid_hotkey_signature")
    result = fetch(
        base_url.rstrip("/") + "/api/miner/funding", body={**body, "signature_hex": signature.hex()}
    )
    try:
        if (
            type(result) is not dict
            or set(result)
            != {
                "schema_version",
                "identity",
                "config_hash",
                "finalized_block",
                "observed_at",
                "terms",
                "account",
            }
            or type(result["schema_version"]) is not int
            or result["schema_version"] != 1
            or result["config_hash"] != body["config_hash"]
            or set(result["identity"]) != {"coldkey", "hotkey", "netuid"}
            or result["identity"]["hotkey"] != hotkey
            or type(result["identity"]["netuid"]) is not int
            or result["identity"]["netuid"] != 37
        ):
            raise ValueError("scope")
        ss58_decode(result["identity"]["coldkey"])
        if (
            type(result["finalized_block"]) is not int
            or result["finalized_block"] < 0
            or type(result["observed_at"]) is not int
            or abs(now - result["observed_at"]) > 60
        ):
            raise ValueError("stale funding response")
        terms = result["terms"]
        if terms["status"] == "unavailable":
            if set(terms) != {"status", "reason"} or not re.fullmatch(
                r"[a-z_]{1,128}", terms["reason"]
            ):
                raise ValueError("unavailable")
        else:
            active = resolve(entries["price_policy"], result["finalized_block"], shipped_catalog())
            policy = active["policy"]
            if (
                policy.get("schema_version") != 2
                or policy["capital"] != {"mode": "none"}
                or policy["conversion"] != {"alpha_rao_per_work_token": 1_000_000_000}
                or policy["lanes"]["interp_sae"]["gpu_sae_verdict"]
                != {"cost_model_version": 2, "work_tokens": 10}
                or canonical_bytes(terms)
                != canonical_bytes(
                    {
                        "status": "available",
                        "chain_id": CHAIN_ID,
                        "genesis_hash": GENESIS_HASH,
                        "netuid": 37,
                        "destination_coldkey": TREASURY,
                        "tariff_version": TARIFF_VERSION,
                        "price_policy_digest": active["policy_digest"],
                        "base_alpha_rao_per_work_token": 1_000_000_000,
                        "evaluation_work_tokens": 10,
                        "retention": "quadratic_fresh_lots_v3",
                        "fractional_carry": "beneficiary",
                        "refundable": False,
                        "runtime_spec_version": 475,
                        "transaction_version": 1,
                    }
                )
                or resolve(
                    entries["price_policy"], result["finalized_block"] + 64, shipped_catalog()
                )["policy_digest"]
                != active["policy_digest"]
            ):
                raise ValueError("unsigned funding policy")
        account = result["account"]
        if (
            type(account) is not dict
            or set(account) != {"balance", "receipts"}
            or set(account["balance"])
            != {
                "available_real",
                "available_synthetic",
                "deposited_lifetime",
                "consumed_lifetime",
                "synthetic_granted_lifetime",
                "synthetic_consumed_lifetime",
            }
            or any(type(value) is not int or value < 0 for value in account["balance"].values())
            or set(account["receipts"])
            != {"real", "synthetic", "more_real", "more_synthetic", "limit_per_ledger"}
            or type(account["receipts"]["limit_per_ledger"]) is not int
            or account["receipts"]["limit_per_ledger"] != 25
        ):
            raise ValueError("account")
        for ledger in ("real", "synthetic"):
            receipts = account["receipts"]
            if (
                type(receipts["more_" + ledger]) is not bool
                or type(receipts[ledger]) is not list
                or len(receipts[ledger]) > 25
            ):
                raise ValueError("receipts")
            for row in receipts[ledger]:
                if (
                    type(row) is not dict
                    or set(row)
                    != {"id", "kind", "amount", "receipt", "signature_hex", "key_id", "inserted_at"}
                    or type(row["receipt"]) is not dict
                    or type(row["amount"]) is not int
                    or type(row["signature_hex"]) is not str
                ):
                    raise ValueError("receipt row")
    except (KeyError, TypeError, ValueError) as exc:
        raise ReadUnavailable("invalid_funding_terms") from exc
    return result


def verified_deposit(response, deposit_key, *, receipt_pubkey=SN37_RECEIPT_SIGNING_PUBKEY):
    """Return a signed own-account v4 deposit receipt, never infer credit from balance changes."""
    for row in response["account"]["receipts"]["real"]:
        if type(row) is not dict or type(row.get("receipt")) is not dict:
            raise ReadUnavailable("invalid_deposit_receipt")
        receipt = row.get("receipt", {})
        if receipt.get("deposit_key") != deposit_key:
            continue
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(receipt_pubkey)).verify(
                bytes.fromhex(row["signature_hex"]), canonical_bytes(receipt)
            )
            identity = response["identity"]
            if (
                type(receipt["receipt_version"]) is not int
                or receipt["receipt_version"] != 4
                or receipt["kind"] != "deposit"
                or any(receipt[key] != identity[key] for key in ("coldkey", "hotkey", "netuid"))
                or receipt["genesis_hash"] != GENESIS_HASH
                or receipt["tariff_version"] != TARIFF_VERSION
                or receipt["rate_version"] != TARIFF_VERSION
                or receipt["base_work_tokens_per_alpha"] != "1"
                or any(
                    type(receipt.get(key)) is not str
                    or not re.fullmatch(r"[0-9a-f]{64}", receipt[key])
                    for key in (
                        "price_policy_digest",
                        "commitment_policy_digest",
                        "commitment_state_digest",
                    )
                )
                or any(
                    type(receipt.get(key)) is not str
                    or len(receipt[key]) > 2100
                    or not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", receipt[key])
                    for key in ("carry_before", "carry_after", "retention_multiple")
                )
                or receipt["denomination"] != "alpha"
                or type(receipt["tokens_credited"]) is not int
                or receipt["tokens_credited"] < 0
                or type(receipt["alpha_amount_rao"]) is not int
                or receipt["alpha_amount_rao"] <= 0
            ):
                raise ValueError("receipt scope")
        except Exception as exc:
            raise ReadUnavailable("invalid_deposit_receipt") from exc
        return receipt
    return None
