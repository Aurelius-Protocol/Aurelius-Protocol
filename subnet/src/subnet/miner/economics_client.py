"""Read own economics with hotkey authority and independently checked signed prices."""

import secrets
import time
from .read_client import ReadUnavailable, request_json
from subnet.shared.config_snapshot import verify_snapshot
from subnet.shared.miner_economics import signed_bytes
from subnet.shared.price_policy import validate_entries, resolve, shipped_catalog
from subnet.shared.ss58 import ss58_decode


def read_economics(base_url, *, hotkey, netuid, signer, config_pubkey, fetch=request_json):
    snapshot = fetch(base_url.rstrip("/") + "/api/config")
    verified = verify_snapshot(snapshot, config_pubkey)
    entries = validate_entries(verified["entries"])
    body = dict(
        hotkey=hotkey,
        netuid=netuid,
        config_hash=snapshot["canonical_hash"],
        timestamp=int(time.time()),
        nonce=secrets.token_hex(32),
    )
    signature = signer(signed_bytes(body))
    if type(signature) is not bytes or len(signature) != 64:
        raise ReadUnavailable("invalid_hotkey_signature")
    response = fetch(
        base_url.rstrip("/") + "/api/miner/economics",
        body={**body, "signature_hex": signature.hex()},
    )
    try:
        if (
            type(response) is not dict
            or set(response)
            != {
                "schema_version",
                "identity",
                "config_hash",
                "finalized_block",
                "observed_at",
                "pricing",
                "account",
                "capital",
                "purchase_conversion",
                "weights",
                "emissions",
            }
            or type(response["schema_version"]) is not int
            or response["schema_version"] != 1
            or response["config_hash"] != body["config_hash"]
            or set(response["identity"]) != {"coldkey", "hotkey", "netuid"}
            or response["identity"]["hotkey"] != hotkey
            or response["identity"]["netuid"] != netuid
        ):
            raise ValueError("scope")
        ss58_decode(response["identity"]["coldkey"])
        for field in ("finalized_block", "observed_at"):
            if type(response[field]) is not int or response[field] < 0:
                raise ValueError("coordinate")
        account = response["account"]
        balance = account["balance"]
        receipts = account["receipts"]
        if (
            set(account) != {"balance", "receipts"}
            or set(balance)
            != {
                "available_real",
                "available_synthetic",
                "deposited_lifetime",
                "consumed_lifetime",
                "synthetic_granted_lifetime",
                "synthetic_consumed_lifetime",
            }
            or any(type(value) is not int or value < 0 for value in balance.values())
            or set(receipts)
            != {"real", "synthetic", "more_real", "more_synthetic", "limit_per_ledger"}
            or receipts["limit_per_ledger"] != 25
        ):
            raise ValueError("account")
        for ledger in ("real", "synthetic"):
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
                ):
                    raise ValueError("receipt")
        pricing = response["pricing"]
        document = entries.get("price_policy")
        if pricing["announced"] != document:
            raise ValueError("announced policy")
        if document is None:
            if pricing != {"status": "unpriced", "announced": None, "active": None}:
                raise ValueError("unpriced")
        elif response["finalized_block"] < document["current"]["effective_from_block"]:
            if pricing != {"status": "pending", "announced": document, "active": None}:
                raise ValueError("pending")
        else:
            expected = resolve(document, response["finalized_block"], shipped_catalog())
            if pricing != {
                "status": "available",
                "announced": document,
                "active": expected,
                "purpose": "published_policy_preview",
            }:
                raise ValueError("active policy")
        for field in ("capital", "purchase_conversion", "weights", "emissions"):
            if (
                type(response[field]) is not dict
                or set(response[field]) != {"status", "reason"}
                or response[field]["status"]
                != ("not_observed" if field == "emissions" else "unavailable")
                or type(response[field]["reason"]) is not str
            ):
                raise ValueError("unsupported economic claim")
    except (TypeError, ValueError, KeyError) as exc:
        raise ReadUnavailable("invalid_economics") from exc
    return response
