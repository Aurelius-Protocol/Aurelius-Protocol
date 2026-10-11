"""Read own exact charge/refund facts with a fresh hotkey proof; never opens a session."""

import secrets
import time

from .read_client import ReadUnavailable, request_json
from subnet.shared.config_snapshot import verify_snapshot
from subnet.shared.miner_observation import signed_bytes


def observe(
    base_url: str, identity: dict, *, netuid: int, signer, config_pubkey: str, fetch=request_json
) -> dict:
    snapshot = fetch(base_url.rstrip("/") + "/api/config")
    verify_snapshot(snapshot, config_pubkey)
    body = {
        "hotkey": identity["hotkey"],
        "netuid": netuid,
        "content_hash": identity["content_hash"],
        "fanout_window": identity["fanout_window"],
        "config_hash": snapshot["canonical_hash"],
        "timestamp": int(time.time()),
        "nonce": secrets.token_hex(32),
    }
    signature = signer(signed_bytes(body))
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ValueError("invalid hotkey signature")
    response = fetch(
        base_url.rstrip("/") + "/api/miner/observation",
        body={**body, "signature_hex": signature.hex()},
    )
    expected = {key: body[key] for key in ("hotkey", "netuid", "content_hash", "fanout_window")}
    observation = response.get("observation", {})
    if (
        response.get("schema_version") != 1
        or response.get("identity") != expected
        or not isinstance(observation, dict)
        or set(observation) != {"charge", "refund", "observed_at"}
        or not isinstance(observation["charge"], str)
        or not isinstance(observation["refund"], str)
        or observation["charge"] not in {"recorded", "absent"}
        or observation["refund"] not in {"recorded", "not_observed"}
        or not isinstance(observation["observed_at"], str)
    ):
        raise ReadUnavailable("invalid_observation")
    validators = response.get("validators")
    if (
        not isinstance(validators, list)
        or len(validators) > 256
        or any(
            not isinstance(row, dict)
            or set(row) != {"id", "validator_hotkey", "verdict", "inserted_at"}
            or any(not isinstance(row[key], str) for key in row)
            for row in validators
        )
    ):
        raise ReadUnavailable("invalid_validator_observation")
    return {**observation, "validators": validators}


def progress(
    base_url: str,
    identity: dict,
    *,
    netuid: int,
    validator_hotkey: str,
    signer,
    config_pubkey: str,
    fetch=request_json,
) -> dict:
    snapshot = fetch(base_url.rstrip("/") + "/api/config")
    verify_snapshot(snapshot, config_pubkey)
    body = {
        "hotkey": identity["hotkey"],
        "netuid": netuid,
        "content_hash": identity["content_hash"],
        "fanout_window": identity["fanout_window"],
        "config_hash": snapshot["canonical_hash"],
        "timestamp": int(time.time()),
        "nonce": secrets.token_hex(32),
    }
    signature = signer(signed_bytes(body))
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ValueError("invalid hotkey signature")
    response = fetch(
        base_url.rstrip("/") + "/api/miner/progress",
        body={**body, "signature_hex": signature.hex()},
    )
    expected = {key: body[key] for key in ("hotkey", "netuid", "content_hash", "fanout_window")}
    rows = response.get("generations")
    states = {
        "attempt_created",
        "novelty_reserved",
        "charge_requested",
        "charge_recorded_locally",
        "novelty_committed",
        "result_recorded",
        "submitted_to_server",
        "finalized",
        "discarded",
        "unknown",
    }
    fields = {
        "generation",
        "state",
        "retryable_precharge",
        "refusal",
        "evaluation",
        "refund_review",
        "created_at",
        "updated_at",
    }
    if (
        response.get("schema_version") != 1
        or response.get("identity") != expected
        or response.get("validator_hotkey") != validator_hotkey
        or type(response.get("observed_at")) is not int
        or not isinstance(rows, list)
        or len(rows) > 8
    ):
        raise ReadUnavailable("invalid_progress")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != fields
            or any(
                type(row[key]) is not int or row[key] < 0
                for key in ("generation", "created_at", "updated_at")
            )
            or type(row["retryable_precharge"]) is not bool
            or row["state"] not in states
            or row["refusal"]
            not in {
                None,
                "fanout_window_mismatch",
                "foreign_charge_already_committed",
                "refused_before_charge",
                "discarded",
            }
            or row["evaluation"] not in {"pending", "dispatched", "not_observed"}
            or row["refund_review"] not in {"requested", "not_observed"}
        ):
            raise ReadUnavailable("invalid_progress")
    return response
