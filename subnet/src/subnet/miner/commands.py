"""Miner CLI command bodies (plan §1.8): `validate`, `balance`, `deposit`.

CONTRACT (capsule header)
- `validate`: check every staged item against the PUBLISHED config_schema (never a lane import);
  invalid configs are surfaced and the command exits non-zero (a pre-submission hygiene gate).
- `balance`: a PUBLIC affordability read (`/api/can_afford`) + the deposit address — the miner
  never authenticates (only validators hold sessions) and never charges (invariant 1).
- `deposit`: the ONLY command that unlocks the coldkey, and it uses it LOCALLY — resolves the
  work-token deposit address and reports the transfer the operator must make on-chain. The coldkey
  is never placed in a network payload. The actual on-chain transfer runs under the `chain` profile.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlencode

from subnet.miner import config_schema
from subnet.miner.staging import StagingPool
from subnet.miner.wallet import load_coldkey


def _central() -> str:
    return os.environ.get("CENTRAL_API_URL", "http://central-api:4000")


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{_central()}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode())


def run_validate(task_type_default: str = "synthetic") -> int:
    """Validate every staged item against the published schema for its task type. Exit 1 if any
    item is invalid (or a schema is unavailable), 0 if all pass (or the pool is empty)."""
    pool = StagingPool()
    items = pool.items()
    if not items:
        print(f"[aurelius-miner] validate: no staged items in {pool.path}", flush=True)
        return 0

    schemas: dict[str, dict] = {}
    invalid = 0
    for i, item in enumerate(items):
        task_type = item.get("task_type", task_type_default)
        try:
            schema = schemas.get(task_type) or schemas.setdefault(
                task_type, config_schema.load_published_schema(task_type)
            )
        except config_schema.SchemaUnavailable as e:
            print(f"[aurelius-miner] validate: item {i}: schema unavailable: {e}", flush=True)
            invalid += 1
            continue
        published_version = schema.get("schema_version")
        if (
            type(published_version) is not int
            or type(item.get("schema_version")) is not int
            or item["schema_version"] != published_version
        ):
            print(
                f"[aurelius-miner] validate: item {i} INVALID: schema_version "
                f"{item.get('schema_version')!r} != published {published_version!r}",
                flush=True,
            )
            invalid += 1
            continue
        errors = config_schema.validate_config(item.get("config", {}), schema)
        if errors:
            invalid += 1
            print(f"[aurelius-miner] validate: item {i} INVALID: {'; '.join(errors)}", flush=True)
        else:
            print(f"[aurelius-miner] validate: item {i} ok ({task_type})", flush=True)

    print(f"[aurelius-miner] validate: {len(items) - invalid}/{len(items)} valid", flush=True)
    return 1 if invalid else 0


def run_balance(hotkey_ss58: str, task_type: str = "synthetic") -> int:
    """Report affordability (public read) + the deposit address. No session, no coldkey."""
    try:
        query = urlencode({"hotkey_ss58": hotkey_ss58, "task_type": task_type})
        afford = _get(f"/api/can_afford?{query}")
        can = afford.get("can_afford")
        if type(can) is not bool:
            raise ValueError("invalid affordability response")
    except (urllib.error.URLError, OSError, ValueError):
        can = None  # unknown is distinct from insufficient; neither authorizes work
    try:
        address = _get("/api/deposit_address").get("address", "<unavailable>")
    except (urllib.error.URLError, OSError, ValueError):
        address = "<unavailable>"

    print(
        f"[aurelius-miner] balance: hotkey={hotkey_ss58} can_afford({task_type})={can if can is not None else 'unavailable'}; "
        f"deposit work-tokens to {address}",
        flush=True,
    )
    return 2 if can is None else 0


def run_readiness(task_type: str, fanout_window: int) -> int:
    """Local candidate readiness only; never a claim about chain discovery or funding."""
    from subnet.miner.staging import item_id

    doc = {
        "ready": False,
        "scope": "local_candidate",
        "task_type": task_type,
        "fanout_window": fanout_window,
        "item_id": None,
    }
    try:
        pool = StagingPool()
        if not pool.active_items():
            doc["reason"] = "empty_or_disabled_pool"
        else:
            schema = config_schema.load_published_schema(task_type)
            item = pool.first_valid(
                task_type=task_type,
                schema_version=schema["schema_version"],
                fanout_window=fanout_window,
                validate=lambda cfg: config_schema.validate_config(cfg, schema),
            )
            if item is not None:
                pool.pin_selection(item, fanout_window, persist=False)
            doc.update(
                ready=item is not None,
                item_id=item_id(item) if item else None,
                reason="eligible" if item else "no_eligible_candidate",
            )
    except (OSError, ValueError, KeyError, config_schema.SchemaUnavailable) as exc:
        doc["reason"] = str(exc)
    print(json.dumps(doc, sort_keys=True))
    return 0 if doc["ready"] else 1


def run_observe(
    central: str,
    *,
    netuid: int,
    journal=None,
    limit: int = 20,
    validators: list[str] | None = None,
) -> int:
    """Lane-agnostic refresh of remote submission observations into the local journal.

    Signs only with the serving hotkey (fresh proof per read, never a session); records
    ``charge_observation`` from central and ``validator_progress`` per validator endpoint.
    Unavailable evidence is journaled as ``status_unavailable`` — absence stays unknown,
    never an inferred verdict. No candidate, payment, or coldkey interaction.
    """
    import sys

    from subnet.miner.journal import AttemptJournal
    from subnet.miner.observation_client import observe, progress
    from subnet.miner.serve import minter_from_env
    from subnet.miner.trust import config_signing_pubkey

    try:
        public_key = config_signing_pubkey()
    except ValueError as exc:
        print(
            f"[aurelius-miner] observe: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if validators and any("=" not in endpoint for endpoint in validators):
        print(
            "[aurelius-miner] observe: --validator endpoints must be HOTKEY=URL",
            file=sys.stderr,
            flush=True,
        )
        return 2
    minter = minter_from_env(os.environ.get("MINER_HOTKEY", "miner-hotkey"))
    book = AttemptJournal(journal)
    results = []
    for attempt in book.attempts(limit=limit):
        identity = attempt["identity"]
        row = {"attempt_id": attempt["attempt_id"]}
        if identity["hotkey"] != minter.hotkey_ss58:
            row["accounting"] = {"status": "different_hotkey"}
            results.append(row)
            continue
        try:
            facts = observe(
                central,
                identity,
                netuid=netuid,
                signer=minter.sign_observation,
                config_pubkey=public_key,
            )
            book.record(identity, "charge_observation", "central", facts)
            row["accounting"] = {key: facts[key] for key in ("charge", "refund", "observed_at")}
            row["validators"] = facts["validators"]
        except (ValueError, RuntimeError, OSError) as exc:
            book.record(identity, "status_unavailable", "central", {"code": str(exc)[:100]})
            row["accounting"] = {"status": "unavailable"}
        for endpoint in validators or []:
            validator, _, url = endpoint.partition("=")
            try:
                facts = progress(
                    url,
                    identity,
                    netuid=netuid,
                    validator_hotkey=validator,
                    signer=minter.sign_observation,
                    config_pubkey=public_key,
                )
                book.record(identity, "validator_progress", "validator", facts)
                row.setdefault("progress", []).append(
                    {"validator_hotkey": validator, "observed_at": facts["observed_at"]}
                )
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                book.record(
                    identity,
                    "status_unavailable",
                    "validator",
                    {"validator_hotkey": validator, "code": str(exc)[:100]},
                )
                row.setdefault("progress", []).append(
                    {"validator_hotkey": validator, "status": "unavailable"}
                )
        results.append(row)
    print(
        json.dumps(
            {"schema_version": 1, "authority": "observations_only", "refresh": results},
            sort_keys=True,
        )
    )
    return 0


def run_deposit(coldkey_name: str, *, unlocked: bool) -> int:
    """Resolve the deposit address and report the transfer to make. Unlocks the coldkey LOCALLY
    (never sent). The on-chain transfer itself runs under the `chain` profile."""
    from subnet.miner.wallet import ColdkeyLocked

    try:
        _pair, coldkey_ss58 = load_coldkey(coldkey_name, unlocked=unlocked)
    except ColdkeyLocked as e:
        print(f"[aurelius-miner] deposit: {e}", flush=True)
        return 2

    try:
        address = _get("/api/deposit_address").get("address", "<unavailable>")
    except (urllib.error.URLError, OSError, ValueError):
        address = "<unavailable>"

    print(
        f"[aurelius-miner] deposit: from coldkey {coldkey_ss58} (unlocked locally) -> "
        f"work-token wallet {address}. The on-chain transfer_stake runs under the `chain` profile; "
        "the coldkey signs it LOCALLY and is never sent to central.",
        flush=True,
    )
    return 0
