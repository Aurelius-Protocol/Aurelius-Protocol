"""Local append-only observation journal (spec §15). No authority over money or weights.

Attempt identity includes the signed content hash and window. Each event retains its source;
absence of evidence is unknown, never an inferred rejection. No keys or work-id signatures.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from .staging import data_dir
from subnet.shared.canonicalization import content_hash
from subnet.shared.cost_model import cost_model_version, derive_cost_class

STAGES = frozenset(
    {
        "prepared",
        "staged",
        "envelope_created",
        "epoch_closed",
        "result",
        "charge_observation",
        "upload_observation",
        "validator_progress",
        "status_unavailable",
    }
)
SOURCES = frozenset({"local", "public_verdict", "central", "ingress", "validator"})


def identity_for(candidate: dict, hotkey: str, *, window: int | None = None) -> dict:
    task, version, cfg = candidate["task_type"], candidate["schema_version"], candidate["config"]
    selected_window = candidate.get("eligible_fanout_window") if window is None else window
    if (
        type(selected_window) is not int
        or selected_window < 0
        or not isinstance(hotkey, str)
        or not hotkey
    ):
        raise ValueError("journal identity requires a hotkey and exact window")
    return {
        "hotkey": hotkey,
        "task_type": task,
        "schema_version": version,
        "content_hash": content_hash(
            task, version, cfg, cost_model_version(task), derive_cost_class(task, cfg)
        ),
        "fanout_window": selected_window,
        "slot_id": cfg.get("slot_id"),
        "slot_epoch": cfg.get("slot_epoch"),
        "artifact_sha256": cfg.get("artifact_sha256"),
    }


def _wire(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > 2 * 1024**2:
        raise ValueError("journal observation exceeds two MiB")
    return encoded


class AttemptJournal:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(
            path or os.environ.get("AURELIUS_MINER_JOURNAL", data_dir() / "attempts.sqlite3")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                  id TEXT PRIMARY KEY, identity TEXT NOT NULL UNIQUE, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES attempts(id),
                  stage TEXT NOT NULL, source TEXT NOT NULL, data TEXT NOT NULL, observed REAL NOT NULL,
                  UNIQUE(attempt_id, stage, source, data));
                CREATE TABLE IF NOT EXISTS uploads (
                  hotkey TEXT NOT NULL, slot_id TEXT NOT NULL, digest TEXT NOT NULL,
                  data TEXT NOT NULL, observed REAL NOT NULL, PRIMARY KEY(hotkey, slot_id, digest));
            """)
        os.chmod(self.path, 0o600)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def ensure(self, identity: dict) -> str:
        wire = _wire(identity)
        identifier = hashlib.sha256(wire.encode()).hexdigest()
        with self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO attempts VALUES (?, ?, ?)", (identifier, wire, time.time())
            )
        return identifier

    def record(self, identity: dict, stage: str, source: str, data: dict | None = None) -> str:
        if stage not in STAGES or source not in SOURCES:
            raise ValueError("unknown observation stage/source")
        identifier = self.ensure(identity)
        with self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO events(attempt_id,stage,source,data,observed) VALUES(?,?,?,?,?)",
                (identifier, stage, source, _wire(data or {}), time.time()),
            )
        return identifier

    def upload(self, hotkey: str, slot: str, digest: str, acknowledgement: dict):
        with self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO uploads VALUES(?,?,?,?,?)",
                (hotkey, slot, digest, _wire(acknowledgement), time.time()),
            )

    def attempts(self, *, limit: int = 100) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("journal limit must be 1..1000")
        with self._db() as db:
            result = []
            for identifier, wire, created in db.execute(
                "SELECT * FROM attempts ORDER BY created DESC LIMIT ?", (limit,)
            ):
                identity = json.loads(wire)
                events = [
                    {
                        "stage": stage,
                        "source": source,
                        "data": json.loads(data),
                        "observed_at": observed,
                    }
                    for stage, source, data, observed in db.execute(
                        "SELECT stage,source,data,observed FROM events WHERE attempt_id=? ORDER BY id",
                        (identifier,),
                    )
                ]
                upload = db.execute(
                    "SELECT data,observed FROM uploads WHERE hotkey=? AND slot_id=? AND digest=?",
                    (identity["hotkey"], identity.get("slot_id"), identity.get("artifact_sha256")),
                ).fetchone()
                result.append(
                    {
                        "attempt_id": identifier,
                        "identity": identity,
                        "created_at": created,
                        "events": events,
                        "upload": None
                        if upload is None
                        else {
                            "data": json.loads(upload[0]),
                            "observed_at": upload[1],
                            "source": "ingress",
                        },
                    }
                )
        return result


def observe_envelope(candidate: dict, hotkey: str, validator: str, window: int) -> None:
    """Best-effort local observation; a full disk cannot interrupt work-id serving."""
    if not os.environ.get("AURELIUS_MINER_JOURNAL"):
        return
    try:
        AttemptJournal().record(
            identity_for(candidate, hotkey, window=window),
            "envelope_created",
            "local",
            {"validator_hotkey": validator},
        )
    except (OSError, ValueError, sqlite3.Error):
        import logging

        logging.getLogger(__name__).warning(
            "miner journal unavailable; envelope observation not recorded"
        )
