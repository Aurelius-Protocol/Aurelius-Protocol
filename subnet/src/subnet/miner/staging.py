"""The miner's local staging pool (plan §1.8): candidate work items the operator intends to submit,
persisted as jsonl under `AURELIUS_MINER_DATA`.

CONTRACT (capsule header)
- Pure local state — NO central calls, NO coldkey, NO lane import. An item has identity fields
  `{task_type, schema_version, config}` plus optional non-identity
  `eligible_fanout_window` metadata.
- A task/version with window-scoped rows ignores legacy, stale, and future rows. The sole
  schema-valid exact-window candidate is stable for every validator in that window; multiple
  eligible candidates fail closed until an ordering policy is frozen.
- Append-only jsonl keeps the pool dependency-free and human-inspectable.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Callable


ELIGIBLE_FANOUT_WINDOW = "eligible_fanout_window"


class StagingSelectionError(ValueError):
    """The local pool has no unambiguous candidate under the frozen selection rules."""


def data_dir() -> Path:
    return Path(os.environ.get("AURELIUS_MINER_DATA", str(Path.home() / ".aurelius" / "miner")))


def item_id(item: dict) -> str:
    """Local pool identity, including eligibility metadata; never a work/content hash."""
    return hashlib.sha256(
        json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class StagingPool:
    """A jsonl-backed pool of candidate items. Safe to open on a missing path (empty pool)."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else data_dir() / "staging.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def _persisted(item: dict) -> dict:
        if not isinstance(item, dict):
            raise ValueError("staged item must be an object")
        for field in ("task_type", "schema_version", "config"):
            if field not in item:
                raise ValueError(f"staged item missing {field!r}")
        if ELIGIBLE_FANOUT_WINDOW in item:
            window = item[ELIGIBLE_FANOUT_WINDOW]
            if isinstance(window, bool) or not isinstance(window, int) or window < 0:
                raise ValueError(f"{ELIGIBLE_FANOUT_WINDOW} must be a non-negative integer")
        persisted = {key: item[key] for key in ("task_type", "schema_version", "config")}
        if ELIGIBLE_FANOUT_WINDOW in item:
            persisted[ELIGIBLE_FANOUT_WINDOW] = item[ELIGIBLE_FANOUT_WINDOW]
        return persisted

    def _append(self, item: dict, *, once: bool) -> bool:
        persisted = self._persisted(item)
        if once and ELIGIBLE_FANOUT_WINDOW not in persisted:
            raise StagingSelectionError("idempotent staging requires an exact fanout window")
        payload = json.dumps(persisted, sort_keys=True, allow_nan=False) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self._path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "a+", encoding="utf-8") as fh:
            info = os.fstat(fh.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise StagingSelectionError("staging file must be a single regular inode")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            existing = fh.read()
            if existing and not existing.endswith("\n"):
                raise StagingSelectionError("staging file has an unterminated row")
            rows = [json.loads(line) for line in existing.splitlines() if line.strip()]
            if any(not isinstance(row, dict) for row in rows):
                raise StagingSelectionError("staging row must be an object")
            if once:
                fields = ("task_type", "schema_version", ELIGIBLE_FANOUT_WINDOW)
                occupied = [
                    row for row in rows if all(row.get(key) == persisted[key] for key in fields)
                ]
                if occupied:
                    if (
                        len(occupied) == 1
                        and json.dumps(occupied[0], sort_keys=True, allow_nan=False) + "\n"
                        == payload
                    ):
                        return False
                    raise StagingSelectionError("fanout window is already occupied")
            elif persisted in rows:
                return False
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        # Preserve the actual file inode: miners may have it bind-mounted individually.
        parent = os.open(self._path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return True

    def add(self, item: dict) -> None:
        """Append one candidate item. Requires the §3 identity fields; config stays opaque data."""
        with self._locked():
            self._append(item, once=False)

    def add_once(self, item: dict) -> bool:
        """Durably append once, or verify an exact replay; refuse competing window candidates."""
        with self._locked():
            return self._append(item, once=True)

    @contextmanager
    def _locked(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.with_suffix(self._path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _append_event(path: Path, value: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @property
    def controls_path(self) -> Path:
        return self._path.with_suffix(self._path.suffix + ".controls")

    def enabled(self) -> dict[str, bool]:
        if not self.controls_path.exists():
            return {}
        states = {}
        for line in self.controls_path.read_text().splitlines():
            event = json.loads(line)
            if set(event) != {"item_id", "enabled"} or type(event["enabled"]) is not bool:
                raise StagingSelectionError("invalid staging control event")
            states[event["item_id"]] = event["enabled"]
        return states

    def set_enabled(self, identity: str, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        with self._locked():
            if identity not in {item_id(item) for item in self.items()}:
                raise ValueError("unknown staging item")
            if self.enabled().get(identity, True) != enabled:
                self._append_event(self.controls_path, {"item_id": identity, "enabled": enabled})

    def active_items(self) -> list[dict]:
        states = self.enabled()
        unique = {item_id(item): item for item in self.items()}
        return [item for identity, item in unique.items() if states.get(identity, True)]

    def items(self) -> list[dict]:
        if not self._path.exists():
            return []
        out = []
        with self._path.open(encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def first_valid(
        self,
        *,
        task_type: str,
        schema_version: int,
        fanout_window: int | None = None,
        validate: Callable[[dict], list[str]],
        require_window_scoped: bool = False,
    ) -> dict | None:
        """Select one miner-staged candidate for an operator-defined task request.

        Legacy pools with no window metadata retain append-order priority. Once a matching row is
        window-scoped, only exact-window rows participate; unscoped, stale, and future rows are
        skipped, and multiple schema-valid rows fail closed. Malformed JSON makes ``items()``
        fail closed. The validator supplies no candidate bytes or selection index. The callback
        keeps published-schema validation outside this persistence capsule, so the pool remains
        free of central, lane, and economics dependencies.

        ``require_window_scoped`` (the production posture) refuses to serve from a pool with no
        window-scoped matching row: without it a miner could rotate divergent candidates per
        validator inside one fanout window.
        """
        if not isinstance(task_type, str) or not task_type or type(schema_version) is not int:
            return None
        if fanout_window is not None and (
            isinstance(fanout_window, bool)
            or not isinstance(fanout_window, int)
            or fanout_window < 0
        ):
            return None

        matching: list[dict] = []
        has_window_scoped_row = False
        for item in self.active_items():
            if not isinstance(item, dict):
                raise StagingSelectionError("staging row must be an object")
            if item.get("task_type") != task_type:
                continue
            if type(item.get("schema_version")) is not int:
                continue
            if item["schema_version"] != schema_version:
                continue
            if ELIGIBLE_FANOUT_WINDOW in item:
                window = item[ELIGIBLE_FANOUT_WINDOW]
                if isinstance(window, bool) or not isinstance(window, int) or window < 0:
                    raise StagingSelectionError(f"staging row has invalid {ELIGIBLE_FANOUT_WINDOW}")
                has_window_scoped_row = True
            matching.append(item)

        # An empty/disabled pool is idle; only actual unscoped content violates policy.
        if not matching:
            return None

        if require_window_scoped and not has_window_scoped_row:
            raise StagingSelectionError(
                "serving outside the local testlab posture requires a window-scoped staged "
                f"candidate ({ELIGIBLE_FANOUT_WINDOW}) for this task"
            )

        eligible: list[dict] = []
        for item in matching:
            if has_window_scoped_row:
                if fanout_window is None:
                    continue
                if item.get(ELIGIBLE_FANOUT_WINDOW) != fanout_window:
                    continue
            config = item.get("config")
            if not validate(config):
                eligible.append(item)

        if has_window_scoped_row and len(eligible) > 1:
            raise StagingSelectionError(
                f"multiple valid staged candidates are eligible for fanout window {fanout_window}"
            )
        return eligible[0] if eligible else None

    def pin_selection(self, item: dict, window: int, *, persist: bool = True) -> None:
        """Once served, changing controls cannot silently switch a scoped window's candidate."""
        if ELIGIBLE_FANOUT_WINDOW not in item:
            return
        key = [item["task_type"], item["schema_version"], window]
        path = self._path.with_suffix(self._path.suffix + ".selections")
        with self._locked():
            rows = (
                [json.loads(line) for line in path.read_text().splitlines()]
                if path.exists()
                else []
            )
            existing = [row for row in rows if row["key"] == key]
            if existing:
                if any(row["item_id"] != item_id(item) for row in existing):
                    raise StagingSelectionError(
                        "candidate already selected for this window; prepare a later window"
                    )
            elif persist:
                self._append_event(path, {"key": key, "item_id": item_id(item)})

    def clear(self) -> None:
        with self._locked():
            self._path.unlink(missing_ok=True)
            self.controls_path.unlink(missing_ok=True)
            self._path.with_suffix(self._path.suffix + ".selections").unlink(missing_ok=True)
