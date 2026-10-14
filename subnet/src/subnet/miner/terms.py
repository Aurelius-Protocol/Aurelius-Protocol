"""Incentive-terms acknowledgement for commands that commit a miner (`serve`, `fund`).

CONTRACT (capsule header)
- Pure local state — NO central calls, NO coldkey. The acceptance record lives under
  `AURELIUS_MINER_DATA` and is valid only for the current `INCENTIVES_TERMS_VERSION`.
- Bump the version when INCENTIVES.md changes materially; every miner is asked again.
- Headless starts opt in with `AURELIUS_ACCEPT_INCENTIVES=yes` or `--accept-incentives`;
  without a TTY or an opt-in the command refuses instead of assuming consent.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, TextIO

from subnet.miner.staging import data_dir

INCENTIVES_TERMS_VERSION = "2026-10-13"
INCENTIVES_URL = "https://github.com/Aurelius-Protocol/Aurelius-Protocol/blob/main/INCENTIVES.md"
ACCEPT_ENV = "AURELIUS_ACCEPT_INCENTIVES"
_TRUTHY = {"y", "yes", "1", "true"}

_NOTICE = f"""\
Aurelius incentives (terms version {INCENTIVES_TERMS_VERSION})
  Incentives are in open beta; values and mechanics can and will change frequently.
  The spirit of the rules matters more than the stated facts.
  Rewards are audited against chain data and may be adjusted retroactively.
  Read: {INCENTIVES_URL}
"""


class TermsNotAccepted(RuntimeError):
    """The miner has not agreed to the current incentive terms."""


def _record_path() -> Path:
    return data_dir() / "incentives-acceptance.json"


def _accepted() -> bool:
    try:
        record = json.loads(_record_path().read_text())
    except (OSError, ValueError):
        return False
    return isinstance(record, dict) and record.get("version") == INCENTIVES_TERMS_VERSION


def _record(method: str) -> None:
    path = _record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "version": INCENTIVES_TERMS_VERSION,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "method": method,
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".incentives-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(body, handle, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def ensure_incentives_accepted(
    *,
    accept_flag: bool = False,
    environ: Mapping[str, str] | None = None,
    stdin: TextIO | None = None,
    stream: TextIO | None = None,
) -> None:
    """Return once the current terms are accepted; raise `TermsNotAccepted` otherwise."""
    if _accepted():
        return
    environ = os.environ if environ is None else environ
    stdin = sys.stdin if stdin is None else stdin
    stream = sys.stderr if stream is None else stream
    print(_NOTICE, file=stream, flush=True)

    if accept_flag:
        _record("flag")
        return
    if environ.get(ACCEPT_ENV, "").strip().lower() in _TRUTHY:
        _record("env")
        return
    if not stdin.isatty():
        raise TermsNotAccepted(
            f"incentive terms not accepted; read {INCENTIVES_URL}, then run interactively, "
            f"set {ACCEPT_ENV}=yes or pass --accept-incentives"
        )
    print(
        "Have you read and do you agree to the incentives? [y/N]: ", end="", file=stream, flush=True
    )
    answer = stdin.readline().strip().lower()
    if answer not in {"y", "yes"}:
        raise TermsNotAccepted(f"incentive terms declined; read {INCENTIVES_URL}")
    _record("prompt")
