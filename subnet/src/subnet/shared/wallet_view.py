# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Structural guard for least-privilege Bittensor hotkey wallet views.

CONTRACT (capsule header)
- The guard inspects metadata and directory entries only; it never opens key or public-document
  bytes. The independently generated testnet-full view is the authority for their contents.
- A guarded long-running process receives exactly ``coldkeypub.txt`` and ``hotkeys/<name>`` plus
  ``hotkeys/<name>pub.txt``. A private ``coldkey`` (including a dangling symlink) is fatal.
- All view directories/files are owner-only, regular, and non-symlink. This module is stdlib-only
  so both miner and validator loaders can enforce the same fail-closed boundary before importing
  Bittensor.
"""

from __future__ import annotations
import os
import stat
from pathlib import Path

_MAX_WALLET_FILE_BYTES = 1_048_576


def _leaf(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be one non-empty path segment")
    return value


def _directory(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink directory")
    mode = stat.S_IMODE(value.st_mode)
    if mode != 0o700:
        raise ValueError(f"{label} must have mode 0700, got {mode:04o}")


def _file(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(value.st_mode) or stat.S_ISLNK(value.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    mode = stat.S_IMODE(value.st_mode)
    if mode != 0o600:
        raise ValueError(f"{label} must have mode 0600, got {mode:04o}")
    if not 0 < value.st_size <= _MAX_WALLET_FILE_BYTES:
        raise ValueError(f"{label} has an invalid byte size")


def assert_hotkey_only_wallet_view(root: str | Path, wallet_name: str, hotkey_name: str) -> None:
    """Fail unless ``root/wallet_name`` is the exact safe Bittensor runtime layout.

    This deliberately checks only filesystem structure/metadata; it does not read any file.
    """

    wallet_name = _leaf(wallet_name, "Bittensor wallet name")
    hotkey_name = _leaf(hotkey_name, "Bittensor hotkey name")
    wallet = Path(root) / wallet_name
    hotkeys = wallet / "hotkeys"
    _directory(wallet, f"{wallet_name} runtime wallet")
    _directory(hotkeys, f"{wallet_name} runtime hotkeys")

    coldkey = wallet / "coldkey"
    if os.path.lexists(coldkey):
        raise ValueError(f"{wallet_name} runtime wallet exposes a private coldkey")
    if set(os.listdir(wallet)) != {"coldkeypub.txt", "hotkeys"}:
        raise ValueError(f"{wallet_name} runtime wallet has an unexpected or missing entry")
    if set(os.listdir(hotkeys)) != {hotkey_name, f"{hotkey_name}pub.txt"}:
        raise ValueError(f"{wallet_name} runtime hotkeys has an unexpected or missing entry")

    _file(wallet / "coldkeypub.txt", f"{wallet_name} coldkey public document")
    _file(hotkeys / hotkey_name, f"{wallet_name} hotkey private file")
    _file(hotkeys / f"{hotkey_name}pub.txt", f"{wallet_name} hotkey public document")
