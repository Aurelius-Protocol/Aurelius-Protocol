"""Copy exactly the hotkey and public documents required by the v5 miner guard."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat

from subnet.shared.ss58 import ss58_decode
from subnet.shared.wallet_view import assert_hotkey_only_wallet_view


def prepare_view(source: Path, output: Path, wallet: str, hotkey: str, address: str) -> Path:
    for label, value in (("wallet", wallet), ("hotkey", hotkey)):
        if value in {"", ".", ".."} or Path(value).name != value:
            raise ValueError(f"{label} must be one path segment")
    public_key = ss58_decode(address)
    files = (source / wallet / "coldkeypub.txt", source / wallet / "hotkeys" / hotkey)
    for path in files:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 1_048_576:
            raise ValueError(f"not a bounded regular wallet file: {path}")
    destination = output / wallet
    # Never overwrite an existing runtime wallet.
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(destination, 0o700)
    (destination / "hotkeys").mkdir(mode=0o700)
    for source_file, relative in zip(files, ("coldkeypub.txt", f"hotkeys/{hotkey}")):
        copied = destination / relative
        shutil.copyfile(source_file, copied, follow_symlinks=False)
        os.chmod(copied, 0o600)
    public_document = destination / "hotkeys" / f"{hotkey}pub.txt"
    with public_document.open("x") as stream:
        json.dump({"publicKey": "0x" + public_key.hex(), "ss58Address": address}, stream)
    os.chmod(public_document, 0o600)
    assert_hotkey_only_wallet_view(output, wallet, hotkey)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / ".bittensor/wallets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wallet", default="miner")
    parser.add_argument("--hotkey", default="default")
    parser.add_argument("--hotkey-ss58", required=True)
    args = parser.parse_args()
    print(prepare_view(args.source, args.output, args.wallet, args.hotkey, args.hotkey_ss58))


if __name__ == "__main__":
    main()
