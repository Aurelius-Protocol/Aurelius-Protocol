# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Descriptor-held custody for the five runtime452 one-shot action journals.

The root installer publishes one immutable completion marker only after all five fixed journal
directories exist.  Runtime processes consume a typed action claim, never a caller-selected path,
and retain open descriptors for the installation root, marker, and their actor directory for the
entire action.  Every filesystem and signing/network boundary re-joins those descriptors to the
compiled path and re-reads the exact marker bytes.

This module has no wallet, RPC, signing, submission, Docker, or SSH capability.
"""

from __future__ import annotations
import hashlib
import json
import os
from contextlib import contextmanager
import base64
from pathlib import Path
import re
import socket
import stat
import struct
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator, Literal

if TYPE_CHECKING:
    from subnet.operator.sn455_runtime452_action_claim import Runtime452ActionClaim


MAX_MARKER_BYTES = 1_048_576


MAX_RECORD_BYTES = 1_048_576


INSTALLER_UID = 0


INSTALLER_GID = 0


MARKER_MODE = 0o444


ROOT_MODE = 0o755


JOURNAL_MODE = 0o750


RECORDER_ROOT_MODE = 0o755


RECORDER_SOCKET_MODE = 0o660


RECORD_MODE = 0o440


INSTALL_LOCK_NAME = ".aurelius-sn455-runtime452-action-journals-install-v1.lock"


INSTALL_INTENT_NAME = ".aurelius-sn455-runtime452-action-journals-install-intent-v1.json"


_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


_CLOSED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Runtime452ActionJournalError(RuntimeError):
    """The installed one-shot journal cannot be proved exactly."""


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Runtime452ActionJournalError("journal evidence is not canonical JSON data") from exc


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _compiled_contract(profile_id: str | None = None):
    # Deliberately lazy: the action-claim module consumes weight burn authority, while the weight
    # package imports its chain client during package initialization.
    from subnet.operator import sn455_runtime452_action_claim as contract

    selected = (
        profile_id
        or os.environ.get(contract.generation_authority.PROFILE_ENV)
        or contract.PROFILE_ID
    )
    if selected == contract.PROFILE_ID:
        return contract
    if selected not in contract.ACTION_PROFILE_SPECS or selected == contract.PLUMBING_PROFILE_ID:
        raise Runtime452ActionJournalError("runtime452 journal profile is not compiled")
    axon_paths, weight_path, recorder_paths = contract._journal_paths(selected)
    spec = contract.ACTION_PROFILE_SPECS[selected]
    root = str(spec["journal_root"])
    return SimpleNamespace(
        ACTION_JOURNAL_ROOT=root,
        ACTION_RUNTIME_IDS=spec["runtime_ids"],
        AXON_ENDPOINTS=contract.AXON_ENDPOINTS if "runtime_ids" not in spec else {},
        AXON_JOURNAL_PATHS=axon_paths,
        JOURNAL_INSTALLATION_MARKER=contract.JOURNAL_INSTALLATION_MARKER,
        JOURNAL_RECORDER_PROTOCOL=contract.JOURNAL_RECORDER_PROTOCOL,
        JOURNAL_RECORDER_SOCKET_PATHS=recorder_paths,
        JOURNAL_RECORDER_SOCKET_ROOT=f"{root}/recorders",
        PROFILE_ID=selected,
        Runtime452ActionClaim=contract.Runtime452ActionClaim,
        SERVE_AXON_ACTION=contract.SERVE_AXON_ACTION,
        WEIGHT_ACTION=contract.WEIGHT_ACTION,
        WEIGHT_JOURNAL_PATH=weight_path,
        generation_authority=SimpleNamespace(
            GENERATION=contract.ACTION_PROFILE_SPECS[selected]["generation"]
        ),
    )


def _expected_roster(contract) -> tuple[tuple[str, str, int, str, int, int], ...]:
    # The k8s generation (ADR-0199) authorizes exactly one action — the validator's weight
    # cycle; the host generations roster four miner ServeAxon journals ahead of it. A contract
    # with no axon endpoints IS the k8s shape.
    weight_index = 5 if contract.AXON_ENDPOINTS else 1
    miners = tuple(
        (
            contract.SERVE_AXON_ACTION,
            actor,
            index,
            contract.AXON_JOURNAL_PATHS[actor],
            contract.ACTION_RUNTIME_IDS[actor]["uid"],
            contract.ACTION_RUNTIME_IDS[actor]["gid"],
        )
        for index, actor in enumerate(contract.AXON_ENDPOINTS, start=1)
    )
    return miners + (
        (
            contract.WEIGHT_ACTION,
            "validator-0",
            weight_index,
            contract.WEIGHT_JOURNAL_PATH,
            contract.ACTION_RUNTIME_IDS["validator-0"]["uid"],
            contract.ACTION_RUNTIME_IDS["validator-0"]["gid"],
        ),
    )


def _validate_installation_intent(
    installation_intent: Any,
    *,
    generation_marker_sha256: str,
    burn_handoff_sha256: str,
    burn_uid: int,
    journals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the embedded genesis binding, without claiming out-of-mount FD custody."""

    contract = _compiled_contract()
    anchor = Path(contract.ACTION_JOURNAL_ROOT).parent.parent
    directory_fields = {"path", "device", "inode", "uid", "gid", "mode"}
    file_fields = directory_fields | {"nlink", "bytes", "sha256", "mtime_ns", "ctime_ns"}

    def directory(value, path: Path, mode: int | None = None):
        if (
            type(value) is not dict
            or set(value) != directory_fields
            or value["path"] != os.fspath(path)
            or value["uid"] != INSTALLER_UID
            or value["gid"] != INSTALLER_GID
            or any(
                type(value[key]) is not int or value[key] < 0 for key in directory_fields - {"path"}
            )
            or value["inode"] == 0
            or value["mode"] > 0o777
            or value["mode"] & 0o022
            or (mode is not None and value["mode"] != mode)
        ):
            raise Runtime452ActionJournalError("embedded installation directory scope differs")
        return value

    def file(value, path: Path, mode: int, payload: bytes):
        if (
            type(value) is not dict
            or set(value) != file_fields
            or value["path"] != os.fspath(path)
            or value["uid"] != INSTALLER_UID
            or value["gid"] != INSTALLER_GID
            or value["mode"] != mode
            or value["nlink"] != 1
            or value["bytes"] != len(payload)
            or value["sha256"] != hashlib.sha256(payload).hexdigest()
            or any(
                type(value[key]) is not int or value[key] < 0
                for key in file_fields - {"path", "sha256"}
            )
            or value["inode"] == 0
        ):
            raise Runtime452ActionJournalError("embedded installation file scope differs")
        return value

    if type(installation_intent) is not dict or set(installation_intent) != {"descriptor", "value"}:
        raise Runtime452ActionJournalError("journal installation lacks its exact genesis binding")
    value = installation_intent["value"]
    if type(value) is not dict:
        raise Runtime452ActionJournalError("embedded installation intent is not one object")
    anchor_document = directory(value.get("anchor"), anchor)
    lock_document = file(value.get("lock"), anchor / INSTALL_LOCK_NAME, 0o600, b"")
    prestate = value.get("profile_parent_prestate")
    if prestate != {"state": "absent"}:
        if (
            type(prestate) is not dict
            or set(prestate) != {"state", "identity"}
            or prestate["state"] != "present"
        ):
            raise Runtime452ActionJournalError("embedded installation parent prestate differs")
        directory(prestate["identity"], Path(contract.ACTION_JOURNAL_ROOT).parent, ROOT_MODE)
    expected_journals = []
    for entry, (action, actor, index, path, runtime_uid, runtime_gid) in zip(
        journals, _expected_roster(contract), strict=True
    ):
        expected_journals.append(
            {
                "actor": actor,
                "action": action,
                "claim_index": index,
                "claim_sha256": entry["claim_sha256"],
                "path": path,
                "owner_uid": INSTALLER_UID,
                "owner_gid": runtime_gid,
                "mode": JOURNAL_MODE,
                "runtime_uid": runtime_uid,
                "runtime_gid": runtime_gid,
                "socket": contract.JOURNAL_RECORDER_SOCKET_PATHS[actor],
                "protocol": contract.JOURNAL_RECORDER_PROTOCOL,
            }
        )
    expected_value = {
        "schema_version": 1,
        "kind": "sn455-runtime452-action-journal-install-intent-v1",
        "status": "prepared-before-first-journal-directory",
        "profile_id": contract.PROFILE_ID,
        "generation": contract.generation_authority.GENERATION,
        "generation_marker_sha256": generation_marker_sha256,
        "burn_handoff_sha256": burn_handoff_sha256,
        "burn_uid": burn_uid,
        "anchor": anchor_document,
        "lock": lock_document,
        "profile_parent_prestate": prestate,
        "root": {
            "path": contract.ACTION_JOURNAL_ROOT,
            "uid": INSTALLER_UID,
            "gid": INSTALLER_GID,
            "mode": ROOT_MODE,
        },
        "recorder_root": {
            "path": contract.JOURNAL_RECORDER_SOCKET_ROOT,
            "uid": INSTALLER_UID,
            "gid": INSTALLER_GID,
            "mode": RECORDER_ROOT_MODE,
        },
        "marker_name": contract.JOURNAL_INSTALLATION_MARKER,
        "marker_mode": MARKER_MODE,
        "journals": expected_journals,
        "private_material": False,
        "authorizes_chain_action": False,
    }
    if _canonical_json(value) != _canonical_json(expected_value):
        raise Runtime452ActionJournalError("embedded installation intent differs from exact claims")
    payload = _canonical_json(expected_value)
    descriptor = file(
        installation_intent["descriptor"], anchor / INSTALL_INTENT_NAME, MARKER_MODE, payload
    )
    if (
        descriptor["device"] != anchor_document["device"]
        or lock_document["device"] != anchor_document["device"]
    ):
        raise Runtime452ActionJournalError("embedded installation files left their anchor device")
    return json.loads(_canonical_json(installation_intent))


def _linux_mountpoints() -> frozenset[str]:
    """Return decoded mount points; Linux runtime452 fails closed if mountinfo is unreadable."""

    if os.name != "posix" or not Path("/proc/self/mountinfo").exists():
        return frozenset()
    try:
        raw = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError as exc:
        raise Runtime452ActionJournalError("runtime452 mount topology is unavailable") from exc
    result: set[str] = set()
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 6 or "-" not in fields:
            raise Runtime452ActionJournalError("runtime452 mount topology is malformed")
        mountpoint = fields[4]
        for escaped, value in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
            mountpoint = mountpoint.replace(escaped, value)
        result.add(mountpoint)
    return frozenset(result)


class Runtime452ActionJournalCustody:
    """Held capability for exactly one actor's installed runtime452 journal."""

    def __init__(self, claim: "Runtime452ActionClaim"):
        contract = _compiled_contract(getattr(claim, "profile_id", None))
        roster = _expected_roster(contract)
        expected = next(
            (entry for entry in roster if entry[1] == getattr(claim, "actor", None)), None
        )
        if (
            type(claim) is not contract.Runtime452ActionClaim
            or expected is None
            or (
                claim.action,
                claim.actor,
                claim.claim_index,
                claim.journal_path,
                claim.journal_runtime_uid,
                claim.journal_runtime_gid,
            )
            != expected
            or claim.journal_installation_marker != contract.JOURNAL_INSTALLATION_MARKER
            or claim.journal_recorder_socket
            != contract.JOURNAL_RECORDER_SOCKET_PATHS.get(claim.actor)
            or os.geteuid() != claim.journal_runtime_uid
            or os.getegid() != claim.journal_runtime_gid
        ):
            raise Runtime452ActionJournalError(
                "runtime452 journal claim or process identity differs from compiled scope"
            )
        self.claim = claim
        self._contract = contract
        self.path = Path(claim.journal_path)
        self.root_path = Path(contract.ACTION_JOURNAL_ROOT)
        self.recorder_root_path = Path(contract.JOURNAL_RECORDER_SOCKET_ROOT)
        self.recorder_socket_path = Path(claim.journal_recorder_socket)
        self.marker_name = contract.JOURNAL_INSTALLATION_MARKER
        self._root_fd: int | None = None
        self._recorder_root_fd: int | None = None
        self._marker_fd: int | None = None
        self._journal_fd: int | None = None
        self._root_identity: tuple[int, int, int, int, int] | None = None
        self._recorder_root_identity: tuple[int, int, int, int, int] | None = None
        self._recorder_socket_identity: (
            tuple[int, int, int, int, int, int, int, int, int] | None
        ) = None
        self._marker_identity: tuple[int, int, int, int, int, int, int, int, int] | None = None
        self._journal_identity: tuple[int, int, int, int, int] | None = None
        self._marker_payload: bytes | None = None
        self._session_connection: socket.socket | None = None
        self._session_owner: int | None = None
        self._session_lock = threading.Lock()
        try:
            self._root_fd = self._open_compiled_root()
            self._root_identity = _identity(os.fstat(self._root_fd))
            self._validate_mount_topology()
            self._recorder_root_fd = os.open(
                self.recorder_root_path.name,
                self._directory_flags(),
                dir_fd=self._root_fd,
            )
            recorder_root_info = os.fstat(self._recorder_root_fd)
            if (
                not stat.S_ISDIR(recorder_root_info.st_mode)
                or stat.S_IMODE(recorder_root_info.st_mode) != RECORDER_ROOT_MODE
                or recorder_root_info.st_uid != INSTALLER_UID
                or recorder_root_info.st_gid != INSTALLER_GID
                or recorder_root_info.st_nlink < 2
            ):
                raise Runtime452ActionJournalError("runtime452 recorder root has an unsafe inode")
            self._recorder_root_identity = _identity(recorder_root_info)
            self._marker_fd = os.open(
                self.marker_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self._root_fd,
            )
            marker_info = os.fstat(self._marker_fd)
            if (
                not stat.S_ISREG(marker_info.st_mode)
                or stat.S_IMODE(marker_info.st_mode) != MARKER_MODE
                or marker_info.st_uid != INSTALLER_UID
                or marker_info.st_gid != INSTALLER_GID
                or marker_info.st_nlink != 1
                or not 1 <= marker_info.st_size <= MAX_MARKER_BYTES
            ):
                raise Runtime452ActionJournalError(
                    "runtime452 journal completion marker has an unsafe inode"
                )
            self._marker_identity = _file_identity(marker_info)
            self._journal_fd = os.open(
                self.path.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self._root_fd,
            )
            journal_info = os.fstat(self._journal_fd)
            if (
                not stat.S_ISDIR(journal_info.st_mode)
                or stat.S_IMODE(journal_info.st_mode) != JOURNAL_MODE
                or journal_info.st_uid != INSTALLER_UID
                or journal_info.st_gid != claim.journal_runtime_gid
                or journal_info.st_nlink < 2
            ):
                raise Runtime452ActionJournalError("runtime452 actor journal has an unsafe inode")
            self._journal_identity = _identity(journal_info)
            self._recorder_socket_identity = self._validate_recorder_socket()
            self._marker_payload = self._read_and_validate_marker()
            self.revalidate()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _directory_flags() -> int:
        required = (getattr(os, "O_DIRECTORY", 0), getattr(os, "O_NOFOLLOW", 0))
        if 0 in required:
            raise Runtime452ActionJournalError(
                "runtime452 descriptor-safe directory opening is unavailable"
            )
        return os.O_RDONLY | required[0] | required[1] | getattr(os, "O_CLOEXEC", 0)

    def _open_compiled_root(self) -> int:
        flags = self._directory_flags()
        descriptor = os.open("/", flags)
        try:
            parts = self.root_path.parts[1:]
            for index, part in enumerate(parts):
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
                info = os.fstat(descriptor)
                final = index == len(parts) - 1
                if not stat.S_ISDIR(info.st_mode):
                    raise Runtime452ActionJournalError(
                        "runtime452 journal path contains a non-directory"
                    )
                if not final and (
                    info.st_uid != INSTALLER_UID
                    or info.st_gid != INSTALLER_GID
                    or stat.S_IMODE(info.st_mode) & 0o022
                ):
                    raise Runtime452ActionJournalError(
                        "runtime452 journal parent is not installer-owned and non-writable"
                    )
                if final and (
                    info.st_uid != INSTALLER_UID
                    or info.st_gid != INSTALLER_GID
                    or stat.S_IMODE(info.st_mode) != ROOT_MODE
                ):
                    raise Runtime452ActionJournalError(
                        "runtime452 journal root differs from its installed identity"
                    )
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _validate_mount_topology(self) -> None:
        mountpoints = _linux_mountpoints()
        if not mountpoints:
            return
        allowed = {"/", os.fspath(self.root_path)}
        prefixes = [
            os.fspath(Path(*self.root_path.parts[:index]))
            for index in range(2, len(self.root_path.parts) + 1)
        ]
        if (
            any(prefix in mountpoints and prefix not in allowed for prefix in prefixes)
            or os.fspath(self.path) in mountpoints
        ):
            raise Runtime452ActionJournalError(
                "runtime452 journal path crosses an unclaimed parent or actor mount"
            )

    def _validate_recorder_socket(self) -> tuple[int, int, int, int, int, int, int, int, int]:
        if self._recorder_root_fd is None:
            raise Runtime452ActionJournalError("runtime452 recorder root custody is closed")
        try:
            info = os.stat(
                self.recorder_socket_path.name,
                dir_fd=self._recorder_root_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise Runtime452ActionJournalError(
                "runtime452 actor recorder socket is unavailable"
            ) from exc
        if (
            not stat.S_ISSOCK(info.st_mode)
            or stat.S_IMODE(info.st_mode) != RECORDER_SOCKET_MODE
            or info.st_uid != INSTALLER_UID
            or info.st_gid != self.claim.journal_runtime_gid
            or info.st_nlink != 1
        ):
            raise Runtime452ActionJournalError(
                "runtime452 actor recorder socket has an unsafe inode"
            )
        return _file_identity(info)

    def _read_marker_bytes(self) -> bytes:
        if self._marker_fd is None:
            raise Runtime452ActionJournalError("runtime452 marker custody is closed")
        before = os.fstat(self._marker_fd)
        try:
            payload = bytearray()
            while len(payload) <= MAX_MARKER_BYTES:
                chunk = os.pread(
                    self._marker_fd,
                    min(65_536, MAX_MARKER_BYTES + 1 - len(payload)),
                    len(payload),
                )
                if not chunk:
                    break
                payload.extend(chunk)
        except OSError as exc:
            raise Runtime452ActionJournalError(
                "runtime452 journal marker cannot be reread"
            ) from exc
        after = os.fstat(self._marker_fd)
        if (
            _file_identity(before) != _file_identity(after)
            or not 1 <= len(payload) <= MAX_MARKER_BYTES
            or len(payload) != after.st_size
        ):
            raise Runtime452ActionJournalError("runtime452 journal marker changed while read")
        return bytes(payload)

    def _read_and_validate_marker(self) -> bytes:
        payload = self._read_marker_bytes()
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Runtime452ActionJournalError(
                "runtime452 journal marker is not canonical JSON"
            ) from exc
        if type(document) is not dict or payload != _canonical_json(document):
            raise Runtime452ActionJournalError("runtime452 journal marker is not canonical JSON")
        self._validate_marker_document(document)
        return payload

    @staticmethod
    def _directory_matches_document(info: os.stat_result, document: Any) -> bool:
        return (
            type(document) is dict
            and set(document)
            == {"path", "device", "inode", "uid", "gid", "mode", "nlink_at_install"}
            and all(
                type(document[key]) is int
                for key in ("device", "inode", "uid", "gid", "mode", "nlink_at_install")
            )
            and document["device"] == info.st_dev
            and document["inode"] == info.st_ino
            and document["uid"] == info.st_uid
            and document["gid"] == info.st_gid
            and document["mode"] == stat.S_IMODE(info.st_mode)
            and type(document["nlink_at_install"]) is int
            and 2 <= document["nlink_at_install"] <= info.st_nlink
        )

    def _validate_marker_document(self, document: dict[str, Any]) -> None:
        contract = self._contract
        roster = _expected_roster(contract)
        if self._root_fd is None or self._recorder_root_fd is None:
            raise Runtime452ActionJournalError("runtime452 root custody is closed")
        if (
            set(document)
            != {
                "schema_version",
                "status",
                "profile_id",
                "generation",
                "generation_marker_sha256",
                "burn_handoff_sha256",
                "burn_uid",
                "installation_intent",
                "journal_installation_marker",
                "journal_root",
                "recorder_root",
                "journals",
                "private_material",
            }
            or document["schema_version"] != 2
            or document["status"] != "installed_runtime452_action_journals"
            or document["profile_id"] != contract.PROFILE_ID
            or document["generation"] != contract.generation_authority.GENERATION
            or document["generation_marker_sha256"] != self.claim.generation_marker_sha256
            or document["burn_handoff_sha256"] != self.claim.burn_handoff_sha256
            or document["burn_uid"] != self.claim.burn_uid
            or document["journal_installation_marker"] != self.marker_name
            or document["private_material"] is not False
            or not self._directory_matches_document(
                os.fstat(self._root_fd), document["journal_root"]
            )
            or document["journal_root"]["path"] != os.fspath(self.root_path)
            or not self._directory_matches_document(
                os.fstat(self._recorder_root_fd), document["recorder_root"]
            )
            or document["recorder_root"]["path"] != os.fspath(self.recorder_root_path)
            or type(document["journals"]) is not list
            or len(document["journals"]) != len(roster)
        ):
            raise Runtime452ActionJournalError(
                "runtime452 journal marker differs from compiled generation"
            )
        own_seen = False
        for entry, (action, actor, index, path, runtime_uid, runtime_gid) in zip(
            document["journals"], roster, strict=True
        ):
            if (
                type(entry) is not dict
                or set(entry)
                != {
                    "action",
                    "actor",
                    "claim_index",
                    "claim_sha256",
                    "directory",
                    "recorder",
                }
                or entry["action"] != action
                or entry["actor"] != actor
                or type(entry["claim_index"]) is not int
                or entry["claim_index"] != index
                or _HEX_DIGEST.fullmatch(str(entry["claim_sha256"])) is None
                or type(entry["directory"]) is not dict
                or entry["directory"].get("path") != path
                or _canonical_json(entry["recorder"])
                != _canonical_json(
                    {
                        "protocol": contract.JOURNAL_RECORDER_PROTOCOL,
                        "socket_path": contract.JOURNAL_RECORDER_SOCKET_PATHS[actor],
                        "socket_uid": INSTALLER_UID,
                        "socket_gid": runtime_gid,
                        "socket_mode": RECORDER_SOCKET_MODE,
                        "peer_uid": runtime_uid,
                        "peer_gid": runtime_gid,
                    }
                )
            ):
                raise Runtime452ActionJournalError(
                    "runtime452 journal marker action roster differs"
                )
            try:
                info = os.stat(Path(path).name, dir_fd=self._root_fd, follow_symlinks=False)
            except OSError as exc:
                raise Runtime452ActionJournalError(
                    "runtime452 installed action journal is unavailable"
                ) from exc
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != INSTALLER_UID
                or info.st_gid != runtime_gid
                or stat.S_IMODE(info.st_mode) != JOURNAL_MODE
                or not self._directory_matches_document(info, entry["directory"])
            ):
                raise Runtime452ActionJournalError(
                    "runtime452 installed action journal inode differs"
                )
            if actor == self.claim.actor:
                own_seen = True
                if entry["claim_sha256"] != self.claim.claim_sha256:
                    raise Runtime452ActionJournalError(
                        "runtime452 actor journal is bound to another claim"
                    )
        if not own_seen:
            raise Runtime452ActionJournalError(
                "runtime452 actor journal is absent from completion marker"
            )
        _validate_installation_intent(
            document["installation_intent"],
            generation_marker_sha256=self.claim.generation_marker_sha256,
            burn_handoff_sha256=self.claim.burn_handoff_sha256,
            burn_uid=self.claim.burn_uid,
            journals=document["journals"],
        )

    def revalidate(self) -> None:
        """Rejoin all held descriptors and exact marker bytes to the compiled path."""

        if (
            self._root_fd is None
            or self._recorder_root_fd is None
            or self._marker_fd is None
            or self._journal_fd is None
            or self._root_identity is None
            or self._recorder_root_identity is None
            or self._recorder_socket_identity is None
            or self._marker_identity is None
            or self._journal_identity is None
            or self._marker_payload is None
        ):
            raise Runtime452ActionJournalError("runtime452 journal custody is closed")
        replacement_root: int | None = None
        try:
            replacement_root = self._open_compiled_root()
            held_root = os.fstat(self._root_fd)
            current_root = os.fstat(replacement_root)
            current_marker = os.stat(
                self.marker_name,
                dir_fd=self._root_fd,
                follow_symlinks=False,
            )
            current_journal = os.stat(
                self.path.name,
                dir_fd=self._root_fd,
                follow_symlinks=False,
            )
            held_journal = os.fstat(self._journal_fd)
            held_recorder_root = os.fstat(self._recorder_root_fd)
            current_recorder_root = os.stat(
                self.recorder_root_path.name,
                dir_fd=self._root_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise Runtime452ActionJournalError(
                "runtime452 held journal cannot be rejoined to its path"
            ) from exc
        finally:
            if replacement_root is not None:
                os.close(replacement_root)
        self._validate_mount_topology()
        if (
            _identity(held_root) != self._root_identity
            or _identity(current_root) != self._root_identity
            or _file_identity(current_marker) != self._marker_identity
            or _identity(held_recorder_root) != self._recorder_root_identity
            or _identity(current_recorder_root) != self._recorder_root_identity
            or self._validate_recorder_socket() != self._recorder_socket_identity
            or _identity(held_journal) != self._journal_identity
            or _identity(current_journal) != self._journal_identity
            or held_journal.st_nlink != current_journal.st_nlink
            or held_journal.st_nlink < 2
            or self._read_and_validate_marker() != self._marker_payload
        ):
            raise Runtime452ActionJournalError(
                "runtime452 journal directory or completion marker changed"
            )

    def _name(self, name: str) -> str:
        if type(name) is not str or _CLOSED_NAME.fullmatch(name) is None:
            raise Runtime452ActionJournalError("runtime452 journal record name is outside scope")
        return name

    def exists(self, name: str) -> bool:
        self.revalidate()
        assert self._journal_fd is not None
        try:
            info = os.stat(self._name(name), dir_fd=self._journal_fd, follow_symlinks=False)
        except FileNotFoundError:
            self.revalidate()
            return False
        except OSError as exc:
            raise Runtime452ActionJournalError(
                "runtime452 journal record cannot be inspected"
            ) from exc
        self._validate_record_info(info)
        self.revalidate()
        return True

    def _validate_record_info(self, info: os.stat_result) -> None:
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != RECORD_MODE
            or info.st_uid != INSTALLER_UID
            or info.st_gid != self.claim.journal_runtime_gid
            or info.st_nlink != 1
            or not 1 <= info.st_size <= MAX_RECORD_BYTES
        ):
            raise Runtime452ActionJournalError("runtime452 journal record has an unsafe inode")

    def list_names(self) -> tuple[str, ...]:
        self.revalidate()
        assert self._journal_fd is not None
        try:
            names = tuple(sorted(os.listdir(self._journal_fd)))
            for name in names:
                self._name(name)
                self._validate_record_info(
                    os.stat(name, dir_fd=self._journal_fd, follow_symlinks=False)
                )
        except OSError as exc:
            raise Runtime452ActionJournalError(
                "runtime452 journal records cannot be enumerated"
            ) from exc
        self.revalidate()
        return names

    @staticmethod
    def _recv_exact(connection: socket.socket, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = connection.recv(size - len(result))
            if not chunk:
                raise Runtime452ActionJournalError(
                    "runtime452 recorder closed a bounded response early"
                )
            result.extend(chunk)
        return bytes(result)

    def _open_recorder_connection(self) -> socket.socket:
        self.revalidate()
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(10)
            connection.connect(os.fspath(self.recorder_socket_path))
            if not hasattr(socket, "SO_PEERCRED"):
                raise Runtime452ActionJournalError(
                    "runtime452 recorder peer credentials are unavailable"
                )
            peer = connection.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                struct.calcsize("3i"),
            )
            peer_pid, peer_uid, peer_gid = struct.unpack("3i", peer)
            if peer_pid <= 0 or peer_uid != INSTALLER_UID or peer_gid != INSTALLER_GID:
                raise Runtime452ActionJournalError(
                    "runtime452 recorder peer is not the fixed root service"
                )
            self.revalidate()
            return connection
        except BaseException:
            connection.close()
            raise

    def _recorder_request(
        self,
        connection: socket.socket,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        contract = self._contract
        payload = _canonical_json(request)
        if not 1 <= len(payload) <= 2 * MAX_RECORD_BYTES:
            raise Runtime452ActionJournalError(
                "runtime452 recorder request exceeds its bounded frame"
            )
        try:
            self.revalidate()
            connection.sendall(struct.pack("!I", len(payload)) + payload)
            size = struct.unpack("!I", self._recv_exact(connection, 4))[0]
            if not 1 <= size <= MAX_MARKER_BYTES:
                raise Runtime452ActionJournalError(
                    "runtime452 recorder returned an invalid frame size"
                )
            raw_response = self._recv_exact(connection, size)
        except Runtime452ActionJournalError:
            raise
        except OSError as exc:
            raise Runtime452ActionJournalError(
                "runtime452 recorder request is unavailable"
            ) from exc
        try:
            response = json.loads(raw_response)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Runtime452ActionJournalError(
                "runtime452 recorder response is not canonical JSON"
            ) from exc
        if (
            type(response) is not dict
            or raw_response != _canonical_json(response)
            or response.get("schema_version") != 1
            or response.get("protocol") != contract.JOURNAL_RECORDER_PROTOCOL
            or response.get("actor") != self.claim.actor
            or response.get("operation") != request.get("operation")
        ):
            raise Runtime452ActionJournalError(
                "runtime452 recorder response differs from the exact request"
            )
        self.revalidate()
        return response

    def claim_bytes(self, name: str, payload: bytes) -> Literal["claimed_exact", "already_exact"]:
        if type(payload) is not bytes or not 1 <= len(payload) <= MAX_RECORD_BYTES:
            raise Runtime452ActionJournalError(
                "runtime452 journal record exceeds its bounded byte posture"
            )
        final_name = self._name(name)
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Runtime452ActionJournalError(
                "runtime452 journal claim is not canonical JSON"
            ) from exc
        if type(document) is not dict or payload != _canonical_json(document):
            raise Runtime452ActionJournalError("runtime452 journal claim is not canonical JSON")
        contract = self._contract
        digest = hashlib.sha256(payload).hexdigest()
        connection = self._session_connection
        if connection is None or self._session_owner != threading.get_ident():
            raise Runtime452ActionJournalError(
                "runtime452 journal claims require this thread's root-recorder transition session"
            )
        response = self._recorder_request(
            connection,
            {
                "schema_version": 1,
                "protocol": contract.JOURNAL_RECORDER_PROTOCOL,
                "operation": "claim_record",
                "actor": self.claim.actor,
                "action": self.claim.action,
                "claim_sha256": self.claim.claim_sha256,
                "name": final_name,
                "payload_base64": base64.b64encode(payload).decode("ascii"),
                "payload_sha256": digest,
            },
        )
        if _canonical_json(response) != _canonical_json(
            {
                "schema_version": 1,
                "protocol": contract.JOURNAL_RECORDER_PROTOCOL,
                "operation": "claim_record",
                "status": response.get("status"),
                "actor": self.claim.actor,
                "name": final_name,
                "payload_sha256": digest,
            }
        ) or response["status"] not in {"claimed_exact", "already_exact"}:
            raise Runtime452ActionJournalError(
                "runtime452 recorder did not acknowledge the exact durable record"
            )
        if self.read_bytes(final_name) != payload:
            raise Runtime452ActionJournalError(
                "runtime452 recorder acknowledgement differs from retained bytes"
            )
        return response["status"]

    def read_bytes(self, name: str) -> bytes:
        self.revalidate()
        assert self._journal_fd is not None
        descriptor: int | None = None
        try:
            expected = os.stat(
                self._name(name),
                dir_fd=self._journal_fd,
                follow_symlinks=False,
            )
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self._journal_fd,
            )
            opened = os.fstat(descriptor)
            self._validate_record_info(opened)
            if _file_identity(expected) != _file_identity(opened):
                raise Runtime452ActionJournalError(
                    "runtime452 journal record changed before opening"
                )
            payload = bytearray()
            while len(payload) <= MAX_RECORD_BYTES:
                chunk = os.read(
                    descriptor,
                    min(65_536, MAX_RECORD_BYTES + 1 - len(payload)),
                )
                if not chunk:
                    break
                payload.extend(chunk)
            after = os.fstat(descriptor)
            if _file_identity(opened) != _file_identity(after) or len(payload) != after.st_size:
                raise Runtime452ActionJournalError(
                    "runtime452 journal record changed while being read"
                )
            self.revalidate()
            return bytes(payload)
        except Runtime452ActionJournalError:
            raise
        except OSError as exc:
            raise Runtime452ActionJournalError("runtime452 journal record cannot be read") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def read_json(self, name: str) -> dict[str, Any]:
        payload = self.read_bytes(name)
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Runtime452ActionJournalError(
                "runtime452 journal record is not canonical JSON"
            ) from exc
        if type(document) is not dict or payload != _canonical_json(document):
            raise Runtime452ActionJournalError("runtime452 journal record is not canonical JSON")
        return document

    @contextmanager
    def locked(self) -> Iterator[None]:
        contract = self._contract
        self._session_lock.acquire()
        connection: socket.socket | None = None
        completed = False
        try:
            if self._session_connection is not None:
                raise Runtime452ActionJournalError(
                    "runtime452 recorder transition session is already active"
                )
            connection = self._open_recorder_connection()
            begin = self._recorder_request(
                connection,
                {
                    "schema_version": 1,
                    "protocol": contract.JOURNAL_RECORDER_PROTOCOL,
                    "operation": "begin_session",
                    "actor": self.claim.actor,
                    "action": self.claim.action,
                    "claim_sha256": self.claim.claim_sha256,
                },
            )
            if _canonical_json(begin) != _canonical_json(
                {
                    "schema_version": 1,
                    "protocol": contract.JOURNAL_RECORDER_PROTOCOL,
                    "operation": "begin_session",
                    "status": "session_begun",
                    "actor": self.claim.actor,
                }
            ):
                raise Runtime452ActionJournalError(
                    "runtime452 recorder did not begin the exact transition session"
                )
            self._session_connection = connection
            self._session_owner = threading.get_ident()
            yield
            self.revalidate()
            completed = True
        finally:
            try:
                if completed and connection is not None:
                    end = self._recorder_request(
                        connection,
                        {
                            "schema_version": 1,
                            "protocol": contract.JOURNAL_RECORDER_PROTOCOL,
                            "operation": "end_session",
                            "actor": self.claim.actor,
                            "action": self.claim.action,
                            "claim_sha256": self.claim.claim_sha256,
                        },
                    )
                    if _canonical_json(end) != _canonical_json(
                        {
                            "schema_version": 1,
                            "protocol": contract.JOURNAL_RECORDER_PROTOCOL,
                            "operation": "end_session",
                            "status": "session_ended",
                            "actor": self.claim.actor,
                        }
                    ):
                        raise Runtime452ActionJournalError(
                            "runtime452 recorder did not end the exact transition session"
                        )
            finally:
                self._session_connection = None
                self._session_owner = None
                if connection is not None:
                    connection.close()
                self._session_lock.release()

    def close(self) -> None:
        connection = getattr(self, "_session_connection", None)
        if connection is not None:
            connection.close()
            self._session_connection = None
            self._session_owner = None
        for name in ("_journal_fd", "_marker_fd", "_recorder_root_fd", "_root_fd"):
            descriptor = getattr(self, name, None)
            if descriptor is not None:
                os.close(descriptor)
                setattr(self, name, None)

    def __enter__(self) -> "Runtime452ActionJournalCustody":
        self.revalidate()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass
