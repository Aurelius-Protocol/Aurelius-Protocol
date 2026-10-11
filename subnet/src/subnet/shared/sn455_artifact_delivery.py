# Miner dependency subset extracted from Aurelius v5 (02375403).
"""ADR0163: path-free, independently pinned SN455 local artifact delivery (inv11/27/28).

This is data, not filesystem or training authority. The trusted immutable deployment must supply
the admission's expected hash independently of the document, schema and candidate. It must first
verify the retained producer/seal evidence and fresh root-owned destination inodes. A Python
value is not an unforgeable capability. No wallet, network or lane import lives here; the ONE
deliberate environment read is the ADR-0205 item 1 ingress-only posture flag
(``interp_sae_ingress_only``), a pure string comparison shared by the validator loop and the
preseed CLI (r2 python: the preseed command must not import a package-private loop name across
packages).
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import re

PROFILE_ID = "gh200-sn455-one-validator"


CHAIN_GENESIS = "0x8f9cf856bf558a14440e75569c9e58594757048d7b3a84b5d25f6bd978263105"


NETUID = 455


VALIDATOR_HOTKEY = "5F1hDD7jveZrr95Kb2sJF27vzZm2348UGYo6HhpssRDZtiFD"


MINER_HOTKEYS = (
    "5CnmfdG2HmzXFrdpdfLq1jMJjhqxVWucxrcowZNE2RoZRzgs",
    "5DnvYrsPtPksiz68EjQsTBE5ZJxrF581eNbgi3wvcMcmZskn",
    "5Df2NGWgGhKHYBWfivty3AaL42bnKXSeFGcjCCdULdXk9yAe",
    "5CkYMTPdifgqzfzYRLTfyhpN37zwGtX2mSycxPqNiqztbFuv",
)


ADMISSION_CONTRACT = "aurelius-sn455-local-artifacts-v1"


EXPORT_CONTRACT = "aurelius-sn455-miner-export-v1"


MAX_ADMISSION_BYTES = 262_144


MAX_RECORD_BYTES = 65_536


MAX_EXPORTS_PER_MINER = 8


MAX_OBJECTS_PER_EXPORT = 16


MAX_LOCAL_ARTIFACT_BYTES = 4 * 1024**3


_DIGEST = re.compile(r"[0-9a-f]{64}")


_OBJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _digest(value: object) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError("artifact delivery digest must be lowercase 64-hex")
    return value


def _hotkey(value: object) -> str:
    # Cold GPU supervisors import lane declarations before their admission gate. The
    # address codec belongs to actual artifact validation, not module initialization.
    from subnet.shared.ss58 import ss58_decode, ss58_encode

    if type(value) is not str or len(value) != 48 or ss58_encode(ss58_decode(value)) != value:
        raise ValueError("artifact delivery hotkey must be canonical SS58-42")
    return value


def _uint(value: object, field: str, *, maximum: int = 2**64 - 1) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"artifact delivery {field} must be a bounded exact unsigned integer")
    return value


def _closed(value: object, names: set[str]) -> dict:
    if type(value) is not dict or set(value) != names:
        raise ValueError("artifact delivery object fields are not closed")
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class MinerArtifactRef:
    miner_hotkey: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _hotkey(self.miner_hotkey)
        _digest(self.artifact_sha256)


def miner_artifact_ref(hotkey: str, digest: str) -> str:
    ref = MinerArtifactRef(hotkey, digest)
    return f"miner-artifact://{ref.miner_hotkey}/{ref.artifact_sha256}"


@dataclass(frozen=True, slots=True)
class SealedFileIdentity:
    """Actual post-seal stat values, pinned externally; no path or file descriptor."""

    device: int
    inode: int
    size_bytes: int
    mode: int
    uid: int
    gid: int
    links: int
    mtime_ns: int
    ctime_ns: int

    def __post_init__(self) -> None:
        for item in fields(self):
            _uint(getattr(self, item.name), item.name)
        if self.inode == 0 or self.links == 0 or self.mode > 0o7777:
            raise ValueError("sealed filesystem identity is invalid")

    @classmethod
    def from_mapping(cls, raw: object) -> SealedFileIdentity:
        return cls(**_closed(raw, {field.name for field in fields(cls)}))


@dataclass(frozen=True, slots=True)
class SealedArtifactObject:
    name: str
    size_bytes: int
    sha256: str
    identity: SealedFileIdentity

    def __post_init__(self) -> None:
        if type(self.name) is not str or _OBJECT_NAME.fullmatch(self.name) is None:
            raise ValueError("local artifact objects require safe flat names")
        _uint(self.size_bytes, "object size", maximum=MAX_LOCAL_ARTIFACT_BYTES)
        _digest(self.sha256)
        if type(self.identity) is not SealedFileIdentity:
            raise ValueError("local artifact object requires a sealed file identity")
        self.identity.__post_init__()
        if (
            self.identity.size_bytes != self.size_bytes
            or self.identity.links != 1
            or self.identity.mode != 0o444
        ):
            raise ValueError("local artifact requires an exact single-link read-only file identity")

    @classmethod
    def from_mapping(cls, raw: object) -> SealedArtifactObject:
        value = _closed(raw, {field.name for field in fields(cls)})
        return cls(**{**value, "identity": SealedFileIdentity.from_mapping(value["identity"])})


@dataclass(frozen=True, slots=True)
class SealedMinerExport:
    miner_hotkey: str
    artifact_sha256: str
    generation_sha256: str
    producer_provenance_sha256: str
    seal_proof_sha256: str
    root_identity: SealedFileIdentity
    objects_identity: SealedFileIdentity
    record_identity: SealedFileIdentity
    objects: tuple[SealedArtifactObject, ...]

    def __post_init__(self) -> None:
        if _hotkey(self.miner_hotkey) not in MINER_HOTKEYS:
            raise ValueError("export miner is outside the fixed SN455 hotkey roster")
        for digest in (
            self.artifact_sha256,
            self.generation_sha256,
            self.producer_provenance_sha256,
            self.seal_proof_sha256,
        ):
            _digest(digest)
        if type(self.objects) is not tuple or not 1 <= len(self.objects) <= MAX_OBJECTS_PER_EXPORT:
            raise ValueError("export object inventory is empty or exceeds its bound")
        for obj in self.objects:
            if type(obj) is not SealedArtifactObject:
                raise ValueError("export object inventory requires typed objects")
            obj.__post_init__()
        names = tuple(obj.name for obj in self.objects)
        if names != tuple(sorted(set(names))):
            raise ValueError("export objects must be unique and canonically ordered")
        if not 0 < sum(obj.size_bytes for obj in self.objects) <= MAX_LOCAL_ARTIFACT_BYTES:
            raise ValueError("export payload exceeds its aggregate bound")
        identities = (self.root_identity, self.objects_identity, self.record_identity)
        for value in identities:
            if type(value) is not SealedFileIdentity:
                raise ValueError("export requires typed sealed identities")
            value.__post_init__()
        for directory in identities[:2]:
            if directory.mode != 0o555 or directory.size_bytes > 1_048_576:
                raise ValueError("export requires bounded read-only directory identities")
        if self.record_identity.mode != 0o444 or self.record_identity.links != 1:
            raise ValueError("export requires a single-link read-only record identity")
        all_identities = (*identities, *(obj.identity for obj in self.objects))
        if any(value.uid != self.root_identity.uid for value in all_identities):
            raise ValueError("sealed export owners differ")
        record = self.record_bytes()
        if len(record) > MAX_RECORD_BYTES or self.record_identity.size_bytes != len(record):
            raise ValueError("sealed record size differs from the canonical record")
        if hashlib.sha256(record).hexdigest() != self.generation_sha256:
            raise ValueError("generation digest differs from the canonical export record")

    def record_bytes(self) -> bytes:
        return _canonical(
            {
                "contract": EXPORT_CONTRACT,
                "profile_id": PROFILE_ID,
                "chain_genesis": CHAIN_GENESIS,
                "netuid": NETUID,
                "validator_hotkey": VALIDATOR_HOTKEY,
                "miner_hotkey": self.miner_hotkey,
                "artifact_sha256": self.artifact_sha256,
                "producer_provenance_sha256": self.producer_provenance_sha256,
                "objects": [
                    {"name": obj.name, "size_bytes": obj.size_bytes, "sha256": obj.sha256}
                    for obj in self.objects
                ],
            }
        )

    @classmethod
    def from_mapping(cls, raw: object) -> SealedMinerExport:
        value = _closed(raw, {field.name for field in fields(cls)})
        if type(value["objects"]) is not list or len(value["objects"]) > MAX_OBJECTS_PER_EXPORT:
            raise ValueError("export objects must be a bounded array")
        return cls(
            **{
                **value,
                "root_identity": SealedFileIdentity.from_mapping(value["root_identity"]),
                "objects_identity": SealedFileIdentity.from_mapping(value["objects_identity"]),
                "record_identity": SealedFileIdentity.from_mapping(value["record_identity"]),
                "objects": tuple(
                    SealedArtifactObject.from_mapping(obj) for obj in value["objects"]
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class Sn455ArtifactAdmission:
    """Trusted-construction input; its hash must NOT come from the document being admitted."""

    descriptor_sha256: str
    exports: tuple[SealedMinerExport, ...]

    def __post_init__(self) -> None:
        _digest(self.descriptor_sha256)
        if type(self.exports) is not tuple or len(self.exports) > 4 * MAX_EXPORTS_PER_MINER:
            raise ValueError("local admission export inventory exceeds its bound")
        for export in self.exports:
            if type(export) is not SealedMinerExport:
                raise ValueError("local admission requires typed exports")
            export.__post_init__()
        keys = tuple((export.miner_hotkey, export.artifact_sha256) for export in self.exports)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("local admission exports must be unique and canonically ordered")
        for hotkey in MINER_HOTKEYS:
            if (
                sum(export.miner_hotkey == hotkey for export in self.exports)
                > MAX_EXPORTS_PER_MINER
            ):
                raise ValueError("per-miner export count exceeds its bound")
        raw = _canonical(self.as_mapping())
        if (
            len(raw) > MAX_ADMISSION_BYTES
            or hashlib.sha256(raw).hexdigest() != self.descriptor_sha256
        ):
            raise ValueError("local admission differs from its independently supplied digest")

    def as_mapping(self) -> dict:
        return {
            "contract": ADMISSION_CONTRACT,
            "profile_id": PROFILE_ID,
            "chain_genesis": CHAIN_GENESIS,
            "netuid": NETUID,
            "validator_hotkey": VALIDATOR_HOTKEY,
            "miner_hotkeys": list(MINER_HOTKEYS),
            "exports": [
                {**asdict(export), "objects": [asdict(obj) for obj in export.objects]}
                for export in self.exports
            ],
        }

    def require_export(self, hotkey: str, digest: str) -> SealedMinerExport:
        require_artifact_admission(self)
        _hotkey(hotkey)
        _digest(digest)
        for export in self.exports:
            if (export.miner_hotkey, export.artifact_sha256) == (hotkey, digest):
                return export
        raise ValueError("this miner has no admitted sealed export for the artifact digest")

    def ref_for(self, hotkey: str, digest: str) -> str:
        self.require_export(hotkey, digest)
        return miner_artifact_ref(hotkey, digest)

    @classmethod
    def from_bytes(cls, raw: bytes, *, expected_sha256: str) -> Sn455ArtifactAdmission:
        _digest(expected_sha256)
        if type(raw) is not bytes or not 0 < len(raw) <= MAX_ADMISSION_BYTES:
            raise ValueError("local admission must be bounded bytes")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("local admission digest differs from the independent input pin")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate local admission JSON key")
                result[key] = value
            return result

        def no_decimal(value):
            del value
            raise ValueError("local admission JSON is integers-only")

        try:
            value = json.loads(
                raw, object_pairs_hook=unique, parse_float=no_decimal, parse_constant=no_decimal
            )
        except (UnicodeError, RecursionError) as exc:
            raise ValueError("invalid local admission JSON") from exc
        expected = {
            "contract": ADMISSION_CONTRACT,
            "profile_id": PROFILE_ID,
            "chain_genesis": CHAIN_GENESIS,
            "netuid": NETUID,
            "validator_hotkey": VALIDATOR_HOTKEY,
            "miner_hotkeys": list(MINER_HOTKEYS),
        }
        _closed(value, {*expected, "exports"})
        for key, pin in expected.items():
            if type(value[key]) is not type(pin) or value[key] != pin:
                raise ValueError("local admission differs from the fixed SN455 scope")
        if type(value["exports"]) is not list or len(value["exports"]) > 4 * MAX_EXPORTS_PER_MINER:
            raise ValueError("local admission exports must be a bounded array")
        admission = cls(
            expected_sha256, tuple(SealedMinerExport.from_mapping(row) for row in value["exports"])
        )
        if _canonical(admission.as_mapping()) != raw:
            raise ValueError("local admission bytes are not canonical")
        return admission


def require_artifact_admission(value: object) -> Sn455ArtifactAdmission:
    if type(value) is not Sn455ArtifactAdmission:
        raise ValueError("an explicit image-owned SN455 artifact admission is required")
    value.__post_init__()
    return value
