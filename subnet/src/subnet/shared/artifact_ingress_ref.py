# Miner dependency subset extracted from Aurelius v5 (02375403).
r"""aurelius-artifact:// ingress ref grammar (path-free artifact upload ingress).

CONTRACT (capsule header)
- Scope: pure data grammar — the exact locator string naming one uploaded artifact object
  (`aurelius-artifact://<ss58-42-hotkey>/<sha256>`) plus the closed set of object names an
  upload may contain. No filesystem, network, wallet or lane import lives here; `subnet.shared`
  stays dependency-free in-repo (CI contract `shared-independent`).
- Guarantees: `parse_aurelius_artifact_ref` is exact-match AND canonical — the hotkey must
  round-trip through SS58 prefix 42 (`ss58_encode(ss58_decode(h)) == h`), so a syntactically
  valid but non-canonical address is rejected, never normalized. SS58 addresses are
  case-sensitive; URL-style hostname normalization would be a confused-deputy bug.
- `ARTIFACT_INGRESS_REF_PATTERN` is the exported single regex: the lane's JSON schema and
  Central's validator are compared against exactly this string (tests pin it). It is
  anchored `\A...\Z` — not `^...$` — because a JSON-schema `pattern` is a search and `$`
  matches before a trailing newline; Central's `@ingress_artifact_ref` carries the identical
  `\A...\z` anchors, and the two grammars changed together (shared shape table, both suites).

The ss58 codec is imported lazily (mirroring `sn455_artifact_delivery._hotkey`): cold GPU
supervisors import lane declarations before their admission gate, and the address codec
belongs to actual artifact validation, not module initialization.
"""

from __future__ import annotations
from dataclasses import dataclass
import re

ARTIFACT_INGRESS_REF_PATTERN = r"\Aaurelius-artifact://[1-9A-HJ-NP-Za-km-z]{48}/[0-9a-f]{64}\Z"


_REF = re.compile(ARTIFACT_INGRESS_REF_PATTERN)


_DIGEST = re.compile(r"[0-9a-f]{64}")


def _hotkey(value: object) -> str:
    from subnet.shared.ss58 import ss58_decode, ss58_encode  # lazy — see module docstring

    if type(value) is not str or len(value) != 48 or ss58_encode(ss58_decode(value)) != value:
        raise ValueError("artifact ingress hotkey must be canonical SS58-42")
    return value


def _digest(value: object) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError("artifact ingress digest must be lowercase 64-hex")
    return value


@dataclass(frozen=True, slots=True)
class IngressArtifactRef:
    """One uploaded artifact: producer hotkey + sha256 digest (lowercase hex)."""

    hotkey: str
    digest: str

    def __post_init__(self) -> None:
        _hotkey(self.hotkey)
        _digest(self.digest)


def parse_aurelius_artifact_ref(value: str) -> IngressArtifactRef:
    """Exact-grammar parse; SS58 addresses are case-sensitive, so no URL normalization."""
    match = _REF.fullmatch(value) if type(value) is str else None
    if match is None:
        raise ValueError(
            "artifact ingress ref must be an exact aurelius-artifact://hotkey/digest locator"
        )
    # The pattern guarantees exactly "<scheme><48 base58>/<64 hex>"; split without URL parsing.
    _, _, body = match.group(0).partition("://")
    hotkey, _, digest = body.partition("/")
    return IngressArtifactRef(hotkey, digest)


def is_aurelius_artifact_ref(value: object) -> bool:
    """Fail-closed grammar+canonicality check — never raises on any input."""
    try:
        parse_aurelius_artifact_ref(value)
    except ValueError:
        return False
    return True
