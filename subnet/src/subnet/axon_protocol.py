# Miner dependency subset extracted from Aurelius v5 (02375403).
"""The validator↔miner request constants for Bittensor v11 btauth HTTP.

CONTRACT (capsule header)
- The request carries the exact v2 authorization context (validator audience,
  64-hex request identity, and authoritative fanout window) plus operator-defined task metadata;
  `candidate_source=staging` carries NO candidate config, while testlab-only `inline` carries
  `config_json`. The response carries the §3.1 envelope VERBATIM as one JSON string
  as the ordinary JSON response. V11 btauth signs the exact raw request bytes and binds them to
  method, path, sender, receiver and nonce; the Aurelius work-id independently signs the response.
- A valid request with no eligible staged work returns HTTP 204 with no body. It carries
  no work authorization and never enters evaluation or charging.
- This module has no Bittensor import. The optional SDK lives only in the btauth transport and
  miner server construction paths.
"""

from __future__ import annotations

MAX_CONFIG_JSON_BYTES = 65_536


PROTOCOL_VERSION = 1


CANDIDATE_SOURCE_INLINE = "inline"


CANDIDATE_SOURCE_STAGING = "staging"
