from pydantic import BaseModel

from aurelius.common.version import PROTOCOL_VERSION


class ScenarioConfigSynapse(BaseModel):
    """Wire model for exchanging moral dilemma scenario configurations.

    Validators set immutable fields before sending; miners populate mutable
    fields. Since bittensor 11 removed the Synapse/axon/dendrite stack, this
    is a plain pydantic model carried as JSON over the subnet's own HTTP
    transport (see `aurelius.transport`); the field set is identical to the
    pre-11 Synapse so the wire payload is unchanged.

    VU2/VU3 VERSIONING POLICY:
    - New fields MUST be Optional with a default value (additive-only).
    - Removing a field or promoting Optional -> required is a BREAKING CHANGE
      that requires a MAJOR version bump in PROTOCOL_VERSION.
    - The version check in pipeline._version_check() enforces major mismatches.
    """

    # Immutable — set by validator
    request_id: str = ""
    validator_version: str = ""
    protocol_version: str = PROTOCOL_VERSION

    # Mutable — set by miner
    scenario_config: dict | None = None
    work_id: str | None = None
    work_id_nonce: str | None = None
    work_id_time_ns: str | None = None
    work_id_signature: str | None = None  # Miner's hotkey signature over work_id (ownership proof)
    miner_version: str | None = None
    miner_protocol_version: str | None = None
