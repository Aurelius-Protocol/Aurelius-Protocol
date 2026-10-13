# Miner dependency subset extracted from Aurelius v5 (02375403).
"""Pure Axon endpoint policy shared by miner publication and validator discovery.

Production admits only literal global-unicast addresses. Private Docker/cohost addresses require
an explicit test-only profile *and* explicit CIDRs; profile typos and environment/profile mismatch
fail startup instead of weakening egress policy.
"""

from __future__ import annotations
import ipaddress
from dataclasses import dataclass


class AxonEndpointPolicyError(ValueError):
    """The endpoint profile is malformed or inappropriate for the selected environment."""


@dataclass(frozen=True)
class AxonEndpointPolicy:
    environment: str
    profile: str
    allowed_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        environment: str,
        profile: str,
        allowed_cidrs: str | None = None,
    ) -> "AxonEndpointPolicy":
        if profile == "public":
            return cls(environment=environment, profile=profile)

        permitted_environment = {
            "testlab-private": "testlab",
            "cohost": "testnet-full",
        }.get(profile)
        if permitted_environment is None:
            raise AxonEndpointPolicyError(
                "AURELIUS_AXON_ENDPOINT_PROFILE must be public, testlab-private, or cohost"
            )
        if environment != permitted_environment:
            raise AxonEndpointPolicyError(
                f"{profile} is a test-only Axon endpoint profile for {permitted_environment}"
            )
        if not isinstance(allowed_cidrs, str) or not allowed_cidrs.strip():
            raise AxonEndpointPolicyError(
                f"{profile} requires non-empty AURELIUS_AXON_ALLOWED_CIDRS"
            )
        networks = []
        try:
            for raw in allowed_cidrs.split(","):
                network = ipaddress.ip_network(raw.strip(), strict=True)
                if not network.is_private:
                    raise AxonEndpointPolicyError(
                        f"{profile} CIDRs must be private networks, got {network}"
                    )
                networks.append(network)
        except ValueError as exc:
            raise AxonEndpointPolicyError(f"invalid AURELIUS_AXON_ALLOWED_CIDRS: {exc}") from None
        if len(networks) != len(set(networks)):
            raise AxonEndpointPolicyError("AURELIUS_AXON_ALLOWED_CIDRS contains duplicates")
        return cls(environment=environment, profile=profile, allowed_networks=tuple(networks))

    def allows(self, raw_ip: object) -> bool:
        try:
            address = ipaddress.ip_address(raw_ip)
        except (TypeError, ValueError):
            return False
        if (
            address.is_unspecified
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
        ):
            return False
        if self.profile == "public":
            return address.is_global
        return any(
            address.version == network.version and address in network
            for network in self.allowed_networks
        )

    def require(self, raw_ip: object) -> str:
        if not self.allows(raw_ip):
            raise AxonEndpointPolicyError(
                f"Axon endpoint {raw_ip!r} is forbidden by profile {self.profile!r}"
            )
        return str(ipaddress.ip_address(raw_ip))
