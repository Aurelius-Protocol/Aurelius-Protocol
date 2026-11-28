"""Pure Axon endpoint policy shared by miner publication and validator discovery.

A miner may advertise a literal global-unicast address. It may advertise another one only when
the operator lists that address's network in ``AURELIUS_AXON_ALLOWED_CIDRS``. A listed network
must consist of addresses that are not globally routable, so private ranges and shared address
space both qualify and a public block does not. Loopback, link-local, multicast, reserved and
unspecified addresses are never admitted, and a malformed value fails startup instead of
widening the policy.
"""

from __future__ import annotations
import ipaddress
from dataclasses import dataclass


class AxonEndpointPolicyError(ValueError):
    """The endpoint address, or the networks allowed for it, are malformed."""


@dataclass(frozen=True)
class AxonEndpointPolicy:
    allowed_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()

    @classmethod
    def build(cls, *, allowed_cidrs: str | None = None) -> "AxonEndpointPolicy":
        if allowed_cidrs is None or not allowed_cidrs.strip():
            return cls()
        networks = []
        try:
            for raw in allowed_cidrs.split(","):
                network = ipaddress.ip_network(raw.strip(), strict=True)
                if network.network_address.is_global or network.broadcast_address.is_global:
                    raise AxonEndpointPolicyError(
                        "AURELIUS_AXON_ALLOWED_CIDRS must list networks of addresses that are "
                        f"not globally routable, got {network}"
                    )
                networks.append(network)
        except ValueError as exc:
            raise AxonEndpointPolicyError(f"invalid AURELIUS_AXON_ALLOWED_CIDRS: {exc}") from None
        if len(networks) != len(set(networks)):
            raise AxonEndpointPolicyError("AURELIUS_AXON_ALLOWED_CIDRS contains duplicates")
        return cls(allowed_networks=tuple(networks))

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
        if address.is_global:
            return True
        return any(
            address.version == network.version and address in network
            for network in self.allowed_networks
        )

    def require(self, raw_ip: object) -> str:
        if not self.allows(raw_ip):
            raise AxonEndpointPolicyError(
                f"Axon endpoint {raw_ip!r} is forbidden: it is not a global-unicast address "
                "and its network is not in AURELIUS_AXON_ALLOWED_CIDRS"
            )
        return str(ipaddress.ip_address(raw_ip))
