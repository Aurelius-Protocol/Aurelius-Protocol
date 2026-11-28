"""The one address rule for a miner's advertised endpoint.

A global-unicast address is always admitted. Any other one is admitted only when the operator
lists its network in ``AURELIUS_AXON_ALLOWED_CIDRS``, and a listed network must consist of
addresses that are not globally routable. Loopback, link-local, multicast, reserved and
unspecified addresses are never admitted.
"""

import pytest

from subnet.shared.network_policy import AxonEndpointPolicy, AxonEndpointPolicyError

pytestmark = pytest.mark.shared

PUBLIC = ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"]
PRIVATE = ["10.0.0.5", "172.16.3.9", "192.168.1.20", "100.100.0.7", "fd00::5"]
NEVER = [
    "0.0.0.0",
    "127.0.0.1",
    "169.254.1.1",
    "224.0.0.1",
    "240.0.0.1",
    "::",
    "::1",
    "fe80::1",
    "ff02::1",
]


@pytest.mark.parametrize("address", PUBLIC)
def test_a_global_address_is_admitted_with_no_allowed_networks(address):
    assert AxonEndpointPolicy.build().allows(address) is True


@pytest.mark.parametrize("address", PRIVATE + NEVER)
def test_every_other_address_is_refused_with_no_allowed_networks(address):
    assert AxonEndpointPolicy.build().allows(address) is False
    with pytest.raises(AxonEndpointPolicyError, match="AURELIUS_AXON_ALLOWED_CIDRS"):
        AxonEndpointPolicy.build().require(address)


@pytest.mark.parametrize("allowed_cidrs", [None, "", "   "])
def test_no_allowed_networks_is_an_empty_policy(allowed_cidrs):
    assert AxonEndpointPolicy.build(allowed_cidrs=allowed_cidrs).allowed_networks == ()


def test_a_listed_private_network_is_admitted_and_an_unlisted_one_is_not():
    policy = AxonEndpointPolicy.build(allowed_cidrs="10.0.0.0/16,10.200.0.9/32")
    assert policy.allows("10.0.5.5") is True
    assert policy.allows("10.200.0.9") is True
    assert policy.allows("10.200.0.10") is False  # one address off the /32
    assert policy.allows("10.1.0.1") is False
    assert policy.allows("172.16.3.9") is False
    assert policy.require("10.0.5.5") == "10.0.5.5"


def test_shared_address_space_is_listable():
    """RFC 6598 shared address space (100.64.0.0/10) is where a cluster puts its service
    addresses. It is neither private nor global in Python's classification, and it is listable."""
    policy = AxonEndpointPolicy.build(allowed_cidrs="100.100.0.0/16")
    assert policy.allows("100.100.0.7") is True
    assert policy.allows("100.101.0.7") is False


def test_a_public_address_stays_admitted_when_private_networks_are_listed():
    policy = AxonEndpointPolicy.build(allowed_cidrs="10.0.0.0/8")
    assert policy.allows("8.8.8.8") is True


def test_a_listed_network_admits_only_addresses_of_its_own_version():
    policy = AxonEndpointPolicy.build(allowed_cidrs="fd00::/8")
    assert policy.allows("fd00::5") is True
    assert policy.allows("10.0.0.5") is False


@pytest.mark.parametrize("address", NEVER)
def test_no_listed_network_admits_an_address_that_is_never_admitted(address):
    """Loopback, link-local and the unspecified address count as private, so an operator could
    list them. Listing them admits nothing."""
    policy = AxonEndpointPolicy.build(
        allowed_cidrs="127.0.0.0/8,169.254.0.0/16,0.0.0.0/8,::1/128,fe80::/10,::/128"
    )
    assert policy.allows(address) is False


@pytest.mark.parametrize("address", [None, "", "x", "999.1.1.1", "10.0.0.5/32", 10])
def test_malformed_input_is_refused_never_raised(address):
    assert AxonEndpointPolicy.build(allowed_cidrs="10.0.0.0/8").allows(address) is False


@pytest.mark.parametrize(
    ("allowed_cidrs", "match"),
    [
        ("8.8.8.0/24", "not globally routable"),  # a public block
        ("100.63.0.0/16", "not globally routable"),  # public space next to the shared block
        ("10.0.0.1/24", "invalid"),  # host bits set: not a network
        ("10.0.0.0/33", "invalid"),
        ("10.0.0.0/8,10.0.0.0/8", "duplicates"),
        ("10.0.0.0/8,", "invalid"),  # an empty entry is malformed
    ],
)
def test_a_malformed_or_public_network_fails_to_build(allowed_cidrs, match):
    with pytest.raises(AxonEndpointPolicyError, match=match):
        AxonEndpointPolicy.build(allowed_cidrs=allowed_cidrs)
