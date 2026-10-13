"""Public miner provider credentials and endpoint ambiguity guards."""

from types import SimpleNamespace

import pytest
from subnet.bittensor_v11 import (
    BittensorV11Error,
    FinalizedRpc,
    open_subtensor,
    validate_subtensor_endpoint,
)

pytestmark = pytest.mark.miner


def test_provider_credentials_survive_both_readers_without_sdk_fallback():
    uri = "wss://fixture.onfinality.io/ws?apikey=synthetic-fixture"
    assert validate_subtensor_endpoint(uri) == uri
    assert FinalizedRpc(uri).endpoint == uri
    assert open_subtensor(SimpleNamespace(Subtensor=lambda **kw: kw), uri) == dict(
        network=uri, fallback_endpoints=[], archive_endpoints=[]
    )


@pytest.mark.parametrize(
    "uri",
    [
        "wss://onfinality.io.attacker.invalid/ws?apikey=synthetic",
        "ws://fixture.onfinality.io/ws?apikey=synthetic",
        "wss://fixture.onfinality.io:444/ws?apikey=synthetic",
        "wss://user:password@fixture.onfinality.io/ws?apikey=synthetic",
        "wss://fixture.onfinality.io/ws?apikey=synthetic#fragment",
        "wss://unreviewed.example/ws?apikey=synthetic",
    ],
)
def test_provider_selection_cannot_relax_other_endpoint_guards(uri):
    with pytest.raises(BittensorV11Error):
        validate_subtensor_endpoint(uri)
