"""Small helpers over the bittensor 11 chain API.

bittensor 11's Metagraph is an immutable snapshot (no .sync()); async
consumers re-fetch via `bt.metagraph.fetch(client, netuid)` on a connected
`bt.Client`. These helpers cover the two conveniences the old API had that
the new one doesn't: a blocking fetch for sync startup paths, and neuron
lookups by uid/hotkey.
"""

import asyncio

import bittensor as bt


def fetch_metagraph_blocking(network: str, netuid: int):
    """One-shot metagraph fetch for sync contexts (constructor/preflight).

    Opens and closes its own connection; use `bt.metagraph.fetch` on a
    long-lived client inside async loops.
    """

    async def _fetch():
        async with bt.Client(network=network) as client:
            return await bt.metagraph.fetch(client, netuid)

    return asyncio.run(_fetch())


def neuron_for_uid(mg, uid: int):
    """Return the MetagraphNeuron with this uid, or None. UIDs are not
    guaranteed to be contiguous list indices, so scan by field."""
    for n in mg.neurons:
        if n.uid == uid:
            return n
    return None


def neuron_for_hotkey(mg, hotkey: str):
    """Return the MetagraphNeuron registered to this hotkey, or None."""
    for n in mg.neurons:
        if n.hotkey == hotkey:
            return n
    return None
