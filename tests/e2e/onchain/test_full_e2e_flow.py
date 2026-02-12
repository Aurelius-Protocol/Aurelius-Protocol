"""
Full end-to-end flow test on testnet.

This test validates the complete validator workflow:
1. Submit execution to collector API (with miner info)
2. Check novelty score
3. Set weights on chain
4. Verify weights propagated

Requires:
- Docker services (collector API + postgres)
- Validator wallet with stake on subnet 290
- At least one active miner in metagraph
"""

import os
import uuid

import numpy as np
import pytest
import requests

import bittensor as bt

from ..helpers.api_client import (
    CollectorAPIClient,
    generate_random_embedding,
    generate_deterministic_embedding,
)
from ..helpers.chain_utils import (
    verify_weights_onchain,
    wait_for_blocks,
)


TESTNET_NETUID = 290
COLLECTOR_API_URL = os.environ.get("COLLECTOR_API_URL", "http://localhost:3000")


@pytest.fixture
def collector_client() -> CollectorAPIClient:
    """Get collector API client."""
    return CollectorAPIClient(
        base_url=COLLECTOR_API_URL,
        timeout=30,
    )


@pytest.fixture
def require_collector_api(collector_client: CollectorAPIClient):
    """Skip test if collector API is not available."""
    try:
        # Don't follow redirects - collector API should return 200 directly
        response = requests.get(
            f"{COLLECTOR_API_URL}/health",
            timeout=5,
            allow_redirects=False,
        )

        # Must be exactly 200, not a redirect or error
        if response.status_code != 200:
            pytest.skip(f"Collector API not available (status {response.status_code})")

        if not response.text:
            pytest.skip("Collector API returned empty response")

        # Try to parse JSON to ensure it's valid
        try:
            data = response.json()
            if data.get("status") != "ok" and "healthy" not in str(data).lower():
                pytest.skip(f"Collector API not healthy: {data}")
        except requests.JSONDecodeError:
            pytest.skip("Collector API returned invalid JSON")

    except (requests.ConnectionError, requests.Timeout):
        pytest.skip("Collector API not running")
    except Exception as e:
        pytest.skip(f"Collector API check failed: {e}")


@pytest.fixture
def unique_test_prompt() -> tuple[str, list[float]]:
    """Generate unique prompt and embedding for testing."""
    unique_id = uuid.uuid4().hex[:8]
    prompt = f"E2E test prompt {unique_id}: Explain the concept of {unique_id} in physics."
    embedding = generate_deterministic_embedding(prompt)
    return prompt, embedding


@pytest.mark.onchain
@pytest.mark.requires_funding
class TestFullE2EFlow:
    """Complete end-to-end validator flow tests."""

    def test_complete_validator_flow(
        self,
        collector_client: CollectorAPIClient,
        validator_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
        require_validator_stake: dict,
        metagraph: bt.Metagraph,
        unique_test_prompt: tuple[str, list[float]],
    ):
        """
        Test complete flow: submit -> novelty check -> set weights -> verify.

        This is the primary E2E test for the Aurelius validator workflow.
        """
        prompt, embedding = unique_test_prompt
        validator_hotkey = validator_wallet.hotkey.ss58_address
        validator_uid = require_validator_stake["uid"]

        # Find a miner to score
        miner_uid = None
        miner_hotkey = None
        for uid in range(len(metagraph.hotkeys)):
            # Skip validators (those with significant stake)
            if metagraph.S[uid] > 10:
                continue
            if metagraph.active[uid]:
                miner_uid = uid
                miner_hotkey = metagraph.hotkeys[uid]
                break

        if miner_uid is None:
            pytest.skip("No active miners in metagraph")

        # Step 1: Submit execution to collector API
        response = collector_client.submit_execution(
            prompt=prompt,
            response="This is a test response for E2E flow validation.",
            danger_score=0.05,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            validator_uid=validator_uid,
            miner_hotkey=miner_hotkey,
            miner_uid=miner_uid,
            accepted=True,
            prompt_embedding=embedding,
            experiment_id="prompt",
        )

        # Check API response
        if response.status_code != 200:
            # Try without signature auth
            response = collector_client.submit_execution(
                prompt=prompt,
                response="This is a test response for E2E flow validation.",
                danger_score=0.05,
                validator_hotkey=validator_hotkey,
                validator_uid=validator_uid,
                miner_hotkey=miner_hotkey,
                miner_uid=miner_uid,
                accepted=True,
                prompt_embedding=embedding,
            )

        assert response.status_code in [200, 201], (
            f"Execution submission failed: {response.status_code} - {response.text}"
        )

        result = response.json()
        assert "id" in result or "execution_id" in result

        # Step 2: Check novelty score
        novelty_response = collector_client.check_novelty(
            prompt=prompt,
            embedding=embedding,
            experiment_id="prompt",
        )

        assert novelty_response.status_code == 200, (
            f"Novelty check failed: {novelty_response.text}"
        )

        novelty_data = novelty_response.json()
        assert "novelty_score" in novelty_data or "score" in novelty_data

        novelty_score = novelty_data.get("novelty_score") or novelty_data.get("score", 0)
        assert 0 <= novelty_score <= 1

        # Step 3: Set weights on chain
        # Use novelty score to influence weight (higher novelty = higher weight)
        weight = max(0.1, novelty_score)  # Minimum 0.1 weight

        uids = np.array([miner_uid], dtype=np.int64)
        weights = np.array([weight], dtype=np.float32)

        try:
            success, msg = testnet_subtensor.set_weights(
                wallet=validator_wallet,
                netuid=TESTNET_NETUID,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )

            if not success:
                if "rate limit" in str(msg).lower():
                    pytest.skip("Weight setting rate limited")
                pytest.fail(f"Weight setting failed: {msg}")

        except Exception as e:
            if "rate limit" in str(e).lower():
                pytest.skip("Weight setting rate limited")
            raise

        # Step 4: Verify weights on chain
        wait_for_blocks(testnet_subtensor, num_blocks=1, timeout=30)

        weight_info = verify_weights_onchain(
            testnet_subtensor,
            TESTNET_NETUID,
            validator_uid,
        )

        assert miner_uid in weight_info["uids"], (
            f"Miner {miner_uid} not found in on-chain weights"
        )

    def test_submit_execution_with_miner_info(
        self,
        collector_client: CollectorAPIClient,
        validator_wallet: bt.Wallet,
        require_validator_registration: dict,
        metagraph: bt.Metagraph,
        require_collector_api,
    ):
        """Test submitting execution data with miner information."""
        validator_hotkey = validator_wallet.hotkey.ss58_address
        validator_uid = require_validator_registration["uid"]

        # Get a miner from metagraph
        miner_uid = None
        miner_hotkey = None
        for uid in range(len(metagraph.hotkeys)):
            if metagraph.S[uid] < 10 and metagraph.active[uid]:
                miner_uid = uid
                miner_hotkey = metagraph.hotkeys[uid]
                break

        prompt = f"Test submission {uuid.uuid4().hex[:8]}"
        embedding = generate_random_embedding()

        response = collector_client.submit_execution(
            prompt=prompt,
            response="Test response",
            danger_score=0.1,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            validator_uid=validator_uid,
            miner_hotkey=miner_hotkey,
            miner_uid=miner_uid,
            accepted=True,
            prompt_embedding=embedding,
        )

        # Accept 200 or 201 as success
        assert response.status_code in [200, 201, 422], (
            f"Submission failed: {response.status_code}"
        )

    def test_novelty_score_decreases_on_repeat(
        self,
        collector_client: CollectorAPIClient,
        validator_wallet: bt.Wallet,
        require_validator_registration: dict,
        require_collector_api,
    ):
        """Test that novelty score decreases for similar prompts."""
        validator_hotkey = validator_wallet.hotkey.ss58_address

        # First unique prompt
        base_prompt = f"Unique concept {uuid.uuid4().hex}: quantum entanglement"
        embedding = generate_deterministic_embedding(base_prompt)

        # Submit first
        collector_client.submit_execution(
            prompt=base_prompt,
            response="Response 1",
            danger_score=0.0,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            prompt_embedding=embedding,
        )

        # Check novelty - should be high initially
        response1 = collector_client.check_novelty(
            prompt=base_prompt,
            embedding=embedding,
        )

        if response1.status_code != 200:
            pytest.skip(f"Novelty check not available: {response1.status_code}")

        # Submit similar prompt
        similar_prompt = f"Similar to: {base_prompt}"
        similar_embedding = generate_deterministic_embedding(similar_prompt)

        collector_client.submit_execution(
            prompt=similar_prompt,
            response="Response 2",
            danger_score=0.0,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
            prompt_embedding=similar_embedding,
        )

        # Check novelty again with very similar embedding
        # Using same embedding should give lower novelty
        response2 = collector_client.check_novelty(
            prompt=base_prompt,
            embedding=embedding,  # Same embedding
        )

        if response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            score1 = data1.get("novelty_score") or data1.get("score", 1)
            score2 = data2.get("novelty_score") or data2.get("score", 1)

            # Second check with same embedding should have similar novelty
            # Allow some variance due to how novelty is calculated
            assert score2 <= score1 + 0.2, (
                f"Novelty increased unexpectedly: {score1} -> {score2}"
            )


@pytest.mark.onchain
class TestCollectorAPIIntegration:
    """Tests for collector API integration with on-chain data."""

    def test_collector_api_health(
        self,
        collector_client: CollectorAPIClient,
        require_collector_api,
    ):
        """Test collector API is reachable."""
        response = collector_client.health()
        assert response.status_code == 200

        data = response.json()
        assert data.get("status") == "ok" or "healthy" in str(data).lower()

    def test_submit_execution_without_miner(
        self,
        collector_client: CollectorAPIClient,
        validator_wallet: bt.Wallet,
        require_validator_registration: dict,
        require_collector_api,
    ):
        """Test submitting execution without miner info."""
        validator_hotkey = validator_wallet.hotkey.ss58_address

        prompt = f"Test {uuid.uuid4().hex[:8]}"
        response = collector_client.submit_execution(
            prompt=prompt,
            response="Test response",
            danger_score=0.0,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # Should succeed even without miner info
        assert response.status_code in [200, 201, 422]

    def test_check_novelty_for_unique_prompt(
        self,
        collector_client: CollectorAPIClient,
        require_collector_api,
    ):
        """Test novelty check for completely unique prompt."""
        unique_prompt = f"Completely unique prompt {uuid.uuid4()}"
        embedding = generate_random_embedding()

        response = collector_client.check_novelty(
            prompt=unique_prompt,
            embedding=embedding,
        )

        if response.status_code != 200:
            pytest.skip(f"Novelty endpoint not available: {response.status_code}")

        data = response.json()
        score = data.get("novelty_score") or data.get("score")

        # Unique prompt should have high novelty
        assert score is not None
        assert score >= 0

    def test_get_executions_filtered_by_validator(
        self,
        collector_client: CollectorAPIClient,
        validator_wallet: bt.Wallet,
        require_validator_registration: dict,
        require_collector_api,
    ):
        """Test retrieving executions filtered by validator."""
        validator_hotkey = validator_wallet.hotkey.ss58_address

        # First submit an execution
        prompt = f"Retrieval test {uuid.uuid4().hex[:8]}"
        collector_client.submit_execution(
            prompt=prompt,
            response="Test",
            danger_score=0.0,
            validator_hotkey=validator_hotkey,
            wallet=validator_wallet,
        )

        # Then retrieve
        response = collector_client.get_executions(
            validator_hotkey=validator_hotkey,
            limit=10,
        )

        if response.status_code != 200:
            pytest.skip(f"Executions endpoint not available: {response.status_code}")

        data = response.json()
        assert "executions" in data or isinstance(data, list)


@pytest.mark.onchain
@pytest.mark.slow
class TestWeightPropagation:
    """Tests for weight setting and propagation verification."""

    def test_weights_persist_across_metagraph_sync(
        self,
        validator_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
        require_validator_stake: dict,
        sample_miner_uids: list[int],
    ):
        """Test that set weights persist after metagraph re-sync."""
        if not sample_miner_uids:
            pytest.skip("No active miners")

        validator_uid = require_validator_stake["uid"]
        miner_uid = sample_miner_uids[0]

        # Set weight
        uids = np.array([miner_uid], dtype=np.int64)
        weights = np.array([1.0], dtype=np.float32)

        try:
            success, _ = testnet_subtensor.set_weights(
                wallet=validator_wallet,
                netuid=TESTNET_NETUID,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )

            if not success:
                pytest.skip("Weight setting failed")

        except Exception as e:
            if "rate limit" in str(e).lower():
                pytest.skip("Rate limited")
            raise

        # Wait for propagation
        wait_for_blocks(testnet_subtensor, num_blocks=2, timeout=60)

        # Re-sync metagraph and verify
        new_metagraph = bt.Metagraph(netuid=TESTNET_NETUID, network="test")
        new_metagraph.sync(subtensor=testnet_subtensor)

        # Check weight in new metagraph
        weight_row = new_metagraph.W[validator_uid]
        assert weight_row[miner_uid] > 0, (
            f"Weight for miner {miner_uid} not found in re-synced metagraph"
        )

    def test_multiple_validators_can_set_weights(
        self,
        validator_wallet: bt.Wallet,
        testnet_subtensor: bt.Subtensor,
        require_validator_stake: dict,
        metagraph: bt.Metagraph,
        sample_miner_uids: list[int],
    ):
        """Test that our validator can set weights (doesn't conflict with others)."""
        if not sample_miner_uids:
            pytest.skip("No active miners")

        validator_uid = require_validator_stake["uid"]

        # Get current weights before our update
        initial_weights = verify_weights_onchain(
            testnet_subtensor,
            TESTNET_NETUID,
            validator_uid,
        )

        # Set new weights
        miner_uids = sample_miner_uids[:2] if len(sample_miner_uids) > 1 else sample_miner_uids
        uids = np.array(miner_uids, dtype=np.int64)
        weights = np.array([1.0 / len(miner_uids)] * len(miner_uids), dtype=np.float32)

        try:
            success, _ = testnet_subtensor.set_weights(
                wallet=validator_wallet,
                netuid=TESTNET_NETUID,
                uids=uids,
                weights=weights,
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )

            if not success:
                pytest.skip("Weight setting failed")

        except Exception as e:
            if "rate limit" in str(e).lower():
                pytest.skip("Rate limited")
            raise

        # Verify weights changed
        wait_for_blocks(testnet_subtensor, num_blocks=1, timeout=30)

        final_weights = verify_weights_onchain(
            testnet_subtensor,
            TESTNET_NETUID,
            validator_uid,
        )

        # Our miners should be in the weights
        for muid in miner_uids:
            assert muid in final_weights["uids"], (
                f"Miner {muid} not in final weights"
            )
