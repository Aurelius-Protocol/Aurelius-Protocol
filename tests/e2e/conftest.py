"""
Shared fixtures for E2E integration tests.

These fixtures handle:
- Docker compose management for local services
- Testnet subtensor connection
- Wallet loading and verification
- Registration status checking
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Generator

import pytest
import requests

# Import bittensor conditionally for environments where it's not available
try:
    import bittensor as bt
except ImportError:
    bt = None

from .helpers.api_client import CollectorAPIClient
from .helpers.wallet_utils import verify_wallet_exists, get_hotkey_address


# Test configuration
TESTNET_NETWORK = "test"
TESTNET_NETUID = 290
MIN_STAKE_TAO = 100
WALLET_NAME = os.environ.get("VALIDATOR_WALLET_NAME", "validator")
HOTKEY_NAME = os.environ.get("VALIDATOR_HOTKEY", "default")
COLLECTOR_API_URL = os.environ.get("COLLECTOR_API_URL", "http://localhost:3000")
DOCKER_COMPOSE_PATH = Path(__file__).parent.parent.parent.parent / "docker-compose.test.yml"


# Docker fixtures

@pytest.fixture(scope="session")
def docker_compose_file() -> Path:
    """Path to docker-compose file."""
    return DOCKER_COMPOSE_PATH


@pytest.fixture(scope="session")
def docker_services(docker_compose_file: Path) -> Generator[dict[str, str], None, None]:
    """
    Start postgres and collector-api containers for testing.

    Yields:
        Dict with service URLs: {api_url, db_url}
    """
    if not docker_compose_file.exists():
        pytest.skip(f"Docker compose file not found: {docker_compose_file}")

    # Start services
    try:
        subprocess.run(
            [
                "docker", "compose",
                "-f", str(docker_compose_file),
                "up", "-d",
                "postgres", "collector-api",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to start Docker services: {e.stderr.decode()}")
    except FileNotFoundError:
        pytest.skip("Docker not installed or not in PATH")

    # Wait for services to be healthy
    api_url = COLLECTOR_API_URL
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        # Clean up on failure
        subprocess.run(
            ["docker", "compose", "-f", str(docker_compose_file), "down"],
            capture_output=True,
        )
        pytest.skip("Docker services failed to become healthy")

    yield {
        "api_url": api_url,
        "db_url": os.environ.get("DATABASE_URL", "postgresql://aurelius:aurelius@localhost:5432/aurelius_test"),
    }

    # Cleanup
    subprocess.run(
        ["docker", "compose", "-f", str(docker_compose_file), "down"],
        capture_output=True,
    )


@pytest.fixture(scope="session")
def collector_api(docker_services: dict[str, str]) -> CollectorAPIClient:
    """
    Collector API client connected to Docker services.

    Requires docker_services fixture.
    """
    api_key = os.environ.get("COLLECTOR_API_KEY")
    return CollectorAPIClient(
        base_url=docker_services["api_url"],
        api_key=api_key,
    )


@pytest.fixture(scope="function")
def api_client() -> CollectorAPIClient:
    """
    Standalone API client without Docker dependency.

    Uses COLLECTOR_API_URL environment variable.
    """
    api_key = os.environ.get("COLLECTOR_API_KEY")
    return CollectorAPIClient(
        base_url=COLLECTOR_API_URL,
        api_key=api_key,
    )


# Wallet fixtures

@pytest.fixture(scope="session")
def wallet_info() -> dict[str, Any]:
    """
    Get wallet information without loading the full wallet.

    Returns:
        Dict with wallet status and paths
    """
    info = verify_wallet_exists(WALLET_NAME, HOTKEY_NAME)
    info["wallet_name"] = WALLET_NAME
    info["hotkey_name"] = HOTKEY_NAME
    info["hotkey_address"] = get_hotkey_address(WALLET_NAME, HOTKEY_NAME)
    return info


@pytest.fixture(scope="session")
def validator_wallet(wallet_info: dict[str, Any]) -> "bt.Wallet":
    """
    Load validator wallet for testing.

    Skips if wallet doesn't exist or bittensor not available.
    """
    if bt is None:
        pytest.skip("bittensor not installed")

    if not wallet_info["exists"]:
        pytest.skip(f"Wallet not found: {wallet_info['path']}")

    if not wallet_info["hotkey_exists"]:
        pytest.skip(f"Hotkey not found for wallet: {wallet_info['wallet_name']}")

    try:
        return bt.Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
    except Exception as e:
        pytest.skip(f"Failed to load wallet: {e}")


@pytest.fixture(scope="session")
def validator_hotkey(wallet_info: dict[str, Any]) -> str:
    """
    Get validator hotkey address.

    Skips if not available.
    """
    hotkey = wallet_info.get("hotkey_address")
    if not hotkey:
        pytest.skip("Hotkey address not available")
    return hotkey


# Bittensor network fixtures

@pytest.fixture(scope="session")
def testnet_subtensor() -> "bt.Subtensor":
    """
    Connect to Bittensor testnet.

    Skips if bittensor not available or connection fails.
    """
    if bt is None:
        pytest.skip("bittensor not installed")

    try:
        subtensor = bt.Subtensor(network=TESTNET_NETWORK)
        # Test connection
        _ = subtensor.block
        return subtensor
    except Exception as e:
        pytest.skip(f"Failed to connect to testnet: {e}")


@pytest.fixture(scope="session")
def metagraph(testnet_subtensor: "bt.Subtensor") -> "bt.Metagraph":
    """
    Get metagraph for subnet 290.

    Skips if sync fails.
    """
    try:
        metagraph = bt.Metagraph(netuid=TESTNET_NETUID, network=TESTNET_NETWORK)
        metagraph.sync(subtensor=testnet_subtensor)
        return metagraph
    except Exception as e:
        pytest.skip(f"Failed to sync metagraph: {e}")


@pytest.fixture(scope="session")
def validator_registered(
    testnet_subtensor: "bt.Subtensor",
    metagraph: "bt.Metagraph",
    validator_hotkey: str,
) -> dict[str, Any]:
    """
    Check if validator is registered on subnet 290.

    Returns:
        Dict with registration status: {registered, uid, stake}

    Skips test if not registered (for tests that require registration).
    """
    # Check if hotkey is in metagraph
    try:
        hotkeys = metagraph.hotkeys
        if validator_hotkey in hotkeys:
            uid = hotkeys.index(validator_hotkey)
            stake = float(metagraph.S[uid])
            return {
                "registered": True,
                "uid": uid,
                "hotkey": validator_hotkey,
                "stake": stake,
            }
    except Exception:
        pass

    return {
        "registered": False,
        "uid": None,
        "hotkey": validator_hotkey,
        "stake": 0,
    }


@pytest.fixture(scope="session")
def validator_uid(validator_registered: dict[str, Any]) -> int:
    """
    Get validator UID from registration status.

    Returns the UID if registered, otherwise a default UID for testing.
    """
    if validator_registered["registered"] and validator_registered["uid"] is not None:
        return validator_registered["uid"]
    # Return a default UID for testing when not registered
    return 15  # Our testnet UID


@pytest.fixture
def require_registration(validator_registered: dict[str, Any]) -> dict[str, Any]:
    """
    Fixture that skips test if validator is not registered.

    Use this fixture for tests that require on-chain registration.
    """
    if not validator_registered["registered"]:
        pytest.skip(
            f"Validator not registered on subnet {TESTNET_NETUID}. "
            f"Run: python scripts/register_testnet.py --register"
        )
    return validator_registered


@pytest.fixture
def require_stake(validator_registered: dict[str, Any]) -> dict[str, Any]:
    """
    Fixture that skips test if validator doesn't have minimum stake.
    """
    if not validator_registered["registered"]:
        pytest.skip("Validator not registered")

    if validator_registered["stake"] < MIN_STAKE_TAO:
        pytest.skip(
            f"Insufficient stake: {validator_registered['stake']} TAO "
            f"(need {MIN_STAKE_TAO} TAO)"
        )

    return validator_registered


# Test data fixtures

@pytest.fixture
def sample_prompt() -> str:
    """Sample prompt for testing."""
    return "What is the meaning of life according to different philosophical traditions?"


@pytest.fixture
def sample_response() -> str:
    """Sample LLM response for testing."""
    return (
        "The meaning of life varies across philosophical traditions. "
        "Existentialists like Sartre argue we create our own meaning through choices. "
        "Buddhist philosophy suggests the cessation of suffering through enlightenment. "
        "Aristotle proposed eudaimonia - flourishing through virtuous activity."
    )


@pytest.fixture
def sample_danger_score() -> float:
    """Sample danger score for testing."""
    return 0.05  # Low danger - safe content


@pytest.fixture
def unique_prompt() -> str:
    """Generate a unique prompt for novelty testing."""
    import uuid
    return f"Unique test prompt {uuid.uuid4()}: Explain quantum entanglement."


# Environment fixtures

@pytest.fixture(scope="session")
def e2e_env() -> dict[str, str]:
    """
    Load E2E test environment variables.

    Returns dict with relevant config.
    """
    return {
        "network": os.environ.get("BT_NETWORK", TESTNET_NETWORK),
        "netuid": int(os.environ.get("BT_NETUID", TESTNET_NETUID)),
        "wallet_name": WALLET_NAME,
        "hotkey_name": HOTKEY_NAME,
        "api_url": COLLECTOR_API_URL,
        "local_mode": os.environ.get("LOCAL_MODE", "false").lower() == "true",
        "skip_weight_setting": os.environ.get("SKIP_WEIGHT_SETTING", "true").lower() == "true",
    }


# Edge case test fixtures

@pytest.fixture
def raw_api_client(docker_services: dict[str, str]):
    """
    Raw API client for edge case tests.

    No automatic validation or signing.
    """
    from .helpers.api_client import RawAPIClient
    return RawAPIClient(base_url=docker_services["api_url"])


@pytest.fixture
def slow_api_client() -> CollectorAPIClient:
    """
    API client configured for slow response testing.

    Has a very long timeout for slow response tests.
    """
    return CollectorAPIClient(
        base_url=COLLECTOR_API_URL,
        timeout=120,  # 2 minute timeout
    )


@pytest.fixture
def fast_timeout_client() -> CollectorAPIClient:
    """
    API client with very short timeout for timeout testing.
    """
    return CollectorAPIClient(
        base_url=COLLECTOR_API_URL,
        timeout=1,  # 1 second timeout
    )


@pytest.fixture
def second_fake_validator_hotkey() -> str:
    """
    Generate a second fake validator hotkey for isolation tests.

    This is a properly formatted but non-existent SS58 address.
    """
    import secrets
    # Generate a fake but properly formatted hotkey
    return "5" + "Test" + secrets.token_hex(22)


@pytest.fixture
def stress_test_data() -> dict[str, Any]:
    """
    Generate large test datasets for stress testing.

    Returns:
        Dict containing various test data sizes
    """
    import uuid
    from .helpers.api_client import generate_random_embedding

    return {
        "prompts": [f"Stress test prompt {i} - {uuid.uuid4()}" for i in range(100)],
        "embeddings": [generate_random_embedding() for _ in range(100)],
        "trace_ids": [f"{i:032x}" for i in range(100)],
        "span_ids": [f"{i:016x}" for i in range(100)],
    }


@pytest.fixture
def boundary_test_values() -> dict[str, Any]:
    """
    Collection of boundary test values for various fields.

    Returns:
        Dict with boundary values for testing
    """
    return {
        "danger_scores": {
            "valid": [0.0, 0.5, 1.0],
            "invalid": [-0.0001, 1.0001, -100, 100, float("inf"), float("-inf")],
        },
        "embedding_dimensions": {
            "valid": [384],
            "invalid": [0, 1, 100, 383, 385, 1000],
        },
        "batch_sizes": {
            "valid": [1, 100, 1000],
            "invalid": [0, 1001, 10000],
        },
        "pagination": {
            "valid_limits": [1, 100, 1000],
            "invalid_limits": [-1, 0, 1001],
            "valid_offsets": [0, 100, 999999],
            "invalid_offsets": [-1],
        },
        "timestamps": {
            "valid_range_seconds": 30,  # Signature valid for 30 seconds
            "max_safe_integer": 9007199254740991,
        },
    }


# Markers for test categorization

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (involves network operations)"
    )
    config.addinivalue_line(
        "markers", "requires_registration: mark test as requiring validator registration"
    )
    config.addinivalue_line(
        "markers", "requires_docker: mark test as requiring Docker services"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as stress test (high load)"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )
    config.addinivalue_line(
        "markers", "boundary: mark test as boundary/edge case test"
    )
