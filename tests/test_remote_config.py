#!/usr/bin/env python3
"""Standalone integration test for remote configuration feature.

This test mocks external dependencies (bittensor) to run without them installed.
It tests the core logic of the remote config client.

Run with: python3 test_remote_config_standalone.py
"""

import http.server
import json
import os
import socketserver
import sys
import threading
import re
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

# Mock bittensor before any imports that use it
sys.modules['bittensor'] = MagicMock()
bt_mock = sys.modules['bittensor']
bt_mock.logging = MagicMock()
bt_mock.logging.info = lambda x: print(f"[bt.info] {x}")
bt_mock.logging.warning = lambda x: print(f"[bt.warn] {x}")
bt_mock.logging.error = lambda x: print(f"[bt.error] {x}")
bt_mock.logging.debug = lambda x: None  # Suppress debug
bt_mock.logging.success = lambda x: print(f"[bt.success] {x}")

# Mock dotenv
sys.modules['dotenv'] = MagicMock()
sys.modules['dotenv'].load_dotenv = lambda: None

# Mock opentelemetry
sys.modules['opentelemetry'] = MagicMock()
sys.modules['opentelemetry.trace'] = MagicMock()

# Mock server configuration
MOCK_PORT = 18766
MOCK_HOST = "127.0.0.1"

# Test configurations
MOCK_CONFIGS = {
    37: {
        "config": {
            "DANGER_THRESHOLD": 0.35,
            "MIN_NOVELTY_THRESHOLD": 0.15,
            "TOP_REWARDED_MINERS": 5,
            "MINER_TIMEOUT": 45,
        },
        "version": 3,
        "description": "Mainnet test config",
    },
    290: {
        "config": {
            "DANGER_THRESHOLD": 0.25,
            "MIN_VALIDATOR_STAKE": 50.0,
            "RATE_LIMIT_REQUESTS": 200,
        },
        "version": 1,
        "description": "Testnet config",
    },
}


class MockRemoteConfigHandler(http.server.BaseHTTPRequestHandler):
    """Mock HTTP handler simulating the TypeScript API."""

    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path.startswith("/api/remote-config/"):
            try:
                netuid = int(self.path.split("/api/remote-config/")[1])
                if netuid in MOCK_CONFIGS:
                    response = {
                        "netuid": netuid,
                        "version": MOCK_CONFIGS[netuid]["version"],
                        "config": MOCK_CONFIGS[netuid]["config"],
                        "description": MOCK_CONFIGS[netuid]["description"],
                        "updated_at": "2024-01-15T10:30:00Z",
                    }
                    self._send_json(response, 200)
                else:
                    self._send_json({"error": f"No configuration found for netuid {netuid}"}, 404)
            except (ValueError, IndexError):
                self._send_json({"error": "Invalid netuid"}, 400)
        else:
            self._send_json({"error": "Not found"}, 404)


def run_mock_server(port: int, ready_event: threading.Event, shutdown_event: threading.Event):
    """Run the mock server."""
    class ReusableServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableServer(("", port), MockRemoteConfigHandler) as httpd:
        ready_event.set()
        while not shutdown_event.is_set():
            httpd.handle_request()


# Now create a minimal Config mock and the actual client code
class MockConfig:
    """Minimal Config class for testing."""
    # Chat Provider
    CHAT_PROVIDER = "chutes"
    CHUTES_API_KEY = "test-key"
    OPENAI_API_KEY = "sk-test"

    # Moderation
    DANGER_THRESHOLD = 0.3
    SINGLE_CATEGORY_THRESHOLD = 0.8

    # Scoring
    MIN_HIT_RATE_THRESHOLD = 0.4
    MIN_NOVELTY_THRESHOLD = 0.3
    NOVELTY_WEIGHT = 1.0
    TOP_REWARDED_MINERS = 3

    # Consensus
    MIN_VALIDATOR_STAKE = 100.0
    CONSENSUS_VALIDATORS = 5

    # Rate Limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW_HOURS = 1.0

    # Timing
    MINER_TIMEOUT = 30
    CHAT_API_TIMEOUT = 60.0

    # Network (excluded fields - should not be remotely configurable)
    BT_NETWORK = "finney"
    BT_NETUID = 37
    CENTRAL_API_ENDPOINT = "https://collector.aureliusaligned.ai/api/collections"

    # Telemetry
    TELEMETRY_ENABLED = False


# Inject our mock config
sys.modules['aurelius'] = MagicMock()
sys.modules['aurelius.shared'] = MagicMock()
sys.modules['aurelius.shared.config'] = MagicMock()
sys.modules['aurelius.shared.config'].Config = MockConfig

# Now import/define the client code inline (simplified version)
import requests

@dataclass
class RemoteConfigResult:
    success: bool
    config: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    error: str | None = None


class RemoteConfigClient:
    """Client for remote configuration."""

    EXCLUDED_FIELDS = frozenset({
        "CHUTES_API_KEY", "OPENAI_API_KEY", "CENTRAL_API_KEY",
        "VALIDATOR_WALLET_NAME", "VALIDATOR_HOTKEY", "MINER_WALLET_NAME", "MINER_HOTKEY",
        "CENTRAL_API_ENDPOINT", "NOVELTY_API_ENDPOINT", "TELEMETRY_TRACES_ENDPOINT",
        "TELEMETRY_LOGS_ENDPOINT", "TELEMETRY_REGISTRY_ENDPOINT", "SUBTENSOR_ENDPOINT",
        "LOCAL_DATASET_PATH", "TELEMETRY_LOCAL_BACKUP_PATH",
        "BT_NETWORK", "BT_NETUID", "BT_PORT_VALIDATOR", "VALIDATOR_HOST",
        "LOCAL_MODE", "ADVANCED_MODE", "SKIP_WEIGHT_SETTING",
    })

    EXCLUDED_PATTERNS = (
        re.compile(r".*_API_KEY$"),
        re.compile(r".*_ENDPOINT$"),
        re.compile(r".*_PATH$"),
    )

    def __init__(self, api_endpoint: str | None = None, netuid: int = 37, timeout: int = 10, poll_interval: int = 300):
        self.api_endpoint = api_endpoint
        self.netuid = netuid
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._cached_config: dict[str, Any] = {}
        self._cached_version: int = 0
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return bool(self.api_endpoint)

    def _is_field_allowed(self, field_name: str) -> bool:
        if field_name in self.EXCLUDED_FIELDS:
            return False
        for pattern in self.EXCLUDED_PATTERNS:
            if pattern.match(field_name):
                return False
        if not hasattr(MockConfig, field_name):
            return False
        return True

    def fetch_config(self) -> RemoteConfigResult:
        if not self.api_endpoint:
            return RemoteConfigResult(success=False, error="No endpoint")

        try:
            response = requests.get(f"{self.api_endpoint}/{self.netuid}", timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                config = data.get("config", {})
                version = data.get("version", 0)
                with self._lock:
                    self._cached_config = config
                    self._cached_version = version
                return RemoteConfigResult(success=True, config=config, version=version)
            elif response.status_code == 404:
                return RemoteConfigResult(success=True, config={}, version=0)
            else:
                return RemoteConfigResult(
                    success=False,
                    config=self._cached_config,
                    version=self._cached_version,
                    error=f"HTTP {response.status_code}"
                )
        except Exception as e:
            return RemoteConfigResult(
                success=False,
                config=self._cached_config,
                version=self._cached_version,
                error=str(e)
            )

    def _convert_value(self, field_name: str, value: Any) -> Any:
        if not hasattr(MockConfig, field_name):
            return value
        current_value = getattr(MockConfig, field_name)
        if current_value is None:
            return value
        current_type = type(current_value)
        if isinstance(value, current_type):
            return value
        if current_type == float and isinstance(value, int):
            return float(value)
        if current_type == int and isinstance(value, float):
            return int(value)
        return None

    def apply_to_config(self, remote_config: dict[str, Any]) -> list[str]:
        updated = []
        for field_name, value in remote_config.items():
            if not self._is_field_allowed(field_name):
                continue
            converted = self._convert_value(field_name, value)
            if converted is None:
                continue
            current = getattr(MockConfig, field_name, None)
            if current != converted:
                setattr(MockConfig, field_name, converted)
                updated.append(field_name)
        return updated


def test_remote_config():
    """Run integration tests."""
    print("=" * 60)
    print("Remote Configuration Integration Test (Standalone)")
    print("=" * 60)
    print()

    # Start mock server
    ready_event = threading.Event()
    shutdown_event = threading.Event()
    server_thread = threading.Thread(
        target=run_mock_server,
        args=(MOCK_PORT, ready_event, shutdown_event),
        daemon=True,
    )
    server_thread.start()
    ready_event.wait(timeout=5)
    print(f"[OK] Mock API server started on port {MOCK_PORT}")

    tests_passed = 0
    tests_failed = 0

    # Store original values
    original_values = {
        "DANGER_THRESHOLD": MockConfig.DANGER_THRESHOLD,
        "MIN_NOVELTY_THRESHOLD": MockConfig.MIN_NOVELTY_THRESHOLD,
        "TOP_REWARDED_MINERS": MockConfig.TOP_REWARDED_MINERS,
        "MINER_TIMEOUT": MockConfig.MINER_TIMEOUT,
    }

    # Test 1: Fetch config for netuid 37
    print()
    print("-" * 40)
    print("Test 1: Fetch config for netuid 37")
    print("-" * 40)

    client = RemoteConfigClient(
        api_endpoint=f"http://{MOCK_HOST}:{MOCK_PORT}/api/remote-config",
        netuid=37,
        timeout=5,
    )

    assert client.is_available(), "Client should be available"
    print(f"[OK] Client initialized")

    result = client.fetch_config()
    if result.success and result.version == 3:
        print(f"[OK] Fetched config version {result.version}")
        print(f"     Config: {result.config}")
        tests_passed += 1
    else:
        print(f"[FAIL] Expected version 3, got: {result}")
        tests_failed += 1

    # Test 2: Apply config
    print()
    print("-" * 40)
    print("Test 2: Apply config to MockConfig")
    print("-" * 40)

    updated = client.apply_to_config(result.config)
    print(f"[INFO] Updated fields: {updated}")

    if MockConfig.DANGER_THRESHOLD == 0.35:
        print(f"[OK] DANGER_THRESHOLD = {MockConfig.DANGER_THRESHOLD}")
        tests_passed += 1
    else:
        print(f"[FAIL] DANGER_THRESHOLD should be 0.35, got {MockConfig.DANGER_THRESHOLD}")
        tests_failed += 1

    if MockConfig.TOP_REWARDED_MINERS == 5:
        print(f"[OK] TOP_REWARDED_MINERS = {MockConfig.TOP_REWARDED_MINERS}")
        tests_passed += 1
    else:
        print(f"[FAIL] TOP_REWARDED_MINERS should be 5")
        tests_failed += 1

    # Test 3: Fetch testnet config
    print()
    print("-" * 40)
    print("Test 3: Fetch config for netuid 290")
    print("-" * 40)

    client_testnet = RemoteConfigClient(
        api_endpoint=f"http://{MOCK_HOST}:{MOCK_PORT}/api/remote-config",
        netuid=290,
        timeout=5,
    )

    result_testnet = client_testnet.fetch_config()
    if result_testnet.success and result_testnet.version == 1:
        print(f"[OK] Fetched testnet config version {result_testnet.version}")
        tests_passed += 1
    else:
        print(f"[FAIL] Expected version 1")
        tests_failed += 1

    # Test 4: Non-existent netuid
    print()
    print("-" * 40)
    print("Test 4: Non-existent netuid (999)")
    print("-" * 40)

    client_unknown = RemoteConfigClient(
        api_endpoint=f"http://{MOCK_HOST}:{MOCK_PORT}/api/remote-config",
        netuid=999,
        timeout=5,
    )

    result_unknown = client_unknown.fetch_config()
    if result_unknown.success and result_unknown.config == {}:
        print(f"[OK] 404 handled gracefully")
        tests_passed += 1
    else:
        print(f"[FAIL] Should return empty config on 404")
        tests_failed += 1

    # Test 5: Excluded fields
    print()
    print("-" * 40)
    print("Test 5: Excluded fields not applied")
    print("-" * 40)

    original_api_key = MockConfig.OPENAI_API_KEY
    original_endpoint = MockConfig.CENTRAL_API_ENDPOINT

    malicious_config = {
        "OPENAI_API_KEY": "stolen-key",
        "CENTRAL_API_ENDPOINT": "http://evil.com",
        "DANGER_THRESHOLD": 0.5,
    }

    updated_malicious = client.apply_to_config(malicious_config)

    if "OPENAI_API_KEY" not in updated_malicious:
        print(f"[OK] OPENAI_API_KEY excluded")
        tests_passed += 1
    else:
        print(f"[FAIL] OPENAI_API_KEY should be excluded")
        tests_failed += 1

    if "CENTRAL_API_ENDPOINT" not in updated_malicious:
        print(f"[OK] CENTRAL_API_ENDPOINT excluded")
        tests_passed += 1
    else:
        print(f"[FAIL] CENTRAL_API_ENDPOINT should be excluded")
        tests_failed += 1

    if MockConfig.OPENAI_API_KEY == original_api_key:
        print(f"[OK] API key unchanged (security check passed)")
        tests_passed += 1
    else:
        print(f"[FAIL] API key was modified!")
        tests_failed += 1

    # Test 6: Type conversion
    print()
    print("-" * 40)
    print("Test 6: Type conversion (int -> float)")
    print("-" * 40)

    type_test = {"MIN_VALIDATOR_STAKE": 100}  # int -> float
    client.apply_to_config(type_test)

    if isinstance(MockConfig.MIN_VALIDATOR_STAKE, float) and MockConfig.MIN_VALIDATOR_STAKE == 100.0:
        print(f"[OK] int->float conversion: {MockConfig.MIN_VALIDATOR_STAKE}")
        tests_passed += 1
    else:
        print(f"[FAIL] Type conversion failed")
        tests_failed += 1

    # Test 7: Cached config on error
    print()
    print("-" * 40)
    print("Test 7: Fallback to cached config")
    print("-" * 40)

    client_bad = RemoteConfigClient(
        api_endpoint="http://127.0.0.1:19999/api/remote-config",
        netuid=37,
        timeout=1,
    )
    client_bad._cached_config = {"DANGER_THRESHOLD": 0.99}
    client_bad._cached_version = 99

    result_bad = client_bad.fetch_config()
    if not result_bad.success and result_bad.config.get("DANGER_THRESHOLD") == 0.99:
        print(f"[OK] Fallback to cached config works")
        tests_passed += 1
    else:
        print(f"[FAIL] Should fallback to cached config")
        tests_failed += 1

    # Cleanup
    shutdown_event.set()

    # Summary
    print()
    print("=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)

    return tests_failed == 0


if __name__ == "__main__":
    try:
        success = test_remote_config()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
