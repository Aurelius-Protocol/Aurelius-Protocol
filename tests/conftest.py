"""Root conftest for Aurelius Protocol tests.

Auto-skips e2e tests and integration tests that require external services,
unless the corresponding environment variables are set.

- tests/e2e/* always skipped unless RUN_E2E_TESTS=1 (need Docker, bittensor, etc.)
- @pytest.mark.integration skipped unless RUN_INTEGRATION_TESTS=1 (need collector API)
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        # Auto-skip e2e tests in tests/e2e/ unless services are available
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.skipif(
                not os.environ.get("RUN_E2E_TESTS"),
                reason="E2E tests require RUN_E2E_TESTS=1 and running services"
            ))
        # Auto-skip tests marked @pytest.mark.integration
        for marker in item.iter_markers(name="integration"):
            item.add_marker(pytest.mark.skipif(
                not os.environ.get("RUN_INTEGRATION_TESTS"),
                reason="Integration tests require RUN_INTEGRATION_TESTS=1 and running services"
            ))


def pytest_configure(config):
    """Register custom markers at root level."""
    config.addinivalue_line(
        "markers", "integration: mark test as requiring external services (collector API)"
    )
