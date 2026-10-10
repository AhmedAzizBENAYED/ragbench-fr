"""Integration-test configuration: every test in this directory is marked `integration`."""

from pathlib import Path

import pytest

_INTEGRATION_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark tests under tests/integration/ so that the default run (-m "not integration") skips
    them."""
    for item in items:
        if item.path.is_relative_to(_INTEGRATION_DIR):
            item.add_marker(pytest.mark.integration)
