"""Unit-test configuration: every test under tests/unit/ runs without network access.

Note: on Windows, asyncio creates its internal sockets as AF_INET (socketpair); to be handled
when async code arrives.
"""

import pytest
from pytest_socket import disable_socket


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Block socket creation for each unit test; pytest-socket restores it at teardown."""
    disable_socket(allow_unix_socket=True)
