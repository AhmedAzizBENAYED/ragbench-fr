"""The network guard of tests/unit/conftest.py must itself be tested."""

import socket

import pytest
from pytest_socket import SocketBlockedError

# pytest-socket emits a UserWarning on every blocked call; here it is the expected outcome.
pytestmark = pytest.mark.filterwarnings("ignore:A test tried to use socket:UserWarning")


def test_inet_socket_is_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_outbound_connection_is_blocked() -> None:
    with pytest.raises(SocketBlockedError):
        socket.create_connection(("example.com", 80), timeout=1)
