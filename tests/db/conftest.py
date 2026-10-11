"""`db` tests: marked automatically; Python sockets may only reach the database host.

pytest-socket restricts Python's `socket` module only: psycopg connects through libpq (C), which
it cannot see. The restriction still keeps any Python-level network call (HTTP clients, model
downloads) out of these tests.
"""

import socket
from pathlib import Path

import pytest
from psycopg.conninfo import conninfo_to_dict
from pytest_socket import socket_allow_hosts

from tests.db_fixtures import TEST_DSN

_DB_DIR = Path(__file__).parent


def _allowed_hosts() -> list[str]:
    host = str(conninfo_to_dict(TEST_DSN).get("host", "localhost"))
    try:
        return sorted({host, socket.gethostbyname(host)})
    except OSError:
        return [host]


_ALLOWED_HOSTS = _allowed_hosts()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.is_relative_to(_DB_DIR):
            item.add_marker(pytest.mark.db)


def pytest_runtest_setup(item: pytest.Item) -> None:
    socket_allow_hosts(_ALLOWED_HOSTS, allow_unix_socket=True)
