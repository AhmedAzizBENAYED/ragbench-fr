"""PostgreSQL fixtures for `db` tests (and integration tests that need the database).

Tests run in a dedicated schema (`ragbench_test`, via the libpq `search_path` option), so they
never touch an index stored in `public` on a developer's database.

Locally an unreachable database skips the tests with a message; with RAGBENCH_REQUIRE_DB=1 (CI)
it fails them: a db test silently skipped in CI would be a false guarantee. Both variables are
read at import time, before the autouse fixture that clears RAGBENCH_* variables.
"""

import os
from collections.abc import Iterator
from urllib.parse import quote

import psycopg
import pytest

from ragbench.index.pgvector_store import PgVectorBackend, describe_server

DEFAULT_TEST_DSN = "postgresql://ragbench:ragbench@127.0.0.1:5433/ragbench"
TEST_DSN = os.environ.get("RAGBENCH_TEST_PG_DSN", DEFAULT_TEST_DSN)
REQUIRE_DB = os.environ.get("RAGBENCH_REQUIRE_DB") == "1"
TEST_SCHEMA = "ragbench_test"


def with_search_path(dsn: str, schema: str) -> str:
    """Append a libpq `options` parameter setting `search_path` to `schema, public`."""
    separator = "&" if "?" in dsn else "?"
    return f"{dsn}{separator}options={quote(f'-csearch_path={schema},public', safe='')}"


@pytest.fixture(scope="session")
def pg_dsn() -> Iterator[str]:
    """DSN of a clean test schema; skips (or fails in CI) if PostgreSQL is unreachable."""
    try:
        admin = psycopg.connect(TEST_DSN, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as error:
        message = (
            f"PostgreSQL unreachable on {describe_server(TEST_DSN)}; run `make db-up` "
            f"({str(error).strip().splitlines()[-1] if str(error).strip() else error!r})"
        )
        if REQUIRE_DB:
            pytest.fail(f"RAGBENCH_REQUIRE_DB=1: {message}")
        pytest.skip(message)
    with admin:
        # The extension lives in `public`; creating it from the test search_path would put
        # its type in the test schema.
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        admin.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")
        admin.execute(f"CREATE SCHEMA {TEST_SCHEMA}")
    yield with_search_path(TEST_DSN, TEST_SCHEMA)
    with psycopg.connect(TEST_DSN, autocommit=True) as admin:
        admin.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")


@pytest.fixture
def pg_backend(pg_dsn: str) -> Iterator[PgVectorBackend]:
    """A connected backend on an empty test schema."""
    with PgVectorBackend.connect(pg_dsn) as backend:
        with psycopg.connect(pg_dsn, autocommit=True) as connection:
            connection.execute("TRUNCATE indexes CASCADE")
        yield backend
