"""Infrastructure configuration: store backends and the local/CI services (no network)."""

from contextlib import ExitStack

import pytest

from ragbench.config import Settings, VectorBackend
from ragbench.index.backends import STORE_BACKENDS
from ragbench.index.store import VectorStoreError


def test_pgvector_backend_requires_a_dsn() -> None:
    with (
        ExitStack() as resources,
        pytest.raises(VectorStoreError, match="RAGBENCH_PG_DSN is not set"),
    ):
        STORE_BACKENDS[VectorBackend.PGVECTOR](Settings(), resources)
