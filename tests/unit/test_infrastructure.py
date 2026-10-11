"""Infrastructure configuration: store backends and the local/CI services (no network)."""

import re
from contextlib import ExitStack
from typing import Any

import pytest
import yaml

from ragbench.config import Settings, VectorBackend
from ragbench.index.backends import STORE_BACKENDS
from ragbench.index.store import VectorStoreError
from tests.constants import REPO_ROOT


def _yaml(relative: str) -> Any:
    return yaml.safe_load((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_postgres_image_is_pinned_and_identical_in_compose_and_ci() -> None:
    compose_image = _yaml("docker-compose.yml")["services"]["postgres"]["image"]
    ci_image = _yaml(".github/workflows/ci.yml")["jobs"]["check"]["services"]["postgres"]["image"]

    assert compose_image == ci_image
    name, _, tag = compose_image.partition(":")
    assert name == "pgvector/pgvector"
    # A precise pgvector version and PostgreSQL major version, e.g. 0.8.7-pg18-trixie.
    assert re.fullmatch(r"\d+\.\d+\.\d+-pg\d+(-[a-z]+)?", tag), tag


def test_pgvector_backend_requires_a_dsn() -> None:
    with (
        ExitStack() as resources,
        pytest.raises(VectorStoreError, match="RAGBENCH_PG_DSN is not set"),
    ):
        STORE_BACKENDS[VectorBackend.PGVECTOR](Settings(), resources)
