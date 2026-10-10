"""Configuration shared by all tests: isolate each test from the developer's environment.

Every test runs in an empty temporary directory, so relative paths (such as `.env`) never reach
the repository. Tests that read repository files must use `REPO_ROOT` from `tests.paths`.
"""

import logging
from collections.abc import Iterator

import pytest
import structlog


@pytest.fixture(autouse=True)
def _isolate_environment(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    root.handlers[:] = handlers
    root.setLevel(level)
