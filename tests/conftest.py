"""Configuration shared by all tests: isolate each test from the developer's environment.

Every test runs in an empty temporary directory, so relative paths (such as `.env`) never reach
the repository, without ragbench environment variables and with a fresh `get_settings()`.
Tests that read repository files must use `REPO_ROOT` from `tests.paths`.
"""

import logging
import os
from collections.abc import Iterator

import pytest
import structlog

from ragbench.config import get_settings

_UNPREFIXED_VARIABLES = frozenset({"GROQ_API_KEY", "GEMINI_API_KEY"})


@pytest.fixture(autouse=True)
def _isolate_environment(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    for name in list(os.environ):
        if name.upper().startswith("RAGBENCH_") or name.upper() in _UNPREFIXED_VARIABLES:
            monkeypatch.delenv(name)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    root.handlers[:] = handlers
    root.setLevel(level)
