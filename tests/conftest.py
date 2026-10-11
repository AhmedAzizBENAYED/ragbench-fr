"""Configuration shared by all tests: isolate each test from the developer's environment.

Every test runs in an empty temporary directory, so relative paths (such as `.env`) never reach
the repository, without ragbench environment variables and with a fresh `get_settings()`.
Tests that read repository files must use `REPO_ROOT` from `tests.constants`.
"""

import logging
import os
from collections.abc import Iterator

import pytest
import structlog

from ragbench.config import get_settings
from ragbench.log import NOISY_LOGGERS

# Shared fixtures (mini PIAF corpus, PostgreSQL) for unit, db and integration tests.
pytest_plugins = ("tests.piaf_fixtures", "tests.db_fixtures")

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
    noisy = {name: logging.getLogger(name) for name in NOISY_LOGGERS}
    saved = {name: (lg.level, list(lg.handlers), lg.propagate) for name, lg in noisy.items()}
    yield
    logging.captureWarnings(False)
    for name, (noisy_level, noisy_handlers, propagate) in saved.items():
        noisy[name].setLevel(noisy_level)
        noisy[name].handlers[:] = noisy_handlers
        noisy[name].propagate = propagate
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()
    root.handlers[:] = handlers
    root.setLevel(level)
