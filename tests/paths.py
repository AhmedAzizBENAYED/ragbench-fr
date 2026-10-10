"""Absolute paths for tests: the working directory is a temporary one (see tests/conftest.py)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
