"""Tests of the ragbench CLI."""

from typer.testing import CliRunner

import ragbench
from ragbench.cli import _version_callback, app

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"ragbench {ragbench.__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "Usage" in result.output


def test_version_callback_does_nothing_when_flag_absent() -> None:
    _version_callback(False)
