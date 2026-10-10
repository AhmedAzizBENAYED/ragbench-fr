"""Tests of the ragbench CLI."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import ragbench
from ragbench.cli import EXIT_INVALID_CONFIG, EXIT_INVALID_ENVIRONMENT, _version_callback, app
from tests.constants import EXAMPLE_CONFIG, EXAMPLE_FINGERPRINT, PIAF_REVISION

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


def _write_config(text: str) -> Path:
    path = Path("config.yaml")
    path.write_text(text, encoding="utf-8")
    return path


def test_validate_config_prints_fingerprint_on_stdout_and_logs_on_stderr() -> None:
    result = runner.invoke(app, ["validate-config", str(EXAMPLE_CONFIG)])

    assert result.exit_code == 0
    assert result.stdout == f"{EXAMPLE_FINGERPRINT}\n"
    [log_line] = [json.loads(line) for line in result.stderr.splitlines()]
    assert log_line["event"] == "config.valid"
    assert log_line["fingerprint"] == EXAMPLE_FINGERPRINT


@pytest.mark.parametrize(
    ("replace", "by", "expected"),
    [
        ("  k: 5", "  k: 0", "retrieval.k: Input should be greater than or equal to 1"),
        ("  k: 5", "  k: 5\n  top_k: 3", "retrieval.top_k: Extra inputs are not permitted"),
        (
            f'revision: "{PIAF_REVISION}"',
            "revision: main",
            "data.revision: Value error, must be a full",
        ),
    ],
)
def test_validate_config_reports_invalid_config_on_stderr(
    replace: str, by: str, expected: str
) -> None:
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    assert replace in text
    path = _write_config(text.replace(replace, by, 1))

    result = runner.invoke(app, ["validate-config", str(path)])

    assert result.exit_code == EXIT_INVALID_CONFIG
    assert result.stdout == ""
    assert f"config.yaml: {expected}" in result.stderr
    assert "Traceback" not in result.stderr


def test_validate_config_reports_missing_file() -> None:
    result = runner.invoke(app, ["validate-config", "missing.yaml"])

    assert result.exit_code == EXIT_INVALID_CONFIG
    assert "missing.yaml: file not found" in result.stderr


def test_invalid_environment_is_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAGBENCH_LOG_LEVEL", "verbose")

    result = runner.invoke(app, ["validate-config", str(EXAMPLE_CONFIG)])

    assert result.exit_code == EXIT_INVALID_ENVIRONMENT
    assert result.stdout == ""
    assert result.stderr.startswith("invalid environment setting log_level: ")


def test_logging_is_configured_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGBENCH_LOG_FORMAT", "console")

    result = runner.invoke(app, ["validate-config", str(EXAMPLE_CONFIG)])

    assert result.exit_code == 0
    assert "config.valid" in result.stderr
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.stderr)
