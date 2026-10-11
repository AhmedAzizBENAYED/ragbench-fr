"""Tests of structured logging."""

import json
import logging
import warnings
from typing import Any

import pytest
import structlog

from ragbench.log import (
    NOISY_LOGGERS,
    REDACTED,
    LogLevel,
    configure_logging,
    get_logger,
    is_sensitive_key,
)


def _json_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line]


def test_json_line_has_timestamp_level_logger_event_and_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO", "json")
    structlog.contextvars.bind_contextvars(run_id="r1", pipeline="crag", question_id="q42")

    get_logger("ragbench.test").info("question.answered", latency_ms=12.5)

    [line] = _json_lines(capsys.readouterr().err)
    assert line["event"] == "question.answered"
    assert line["level"] == "info"
    assert line["logger"] == "ragbench.test"
    assert line["timestamp"].endswith("Z")
    assert line["latency_ms"] == 12.5
    assert (line["run_id"], line["pipeline"], line["question_id"]) == ("r1", "crag", "q42")


def test_stdlib_log_is_rendered_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")

    logging.getLogger("thirdparty").warning("retrying in %d s", 3)

    [line] = _json_lines(capsys.readouterr().err)
    assert line["event"] == "retrying in 3 s"
    assert line["level"] == "warning"
    assert line["logger"] == "thirdparty"
    assert "timestamp" in line


def test_logs_go_to_stderr_not_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")

    get_logger(__name__).info("hello")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hello" in captured.err


def test_non_ascii_characters_are_escaped_in_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")

    get_logger(__name__).info("réponse générée")

    err = capsys.readouterr().err
    assert err.isascii()
    assert _json_lines(err)[0]["event"] == "réponse générée"


def test_level_filters_lower_records(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", "json")

    get_logger(__name__).info("hidden")
    logging.getLogger("thirdparty").info("hidden too")
    get_logger(__name__).warning("shown")

    assert [line["event"] for line in _json_lines(capsys.readouterr().err)] == ["shown"]


def test_reconfiguring_does_not_duplicate_lines(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    configure_logging("INFO", "json")

    get_logger(__name__).info("once")

    assert len(_json_lines(capsys.readouterr().err)) == 1


def test_exception_is_rendered_in_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")

    try:
        raise ValueError("boom")
    except ValueError:
        get_logger(__name__).exception("failed")

    [line] = _json_lines(capsys.readouterr().err)
    assert "ValueError: boom" in line["exception"]


def test_console_format_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "console")

    get_logger(__name__).info("hello", k=5)

    err = capsys.readouterr().err
    assert "hello" in err
    assert "k=5" in err
    with pytest.raises(json.JSONDecodeError):
        json.loads(err)


def test_secrets_are_redacted_recursively(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")

    get_logger(__name__).info(
        "request.sent",
        api_key="sk-fake",
        headers={"Authorization": "Bearer fake", "Accept": "application/json"},
        attempts=[{"access_token": "fake"}],
        tokens_in=120,
        tokens_out=30,
        max_tokens=512,
    )

    [line] = _json_lines(capsys.readouterr().err)
    assert line["api_key"] == REDACTED
    assert line["headers"] == {"Authorization": REDACTED, "Accept": "application/json"}
    assert line["attempts"] == [{"access_token": REDACTED}]
    assert (line["tokens_in"], line["tokens_out"], line["max_tokens"]) == (120, 30, 512)


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "GROQ_API_KEY",
        "apiKey",
        "X-Api-Key",
        "access_token",
        "client_secret",
        "password",
        "Authorization",
    ],
)
def test_sensitive_keys(key: str) -> None:
    assert is_sensitive_key(key)


@pytest.mark.parametrize(
    "key", ["tokens_in", "tokens_out", "max_tokens", "keyword", "k", "event", "monkey"]
)
def test_non_sensitive_keys(key: str) -> None:
    assert not is_sensitive_key(key)


@pytest.mark.parametrize(("level", "expected"), [("DEBUG", "WARNING"), ("ERROR", "ERROR")])
def test_noisy_third_party_loggers_are_quieted(
    capsys: pytest.CaptureFixture[str], level: LogLevel, expected: str
) -> None:
    configure_logging(level, "json")

    for name in NOISY_LOGGERS:
        assert logging.getLevelName(logging.getLogger(name).level) == expected
    logging.getLogger("httpx2").info("HTTP Request: GET https://huggingface.co")
    logging.getLogger("huggingface_hub.file_download").warning("disk almost full")

    events = [line["event"] for line in _json_lines(capsys.readouterr().err)]
    assert events == (["disk almost full"] if level == "DEBUG" else [])


def test_library_warnings_are_rendered_once_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("set HF_TOKEN for higher rate limits", UserWarning, stacklevel=1)

    err = capsys.readouterr().err
    [line] = _json_lines(err)
    assert line["logger"] == "py.warnings"
    assert line["level"] == "warning"
    assert "UserWarning: set HF_TOKEN for higher rate limits" in line["event"]
    assert len(err.splitlines()) == 1  # not also printed raw


def test_library_handlers_are_removed_so_records_render_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    library = logging.getLogger("huggingface_hub")
    library.addHandler(logging.StreamHandler())  # what huggingface_hub does at import
    library.propagate = False

    configure_logging("INFO", "json")
    logging.getLogger("huggingface_hub.utils._http").warning("set HF_TOKEN")

    err = capsys.readouterr().err
    assert library.handlers == []
    assert library.propagate is True
    assert [line["event"] for line in _json_lines(err)] == ["set HF_TOKEN"]
