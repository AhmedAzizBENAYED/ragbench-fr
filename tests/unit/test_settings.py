"""Tests of environment settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ragbench.config import Settings, get_settings
from ragbench.log import configure_logging, get_logger
from tests.constants import REPO_ROOT

FAKE_KEY = "gsk-fake-0123456789"


def test_defaults() -> None:
    settings = Settings()

    assert settings.groq_api_key is None
    assert settings.gemini_api_key is None
    assert str(settings.ollama_base_url) == "http://localhost:11434/"
    assert settings.data_dir == Path("data")
    assert settings.cache_dir == Path(".cache")
    assert settings.results_dir == Path("results")
    assert settings.log_level == "INFO"
    assert settings.log_format == "json"


def test_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGBENCH_LOG_LEVEL", "debug")
    monkeypatch.setenv("RAGBENCH_LOG_FORMAT", "CONSOLE")
    monkeypatch.setenv("RAGBENCH_CACHE_DIR", "/tmp/ragbench-cache")
    monkeypatch.setenv("RAGBENCH_OLLAMA_BASE_URL", "http://ollama:11434")

    settings = Settings()

    assert settings.log_level == "DEBUG"
    assert settings.log_format == "console"
    assert settings.cache_dir == Path("/tmp/ragbench-cache")
    assert settings.ollama_base_url.host == "ollama"


def test_api_keys_use_standard_unprefixed_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("RAGBENCH_GEMINI_API_KEY", FAKE_KEY)

    settings = Settings()

    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == FAKE_KEY
    assert settings.gemini_api_key is None


def test_invalid_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGBENCH_LOG_LEVEL", "verbose")

    with pytest.raises(ValidationError, match="log_level"):
        Settings()


def test_dotenv_in_working_directory_is_read() -> None:
    Path(".env").write_text(f"GROQ_API_KEY={FAKE_KEY}\nRAGBENCH_LOG_LEVEL=\n", encoding="utf-8")

    settings = Settings()

    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == FAKE_KEY
    assert settings.log_level == "INFO"  # empty values fall back to defaults


def test_environment_takes_precedence_over_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    Path(".env").write_text("RAGBENCH_LOG_LEVEL=ERROR\n", encoding="utf-8")
    monkeypatch.setenv("RAGBENCH_LOG_LEVEL", "WARNING")

    assert Settings().log_level == "WARNING"


def test_tests_never_see_the_repository_dotenv() -> None:
    assert Path.cwd() != REPO_ROOT
    assert not Path(".env").resolve().is_relative_to(REPO_ROOT)
    assert not Path(".env").exists()


def test_secret_is_hidden_from_repr_str_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)

    settings = Settings()

    assert FAKE_KEY not in repr(settings)
    assert FAKE_KEY not in str(settings)
    assert FAKE_KEY not in settings.model_dump_json()


def test_secret_is_hidden_from_logs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    settings = Settings()
    configure_logging("DEBUG", "json")

    logger = get_logger(__name__)
    logger.info("settings.loaded", settings=settings)
    logger.info("provider.configured", groq_api_key=settings.groq_api_key)
    assert settings.groq_api_key is not None
    logger.info("provider.configured", api_key=settings.groq_api_key.get_secret_value())

    err = capsys.readouterr().err
    assert len(err.splitlines()) == 3
    assert FAKE_KEY not in err


def test_settings_are_immutable() -> None:
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.log_level = "DEBUG"  # type: ignore[misc]


def test_get_settings_is_cached_and_cache_can_be_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    assert get_settings() is first

    monkeypatch.setenv("RAGBENCH_LOG_LEVEL", "ERROR")
    get_settings.cache_clear()

    assert get_settings().log_level == "ERROR"


def test_non_string_log_level_is_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        Settings(log_level=10)  # type: ignore[arg-type]
