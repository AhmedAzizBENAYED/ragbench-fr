"""Configuration: environment `Settings` versus experiment `RunConfig`.

Guiding rule: anything that can change a result belongs to `RunConfig` (a versioned YAML file).
`Settings` only holds secrets, paths and infrastructure knobs; no environment variable may change
a result.
"""

from functools import cache
from pathlib import Path
from typing import Any

from pydantic import Field, HttpUrl, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragbench.log import LogFormat, LogLevel


class Settings(BaseSettings):
    """Environment settings, read from environment variables and an optional `.env` file.

    API keys use their providers' standard names; every other variable is prefixed `RAGBENCH_`.
    Missing API keys are only an error when a provider is actually used.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAGBENCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        # Empty values (as in a `.env` copied from `.env.example`) fall back to the defaults.
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    ollama_base_url: HttpUrl = HttpUrl("http://localhost:11434")
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    results_dir: Path = Path("results")
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "json"

    @field_validator("log_level", "log_format", mode="before")
    @classmethod
    def _normalize_case(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, str):
            return value
        return value.upper() if info.field_name == "log_level" else value.lower()


@cache
def get_settings() -> Settings:
    """Return the process-wide settings, built on first call (tests clear the cache)."""
    return Settings()
