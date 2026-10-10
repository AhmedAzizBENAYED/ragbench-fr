"""Configuration: environment `Settings` versus experiment `RunConfig`.

Guiding rule: anything that can change a result belongs to `RunConfig` (a versioned YAML file).
`Settings` only holds secrets, paths and infrastructure knobs; no environment variable may change
a result.
"""

import re
from collections.abc import Sequence
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragbench.hashing import canonical_hash
from ragbench.log import LogFormat, LogLevel

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def _require_commit_sha(value: str) -> str:
    if not _COMMIT_SHA.fullmatch(value):
        raise ValueError(
            "must be a full 40-character lowercase hex commit SHA, not a branch or tag name"
        )
    return value


CommitSha = Annotated[str, AfterValidator(_require_commit_sha)]
"""An immutable revision: a full lowercase hex commit SHA (branch and tag names are rejected)."""


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


class ConfigError(Exception):
    """A run configuration file cannot be read or is invalid.

    The message names the file and each faulty field, one error per line.
    """

    def __init__(self, path: Path, errors: Sequence[str]) -> None:
        self.path = path
        self.errors = tuple(errors)
        super().__init__("\n".join(f"{path}: {error}" for error in self.errors))

    @classmethod
    def from_validation_error(cls, path: Path, error: ValidationError) -> "ConfigError":
        return cls(path, [f"{_format_loc(e['loc'])}: {e['msg']}" for e in error.errors()])


def _format_loc(loc: tuple[int | str, ...]) -> str:
    parts: list[str] = []
    for item in loc:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(f".{item}" if parts else item)
    return "".join(parts) or "<root>"


class FrozenModel(BaseModel):
    """Base for YAML-backed configs: immutable, and a typo is an error, not an ignored key."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Split(StrEnum):
    """Frozen question splits (spec §5.3)."""

    DEV = "dev"
    TEST = "test"
    CI = "ci"


class Provider(StrEnum):
    """LLM providers reachable through the LLM gateway."""

    OLLAMA = "ollama"
    GROQ = "groq"
    GEMINI = "gemini"


class PipelineName(StrEnum):
    """The five evaluated architectures (spec §7)."""

    CLOSED_BOOK = "closed-book"
    NAIVE_RAG = "naive-rag"
    ADAPTIVE_RAG = "adaptive-rag"
    CRAG = "crag"
    SELF_RAG = "self-rag"


class DataConfig(FrozenModel):
    """Dataset, pinned to an immutable revision, and the split to run."""

    dataset: str = Field(default="AgentPublic/piaf", min_length=1)
    revision: CommitSha
    split: Split


class EmbeddingConfig(FrozenModel):
    """Embedding model shared by every pipeline of the run."""

    model: str = Field(default="intfloat/multilingual-e5-base", min_length=1)


class RetrievalConfig(FrozenModel):
    """Retrieval parameters shared by every pipeline of the run."""

    k: int = Field(default=5, ge=1)


class LLMSpec(FrozenModel):
    """A model on a provider, with its generation parameters."""

    provider: Provider
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)


class ClosedBookPipeline(FrozenModel):
    """Answers without context; one LLM call by construction."""

    name: Literal[PipelineName.CLOSED_BOOK]


class NaiveRagPipeline(FrozenModel):
    """Retrieves top-k then generates; one LLM call by construction."""

    name: Literal[PipelineName.NAIVE_RAG]


class AdaptiveRagPipeline(FrozenModel):
    """Routes the question, then answers directly, retrieves or reformulates."""

    name: Literal[PipelineName.ADAPTIVE_RAG]
    max_llm_calls: int = Field(default=3, ge=0)


class CragPipeline(FrozenModel):
    """Grades retrieved documents and reformulates the query when none is relevant."""

    name: Literal[PipelineName.CRAG]
    max_reformulations: int = Field(default=1, ge=0)


class SelfRagPipeline(FrozenModel):
    """Generates, grades the answer, then regenerates or reformulates if needed."""

    name: Literal[PipelineName.SELF_RAG]
    max_corrections: int = Field(default=2, ge=0)


PipelineSpec = Annotated[
    ClosedBookPipeline | NaiveRagPipeline | AdaptiveRagPipeline | CragPipeline | SelfRagPipeline,
    Field(discriminator="name"),
]


class RunConfig(FrozenModel):
    """Everything that can change a result (spec §6). Loaded from a versioned YAML file."""

    name: str = Field(min_length=1)
    description: str = ""
    seed: int
    data: DataConfig
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generator: LLMSpec
    judge: LLMSpec
    pipelines: list[PipelineSpec] = Field(min_length=1)

    @field_validator("pipelines")
    @classmethod
    def _reject_duplicate_pipelines(cls, pipelines: list[PipelineSpec]) -> list[PipelineSpec]:
        names = [pipeline.name for pipeline in pipelines]
        duplicates = sorted({name.value for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate pipelines: {', '.join(duplicates)}")
        return pipelines

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON of the whole config except `name` and `description`.

        Defaults are included and pipelines are sorted by name, since their order does not
        change results.
        """
        data = self.model_dump(mode="json", exclude={"name", "description"})
        data["pipelines"] = sorted(data["pipelines"], key=lambda pipeline: pipeline["name"])
        return canonical_hash(data)


def load_yaml_model[M: BaseModel](path: Path | str, model: type[M]) -> M:
    """Read a YAML file (UTF-8) and validate it against a pydantic model.

    Raises:
        ConfigError: the file is missing, unreadable, not valid YAML or not a valid `model`.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(path, ["file not found"]) from None
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(path, [f"cannot read file: {error}"]) from error

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(path, [f"invalid YAML: {error}"]) from error
    if not isinstance(data, dict):
        raise ConfigError(path, ["the top level must be a mapping of fields"])

    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise ConfigError.from_validation_error(path, error) from None


def load_run_config(path: Path | str) -> RunConfig:
    """Read and validate a run configuration YAML file (UTF-8).

    Raises:
        ConfigError: the file is missing, unreadable, not valid YAML or not a valid config.
    """
    return load_yaml_model(path, RunConfig)


def dump_run_config(config: RunConfig, path: Path | str) -> None:
    """Write the fully resolved config (defaults included) as canonical YAML, UTF-8, LF."""
    text = yaml.safe_dump(
        config.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    Path(path).write_text(text, encoding="utf-8", newline="\n")
