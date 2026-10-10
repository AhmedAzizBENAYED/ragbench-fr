"""Corpus build parameters, read from a versioned YAML file (`configs/data.yaml`)."""

from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from ragbench.config import CommitSha, FrozenModel, load_yaml_model

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class SplitSizes(FrozenModel):
    """Number of questions per split; `ci` is drawn from `test`."""

    dev: int = Field(ge=1)
    test: int = Field(ge=1)
    ci: int = Field(ge=1)

    @model_validator(mode="after")
    def _ci_within_test(self) -> Self:
        if self.ci > self.test:
            raise ValueError("ci cannot be larger than test, since ci is a subset of test")
        return self


class CorpusBuildConfig(FrozenModel):
    """Everything that determines the corpus, the splits and the manifest."""

    dataset: str = Field(default="AgentPublic/piaf", min_length=1)
    revision: CommitSha
    filename: str = Field(min_length=1)
    raw_sha256: Sha256Hex
    seed: int
    sizes: SplitSizes
    # Versioned artefact: set here, never by an environment variable (ADR 0003).
    manifest_path: Path = Path("data/manifest.json")


def load_corpus_build_config(path: Path | str) -> CorpusBuildConfig:
    """Read and validate a corpus build YAML file.

    Raises:
        ConfigError: the file is missing, unreadable, not valid YAML or not a valid config.
    """
    return load_yaml_model(path, CorpusBuildConfig)
