"""Tests of the corpus build configuration."""

from pathlib import Path

import pytest
import yaml

from ragbench.config import ConfigError
from ragbench.data.build_config import CorpusBuildConfig, load_corpus_build_config
from tests.constants import PIAF_REVISION, REPO_ROOT

VALID = {
    "revision": PIAF_REVISION,
    "filename": "plain_text/train.parquet",
    "raw_sha256": "0" * 64,
    "seed": 1,
    "sizes": {"dev": 100, "test": 300, "ci": 30},
}


def _load(data: dict[str, object]) -> CorpusBuildConfig:
    path = Path("data.yaml")
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_corpus_build_config(path)


def test_repository_config_is_valid() -> None:
    config = load_corpus_build_config(REPO_ROOT / "configs" / "data.yaml")

    assert config.revision == PIAF_REVISION
    assert config.sizes.model_dump() == {"dev": 100, "test": 300, "ci": 30}
    assert config.manifest_path == Path("data/manifest.json")


def test_defaults() -> None:
    config = _load(VALID)

    assert config.dataset == "AgentPublic/piaf"
    assert config.manifest_path == Path("data/manifest.json")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"revision": "main"}, "revision: Value error, must be a full"),
        ({"raw_sha256": "abc"}, "raw_sha256: String should match pattern"),
        ({"sizes": {"dev": 1, "test": 2, "ci": 3}}, "sizes: Value error, ci cannot be larger"),
        ({"sizes": {"dev": 0, "test": 2, "ci": 1}}, "sizes.dev: Input should be greater"),
        ({"seeds": 1}, "seeds: Extra inputs are not permitted"),
    ],
)
def test_invalid_config_is_rejected(change: dict[str, object], expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        _load(VALID | change)
