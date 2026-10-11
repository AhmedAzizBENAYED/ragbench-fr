"""Tests of `build_index` and `ragbench build-index` (fake embedder, no network, no torch)."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cache import SqliteStore
from ragbench.cli import EXIT_DATA_ERROR, EXIT_INVALID_CONFIG, app
from ragbench.config import EmbeddingConfig, RunConfig, load_run_config
from ragbench.data.build import BuildResult, Downloader, build_dataset
from ragbench.data.build_config import CorpusBuildConfig
from ragbench.data.processed import CORPUS_FILE, RunDataMismatchError
from ragbench.index import factory
from ragbench.index.build import build_index
from ragbench.index.cached import CachedEmbedder, CacheStats
from tests.constants import EXAMPLE_CONFIG
from tests.fakes import HashingEmbedder

runner = CliRunner()


def _fake_embedder(config: EmbeddingConfig, cache_dir: Path) -> CachedEmbedder:
    return CachedEmbedder(HashingEmbedder(), SqliteStore(cache_dir / "embeddings.sqlite3"))


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "make_embedder", _fake_embedder)


@pytest.fixture
def built(build_config: CorpusBuildConfig, fake_download: Downloader) -> BuildResult:
    return build_dataset(build_config, Path("data"), download=fake_download)


@pytest.fixture
def run_config(build_config: CorpusBuildConfig) -> RunConfig:
    config = load_run_config(EXAMPLE_CONFIG)
    data = config.data.model_copy(update={"revision": build_config.revision})
    return config.model_copy(update={"data": data})


def _build(run_config: RunConfig, built: BuildResult) -> Any:
    return build_index(run_config, built.manifest_path, Path("data"), Path("cache"))


def test_build_index_returns_a_working_retriever(run_config: RunConfig, built: BuildResult) -> None:
    result = _build(run_config, built)

    assert result.documents == 8
    assert result.dim == 256
    assert result.embedder_identity == HashingEmbedder().identity
    assert result.corpus_sha256 == built.manifest["corpus"]["sha256"]
    assert result.cache == CacheStats(hits=0, misses=8)
    [best] = result.retriever.retrieve("Qui a écrit Les Misérables ?", 1)
    assert best.title == "Victor Hugo"


def test_second_build_only_hits_the_cache(run_config: RunConfig, built: BuildResult) -> None:
    _build(run_config, built)

    assert _build(run_config, built).cache == CacheStats(hits=8, misses=0)


def test_revision_differing_from_manifest_is_rejected(
    run_config: RunConfig, built: BuildResult
) -> None:
    data = run_config.data.model_copy(update={"revision": "0" * 40})

    with pytest.raises(RunDataMismatchError, match="but the data manifest was built from"):
        _build(run_config.model_copy(update={"data": data}), built)


# --- CLI ------------------------------------------------------------------------------------------


@pytest.fixture
def config_files(run_config: RunConfig, build_config: CorpusBuildConfig) -> list[str]:
    Path("run.yaml").write_text(
        yaml.safe_dump(run_config.model_dump(mode="json")), encoding="utf-8"
    )
    Path("data.yaml").write_text(
        yaml.safe_dump(build_config.model_dump(mode="json")), encoding="utf-8"
    )
    return ["--config", "run.yaml", "--data-config", "data.yaml"]


def test_cli_prints_summary_and_hits_cache_on_second_run(
    built: BuildResult, config_files: list[str]
) -> None:
    first = runner.invoke(app, ["build-index", *config_files])
    second = runner.invoke(app, ["build-index", *config_files])

    assert first.exit_code == 0, first.stderr
    assert "documents: 8\n" in first.stdout
    assert "dimension: 256\n" in first.stdout
    assert f"embedder:  {HashingEmbedder().identity}\n" in first.stdout
    assert f"corpus:    sha256 {built.manifest['corpus']['sha256']}\n" in first.stdout
    assert "duration:  " in first.stdout
    assert "cache:     0 hits, 8 misses" in first.stdout
    assert "cache:     8 hits, 0 misses" in second.stdout


def test_cli_rejects_run_revision_differing_from_manifest(
    built: BuildResult, config_files: list[str]
) -> None:
    data = yaml.safe_load(Path("run.yaml").read_text(encoding="utf-8"))
    data["data"]["revision"] = "0" * 40
    Path("run.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    result = runner.invoke(app, ["build-index", *config_files])

    assert result.exit_code == EXIT_INVALID_CONFIG
    assert result.stdout == ""
    assert result.stderr.startswith("run.yaml: the run config targets AgentPublic/piaf@0000")


def test_cli_rejects_stale_corpus(built: BuildResult, config_files: list[str]) -> None:
    path = Path("data/processed") / CORPUS_FILE
    path.write_text(path.read_text(encoding="utf-8").replace("Jupiter", "Saturne"), "utf-8")

    result = runner.invoke(app, ["build-index", *config_files])

    assert result.exit_code == EXIT_DATA_ERROR
    assert "does not match the manifest" in result.stderr


def test_cli_rejects_missing_manifest(config_files: list[str]) -> None:
    result = runner.invoke(app, ["build-index", *config_files])

    assert result.exit_code == EXIT_DATA_ERROR
    assert "no data manifest" in result.stderr


def test_cli_rejects_invalid_config(config_files: list[str]) -> None:
    Path("run.yaml").write_text("seed: 1\n", encoding="utf-8")

    result = runner.invoke(app, ["build-index", *config_files])

    assert result.exit_code == EXIT_INVALID_CONFIG
    assert "run.yaml: name: Field required" in result.stderr
