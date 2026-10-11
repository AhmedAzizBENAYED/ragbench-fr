"""`ragbench build-index` with `retrieval.backend: pgvector` is idempotent (marker `db`)."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cache import SqliteStore
from ragbench.cli import app
from ragbench.config import EmbeddingConfig, load_run_config
from ragbench.data.build import Downloader, build_dataset
from ragbench.data.build_config import CorpusBuildConfig
from ragbench.index import factory
from ragbench.index.cached import CachedEmbedder
from ragbench.index.pgvector_store import PgVectorBackend
from tests.constants import EXAMPLE_CONFIG
from tests.fakes import HashingEmbedder

runner = CliRunner()


@pytest.fixture
def cli_args(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str,
    pg_backend: PgVectorBackend,  # creates the schema and empties it
    build_config: CorpusBuildConfig,
    fake_download: Downloader,
) -> list[str]:
    def fake_embedder(config: EmbeddingConfig, cache_dir: Path) -> CachedEmbedder:
        return CachedEmbedder(HashingEmbedder(), SqliteStore(cache_dir / "embeddings.sqlite3"))

    monkeypatch.setattr(factory, "make_embedder", fake_embedder)
    monkeypatch.setenv("RAGBENCH_PG_DSN", pg_dsn)

    build_dataset(build_config, Path("data"), download=fake_download)
    run = load_run_config(EXAMPLE_CONFIG).model_dump(mode="json")
    run["data"]["revision"] = build_config.revision
    run["retrieval"]["backend"] = "pgvector"
    Path("run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    Path("data.yaml").write_text(
        yaml.safe_dump(build_config.model_dump(mode="json")), encoding="utf-8"
    )
    return ["build-index", "--config", "run.yaml", "--data-config", "data.yaml"]


def _run(args: list[str]) -> Any:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    return result


def test_first_run_fills_the_index_then_reruns_do_nothing(cli_args: list[str]) -> None:
    first = _run(cli_args)
    second = _run(cli_args)

    assert "index:     created (pgvector backend)" in first.stdout
    assert "cache:     0 hits, 8 misses" in first.stdout
    assert "index:     existing (pgvector backend)" in second.stdout
    assert "cache:     0 hits, 0 misses" in second.stdout  # no passage embedded at all


def test_rebuild_recreates_the_index(cli_args: list[str]) -> None:
    _run(cli_args)

    rebuilt = _run([*cli_args, "--rebuild"])

    assert "index:     rebuilt (pgvector backend)" in rebuilt.stdout
    assert "rebuild:" not in rebuilt.stdout
