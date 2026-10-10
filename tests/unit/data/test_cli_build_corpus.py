"""Tests of `ragbench build-corpus`; only the Hugging Face call is faked (no network)."""

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ragbench.cli import EXIT_DATA_ERROR, EXIT_INVALID_CONFIG, app
from ragbench.data import piaf
from ragbench.data.build_config import CorpusBuildConfig

runner = CliRunner()


@pytest.fixture(autouse=True)
def _fake_hub(monkeypatch: pytest.MonkeyPatch, piaf_parquet: Path) -> None:
    def fake_hf_hub_download(**kwargs: Any) -> str:
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(piaf_parquet, target)
        return str(target)

    monkeypatch.setattr(piaf, "hf_hub_download", fake_hf_hub_download)


@pytest.fixture
def config_file(build_config: CorpusBuildConfig) -> Path:
    path = Path("data.yaml")
    path.write_text(yaml.safe_dump(build_config.model_dump(mode="json")), encoding="utf-8")
    return path


def _run(*args: str) -> Any:
    return runner.invoke(app, ["build-corpus", *args])


def test_build_writes_manifest_and_prints_summary(config_file: Path) -> None:
    result = _run("--config", str(config_file))

    assert result.exit_code == 0, result.stderr
    assert "(created, sha256 " in result.stdout
    assert "questions: raw 16, kept 9" in result.stdout
    assert "duplicate_question=2" in result.stdout
    assert "documents: 8" in result.stdout
    assert "splits:    dev 2 " in result.stdout
    assert Path("manifest.json").is_file()
    assert Path("data/processed/corpus.jsonl").is_file()
    assert Path("data/raw/plain_text/piaf_mini.parquet").is_file()


def test_second_build_is_byte_identical(config_file: Path) -> None:
    _run("--config", str(config_file))
    first = Path("manifest.json").read_bytes()

    result = _run("--config", str(config_file))

    assert result.exit_code == 0
    assert "(unchanged, sha256 " in result.stdout
    assert Path("manifest.json").read_bytes() == first


def test_changed_result_is_refused_without_force(config_file: Path) -> None:
    _run("--config", str(config_file))
    first = Path("manifest.json").read_bytes()
    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) | {"seed": 8}
    config_file.write_text(yaml.safe_dump(data), encoding="utf-8")

    refused = _run("--config", str(config_file))

    assert refused.exit_code == EXIT_DATA_ERROR
    assert refused.stdout == ""
    assert "build.seed" in refused.stderr
    assert "--force" in refused.stderr
    assert Path("manifest.json").read_bytes() == first

    forced = _run("--config", str(config_file), "--force")

    assert forced.exit_code == 0
    assert "(overwritten, sha256 " in forced.stdout


def test_invalid_config_is_reported(config_file: Path) -> None:
    config_file.write_text("revision: main\n", encoding="utf-8")

    result = _run("--config", str(config_file))

    assert result.exit_code == EXIT_INVALID_CONFIG
    assert "data.yaml: revision: Value error" in result.stderr


def test_unexpected_raw_file_is_a_data_error(config_file: Path) -> None:
    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) | {"raw_sha256": "0" * 64}
    config_file.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = _run("--config", str(config_file))

    assert result.exit_code == EXIT_DATA_ERROR
    assert "SHA-256 is" in result.stderr
