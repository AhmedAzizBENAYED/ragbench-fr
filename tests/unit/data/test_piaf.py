"""Tests of the PIAF source loader (download is faked: no network)."""

import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from ragbench.data import DataError, piaf
from ragbench.data.build_config import CorpusBuildConfig
from ragbench.data.piaf import download_raw, file_sha256, read_raw


def _write_rows(rows: list[dict[str, Any]]) -> Path:
    path = Path("rows.parquet")
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_read_raw_parses_every_row(piaf_parquet: Path, piaf_rows: list[dict[str, Any]]) -> None:
    records = read_raw(piaf_parquet)

    assert len(records) == len(piaf_rows) == 16
    first = records[0]
    assert (first.id, first.title) == ("p001", "Tour Eiffel")
    assert first.answer_texts == ("Gustave Eiffel",)
    assert first.answer_starts == (36,)


def test_read_raw_rejects_a_non_parquet_file() -> None:
    path = Path("not.parquet")
    path.write_text("not parquet", encoding="utf-8")

    with pytest.raises(DataError, match="cannot read parquet file"):
        read_raw(path)


def test_read_raw_rejects_inconsistent_answers(piaf_rows: list[dict[str, Any]]) -> None:
    row = piaf_rows[0] | {"answers": {"text": ["a", "b"], "answer_start": [0]}}

    with pytest.raises(DataError, match=r"p001.*mismatch"):
        read_raw(_write_rows([row]))


def test_read_raw_rejects_missing_strings(piaf_rows: list[dict[str, Any]]) -> None:
    with pytest.raises(DataError, match="must be strings"):
        read_raw(_write_rows([piaf_rows[0] | {"title": None}]))


def test_read_raw_rejects_non_mapping_answers(piaf_rows: list[dict[str, Any]]) -> None:
    with pytest.raises(DataError, match="answers must be a mapping"):
        read_raw(_write_rows([piaf_rows[0] | {"answers": None}]))


def test_file_sha256_known_value() -> None:
    path = Path("abc.txt")
    path.write_bytes(b"abc")

    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_download_raw_pins_revision_and_copies_into_raw_dir(
    monkeypatch: pytest.MonkeyPatch, build_config: CorpusBuildConfig, piaf_parquet: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(piaf_parquet, target)
        return str(target)

    monkeypatch.setattr(piaf, "hf_hub_download", fake_hf_hub_download)

    path = download_raw(build_config, Path("data/raw"))

    assert path == Path("data/raw") / build_config.filename
    assert calls == [
        {
            "repo_id": "AgentPublic/piaf",
            "filename": build_config.filename,
            "repo_type": "dataset",
            "revision": build_config.revision,
            "local_dir": Path("data/raw"),
        }
    ]


def test_download_raw_rejects_unexpected_sha256(
    monkeypatch: pytest.MonkeyPatch, build_config: CorpusBuildConfig
) -> None:
    tampered = Path("tampered.parquet")
    tampered.write_bytes(b"tampered")
    monkeypatch.setattr(piaf, "hf_hub_download", lambda **_: str(tampered))

    with pytest.raises(DataError, match=f"expected {build_config.raw_sha256}"):
        download_raw(build_config, Path("raw"))


@pytest.mark.parametrize("error", [OSError("disk full"), LocalEntryNotFoundError("offline")])
def test_download_errors_become_data_errors(
    monkeypatch: pytest.MonkeyPatch, build_config: CorpusBuildConfig, error: Exception
) -> None:
    def failing(**_: Any) -> str:
        raise error

    monkeypatch.setattr(piaf, "hf_hub_download", failing)

    with pytest.raises(DataError, match="cannot download"):
        download_raw(build_config, Path("raw"))


def test_download_rejects_dry_run_result(
    monkeypatch: pytest.MonkeyPatch, build_config: CorpusBuildConfig
) -> None:
    monkeypatch.setattr(piaf, "hf_hub_download", lambda **_: object())

    with pytest.raises(DataError, match="unexpected download result"):
        download_raw(build_config, Path("raw"))
