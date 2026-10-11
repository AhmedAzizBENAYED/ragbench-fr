"""Unit-test configuration: no network access, and fixtures built from the mini PIAF file.

Note: on Windows, asyncio creates its internal sockets as AF_INET (socketpair); to be handled
when async code arrives.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest_socket import disable_socket

from ragbench.data.build import Downloader
from ragbench.data.build_config import CorpusBuildConfig, SplitSizes
from ragbench.data.corpus import Corpus, build_corpus
from ragbench.data.piaf import RawRecord, file_sha256, read_raw
from tests.constants import PIAF_MINI, PIAF_REVISION


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Block socket creation for each unit test; pytest-socket restores it at teardown."""
    disable_socket(allow_unix_socket=True)


@pytest.fixture
def piaf_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(PIAF_MINI.read_text(encoding="utf-8"))["rows"]
    return [{key: value for key, value in row.items() if key != "_note"} for row in rows]


@pytest.fixture
def piaf_parquet(piaf_rows: list[dict[str, Any]], tmp_path: Path) -> Path:
    path = tmp_path / "source" / "piaf_mini.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(piaf_rows), path)
    return path


@pytest.fixture
def piaf_records(piaf_parquet: Path) -> list[RawRecord]:
    return read_raw(piaf_parquet)


@pytest.fixture
def corpus(piaf_records: list[RawRecord]) -> Corpus:
    return build_corpus(piaf_records)


@pytest.fixture
def sizes() -> SplitSizes:
    return SplitSizes(dev=2, test=3, ci=2)


@pytest.fixture
def build_config(piaf_parquet: Path, sizes: SplitSizes) -> CorpusBuildConfig:
    return CorpusBuildConfig(
        revision=PIAF_REVISION,
        filename="plain_text/piaf_mini.parquet",
        raw_sha256=file_sha256(piaf_parquet),
        seed=7,
        sizes=sizes,
        manifest_path=Path("manifest.json"),
    )


@pytest.fixture
def fake_download(piaf_parquet: Path) -> Downloader:
    def download(config: CorpusBuildConfig, raw_dir: Path) -> Path:
        target = raw_dir / config.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(piaf_parquet, target)
        return target

    return download
