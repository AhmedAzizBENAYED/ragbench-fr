"""Tests of reading the processed corpus back."""

from pathlib import Path

import pytest

from ragbench.data import DataError
from ragbench.data.build import BuildResult, Downloader, build_dataset
from ragbench.data.build_config import CorpusBuildConfig
from ragbench.data.processed import CORPUS_FILE, load_corpus, require_manifest

PROCESSED = Path("data/processed")


@pytest.fixture
def built(build_config: CorpusBuildConfig, fake_download: Downloader) -> BuildResult:
    return build_dataset(build_config, Path("data"), download=fake_download)


def test_load_corpus_returns_the_built_documents(built: BuildResult) -> None:
    documents = load_corpus(PROCESSED, require_manifest(built.manifest_path))

    assert documents == list(built.corpus.documents)


def test_modified_corpus_is_rejected(built: BuildResult) -> None:
    path = PROCESSED / CORPUS_FILE
    path.write_text(
        path.read_text(encoding="utf-8").replace("Gustave Eiffel", "G. Eiffel"), encoding="utf-8"
    )

    with pytest.raises(
        DataError, match=r"does not match the manifest.*rerun `ragbench build-corpus`"
    ):
        load_corpus(PROCESSED, built.manifest)


def test_truncated_corpus_is_rejected(built: BuildResult) -> None:
    path = PROCESSED / CORPUS_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(DataError, match="does not match the manifest"):
        load_corpus(PROCESSED, built.manifest)


def test_missing_corpus_is_rejected(built: BuildResult) -> None:
    with pytest.raises(DataError, match="corpus not found; run `ragbench build-corpus`"):
        load_corpus(Path("elsewhere"), built.manifest)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("{not json", "invalid JSON"),
        ('{"doc_id": "d"}', "expected an object with"),
        ('{"doc_id": "d", "title": "t", "text": 3}', "doc_id, title and text must be strings"),
    ],
)
def test_malformed_line_is_rejected(built: BuildResult, line: str, message: str) -> None:
    (PROCESSED / CORPUS_FILE).write_text(line + "\n", encoding="utf-8")

    with pytest.raises(DataError, match=rf"{CORPUS_FILE}:1: {message}"):
        load_corpus(PROCESSED, built.manifest)


def test_unreadable_corpus_is_rejected(built: BuildResult) -> None:
    (PROCESSED / CORPUS_FILE).write_bytes(b"\xff\xfe invalid utf-8")

    with pytest.raises(DataError, match="cannot read corpus"):
        load_corpus(PROCESSED, built.manifest)


def test_missing_manifest_is_rejected() -> None:
    with pytest.raises(DataError, match="no data manifest; run `ragbench build-corpus`"):
        require_manifest(Path("missing.json"))
