"""End-to-end corpus build: download, clean, split, then write artefacts and the manifest.

Everything is computed in memory first. If a manifest already exists and differs, nothing is
written unless `force` is set: a frozen split must never change silently.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragbench.data import DataError
from ragbench.data.build_config import CorpusBuildConfig
from ragbench.data.corpus import Corpus, build_corpus
from ragbench.data.manifest import (
    Manifest,
    build_manifest,
    diff_manifests,
    document_record,
    question_record,
    read_manifest,
    render_manifest,
    write_text,
)
from ragbench.data.piaf import download_raw, file_sha256, read_raw
from ragbench.data.splits import Splits, make_splits
from ragbench.log import get_logger

logger = get_logger(__name__)

Downloader = Callable[[CorpusBuildConfig, Path], Path]


class ManifestMismatchError(DataError):
    """The rebuilt manifest differs from the committed one."""

    def __init__(self, path: Path, differences: list[str]) -> None:
        self.path = path
        self.differences = differences
        super().__init__(
            f"{path}: the rebuilt data differs from the existing manifest in: "
            f"{', '.join(differences)}. Frozen splits must not change silently; "
            "rerun with --force to overwrite the manifest on purpose."
        )


@dataclass(frozen=True)
class BuildResult:
    """Outcome of a build."""

    corpus: Corpus
    splits: Splits
    manifest: Manifest
    manifest_path: Path
    status: str  # "created", "unchanged" or "overwritten"


def _jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)


def build_dataset(
    config: CorpusBuildConfig,
    data_dir: Path,
    *,
    force: bool = False,
    download: Downloader = download_raw,
) -> BuildResult:
    """Build `processed/corpus.jsonl`, `processed/questions.jsonl` and the manifest.

    Raises:
        DataError: download, parsing or consistency failure.
        ManifestMismatchError: the manifest exists, differs and `force` is False.
    """
    raw_path = download(config, data_dir / "raw")
    corpus = build_corpus(read_raw(raw_path))
    splits = make_splits(corpus.questions, config.sizes, config.seed)
    manifest = build_manifest(config, file_sha256(raw_path), corpus, splits)

    existing = read_manifest(config.manifest_path)
    differences = [] if existing is None else diff_manifests(existing, manifest)
    if differences and not force:
        raise ManifestMismatchError(config.manifest_path, differences)

    processed = data_dir / "processed"
    write_text(processed / "corpus.jsonl", _jsonl(map(document_record, corpus.documents)))
    write_text(processed / "questions.jsonl", _jsonl(map(question_record, corpus.questions)))
    write_text(config.manifest_path, render_manifest(manifest))

    status = "created" if existing is None else "overwritten" if differences else "unchanged"
    logger.info(
        "corpus.built",
        status=status,
        documents=len(corpus.documents),
        questions=len(corpus.questions),
        manifest_sha256=manifest["sha256"],
        differences=differences,
    )
    return BuildResult(corpus, splits, manifest, config.manifest_path, status)
