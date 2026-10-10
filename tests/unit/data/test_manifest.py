"""Tests of the manifest and of the end-to-end build (download faked: no network)."""

import copy
import json
from pathlib import Path

import pytest

from ragbench.data import DataError
from ragbench.data.build import Downloader, ManifestMismatchError, build_dataset
from ragbench.data.build_config import CorpusBuildConfig
from ragbench.data.manifest import (
    diff_manifests,
    read_manifest,
    render_manifest,
    verify_manifest,
)
from ragbench.hashing import canonical_hash

SIZES = {"dev": 2, "test": 3, "ci": 2}


def _build(config: CorpusBuildConfig, download: Downloader, **kwargs: bool) -> bytes:
    build_dataset(config, Path("data"), download=download, **kwargs)
    return config.manifest_path.read_bytes()


def test_two_builds_give_byte_identical_manifests(
    build_config: CorpusBuildConfig, fake_download: Downloader, tmp_path: Path
) -> None:
    first = _build(build_config, fake_download)
    config_elsewhere = build_config.model_copy(update={"manifest_path": tmp_path / "m2.json"})

    assert _build(config_elsewhere, fake_download) == first


def test_manifest_content(build_config: CorpusBuildConfig, fake_download: Downloader) -> None:
    result = build_dataset(build_config, Path("data"), download=fake_download)
    manifest = result.manifest

    assert result.status == "created"
    assert list(manifest) == [
        "format_version", "normalization_version", "source", "build", "counts",
        "corpus", "questions", "splits", "sha256",
    ]  # fmt: skip
    assert manifest["source"] == {
        "dataset": "AgentPublic/piaf",
        "revision": build_config.revision,
        "filename": build_config.filename,
        "sha256": build_config.raw_sha256,
    }
    assert manifest["build"] == {"seed": 7, "sizes": SIZES}
    assert manifest["counts"] == {
        "raw_questions": 16,
        "kept_questions": 9,
        "excluded": {
            "empty_question": 1,
            "empty_answer": 1,
            "answer_not_found": 2,
            "answer_offset_mismatch": 1,
            "duplicate_question": 2,
        },
        "offsets_repaired": 1,
        "answers_merged": 1,
        "documents": 8,
    }
    assert verify_manifest(manifest, SIZES) == []


def test_manifest_has_no_timestamp_or_absolute_path(
    build_config: CorpusBuildConfig, fake_download: Downloader
) -> None:
    text = _build(build_config, fake_download).decode("utf-8")

    assert str(Path.cwd()) not in text
    assert "\\\\" not in text
    assert "2026" not in text
    assert b"\r\n" not in _build(build_config, fake_download)


def test_processed_files(build_config: CorpusBuildConfig, fake_download: Downloader) -> None:
    build_dataset(build_config, Path("data"), download=fake_download)

    corpus = Path("data/processed/corpus.jsonl").read_text(encoding="utf-8").splitlines()
    questions = Path("data/processed/questions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(corpus) == 8
    assert len(questions) == 9
    assert set(json.loads(corpus[0])) == {"doc_id", "title", "text"}
    assert set(json.loads(questions[0])) == {"question_id", "question", "answers", "doc_id"}
    assert "é" in Path("data/processed/corpus.jsonl").read_text(encoding="utf-8")


def test_unchanged_rebuild_is_accepted(
    build_config: CorpusBuildConfig, fake_download: Downloader
) -> None:
    build_dataset(build_config, Path("data"), download=fake_download)

    result = build_dataset(build_config, Path("data"), download=fake_download)

    assert result.status == "unchanged"


def test_changed_manifest_is_refused_without_force(
    build_config: CorpusBuildConfig, fake_download: Downloader
) -> None:
    original = _build(build_config, fake_download)
    reseeded = build_config.model_copy(update={"seed": 8})

    with pytest.raises(ManifestMismatchError) as excinfo:
        build_dataset(reseeded, Path("data"), download=fake_download)

    assert "build.seed" in excinfo.value.differences
    assert "--force" in str(excinfo.value)
    assert build_config.manifest_path.read_bytes() == original


def test_force_overwrites_a_changed_manifest(
    build_config: CorpusBuildConfig, fake_download: Downloader
) -> None:
    original = _build(build_config, fake_download)
    reseeded = build_config.model_copy(update={"seed": 8})

    result = build_dataset(reseeded, Path("data"), force=True, download=fake_download)

    assert result.status == "overwritten"
    assert build_config.manifest_path.read_bytes() != original


def test_diff_manifests_reports_dotted_paths() -> None:
    old = {"a": 1, "b": {"c": [1], "d": 2}}
    new = {"a": 1, "b": {"c": [2], "d": 2}, "e": 3}

    assert diff_manifests(old, new) == ["b.c", "e"]
    assert diff_manifests(old, copy.deepcopy(old)) == []


def test_render_is_stable_utf8_with_final_newline() -> None:
    text = render_manifest({"b": "é", "a": 1})

    assert text == '{\n  "b": "é",\n  "a": 1\n}\n'


def test_read_manifest(tmp_path: Path) -> None:
    assert read_manifest(tmp_path / "missing.json") is None
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(DataError, match="must be a JSON object"):
        read_manifest(tmp_path / "list.json")
    with pytest.raises(DataError, match="cannot read manifest"):
        read_manifest(tmp_path / "broken.json")


@pytest.fixture
def manifest(build_config: CorpusBuildConfig, fake_download: Downloader) -> dict[str, object]:
    return build_dataset(build_config, Path("data"), download=fake_download).manifest


def _resign(manifest: dict[str, object]) -> dict[str, object]:
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    return body | {"sha256": canonical_hash(body)}


def test_verify_detects_tampered_content(manifest: dict[str, object]) -> None:
    tampered = copy.deepcopy(manifest) | {"format_version": 99}

    assert verify_manifest(tampered, SIZES) == ["sha256 does not match the manifest content"]


@pytest.mark.parametrize(
    ("mutate", "problem"),
    [
        (lambda s: s["test"]["question_ids"].pop(), "splits.test.sha256 does not match"),
        (lambda s: s["dev"]["question_ids"].reverse(), "splits.dev.question_ids must be sorted"),
        (lambda s: s["ci"].update(size=5), "splits.ci.size must be 2"),
        (lambda s: s["test"]["doc_ids"].pop(), "splits.test must have exactly one question"),
        (lambda s: s["ci"]["question_ids"].append("zzz"), "splits.ci must be a subset"),
        (
            lambda s: s["dev"]["doc_ids"].__setitem__(0, s["test"]["doc_ids"][0]),
            "splits.dev and splits.test must not share documents",
        ),
    ],
)
def test_verify_detects_inconsistent_splits(
    manifest: dict[str, object], mutate: object, problem: str
) -> None:
    broken = copy.deepcopy(manifest)
    mutate(broken["splits"])  # type: ignore[operator]

    problems = verify_manifest(_resign(broken), SIZES)

    assert any(p.startswith(problem) for p in problems), problems
