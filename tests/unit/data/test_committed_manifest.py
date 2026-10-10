"""Consistency of the committed data/manifest.json with configs/data.yaml (no network).

The integration test rebuilds it from the source; this one only checks its internal coherence.
"""

import json
from typing import Any

import pytest

from ragbench.data.build_config import CorpusBuildConfig, load_corpus_build_config
from ragbench.data.corpus import NORMALIZATION_VERSION
from ragbench.data.manifest import FORMAT_VERSION, verify_manifest
from tests.constants import REPO_ROOT


@pytest.fixture
def config() -> CorpusBuildConfig:
    return load_corpus_build_config(REPO_ROOT / "configs" / "data.yaml")


@pytest.fixture
def manifest(config: CorpusBuildConfig) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((REPO_ROOT / config.manifest_path).read_text("utf-8"))
    return data


def test_manifest_is_internally_consistent(
    manifest: dict[str, Any], config: CorpusBuildConfig
) -> None:
    assert verify_manifest(manifest, config.sizes.model_dump()) == []


def test_manifest_matches_the_build_config(
    manifest: dict[str, Any], config: CorpusBuildConfig
) -> None:
    assert manifest["source"] == {
        "dataset": config.dataset,
        "revision": config.revision,
        "filename": config.filename,
        "sha256": config.raw_sha256,
    }
    assert manifest["build"] == {"seed": config.seed, "sizes": config.sizes.model_dump()}


def test_manifest_matches_the_code_versions(manifest: dict[str, Any]) -> None:
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["normalization_version"] == NORMALIZATION_VERSION


def test_counts_add_up(manifest: dict[str, Any]) -> None:
    counts = manifest["counts"]

    assert counts["raw_questions"] == counts["kept_questions"] + sum(counts["excluded"].values())
    assert counts["documents"] == manifest["corpus"]["documents"]
    assert counts["kept_questions"] == manifest["questions"]["count"]
