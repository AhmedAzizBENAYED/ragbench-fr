"""Rebuild the corpus from PIAF at the pinned revision (network) and compare to the commit."""

from pathlib import Path

from ragbench.data.build import build_dataset
from ragbench.data.build_config import load_corpus_build_config
from tests.constants import REPO_ROOT


def test_rebuild_matches_committed_manifest(tmp_path: Path) -> None:
    config = load_corpus_build_config(REPO_ROOT / "configs" / "data.yaml")
    committed = (REPO_ROOT / config.manifest_path).read_bytes()
    isolated = config.model_copy(update={"manifest_path": tmp_path / "manifest.json"})

    result = build_dataset(isolated, tmp_path / "data")

    assert result.status == "created"
    assert isolated.manifest_path.read_bytes() == committed
