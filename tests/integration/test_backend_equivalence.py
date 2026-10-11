"""pgvector and the in-memory store rank the dev questions identically (real e5, PostgreSQL).

Needs the network (model, PIAF) and PostgreSQL (`make db-up`). A rank difference is reported
with the question, both documents and the score gap, never relaxed silently (ADR 0006).
"""

import json
from contextlib import ExitStack
from pathlib import Path

from ragbench.config import load_run_config
from ragbench.data.build import build_dataset
from ragbench.data.build_config import load_corpus_build_config
from ragbench.data.processed import PROCESSED_DIR, load_corpus
from ragbench.index.backends import MemoryBackend
from ragbench.index.factory import make_embedder
from ragbench.index.pgvector_store import PgVectorBackend
from ragbench.index.retriever import Retriever
from ragbench.log import get_logger
from tests.constants import EXAMPLE_CONFIG, REPO_ROOT

TOP_K = 10
logger = get_logger(__name__)


def test_pgvector_and_memory_rankings_agree_on_dev(tmp_path: Path, pg_dsn: str) -> None:
    data_config = load_corpus_build_config(REPO_ROOT / "configs" / "data.yaml")
    isolated = data_config.model_copy(update={"manifest_path": tmp_path / "manifest.json"})
    built = build_dataset(isolated, tmp_path / "data")
    documents = load_corpus(tmp_path / "data" / PROCESSED_DIR, built.manifest)
    corpus_sha = built.manifest["corpus"]["sha256"]
    questions = {
        record["question_id"]: record["question"]
        for record in map(
            json.loads,
            (tmp_path / "data" / PROCESSED_DIR / "questions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    dev = [questions[qid] for qid in built.manifest["splits"]["dev"]["question_ids"]]
    embedding = load_run_config(EXAMPLE_CONFIG).embedding

    with ExitStack() as resources:
        embedder = resources.enter_context(make_embedder(embedding, tmp_path / "cache"))
        pg = resources.enter_context(PgVectorBackend.connect(pg_dsn))
        memory, _ = Retriever.open(embedder, documents, corpus_sha, MemoryBackend())
        postgres, _ = Retriever.open(embedder, documents, corpus_sha, pg, rebuild=True)

        mismatches: list[str] = []
        max_gap = 0.0
        for question in dev:
            in_memory = memory.retrieve(question, TOP_K)
            in_pg = postgres.retrieve(question, TOP_K)
            for rank, (a, b) in enumerate(zip(in_memory, in_pg, strict=True), start=1):
                max_gap = max(max_gap, abs(a.score - b.score))
                if a.doc_id != b.doc_id:
                    mismatches.append(
                        f"{question!r} rank {rank}: memory {a.doc_id} ({a.score:.9f}) vs "
                        f"pgvector {b.doc_id} ({b.score:.9f}), gap {abs(a.score - b.score):.2e}"
                    )

    logger.info(
        "backends.equivalence",
        questions=len(dev),
        top_k=TOP_K,
        mismatches=len(mismatches),
        max_score_gap=max_gap,
    )
    assert len(dev) == 100
    assert not mismatches, "\n".join(mismatches)
    assert max_gap < 1e-5
