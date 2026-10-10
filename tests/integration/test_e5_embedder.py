"""Real e5 model on the PIAF corpus and the 100 `dev` questions (network, ~1 GB, CPU).

Measured at T04 (ADR 0005): top-1 0.69 / MRR 0.779 with prefixes, 0.70 / 0.785 without,
0.60 / 0.723 with swapped prefixes. The `test` split is never used here.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragbench.data.build import build_dataset
from ragbench.data.build_config import load_corpus_build_config
from ragbench.index.e5 import PREFIXES, E5Embedder, EncoderBackend, count_truncated
from ragbench.index.embedder import TextKind, Vectors, l2_normalize
from ragbench.log import get_logger
from tests.constants import E5_REVISION, REPO_ROOT

MODEL = "intfloat/multilingual-e5-base"
MIN_TOP1_WITH_PREFIXES = 0.60  # 0.69 measured on dev at T04, minus a safety margin
logger = get_logger(__name__)


@dataclass(frozen=True)
class DevData:
    doc_ids: list[str]
    passages: list[str]
    questions: list[str]
    gold: np.ndarray


@pytest.fixture(scope="module")
def dev_data(tmp_path_factory: pytest.TempPathFactory) -> DevData:
    workdir = tmp_path_factory.mktemp("piaf")
    config = load_corpus_build_config(REPO_ROOT / "configs" / "data.yaml")
    isolated = config.model_copy(update={"manifest_path": workdir / "manifest.json"})
    result = build_dataset(isolated, workdir / "data")

    def jsonl(name: str) -> list[dict[str, Any]]:
        path = workdir / "data" / "processed" / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    docs = jsonl("corpus.jsonl")
    questions = {q["question_id"]: q for q in jsonl("questions.jsonl")}
    dev = [questions[qid] for qid in result.manifest["splits"]["dev"]["question_ids"]]
    doc_ids = [doc["doc_id"] for doc in docs]
    return DevData(
        doc_ids=doc_ids,
        passages=[doc["text"] for doc in docs],
        questions=[q["question"] for q in dev],
        gold=np.array([doc_ids.index(q["doc_id"]) for q in dev]),
    )


@pytest.fixture(scope="module")
def backend() -> EncoderBackend:
    from ragbench.index._st_backend import SentenceTransformerBackend

    return SentenceTransformerBackend(MODEL, E5_REVISION)


@pytest.fixture(scope="module")
def embedder(backend: EncoderBackend) -> E5Embedder:
    return E5Embedder(MODEL, E5_REVISION, loader=lambda model, revision: backend)


def _ranks(queries: Vectors, passages: Vectors, gold: np.ndarray) -> np.ndarray:
    scores = queries @ passages.T
    gold_scores = scores[np.arange(len(gold)), gold]
    return np.asarray((scores > gold_scores[:, None]).sum(axis=1) + 1)


def _metrics(ranks: np.ndarray) -> dict[str, float]:
    return {
        "top1": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def _encode(backend: EncoderBackend, prefix: str, texts: list[str]) -> Vectors:
    return l2_normalize(backend.encode([prefix + text for text in texts], 32))


@pytest.fixture(scope="module")
def correct(embedder: E5Embedder, dev_data: DevData) -> dict[str, float]:
    passages = embedder.embed_passages(dev_data.passages)
    queries = embedder.embed_queries(dev_data.questions)
    return _metrics(_ranks(queries, passages, dev_data.gold))


def test_right_document_ranks_first_for_most_dev_questions(correct: dict[str, float]) -> None:
    logger.info("e5.dev.with_prefixes", **correct)

    assert correct["top1"] >= MIN_TOP1_WITH_PREFIXES


def test_correct_prefixes_beat_swapped_prefixes(
    backend: EncoderBackend, dev_data: DevData, correct: dict[str, float]
) -> None:
    swapped = _metrics(
        _ranks(
            _encode(backend, PREFIXES[TextKind.PASSAGE], dev_data.questions),
            _encode(backend, PREFIXES[TextKind.QUERY], dev_data.passages),
            dev_data.gold,
        )
    )
    # Measured, not asserted: on dev, removing the prefixes made no significant difference.
    unprefixed = _metrics(
        _ranks(
            _encode(backend, "", dev_data.questions),
            _encode(backend, "", dev_data.passages),
            dev_data.gold,
        )
    )
    logger.info("e5.dev.swapped_prefixes", **swapped)
    logger.info("e5.dev.without_prefixes", **unprefixed)

    assert correct["mrr"] > swapped["mrr"]


def test_single_and_batched_embeddings_agree(embedder: E5Embedder, dev_data: DevData) -> None:
    batch = embedder.embed_passages(dev_data.passages[:16])
    single = np.stack([embedder.embed_passages([text])[0] for text in dev_data.passages[:16:5]])

    np.testing.assert_allclose(single, batch[::5], atol=1e-5)


def test_no_passage_is_truncated(backend: EncoderBackend, dev_data: DevData) -> None:
    lengths = backend.token_lengths([PREFIXES[TextKind.PASSAGE] + t for t in dev_data.passages])

    assert count_truncated(lengths, backend.max_length) == 0, max(lengths)


def test_cached_embedder_round_trip_with_real_model(tmp_path: Path, embedder: E5Embedder) -> None:
    from ragbench.cache import SqliteStore
    from ragbench.index.cached import CachedEmbedder

    with SqliteStore(tmp_path / "embeddings.sqlite3") as store:
        cached = CachedEmbedder(embedder, store)
        first = cached.embed_queries(["Qui a construit la tour Eiffel ?"])
        second = cached.embed_queries(["Qui a construit la tour Eiffel ?"])

    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 768)
