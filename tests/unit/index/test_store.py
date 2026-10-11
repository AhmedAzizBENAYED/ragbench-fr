"""Tests of the exact in-memory vector store."""

import numpy as np
import pytest

from ragbench.index.store import InMemoryVectorStore, VectorStoreError, make_index_key
from tests.fakes import HashingEmbedder

KEY = make_index_key("fake", "corpus")


def _store(doc_ids: list[str], rows: list[list[float]]) -> InMemoryVectorStore:
    return InMemoryVectorStore.from_vectors(doc_ids, np.array(rows, dtype=np.float32), KEY)


def test_closest_document_comes_first() -> None:
    embedder = HashingEmbedder()
    passages = [
        "La tour Eiffel a été construite par Gustave Eiffel.",
        "Le Rhône prend sa source en Suisse.",
        "Marie Curie a reçu deux prix Nobel.",
    ]
    store = InMemoryVectorStore.from_vectors(
        ["d1", "d2", "d3"], embedder.embed_passages(passages), KEY
    )

    [query] = embedder.embed_queries(["Où le Rhône prend-il sa source ?"])

    assert store.search(query, 1)[0][0] == "d2"


def test_scores_are_decreasing_dot_products() -> None:
    store = _store(["a", "b", "c"], [[1, 0], [0.6, 0.8], [0, 1]])

    hits = store.search(np.array([1.0, 0.0]), 3)

    assert [doc_id for doc_id, _ in hits] == ["a", "b", "c"]
    assert [score for _, score in hits] == pytest.approx([1.0, 0.6, 0.0])
    assert all(isinstance(score, float) for _, score in hits)


def test_ties_are_broken_by_doc_id() -> None:
    store = _store(["d3", "d1", "d2", "d0"], [[1, 0], [1, 0], [1, 0], [0, 1]])

    hits = store.search(np.array([1.0, 0.0], dtype=np.float32), 4)

    assert [doc_id for doc_id, _ in hits] == ["d1", "d2", "d3", "d0"]
    assert hits[0][1] == hits[1][1] == hits[2][1]


def test_k_larger_than_corpus_returns_everything() -> None:
    store = _store(["a", "b"], [[1, 0], [0, 1]])

    assert len(store.search(np.array([1.0, 0.0]), 10)) == 2


def test_scores_are_computed_in_float64() -> None:
    store = _store(["a"], [[0.1, 0.2]])

    [(_, score)] = store.search(np.array([0.3, 0.4], dtype=np.float32), 1)

    expected = float(np.float64(np.float32(0.1)) * np.float64(np.float32(0.3))) + float(
        np.float64(np.float32(0.2)) * np.float64(np.float32(0.4))
    )
    assert score == pytest.approx(expected, abs=1e-15)


def test_properties() -> None:
    store = _store(["a", "b"], [[1, 0, 0], [0, 1, 0]])

    assert (store.dim, len(store), store.index_key) == (3, 2, KEY)


def test_index_key_depends_on_embedder_and_corpus() -> None:
    assert make_index_key("e5", "c1") != make_index_key("e5", "c2") != make_index_key("x", "c1")


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        _store(["a"], [[1, 0]]).search(np.array([1.0, 0.0]), k)


@pytest.mark.parametrize("query", [np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0]])])
def test_query_of_wrong_shape_is_rejected(query: np.ndarray) -> None:
    with pytest.raises(VectorStoreError, match=r"query vector has shape .* expected \(2,\)"):
        _store(["a"], [[1, 0]]).search(query, 1)


def test_duplicate_doc_ids_are_rejected() -> None:
    with pytest.raises(VectorStoreError, match="duplicate doc_ids: a"):
        _store(["a", "b", "a"], [[1, 0], [0, 1], [1, 1]])


@pytest.mark.parametrize(
    "vectors", [np.ones((3, 2)), np.ones(2), np.ones((2, 2, 2))], ids=["count", "1d", "3d"]
)
def test_vectors_inconsistent_with_ids_are_rejected(vectors: np.ndarray) -> None:
    with pytest.raises(VectorStoreError, match="expected one vector per document"):
        InMemoryVectorStore.from_vectors(["a", "b"], vectors.astype(np.float32), KEY)


def test_empty_store_is_rejected() -> None:
    with pytest.raises(VectorStoreError, match="without documents"):
        InMemoryVectorStore.from_vectors([], np.zeros((0, 2), dtype=np.float32), KEY)
