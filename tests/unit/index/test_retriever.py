"""Tests of the retriever with the hashing fake embedder."""

import numpy as np
import pytest

from ragbench.data.corpus import Document
from ragbench.index.backends import IndexSpec, IndexStatus, MemoryBackend, VectorsProvider
from ragbench.index.retriever import RetrievedDoc, Retriever
from ragbench.index.store import (
    InMemoryVectorStore,
    VectorStore,
    VectorStoreError,
    make_index_key,
)
from tests.fakes import HashingEmbedder

CORPUS_SHA = "c" * 64
DOCUMENTS = [
    Document("doc_a", "Tour Eiffel", "La tour Eiffel a été construite par Gustave Eiffel."),
    Document("doc_b", "Rhône", "Le Rhône prend sa source en Suisse."),
    Document("doc_c", "Marie Curie", "Marie Curie a reçu deux prix Nobel."),
    Document("doc_d", "Jupiter", "Jupiter est la plus grande planète du Système solaire."),
]


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def retriever(embedder: HashingEmbedder) -> Retriever:
    return Retriever.build(embedder, DOCUMENTS, CORPUS_SHA)


def test_closest_document_comes_first_with_its_text(retriever: Retriever) -> None:
    [best, *_] = retriever.retrieve("Où le Rhône prend-il sa source ?", 2)

    assert best == RetrievedDoc(
        "doc_b", "Rhône", "Le Rhône prend sa source en Suisse.", best.score, 1
    )


def test_ranks_start_at_one_and_scores_decrease(retriever: Retriever) -> None:
    results = retriever.retrieve("Qui a reçu le prix Nobel ?", 4)

    assert [doc.rank for doc in results] == [1, 2, 3, 4]
    scores = [doc.score for doc in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].doc_id == "doc_c"


def test_k_larger_than_corpus_returns_every_document(retriever: Retriever) -> None:
    assert len(retriever.retrieve("Jupiter", 50)) == len(retriever) == 4


def test_equal_scores_are_ordered_by_doc_id(embedder: HashingEmbedder) -> None:
    twins = [Document(doc_id, "T", "même texte") for doc_id in ("doc_z", "doc_m", "doc_a")]
    retriever = Retriever.build(embedder, twins, CORPUS_SHA)

    assert [doc.doc_id for doc in retriever.retrieve("même texte", 3)] == [
        "doc_a",
        "doc_m",
        "doc_z",
    ]


def test_passages_are_embedded_once_then_only_queries(embedder: HashingEmbedder) -> None:
    retriever = Retriever.build(embedder, DOCUMENTS, CORPUS_SHA)
    retriever.retrieve("Jupiter", 1)

    assert [kind for kind, _ in embedder.calls] == ["passage", "query"]


def test_empty_question_and_bad_k_are_rejected(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="empty text"):
        retriever.retrieve("  ", 3)
    with pytest.raises(ValueError, match="k must be at least 1"):
        retriever.retrieve("Jupiter", 0)


def _store(embedder: HashingEmbedder, key: str, docs: list[Document]) -> InMemoryVectorStore:
    vectors = embedder.embed_passages([doc.text for doc in docs])
    return InMemoryVectorStore.from_vectors([doc.doc_id for doc in docs], vectors, key)


def test_store_built_for_another_corpus_is_rejected(embedder: HashingEmbedder) -> None:
    store = _store(embedder, make_index_key(embedder.identity, "other corpus"), DOCUMENTS)

    with pytest.raises(VectorStoreError, match="built for another embedder or corpus"):
        Retriever(embedder, store, DOCUMENTS, CORPUS_SHA)


def test_store_built_with_another_embedder_is_rejected(embedder: HashingEmbedder) -> None:
    other = HashingEmbedder(name="other")
    store = _store(other, make_index_key(other.identity, CORPUS_SHA), DOCUMENTS)

    with pytest.raises(VectorStoreError, match="built for another embedder or corpus"):
        Retriever(embedder, store, DOCUMENTS, CORPUS_SHA)


def test_store_with_other_size_is_rejected(embedder: HashingEmbedder) -> None:
    store = _store(embedder, make_index_key(embedder.identity, CORPUS_SHA), DOCUMENTS[:2])

    with pytest.raises(VectorStoreError, match="does not match the corpus"):
        Retriever(embedder, store, DOCUMENTS, CORPUS_SHA)


def test_unknown_doc_id_from_store_is_an_error(embedder: HashingEmbedder) -> None:
    key = make_index_key(embedder.identity, CORPUS_SHA)
    vectors = embedder.embed_passages([doc.text for doc in DOCUMENTS])
    ids = [doc.doc_id for doc in DOCUMENTS[:-1]] + ["doc_unknown"]
    store = InMemoryVectorStore.from_vectors(ids, vectors, key)
    retriever = Retriever(embedder, store, DOCUMENTS, CORPUS_SHA)

    with pytest.raises(VectorStoreError, match="unknown doc_id 'doc_unknown'"):
        retriever.retrieve("Jupiter est la plus grande planète", 1)


def test_scores_are_python_floats(retriever: Retriever) -> None:
    [doc] = retriever.retrieve("Jupiter", 1)

    assert type(doc.score) is float
    assert np.isfinite(doc.score)


class ExistingIndexBackend:
    """Persistent backend whose index already exists: it must not need the vectors."""

    persistent = True

    def __init__(self, store: InMemoryVectorStore) -> None:
        self.store = store
        self.specs: list[IndexSpec] = []

    def open_or_create(
        self, spec: IndexSpec, vectors: VectorsProvider, *, rebuild: bool = False
    ) -> tuple[VectorStore, IndexStatus]:
        self.specs.append(spec)
        return self.store, IndexStatus.EXISTING


def test_opening_an_existing_index_does_not_embed_passages(embedder: HashingEmbedder) -> None:
    store = _store(embedder, make_index_key(embedder.identity, CORPUS_SHA), DOCUMENTS)
    backend = ExistingIndexBackend(store)
    embedder.calls.clear()

    retriever, status = Retriever.open(embedder, DOCUMENTS, CORPUS_SHA, backend)

    assert status is IndexStatus.EXISTING
    assert embedder.calls == []
    [spec] = backend.specs
    assert spec == IndexSpec(
        index_key=make_index_key(embedder.identity, CORPUS_SHA),
        embedder_identity=embedder.identity,
        corpus_sha256=CORPUS_SHA,
        doc_ids=tuple(doc.doc_id for doc in DOCUMENTS),
        dim=embedder.dim,
    )
    assert retriever.retrieve("Jupiter", 1)[0].doc_id == "doc_d"


def test_memory_backend_always_creates_from_vectors(embedder: HashingEmbedder) -> None:
    backend = MemoryBackend()

    _, status = Retriever.open(embedder, DOCUMENTS, CORPUS_SHA, backend, rebuild=True)

    assert status is IndexStatus.CREATED
    assert backend.persistent is False
    assert [kind for kind, _ in embedder.calls] == ["passage"]
