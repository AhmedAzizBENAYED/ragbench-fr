"""The retriever: the only retrieval interface pipelines see (no vectors, no backend)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from ragbench.data.corpus import Document
from ragbench.index.backends import IndexSpec, IndexStatus, MemoryBackend, StoreBackend
from ragbench.index.embedder import Embedder
from ragbench.index.store import VectorStore, VectorStoreError, make_index_key


@dataclass(frozen=True)
class RetrievedDoc:
    """A retrieved passage. `rank` starts at 1.

    `title` and `text` are what pipelines generate from; run artefacts keep only `doc_id`,
    `score` and `rank` (T15).
    """

    doc_id: str
    title: str
    text: str
    score: float
    rank: int


class Retriever:
    """Embeds a question and returns the top-k documents of the corpus."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        documents: Sequence[Document],
        corpus_sha256: str,
    ) -> None:
        """Bind a filled store to its embedder and corpus.

        Raises:
            VectorStoreError: the store was filled with another embedder or corpus.
        """
        expected_key = make_index_key(embedder.identity, corpus_sha256)
        if store.index_key != expected_key:
            raise VectorStoreError(
                "the vector store was built for another embedder or corpus "
                f"(index key {store.index_key}, expected {expected_key})"
            )
        if store.dim != embedder.dim or len(store) != len(documents):
            raise VectorStoreError(
                f"vector store ({len(store)} documents, dim {store.dim}) does not match "
                f"the corpus ({len(documents)} documents) and embedder (dim {embedder.dim})"
            )
        self._embedder = embedder
        self._store = store
        self._documents = {document.doc_id: document for document in documents}

    @classmethod
    def open(
        cls,
        embedder: Embedder,
        documents: Sequence[Document],
        corpus_sha256: str,
        backend: StoreBackend,
        *,
        rebuild: bool = False,
    ) -> tuple[Self, IndexStatus]:
        """Open the backend's index for this embedder and corpus, creating it if needed.

        Passages are embedded only if the backend has to fill the index.
        """
        spec = IndexSpec(
            index_key=make_index_key(embedder.identity, corpus_sha256),
            embedder_identity=embedder.identity,
            corpus_sha256=corpus_sha256,
            doc_ids=tuple(document.doc_id for document in documents),
            dim=embedder.dim,
        )
        store, status = backend.open_or_create(
            spec,
            lambda: embedder.embed_passages([document.text for document in documents]),
            rebuild=rebuild,
        )
        return cls(embedder, store, documents, corpus_sha256), status

    @classmethod
    def build(cls, embedder: Embedder, documents: Sequence[Document], corpus_sha256: str) -> Self:
        """In-memory retriever: embed every passage (cache hits after the first time)."""
        retriever, _ = cls.open(embedder, documents, corpus_sha256, MemoryBackend())
        return retriever

    def __len__(self) -> int:
        return len(self._documents)

    def retrieve(self, question: str, k: int) -> list[RetrievedDoc]:
        """Top-k documents for a question, by decreasing score (ties by `doc_id`).

        Raises:
            ValueError: empty question or `k < 1`.
            VectorStoreError: the backend failed or returned an unknown document.
        """
        [vector] = self._embedder.embed_queries([question])
        retrieved: list[RetrievedDoc] = []
        for rank, (doc_id, score) in enumerate(self._store.search(vector, k), start=1):
            document = self._documents.get(doc_id)
            if document is None:
                raise VectorStoreError(f"the vector store returned an unknown doc_id {doc_id!r}")
            retrieved.append(RetrievedDoc(doc_id, document.title, document.text, score, rank))
        return retrieved
