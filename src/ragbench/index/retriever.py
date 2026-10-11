"""The retriever: the only retrieval interface pipelines see (no vectors, no backend)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self

from ragbench.config import VectorBackend
from ragbench.data.corpus import Document
from ragbench.index.embedder import Embedder, Vectors
from ragbench.index.store import InMemoryVectorStore, VectorStore, VectorStoreError, make_index_key

StoreFactory = Callable[[Sequence[str], Vectors, str], VectorStore]

# Extension point: T06 registers the pgvector backend here.
STORE_BACKENDS: dict[VectorBackend, StoreFactory] = {
    VectorBackend.MEMORY: InMemoryVectorStore.from_vectors,
}


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
    def build(
        cls,
        embedder: Embedder,
        documents: Sequence[Document],
        corpus_sha256: str,
        backend: VectorBackend = VectorBackend.MEMORY,
    ) -> Self:
        """Embed every passage (cache hits after the first time) and fill a store."""
        vectors = embedder.embed_passages([document.text for document in documents])
        store = STORE_BACKENDS[backend](
            [document.doc_id for document in documents],
            vectors,
            make_index_key(embedder.identity, corpus_sha256),
        )
        return cls(embedder, store, documents, corpus_sha256)

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
