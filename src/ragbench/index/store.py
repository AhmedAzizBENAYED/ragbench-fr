"""Vector stores: exact top-k search over passage vectors (ADR 0006).

The protocol is read-only. Each backend has its own way to be filled once (in memory now; a
persistent, costly SQL table with T06), after which it is immutable. Search is exact (no
approximate index) and ties are broken by `doc_id`, so results are deterministic.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray

from ragbench.hashing import canonical_hash
from ragbench.index.embedder import Vectors

Hit = tuple[str, float]


class VectorStoreError(Exception):
    """A vector store cannot be built or queried (inconsistent input, unreachable backend)."""


class VectorStore(Protocol):
    """Exact top-k search over a fixed set of documents.

    `search` returns `(doc_id, score)` pairs sorted by decreasing score, ties broken by
    increasing `doc_id`. It may raise `VectorStoreError` (e.g. a lost database connection).
    """

    @property
    def dim(self) -> int:
        """Vector dimension."""
        ...

    @property
    def index_key(self) -> str:
        """Identifies the content (embedder identity and corpus hash) the store was filled with."""
        ...

    def __len__(self) -> int: ...

    def search(self, query: NDArray[np.floating], k: int) -> list[Hit]:
        """Return the `min(k, len(self))` best documents for a query vector."""
        ...


def make_index_key(embedder_identity: str, corpus_sha256: str) -> str:
    """Key binding a store's vectors to the embedder and the corpus they come from."""
    return canonical_hash([embedder_identity, corpus_sha256])


class InMemoryVectorStore:
    """Exact search on a dense matrix. Scores are dot products computed in float64.

    float64 keeps scores identical across BLAS implementations (Windows, Linux), so two nearly
    tied documents cannot swap ranks from one machine to another.
    """

    def __init__(self, doc_ids: Sequence[str], matrix: NDArray[np.float64], index_key: str) -> None:
        self._doc_ids = np.asarray(doc_ids, dtype=np.str_)
        self._matrix = matrix
        self._index_key = index_key

    @classmethod
    def from_vectors(cls, doc_ids: Sequence[str], vectors: Vectors, index_key: str) -> Self:
        """Fill a store once.

        Raises:
            VectorStoreError: no document, duplicate IDs, or vectors inconsistent with the IDs.
        """
        ids = list(doc_ids)
        if not ids:
            raise VectorStoreError("cannot build a vector store without documents")
        duplicates = sorted(doc_id for doc_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise VectorStoreError(f"duplicate doc_ids: {', '.join(duplicates[:5])}")
        matrix = np.asarray(vectors)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise VectorStoreError(
                f"expected one vector per document ({len(ids)}), got shape {matrix.shape}"
            )
        return cls(ids, matrix.astype(np.float64), index_key)

    @property
    def dim(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def index_key(self) -> str:
        return self._index_key

    def __len__(self) -> int:
        return len(self._doc_ids)

    def search(self, query: NDArray[np.floating], k: int) -> list[Hit]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        vector = np.asarray(query, dtype=np.float64)
        if vector.shape != (self.dim,):
            raise VectorStoreError(f"query vector has shape {vector.shape}, expected ({self.dim},)")
        scores = self._matrix @ vector
        # lexsort sorts by the last key first: decreasing score, then increasing doc_id.
        order = np.lexsort((self._doc_ids, -scores))[:k]
        return [(str(self._doc_ids[i]), float(scores[i])) for i in order]
