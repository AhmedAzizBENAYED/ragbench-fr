"""Store backends: open an existing index, or create it from lazily computed vectors.

A persistent backend (pgvector, T06) can open a matching index without the passage vectors,
so an idempotent `build-index` never recomputes embeddings it does not need.
"""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ragbench.config import Settings, VectorBackend
from ragbench.index.embedder import Vectors
from ragbench.index.store import InMemoryVectorStore, VectorStore

VectorsProvider = Callable[[], Vectors]


@dataclass(frozen=True)
class IndexSpec:
    """What an index must contain: identifies it and lets a backend check an existing one."""

    index_key: str
    embedder_identity: str
    corpus_sha256: str
    doc_ids: tuple[str, ...]
    dim: int


class IndexStatus(StrEnum):
    """What `open_or_create` did."""

    CREATED = "created"
    EXISTING = "existing"
    REBUILT = "rebuilt"


class StoreBackend(Protocol):
    """Gives a filled `VectorStore` for an index spec."""

    @property
    def persistent(self) -> bool:
        """Whether indexes outlive the process (otherwise every run creates the index)."""
        ...

    def open_or_create(
        self, spec: IndexSpec, vectors: VectorsProvider, *, rebuild: bool = False
    ) -> tuple[VectorStore, IndexStatus]:
        """Open the index of `spec` if it exists and matches, else fill it from `vectors()`.

        `rebuild` drops and recreates this index only. `vectors` is called only when filling.
        """
        ...


class MemoryBackend:
    """Non-persistent backend: the index is filled on every run (from the embedding cache)."""

    @property
    def persistent(self) -> bool:
        return False

    def open_or_create(
        self, spec: IndexSpec, vectors: VectorsProvider, *, rebuild: bool = False
    ) -> tuple[VectorStore, IndexStatus]:
        store = InMemoryVectorStore.from_vectors(spec.doc_ids, vectors(), spec.index_key)
        return store, IndexStatus.CREATED


BackendFactory = Callable[[Settings, ExitStack], StoreBackend]


def _memory(settings: Settings, resources: ExitStack) -> StoreBackend:
    return MemoryBackend()


# Extension point: one factory per `retrieval.backend` value. A factory registers on
# `resources` whatever it opens (e.g. a database connection).
STORE_BACKENDS: dict[VectorBackend, BackendFactory] = {
    VectorBackend.MEMORY: _memory,
}
