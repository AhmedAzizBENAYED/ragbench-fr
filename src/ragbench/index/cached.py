"""Disk cache for any `Embedder`, by composition: a known embedding is never recomputed."""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import numpy as np

from ragbench.cache import CacheError, SqliteStore
from ragbench.hashing import canonical_hash
from ragbench.index.embedder import (
    Embedder,
    TextKind,
    Vectors,
    check_shape,
    check_texts,
    empty_vectors,
)

# Explicit little-endian float32, so cached bytes mean the same on every platform.
_STORED_DTYPE = np.dtype("<f4")


@dataclass(frozen=True)
class CacheStats:
    """Texts served from the cache (`hits`) and distinct texts computed by `inner` (`misses`)."""

    hits: int
    misses: int


class CachedEmbedder:
    """`Embedder` that serves known vectors from a `SqliteStore` and delegates the rest.

    Key: SHA-256 of `[identity, kind, text]` (canonical JSON). Only texts missing from the
    cache reach `inner`, in a single call; the output keeps the input order.
    """

    def __init__(self, inner: Embedder, store: SqliteStore) -> None:
        self.inner = inner
        self.store = store
        self._lock = threading.Lock()  # counters are updated from several threads (T15)
        self._hits = 0
        self._misses = 0

    @property
    def identity(self) -> str:
        return self.inner.identity

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._embed(texts, TextKind.QUERY, self.inner.embed_queries)

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self._embed(texts, TextKind.PASSAGE, self.inner.embed_passages)

    def close(self) -> None:
        """Close the cache store (the inner embedder holds no resource to release)."""
        self.store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def stats(self) -> CacheStats:
        """Cumulative hits and misses since creation."""
        with self._lock:
            return CacheStats(self._hits, self._misses)

    def key(self, kind: TextKind, text: str) -> str:
        """Cache key of one text."""
        return canonical_hash([self.identity, kind.value, text])

    def _embed(
        self, texts: Sequence[str], kind: TextKind, compute: Callable[[Sequence[str]], Vectors]
    ) -> Vectors:
        checked = check_texts(texts)
        if not checked:
            return empty_vectors(self.dim)
        keys = [self.key(kind, text) for text in checked]
        stored = self.store.get_many(keys)

        missing = list(
            dict.fromkeys(t for t, k in zip(checked, keys, strict=True) if k not in stored)
        )
        with self._lock:
            self._hits += sum(key in stored for key in keys)
            self._misses += len(missing)
        if missing:
            computed = check_shape(compute(missing), len(missing), self.dim)
            new = {
                self.key(kind, text): vector.astype(_STORED_DTYPE).tobytes()
                for text, vector in zip(missing, computed, strict=True)
            }
            self.store.put_many(new)
            stored |= new

        return np.stack([self._decode(stored[key]) for key in keys]).astype(np.float32)

    def _decode(self, value: bytes) -> Vectors:
        vector = np.frombuffer(value, dtype=_STORED_DTYPE)
        if vector.shape != (self.dim,):
            raise CacheError(
                f"{self.store.path}: cached vector has {vector.size} values, expected {self.dim}"
            )
        return vector.astype(np.float32)
