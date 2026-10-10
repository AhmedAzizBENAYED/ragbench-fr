"""The `Embedder` contract and helpers shared by its implementations."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]


class TextKind(StrEnum):
    """Role of an embedded text; some models (e5) encode queries and passages differently."""

    QUERY = "query"
    PASSAGE = "passage"


class EmbedderError(Exception):
    """The embedding model returned something inconsistent with the embedder's contract."""


class Embedder(Protocol):
    """Turns texts into L2-normalised float32 vectors, so cosine similarity is a dot product.

    Queries and passages have separate methods: any model-specific handling (such as e5's
    `query: ` / `passage: ` prefixes) stays inside the implementation.
    """

    @property
    def identity(self) -> str:
        """Model, revision and implementation version; part of every cache key."""
        ...

    @property
    def dim(self) -> int:
        """Vector dimension."""
        ...

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        """Embed search queries; shape `(len(texts), dim)`. Raises ValueError on empty text."""
        ...

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        """Embed passages to search; shape `(len(texts), dim)`. Raises ValueError on empty text."""
        ...


def check_texts(texts: Sequence[str]) -> list[str]:
    """Return the texts as a list.

    Raises:
        ValueError: a text is empty or only whitespace.
    """
    checked = list(texts)
    for index, text in enumerate(checked):
        if not text.strip():
            raise ValueError(f"cannot embed an empty text (position {index})")
    return checked


def empty_vectors(dim: int) -> Vectors:
    """Result for an empty list of texts."""
    return np.zeros((0, dim), dtype=np.float32)


def l2_normalize(vectors: NDArray[np.floating]) -> Vectors:
    """Scale each row to unit L2 norm.

    Raises:
        EmbedderError: a row is all zeros (it has no direction).
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbedderError("cannot normalise a zero vector")
    normalized: Vectors = (matrix / norms).astype(np.float32)
    return normalized


def check_shape(vectors: Vectors, count: int, dim: int) -> Vectors:
    """Ensure an embedding model returned one `dim`-sized vector per text.

    Raises:
        EmbedderError: unexpected shape.
    """
    if vectors.shape != (count, dim):
        raise EmbedderError(f"expected vectors of shape {(count, dim)}, got {vectors.shape}")
    return vectors
