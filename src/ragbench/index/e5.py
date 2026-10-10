"""e5 embedder (`intfloat/multilingual-e5-*`): prefixes, truncation report, normalisation.

e5 models expect `query: ` before queries and `passage: ` before passages; forgetting them
silently degrades retrieval. They are added here and never exposed to callers.
The model is loaded on first use, so a fully cached run never imports torch.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ragbench.index.embedder import (
    EmbedderError,
    TextKind,
    Vectors,
    check_shape,
    check_texts,
    empty_vectors,
    l2_normalize,
)
from ragbench.log import get_logger

logger = get_logger(__name__)

# Bump on any change of embedding logic (prefixes, normalisation, pooling, truncation): the
# version is part of `identity`, hence of cache keys; otherwise stale vectors would be served.
IMPLEMENTATION_VERSION = 1

PREFIXES = {TextKind.QUERY: "query: ", TextKind.PASSAGE: "passage: "}
E5_BASE_DIM = 768
DEFAULT_BATCH_SIZE = 32


class EncoderBackend(Protocol):
    """Thin access to the underlying model (sentence-transformers in production)."""

    @property
    def dim(self) -> int: ...

    @property
    def max_length(self) -> int: ...

    def encode(self, texts: list[str], batch_size: int) -> NDArray[np.floating]: ...

    def token_lengths(self, texts: list[str]) -> list[int]: ...


BackendLoader = Callable[[str, str], EncoderBackend]


# Not unit-tested on purpose: it imports torch and loads the real model (integration tests).
def _load_sentence_transformer(model: str, revision: str) -> EncoderBackend:  # pragma: no cover
    from ragbench.index._st_backend import SentenceTransformerBackend  # imports torch

    return SentenceTransformerBackend(model, revision)


def count_truncated(token_lengths: Sequence[int], max_length: int) -> int:
    """Number of texts longer than the model's maximum length (they get truncated)."""
    return sum(length > max_length for length in token_lengths)


class E5Embedder:
    """`Embedder` for e5 models, on CPU, loaded lazily."""

    def __init__(
        self,
        model: str,
        revision: str,
        *,
        dim: int = E5_BASE_DIM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        loader: BackendLoader = _load_sentence_transformer,
    ) -> None:
        self.model = model
        self.revision = revision
        self.batch_size = batch_size
        self._dim = dim
        self._loader = loader
        self._backend: EncoderBackend | None = None

    @property
    def identity(self) -> str:
        return f"e5:{self.model}@{self.revision}:v{IMPLEMENTATION_VERSION}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._embed(texts, TextKind.QUERY)

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self._embed(texts, TextKind.PASSAGE)

    def _load(self) -> EncoderBackend:
        if self._backend is None:
            logger.info("embedder.loading", model=self.model, revision=self.revision)
            backend = self._loader(self.model, self.revision)
            if backend.dim != self._dim:
                raise EmbedderError(
                    f"{self.model}@{self.revision} has dimension {backend.dim}, "
                    f"expected {self._dim}"
                )
            self._backend = backend
        return self._backend

    def _embed(self, texts: Sequence[str], kind: TextKind) -> Vectors:
        checked = check_texts(texts)
        if not checked:
            return empty_vectors(self._dim)
        prefixed = [PREFIXES[kind] + text for text in checked]
        backend = self._load()

        truncated = count_truncated(backend.token_lengths(prefixed), backend.max_length)
        if truncated:
            logger.warning(
                "embedder.truncated",
                kind=kind.value,
                truncated=truncated,
                total=len(prefixed),
                max_length=backend.max_length,
            )
        # sentence-transformers sorts each call's texts by length into fixed-size batches.
        vectors = backend.encode(prefixed, self.batch_size)
        return l2_normalize(check_shape(np.asarray(vectors), len(prefixed), self._dim))
