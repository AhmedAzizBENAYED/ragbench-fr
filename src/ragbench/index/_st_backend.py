"""sentence-transformers adapter for `E5Embedder`. No logic: it only forwards calls.

Excluded from unit-test coverage (like provider adapters, NFR-6): it loads torch and a 1 GB
model, so only integration tests exercise it. Every decision lives in `e5.py`.
"""

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging


class SentenceTransformerBackend:
    """A sentence-transformers model on CPU, at a pinned revision."""

    def __init__(self, model: str, revision: str) -> None:
        # Route transformers' logs through the ragbench renderer instead of its own handler.
        transformers_logging.disable_default_handler()
        transformers_logging.enable_propagation()
        self._model = SentenceTransformer(model, revision=revision, device="cpu")

    @property
    def dim(self) -> int:
        return int(self._model.get_embedding_dimension() or 0)

    @property
    def max_length(self) -> int:
        return int(self._model.max_seq_length)

    def encode(self, texts: list[str], batch_size: int) -> NDArray[np.floating]:
        vectors: NDArray[np.floating] = self._model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors

    def token_lengths(self, texts: list[str]) -> list[int]:
        encoded = self._model.tokenizer(texts, add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]
