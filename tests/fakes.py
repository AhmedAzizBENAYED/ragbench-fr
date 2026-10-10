"""Deterministic test doubles: no network, no torch."""

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from ragbench.index.embedder import Vectors, check_texts, empty_vectors, l2_normalize

_WORD = re.compile(r"\w+")


class HashingEmbedder:
    """Normalised bag of hashed words (hashing trick): texts sharing words get close vectors.

    Queries and passages are embedded the same way. Records every call for assertions.
    """

    def __init__(self, dim: int = 256, name: str = "fake-hashing") -> None:
        self._dim = dim
        self._name = name
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def identity(self) -> str:
        return f"{self._name}:dim={self._dim}:v1"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_queries(self, texts: Sequence[str]) -> Vectors:
        return self._embed("query", texts)

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self._embed("passage", texts)

    def _embed(self, kind: str, texts: Sequence[str]) -> Vectors:
        checked = check_texts(texts)
        self.calls.append((kind, checked))
        if not checked:
            return empty_vectors(self._dim)
        vectors = np.zeros((len(checked), self._dim), dtype=np.float32)
        for row, text in enumerate(checked):
            for word in _WORD.findall(text.lower()) or [text]:
                digest = hashlib.sha256(word.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "little") % self._dim
                vectors[row, column] += 1.0 if digest[4] & 1 else -1.0
            if not vectors[row].any():  # opposite signs cancelled out: keep a direction
                vectors[row, 0] = 1.0
        return l2_normalize(vectors)
