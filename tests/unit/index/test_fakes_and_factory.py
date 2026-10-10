"""Tests of the fake embedder used by later tickets, and of the embedder factory."""

from pathlib import Path

import numpy as np
import pytest

from ragbench.config import EmbeddingConfig
from ragbench.index.cached import CachedEmbedder
from ragbench.index.factory import EMBEDDING_CACHE_FILE, make_embedder
from tests.constants import E5_REVISION
from tests.fakes import HashingEmbedder


def test_fake_embedder_is_deterministic_and_normalised() -> None:
    texts = ["La tour Eiffel est à Paris.", "Le Rhône prend sa source en Suisse."]

    first = HashingEmbedder().embed_passages(texts)
    second = HashingEmbedder().embed_passages(texts)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-6)


def test_fake_embedder_brings_related_texts_closer() -> None:
    embedder = HashingEmbedder()
    [query] = embedder.embed_queries(["Où le Rhône prend-il sa source ?"])
    passages = embedder.embed_passages(
        [
            "La tour Eiffel a été construite par Gustave Eiffel.",
            "Le Rhône prend sa source en Suisse.",
            "Marie Curie a reçu deux prix Nobel.",
        ]
    )

    scores = passages @ query

    assert int(np.argmax(scores)) == 1


def test_fake_embedder_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty text"):
        HashingEmbedder().embed_queries([" "])


def test_fake_embedder_handles_text_without_words() -> None:
    [vector] = HashingEmbedder().embed_passages(["?!"])

    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_make_embedder_builds_a_cached_e5_without_loading(tmp_path: Path) -> None:
    embedder = make_embedder(EmbeddingConfig(revision=E5_REVISION), tmp_path)

    assert isinstance(embedder, CachedEmbedder)
    assert embedder.identity == f"e5:intfloat/multilingual-e5-base@{E5_REVISION}:v1"
    assert embedder.dim == 768
    assert (tmp_path / EMBEDDING_CACHE_FILE).is_file()
