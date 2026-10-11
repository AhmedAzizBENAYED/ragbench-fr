"""Tests of the embedding disk cache."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from ragbench.cache import CacheError, SqliteStore
from ragbench.index.cached import CachedEmbedder, CacheStats
from ragbench.index.e5 import E5Embedder, EncoderBackend
from ragbench.index.embedder import EmbedderError, TextKind
from tests.constants import E5_REVISION
from tests.fakes import HashingEmbedder


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "embeddings.sqlite3")


@pytest.fixture
def inner() -> HashingEmbedder:
    return HashingEmbedder(dim=16)


@pytest.fixture
def cached(inner: HashingEmbedder, store: SqliteStore) -> CachedEmbedder:
    return CachedEmbedder(inner, store)


def test_delegates_identity_and_dim(cached: CachedEmbedder, inner: HashingEmbedder) -> None:
    assert (cached.identity, cached.dim) == (inner.identity, inner.dim)


def test_second_call_does_not_reach_inner(cached: CachedEmbedder, inner: HashingEmbedder) -> None:
    first = cached.embed_passages(["La tour Eiffel.", "Le Rhône."])
    second = cached.embed_passages(["La tour Eiffel.", "Le Rhône."])

    assert inner.calls == [("passage", ["La tour Eiffel.", "Le Rhône."])]
    np.testing.assert_array_equal(first, second)
    assert second.dtype == np.float32


def test_only_missing_texts_are_computed_in_one_call_and_order_is_kept(
    cached: CachedEmbedder, inner: HashingEmbedder
) -> None:
    cached.embed_queries(["b"])

    vectors = cached.embed_queries(["a", "b", "c", "a"])

    assert inner.calls[1:] == [("query", ["a", "c"])]
    expected = HashingEmbedder(dim=16).embed_queries(["a", "b", "c", "a"])
    np.testing.assert_allclose(vectors, expected, rtol=1e-6)


def test_queries_and_passages_have_distinct_keys(
    cached: CachedEmbedder, inner: HashingEmbedder
) -> None:
    cached.embed_passages(["même texte"])
    cached.embed_queries(["même texte"])

    assert [kind for kind, _ in inner.calls] == ["passage", "query"]
    assert cached.key(TextKind.QUERY, "x") != cached.key(TextKind.PASSAGE, "x")


def test_identities_have_distinct_keys(store: SqliteStore) -> None:
    first, second = HashingEmbedder(dim=16, name="one"), HashingEmbedder(dim=16, name="two")

    CachedEmbedder(first, store).embed_passages(["texte"])
    CachedEmbedder(second, store).embed_passages(["texte"])

    assert len(first.calls) == len(second.calls) == 1
    assert len(store) == 2


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "persist.sqlite3"
    with SqliteStore(path) as store:
        CachedEmbedder(HashingEmbedder(dim=16), store).embed_passages(["texte"])

    later = HashingEmbedder(dim=16)
    with SqliteStore(path) as store:
        CachedEmbedder(later, store).embed_passages(["texte"])

    assert later.calls == []


def test_empty_text_is_rejected_and_empty_list_is_fine(
    cached: CachedEmbedder, inner: HashingEmbedder
) -> None:
    with pytest.raises(ValueError, match="empty text"):
        cached.embed_queries(["a", ""])

    assert cached.embed_queries([]).shape == (0, 16)
    assert inner.calls == []


def test_corrupted_cached_vector_is_an_error(cached: CachedEmbedder, store: SqliteStore) -> None:
    store.put_many({cached.key(TextKind.PASSAGE, "texte"): b"\x00\x00\x80\x3f"})

    with pytest.raises(CacheError, match="cached vector has 1 values, expected 16"):
        cached.embed_passages(["texte"])


def test_inner_returning_wrong_shape_is_an_error(store: SqliteStore) -> None:
    class Broken(HashingEmbedder):
        def embed_passages(self, texts: object) -> np.ndarray:
            return np.ones((1, 16), dtype=np.float32)

    with pytest.raises(EmbedderError, match="expected vectors of shape"):
        CachedEmbedder(Broken(dim=16), store).embed_passages(["a", "b"])


def test_fully_cached_e5_never_loads_the_model(store: SqliteStore) -> None:
    calls: list[str] = []

    class Backend:
        dim = 2
        max_length = 512

        def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
            return np.ones((len(texts), 2))

        def token_lengths(self, texts: list[str]) -> list[int]:
            return [1] * len(texts)

    def loader(model: str, revision: str) -> EncoderBackend:
        calls.append(model)
        return Backend()

    def e5() -> CachedEmbedder:
        return CachedEmbedder(E5Embedder("m", E5_REVISION, dim=2, loader=loader), store)

    e5().embed_passages(["texte"])
    assert calls == ["m"]

    e5().embed_passages(["texte"])
    assert calls == ["m"]  # second embedder: everything cached, no model loaded


def test_stats_count_hits_and_distinct_misses(cached: CachedEmbedder) -> None:
    cached.embed_passages(["a", "b", "a"])
    cached.embed_passages(["a", "c"])

    assert cached.stats == CacheStats(hits=1, misses=3)


def test_stats_are_consistent_under_threads(cached: CachedEmbedder) -> None:
    texts = [f"texte {i}" for i in range(50)]
    cached.embed_queries(texts)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: cached.embed_queries(texts), range(16)))

    assert cached.stats == CacheStats(hits=16 * 50, misses=50)
