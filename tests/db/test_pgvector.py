"""pgvector backend against a real PostgreSQL (marker `db`), with the hashing fake embedder."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import psycopg
import pytest

from ragbench.data.corpus import Corpus
from ragbench.index import pgvector_store
from ragbench.index.backends import IndexSpec, IndexStatus
from ragbench.index.embedder import Vectors
from ragbench.index.pgvector_store import SCHEMA_VERSION, PgVectorBackend
from ragbench.index.store import InMemoryVectorStore, VectorStoreError, make_index_key
from tests.fakes import HashingEmbedder

QUERIES = [
    "Qui a construit la tour Eiffel ?",
    "Où le Rhône prend-il sa source ?",
    "Combien de prix Nobel Marie Curie a-t-elle reçus ?",
    "Quelle est la plus grande planète ?",
    "D'où vient le camembert ?",
]


class CountingVectors:
    """Vector provider that records how often the backend asks for vectors."""

    def __init__(self, vectors: Vectors) -> None:
        self.vectors = vectors
        self.calls = 0

    def __call__(self) -> Vectors:
        self.calls += 1
        return self.vectors


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def spec(corpus: Corpus, embedder: HashingEmbedder) -> IndexSpec:
    corpus_sha = "c" * 64
    return IndexSpec(
        index_key=make_index_key(embedder.identity, corpus_sha),
        embedder_identity=embedder.identity,
        corpus_sha256=corpus_sha,
        doc_ids=tuple(doc.doc_id for doc in corpus.documents),
        dim=embedder.dim,
    )


@pytest.fixture
def vectors(corpus: Corpus, embedder: HashingEmbedder) -> CountingVectors:
    return CountingVectors(embedder.embed_passages([doc.text for doc in corpus.documents]))


def _rows(dsn: str, sql: str) -> list[tuple[Any, ...]]:
    with psycopg.connect(dsn) as connection:
        return connection.execute(sql).fetchall()


def test_fill_then_search(
    pg_backend: PgVectorBackend, spec: IndexSpec, vectors: CountingVectors, corpus: Corpus
) -> None:
    store, status = pg_backend.open_or_create(spec, vectors)

    assert status is IndexStatus.CREATED
    assert (len(store), store.dim, store.index_key) == (8, 256, spec.index_key)
    [query] = HashingEmbedder().embed_queries(["Qui a écrit Les Misérables ?"])
    hits = store.search(query, 3)
    titles = {doc.doc_id: doc.title for doc in corpus.documents}
    assert titles[hits[0][0]] == "Victor Hugo"
    assert [score for _, score in hits] == sorted((score for _, score in hits), reverse=True)


def test_existing_index_is_opened_without_vectors(
    pg_backend: PgVectorBackend, spec: IndexSpec, vectors: CountingVectors
) -> None:
    pg_backend.open_or_create(spec, vectors)

    _, status = pg_backend.open_or_create(spec, vectors)

    assert status is IndexStatus.EXISTING
    assert vectors.calls == 1


def test_rebuild_replaces_only_this_index(
    pg_backend: PgVectorBackend, pg_dsn: str, spec: IndexSpec, vectors: CountingVectors
) -> None:
    other = IndexSpec("other-key", "other", "d" * 64, spec.doc_ids, spec.dim)
    pg_backend.open_or_create(other, vectors)
    pg_backend.open_or_create(spec, vectors)

    _, status = pg_backend.open_or_create(spec, vectors, rebuild=True)

    assert status is IndexStatus.REBUILT
    assert vectors.calls == 3
    counts = dict(_rows(pg_dsn, "SELECT index_key, count(*) FROM vectors GROUP BY index_key"))
    assert counts == {"other-key": 8, spec.index_key: 8}


def test_ties_are_broken_by_doc_id_in_c_collation(pg_backend: PgVectorBackend) -> None:
    # The database collation (en_US.utf8) sorts doc_a before doc_B; Python and "C" do not.
    doc_ids = ("doc_a", "doc_B", "doc-1", "doc_1", "doc_z")
    vectors = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    spec = IndexSpec("ties", "fake", "e" * 64, doc_ids, 2)
    store, _ = pg_backend.open_or_create(spec, lambda: vectors)
    memory = InMemoryVectorStore.from_vectors(doc_ids, vectors, "ties")
    query = np.array([1.0, 0.0], dtype=np.float32)

    assert [doc_id for doc_id, _ in store.search(query, 5)] == [
        "doc-1",
        "doc_1",
        "doc_B",
        "doc_a",
        "doc_z",
    ]
    assert [d for d, _ in store.search(query, 5)] == [d for d, _ in memory.search(query, 5)]


def test_rankings_match_the_in_memory_store(
    pg_backend: PgVectorBackend, spec: IndexSpec, vectors: CountingVectors
) -> None:
    store, _ = pg_backend.open_or_create(spec, vectors)
    memory = InMemoryVectorStore.from_vectors(spec.doc_ids, vectors.vectors, spec.index_key)

    for query in HashingEmbedder().embed_queries(QUERIES):
        pg_hits, memory_hits = store.search(query, 8), memory.search(query, 8)
        assert [d for d, _ in pg_hits] == [d for d, _ in memory_hits]
        np.testing.assert_allclose(
            [s for _, s in pg_hits], [s for _, s in memory_hits], rtol=0, atol=1e-6
        )


def test_filling_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
    pg_backend: PgVectorBackend,
    pg_dsn: str,
    spec: IndexSpec,
    vectors: CountingVectors,
) -> None:
    real_insert = pgvector_store._insert_vectors

    def insert_then_fail(cursor: Any, index_key: str, doc_ids: Any, matrix: Any) -> None:
        real_insert(cursor, index_key, doc_ids[:3], matrix[:3])  # some rows are written...
        raise psycopg.errors.QueryCanceled("simulated failure during insertion")

    monkeypatch.setattr(pgvector_store, "_insert_vectors", insert_then_fail)

    with pytest.raises(VectorStoreError, match="simulated failure"):
        pg_backend.open_or_create(spec, vectors)

    assert _rows(pg_dsn, "SELECT count(*) FROM indexes") == [(0,)]
    assert _rows(pg_dsn, "SELECT count(*) FROM vectors") == [(0,)]


def test_open_missing_index_names_the_command(pg_backend: PgVectorBackend, spec: IndexSpec) -> None:
    with pytest.raises(
        VectorStoreError, match=r"no PostgreSQL index .* run `ragbench build-index`"
    ):
        pg_backend.open(spec)


@pytest.mark.parametrize(
    "change",
    [
        {"dim": 128},
        {"embedder_identity": "other-embedder"},
        {"corpus_sha256": "f" * 64},
    ],
)
def test_mismatching_index_asks_for_rebuild(
    pg_backend: PgVectorBackend,
    spec: IndexSpec,
    vectors: CountingVectors,
    change: dict[str, Any],
) -> None:
    pg_backend.open_or_create(spec, vectors)
    altered = IndexSpec(**(spec.__dict__ | change))

    with pytest.raises(VectorStoreError, match=r"does not match the run.*--rebuild"):
        pg_backend.open(altered)


def test_mismatching_count_and_schema_version_ask_for_rebuild(
    pg_backend: PgVectorBackend, pg_dsn: str, spec: IndexSpec, vectors: CountingVectors
) -> None:
    pg_backend.open_or_create(spec, vectors)
    shorter = IndexSpec(**(spec.__dict__ | {"doc_ids": spec.doc_ids[:-1]}))
    with pytest.raises(VectorStoreError, match="--rebuild"):
        pg_backend.open(shorter)

    with psycopg.connect(pg_dsn, autocommit=True) as connection:
        connection.execute("UPDATE indexes SET schema_version = %s", (SCHEMA_VERSION + 1,))
    with pytest.raises(VectorStoreError, match=f"schema v{SCHEMA_VERSION + 1}"):
        pg_backend.open(spec)


def test_inconsistent_vectors_are_rejected_before_writing(
    pg_backend: PgVectorBackend, pg_dsn: str, spec: IndexSpec
) -> None:
    with pytest.raises(VectorStoreError, match="expected vectors of shape"):
        pg_backend.open_or_create(spec, lambda: np.ones((2, 256), dtype=np.float32))
    duplicated = IndexSpec(**(spec.__dict__ | {"doc_ids": (spec.doc_ids[0],) * 8}))
    with pytest.raises(VectorStoreError, match="duplicate doc_ids"):
        pg_backend.open_or_create(duplicated, lambda: np.ones((8, 256), dtype=np.float32))

    assert _rows(pg_dsn, "SELECT count(*) FROM indexes") == [(0,)]


def test_only_btree_indexes_exist(pg_backend: PgVectorBackend, pg_dsn: str) -> None:
    methods = _rows(
        pg_dsn,
        """SELECT DISTINCT am.amname
           FROM pg_index i
           JOIN pg_class idx ON idx.oid = i.indexrelid
           JOIN pg_class tbl ON tbl.oid = i.indrelid
           JOIN pg_namespace ns ON ns.oid = tbl.relnamespace
           JOIN pg_am am ON am.oid = idx.relam
           WHERE ns.nspname = current_schema() AND tbl.relname IN ('indexes', 'vectors')""",
    )

    assert methods == [("btree",)]
    assert not {"hnsw", "ivfflat"} & {name for (name,) in methods}


def test_search_validates_arguments(
    pg_backend: PgVectorBackend, spec: IndexSpec, vectors: CountingVectors
) -> None:
    store, _ = pg_backend.open_or_create(spec, vectors)

    with pytest.raises(ValueError, match="k must be at least 1"):
        store.search(np.zeros(256), 0)
    with pytest.raises(VectorStoreError, match=r"expected \(256,\)"):
        store.search(np.zeros(3), 1)


def test_concurrent_searches_share_the_connection_safely(
    pg_backend: PgVectorBackend, spec: IndexSpec, vectors: CountingVectors
) -> None:
    store, _ = pg_backend.open_or_create(spec, vectors)
    queries = HashingEmbedder().embed_queries(QUERIES * 8)
    expected = [store.search(query, 3) for query in queries]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda query: store.search(query, 3), queries))

    assert results == expected


def test_search_after_close_is_a_vector_store_error(
    pg_dsn: str, spec: IndexSpec, vectors: CountingVectors
) -> None:
    with PgVectorBackend.connect(pg_dsn) as backend:
        store, _ = backend.open_or_create(spec, vectors, rebuild=True)

    with pytest.raises(VectorStoreError, match="PostgreSQL search failed"):
        store.search(np.ones(256, dtype=np.float32), 1)


def test_database_errors_become_vector_store_errors(
    pg_backend: PgVectorBackend, pg_dsn: str, spec: IndexSpec, vectors: CountingVectors
) -> None:
    pg_backend.close()

    with pytest.raises(VectorStoreError, match="PostgreSQL error"):
        pg_backend.open_or_create(spec, vectors)
    with pytest.raises(VectorStoreError, match="PostgreSQL error"):
        pg_backend.open(spec)


def test_unreachable_server_names_the_fix() -> None:
    dsn = "postgresql://user:hunter2@127.0.0.1:1/db"

    with pytest.raises(VectorStoreError) as excinfo:
        PgVectorBackend.connect(dsn, connect_timeout=2)

    message = str(excinfo.value)
    assert "PostgreSQL unreachable on 127.0.0.1:1" in message
    assert "docker compose up -d" in message
    assert "hunter2" not in message


def test_describe_server_handles_invalid_dsn() -> None:
    assert pgvector_store.describe_server("not a = valid = dsn") == "<invalid DSN>"
    assert pgvector_store.describe_server("postgresql://u:p@db/x") == "db:5432"
