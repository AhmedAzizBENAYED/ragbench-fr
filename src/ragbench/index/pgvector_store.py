"""pgvector backend: persistent, exact search in PostgreSQL (ADR 0006).

No approximate index (HNSW, IVFFlat): only the btree of the primary key. Scores are inner
products of normalised vectors (`<#>` returns the negated inner product), ordered by distance then
`doc_id COLLATE "C"`: the database collation (e.g. en_US.utf8) would otherwise sort doc_ids
differently from Python and break the tie rule shared with the in-memory store.
"""

import threading
from collections import Counter
from collections.abc import Sequence
from types import TracebackType
from typing import Self

import numpy as np
import psycopg
from numpy.typing import NDArray
from pgvector.psycopg import register_vector
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import TupleRow

from ragbench.index.backends import IndexSpec, IndexStatus, VectorsProvider
from ragbench.index.embedder import Vectors
from ragbench.index.store import Hit, VectorStoreError
from ragbench.log import get_logger

logger = get_logger(__name__)

# Bump when the tables change; an index written with another version must be rebuilt.
SCHEMA_VERSION = 1

_SCHEMA = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS indexes (
        index_key TEXT PRIMARY KEY,
        identity TEXT NOT NULL,
        corpus_sha256 TEXT NOT NULL,
        dim INTEGER NOT NULL,
        count INTEGER NOT NULL,
        schema_version INTEGER NOT NULL
    )""",
    # `vector` without a dimension, so several embedding models can coexist.
    """CREATE TABLE IF NOT EXISTS vectors (
        index_key TEXT NOT NULL REFERENCES indexes (index_key) ON DELETE CASCADE,
        doc_id TEXT NOT NULL,
        embedding vector NOT NULL,
        PRIMARY KEY (index_key, doc_id)
    )""",
)

_SEARCH = """
    SELECT doc_id, embedding <#> %s AS distance
    FROM vectors
    WHERE index_key = %s
    ORDER BY distance, doc_id COLLATE "C"
    LIMIT %s
"""


def describe_server(dsn: str) -> str:
    """`host:port` of a DSN, for error messages (never the password)."""
    try:
        params = conninfo_to_dict(dsn)
    except psycopg.ProgrammingError:
        return "<invalid DSN>"
    return f"{params.get('host', 'localhost')}:{params.get('port', '5432')}"


def _insert_vectors(
    cursor: psycopg.Cursor[TupleRow],
    index_key: str,
    doc_ids: Sequence[str],
    vectors: Vectors,
) -> None:
    cursor.executemany(
        "INSERT INTO vectors (index_key, doc_id, embedding) VALUES (%s, %s, %s)",
        [(index_key, doc_id, vector) for doc_id, vector in zip(doc_ids, vectors, strict=True)],
    )


class PgVectorStore:
    """A filled index in PostgreSQL. Searches share the backend's connection under its lock."""

    def __init__(
        self,
        connection: psycopg.Connection[TupleRow],
        lock: threading.Lock,
        server: str,
        index_key: str,
        dim: int,
        count: int,
    ) -> None:
        self._connection = connection
        self._lock = lock
        self._server = server
        self._index_key = index_key
        self._dim = dim
        self._count = count

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def index_key(self) -> str:
        return self._index_key

    def __len__(self) -> int:
        return self._count

    def search(self, query: NDArray[np.floating], k: int) -> list[Hit]:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (self._dim,):
            raise VectorStoreError(
                f"query vector has shape {vector.shape}, expected ({self._dim},)"
            )
        try:
            with self._lock:
                rows = self._connection.execute(_SEARCH, (vector, self._index_key, k)).fetchall()
        except psycopg.Error as error:
            raise VectorStoreError(
                f"PostgreSQL search failed on {self._server}: {error}"
            ) from error
        return [(str(doc_id), -float(distance)) for doc_id, distance in rows]


class PgVectorBackend:
    """Persistent `StoreBackend`: one connection, protected by a lock (shared with its stores)."""

    def __init__(self, connection: psycopg.Connection[TupleRow], server: str) -> None:
        self._connection = connection
        self._server = server
        self._lock = threading.Lock()

    @classmethod
    def connect(cls, dsn: str, *, connect_timeout: int = 5) -> Self:
        """Connect, create the schema if needed and register the vector type.

        Raises:
            VectorStoreError: the server is unreachable or rejects the connection.
        """
        server = describe_server(dsn)
        try:
            connection = psycopg.connect(dsn, autocommit=True, connect_timeout=connect_timeout)
        except psycopg.Error as error:
            reason = str(error).strip().splitlines()[-1] if str(error).strip() else type(error)
            raise VectorStoreError(
                f"PostgreSQL unreachable on {server}: start it with `docker compose up -d` "
                f"(`make db-up`) and check RAGBENCH_PG_DSN ({reason})"
            ) from error
        try:
            for statement in _SCHEMA:
                connection.execute(statement)
            register_vector(connection)
        except psycopg.Error as error:
            connection.close()
            raise VectorStoreError(
                f"cannot prepare the pgvector schema on {server}: {error}"
            ) from error
        return cls(connection, server)

    @property
    def persistent(self) -> bool:
        return True

    def close(self) -> None:
        """Close the connection (stores opened from this backend stop working)."""
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _store(self, spec: IndexSpec) -> PgVectorStore:
        return PgVectorStore(
            self._connection, self._lock, self._server, spec.index_key, spec.dim, len(spec.doc_ids)
        )

    def _metadata(self, index_key: str) -> TupleRow | None:
        return self._connection.execute(
            "SELECT identity, corpus_sha256, dim, count, schema_version FROM indexes "
            "WHERE index_key = %s",
            (index_key,),
        ).fetchone()

    def _check(self, spec: IndexSpec, metadata: TupleRow) -> None:
        """Raise unless the stored index is exactly the one `spec` describes."""
        expected = (spec.embedder_identity, spec.corpus_sha256, spec.dim, len(spec.doc_ids))
        stored = tuple(metadata[:4])
        schema_version = metadata[4]
        row = self._connection.execute(
            "SELECT count(*) FROM vectors WHERE index_key = %s", (spec.index_key,)
        ).fetchone()
        vector_count = row[0] if row is not None else 0
        if (
            stored != expected
            or schema_version != SCHEMA_VERSION
            or vector_count != len(spec.doc_ids)
        ):
            raise VectorStoreError(
                f"PostgreSQL index {spec.index_key} on {self._server} does not match the run "
                f"(stored identity, corpus, dim, count: {stored}, {vector_count} vectors, schema "
                f"v{schema_version}; expected {expected}, schema v{SCHEMA_VERSION}): "
                "run `ragbench build-index --rebuild`"
            )

    def open(self, spec: IndexSpec) -> PgVectorStore:
        """Open an existing index without writing anything.

        Raises:
            VectorStoreError: the index is missing or does not match `spec`, or a database error.
        """
        try:
            with self._lock:
                metadata = self._metadata(spec.index_key)
                if metadata is None:
                    raise VectorStoreError(
                        f"no PostgreSQL index {spec.index_key} on {self._server}: "
                        "run `ragbench build-index`"
                    )
                self._check(spec, metadata)
        except psycopg.Error as error:
            raise VectorStoreError(f"PostgreSQL error on {self._server}: {error}") from error
        return self._store(spec)

    def open_or_create(
        self, spec: IndexSpec, vectors: VectorsProvider, *, rebuild: bool = False
    ) -> tuple[PgVectorStore, IndexStatus]:
        try:
            with self._lock:
                metadata = self._metadata(spec.index_key)
            if metadata is not None and not rebuild:
                return self.open(spec), IndexStatus.EXISTING
            self._fill(spec, vectors(), replace=metadata is not None)
        except psycopg.Error as error:
            raise VectorStoreError(f"PostgreSQL error on {self._server}: {error}") from error
        status = IndexStatus.REBUILT if metadata is not None else IndexStatus.CREATED
        logger.info("pgvector.filled", index_key=spec.index_key, status=status.value)
        return self._store(spec), status

    def _fill(self, spec: IndexSpec, vectors: Vectors, *, replace: bool) -> None:
        """Write metadata and every vector in one transaction: all or nothing."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(spec.doc_ids), spec.dim):
            raise VectorStoreError(
                f"expected vectors of shape {(len(spec.doc_ids), spec.dim)}, got {matrix.shape}"
            )
        duplicates = sorted(d for d, count in Counter(spec.doc_ids).items() if count > 1)
        if duplicates:
            raise VectorStoreError(f"duplicate doc_ids: {', '.join(duplicates[:5])}")
        with self._lock, self._connection.transaction(), self._connection.cursor() as cursor:
            if replace:
                cursor.execute("DELETE FROM indexes WHERE index_key = %s", (spec.index_key,))
            cursor.execute(
                "INSERT INTO indexes (index_key, identity, corpus_sha256, dim, count, "
                "schema_version) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    spec.index_key,
                    spec.embedder_identity,
                    spec.corpus_sha256,
                    spec.dim,
                    len(spec.doc_ids),
                    SCHEMA_VERSION,
                ),
            )
            _insert_vectors(cursor, spec.index_key, spec.doc_ids, matrix)
