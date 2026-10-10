"""Persistent key-value cache on SQLite, shared by the embedding cache and the LLM cache (T08).

One table, `entries(key TEXT PRIMARY KEY, value BLOB)`. Safe to use from several threads: each
thread gets its own connection, and the database runs in WAL mode so readers do not block the
writer. Writes are atomic (one transaction per `put_many`).
"""

import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

# Stay well below SQLite's limit on bound parameters per statement.
_CHUNK_SIZE = 500
_BUSY_TIMEOUT_SECONDS = 30.0


class CacheError(Exception):
    """The cache database cannot be opened, read or written."""


class SqliteStore:
    """Thread-safe persistent mapping from string keys to bytes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connection()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
                )
        except sqlite3.Error as error:
            raise CacheError(f"{path}: cannot initialise cache: {error}") from error

    def _connection(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = getattr(self._local, "connection", None)
        if connection is None:
            try:
                # One connection per thread; check_same_thread=False only so that close() can
                # close every connection from the calling thread.
                connection = sqlite3.connect(
                    self.path, timeout=_BUSY_TIMEOUT_SECONDS, check_same_thread=False
                )
                connection.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.Error as error:
                raise CacheError(f"{self.path}: cannot open cache: {error}") from error
            self._local.connection = connection
            with self._lock:
                self._connections.append(connection)
        return connection

    def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        """Return the stored values of the keys that are present."""
        connection = self._connection()
        found: dict[str, bytes] = {}
        unique = list(dict.fromkeys(keys))
        try:
            for start in range(0, len(unique), _CHUNK_SIZE):
                chunk = unique[start : start + _CHUNK_SIZE]
                placeholders = ",".join("?" * len(chunk))
                rows = connection.execute(
                    f"SELECT key, value FROM entries WHERE key IN ({placeholders})", chunk
                )
                found.update((key, bytes(value)) for key, value in rows)
        except sqlite3.Error as error:
            raise CacheError(f"{self.path}: cannot read cache: {error}") from error
        return found

    def put_many(self, items: Mapping[str, bytes] | Iterable[tuple[str, bytes]]) -> None:
        """Store values in one transaction (existing keys are replaced)."""
        pairs = list(items.items() if isinstance(items, Mapping) else items)
        if not pairs:
            return
        connection = self._connection()
        try:
            with connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO entries (key, value) VALUES (?, ?)", pairs
                )
        except sqlite3.Error as error:
            raise CacheError(f"{self.path}: cannot write cache: {error}") from error

    def __len__(self) -> int:
        try:
            [(count,)] = self._connection().execute("SELECT COUNT(*) FROM entries")
        except sqlite3.Error as error:
            raise CacheError(f"{self.path}: cannot read cache: {error}") from error
        return int(count)

    def close(self) -> None:
        """Close the connections of every thread."""
        with self._lock:
            for connection in self._connections:
                connection.close()
            self._connections.clear()
        self._local = threading.local()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
