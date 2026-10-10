"""Tests of the SQLite key-value cache."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ragbench.cache import CacheError, SqliteStore


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "cache" / "test.sqlite3")


def test_put_then_get(store: SqliteStore) -> None:
    store.put_many({"a": b"1", "b": b"2"})

    assert store.get_many(["a", "b", "missing"]) == {"a": b"1", "b": b"2"}
    assert len(store) == 2


def test_empty_inputs(store: SqliteStore) -> None:
    store.put_many({})

    assert store.get_many([]) == {}
    assert len(store) == 0


def test_put_replaces_existing_key(store: SqliteStore) -> None:
    store.put_many([("a", b"1")])
    store.put_many([("a", b"2")])

    assert store.get_many(["a"]) == {"a": b"2"}


def test_more_keys_than_one_statement_allows(store: SqliteStore) -> None:
    items = {f"k{i}": i.to_bytes(4, "little") for i in range(1_234)}
    store.put_many(items)

    assert store.get_many([*items, "k0"]) == items


def test_values_persist_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "persist.sqlite3"
    with SqliteStore(path) as first:
        first.put_many({"a": b"\x00\xff"})

    with SqliteStore(path) as second:
        assert second.get_many(["a"]) == {"a": b"\x00\xff"}


def test_database_runs_in_wal_mode(store: SqliteStore) -> None:
    with sqlite3.connect(store.path) as connection:
        [(mode,)] = connection.execute("PRAGMA journal_mode")

    assert mode == "wal"


def test_concurrent_writes_and_reads_from_several_threads(store: SqliteStore) -> None:
    def work(worker: int) -> dict[str, bytes]:
        items = {f"w{worker}-{i}": f"{worker}:{i}".encode() for i in range(200)}
        for start in range(0, 200, 20):
            store.put_many(dict(list(items.items())[start : start + 20]))
            store.get_many(list(items)[:start])
        return items

    with ThreadPoolExecutor(max_workers=8) as pool:
        written = list(pool.map(work, range(8)))

    expected = {key: value for items in written for key, value in items.items()}
    assert store.get_many(list(expected)) == expected
    assert len(store) == 1_600
    store.close()


def test_close_then_reuse_reopens(store: SqliteStore) -> None:
    store.put_many({"a": b"1"})
    store.close()

    assert store.get_many(["a"]) == {"a": b"1"}


def test_unopenable_database_is_a_cache_error(tmp_path: Path) -> None:
    directory = tmp_path / "is_a_directory"
    directory.mkdir()

    with pytest.raises(CacheError, match="cannot"):
        SqliteStore(directory)


def test_corrupted_database_is_a_cache_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupted.sqlite3"
    path.write_bytes(b"this is not a database" * 100)

    with pytest.raises(CacheError, match="cannot"):
        SqliteStore(path)


def test_read_and_write_errors_are_cache_errors(store: SqliteStore) -> None:
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE entries")

    with pytest.raises(CacheError, match="cannot read cache"):
        store.get_many(["a"])
    with pytest.raises(CacheError, match="cannot write cache"):
        store.put_many({"a": b"1"})
    with pytest.raises(CacheError, match="cannot read cache"):
        len(store)
