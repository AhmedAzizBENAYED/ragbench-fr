"""Tests of canonical hashing."""

from ragbench.hashing import canonical_hash, canonical_json, sha256_hex


def test_canonical_json_sorts_keys_without_whitespace_and_escapes_non_ascii() -> None:
    assert canonical_json({"b": [1, 2], "a": "é"}) == r'{"a":"\u00e9","b":[1,2]}'


def test_canonical_hash_ignores_key_order() -> None:
    assert canonical_hash({"a": 1, "b": {"c": 2, "d": 3}}) == canonical_hash(
        {"b": {"d": 3, "c": 2}, "a": 1}
    )


def test_sha256_hex_known_value() -> None:
    assert sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
