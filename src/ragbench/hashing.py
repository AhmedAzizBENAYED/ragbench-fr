"""Canonical hashing, identical on every platform.

Hashes are computed on an in-memory canonical serialisation (JSON with sorted keys, fixed
separators, ASCII escapes, UTF-8), never on bytes read from disk.
"""

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialise a JSON-compatible value canonically: sorted keys, no whitespace, ASCII only."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(text: str) -> str:
    """SHA-256 hex digest of a string encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    """SHA-256 hex digest of the canonical JSON of a value."""
    return sha256_hex(canonical_json(value))
