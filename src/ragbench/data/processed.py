"""Read the processed data back, refusing anything that no longer matches the manifest.

A stale `corpus.jsonl` (rebuilt with other rules, edited by hand, truncated) must fail loudly,
never produce silently wrong retrieval results.
"""

import json
from pathlib import Path

from ragbench.data import DataError
from ragbench.data.corpus import Document
from ragbench.data.manifest import Manifest, document_record, read_manifest
from ragbench.hashing import canonical_hash

PROCESSED_DIR = "processed"
CORPUS_FILE = "corpus.jsonl"

_DOCUMENT_FIELDS = {"doc_id", "title", "text"}


def require_manifest(path: Path) -> Manifest:
    """Read the manifest, which must exist.

    Raises:
        DataError: no manifest at `path`, or it is not a JSON object.
    """
    manifest = read_manifest(path)
    if manifest is None:
        raise DataError(f"{path}: no data manifest; run `ragbench build-corpus` first")
    return manifest


def _document(path: Path, number: int, line: str) -> Document:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise DataError(f"{path}:{number}: invalid JSON: {error}") from error
    if not isinstance(record, dict) or set(record) != _DOCUMENT_FIELDS:
        raise DataError(f"{path}:{number}: expected an object with {sorted(_DOCUMENT_FIELDS)}")
    if not all(isinstance(value, str) for value in record.values()):
        raise DataError(f"{path}:{number}: doc_id, title and text must be strings")
    return Document(doc_id=record["doc_id"], title=record["title"], text=record["text"])


def load_corpus(processed_dir: Path, manifest: Manifest) -> list[Document]:
    """Load `corpus.jsonl` and check its hash against the manifest.

    Raises:
        DataError: the file is missing or malformed, or its content differs from the manifest.
    """
    path = processed_dir / CORPUS_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise DataError(f"{path}: corpus not found; run `ragbench build-corpus` first") from None
    except (OSError, UnicodeDecodeError) as error:
        raise DataError(f"{path}: cannot read corpus: {error}") from error

    documents = [_document(path, number, line) for number, line in enumerate(lines, 1) if line]
    actual = canonical_hash([document_record(document) for document in documents])
    expected = manifest["corpus"]["sha256"]
    if actual != expected:
        raise DataError(
            f"{path}: corpus does not match the manifest (sha256 {actual}, expected {expected}); "
            "it is stale or modified: rerun `ragbench build-corpus`"
        )
    return documents
