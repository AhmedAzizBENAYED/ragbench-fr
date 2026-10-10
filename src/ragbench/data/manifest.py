"""Data manifest: the committed fingerprint of the source, the corpus and the frozen splits.

Two identical builds produce byte-identical manifests: no timestamp, no absolute path, a fixed
key order, hashes computed on canonical in-memory serialisations, UTF-8 and LF line endings.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ragbench.data import DataError
from ragbench.data.build_config import CorpusBuildConfig
from ragbench.data.corpus import NORMALIZATION_VERSION, Corpus, Document, ExclusionReason, Question
from ragbench.data.splits import Split, Splits
from ragbench.hashing import canonical_hash

FORMAT_VERSION = 1
SPLIT_NAMES = ("dev", "test", "ci")

Manifest = dict[str, Any]


def document_record(document: Document) -> dict[str, str]:
    """JSON record of a document, as written to `corpus.jsonl`."""
    return asdict(document)


def question_record(question: Question) -> dict[str, Any]:
    """JSON record of a question, as written to `questions.jsonl`."""
    return {
        "question_id": question.question_id,
        "question": question.question,
        "answers": list(question.answers),
        "doc_id": question.doc_id,
    }


def _split_entry(split: Split) -> dict[str, Any]:
    return {
        "size": len(split.question_ids),
        "sha256": canonical_hash(list(split.question_ids)),
        "question_ids": list(split.question_ids),
        "doc_ids": list(split.doc_ids),
    }


def build_manifest(
    config: CorpusBuildConfig, raw_sha256: str, corpus: Corpus, splits: Splits
) -> Manifest:
    """Assemble the manifest; its `sha256` field hashes every other field."""
    stats = corpus.stats
    body: Manifest = {
        "format_version": FORMAT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "source": {
            "dataset": config.dataset,
            "revision": config.revision,
            "filename": config.filename,
            "sha256": raw_sha256,
        },
        "build": {"seed": config.seed, "sizes": config.sizes.model_dump()},
        "counts": {
            "raw_questions": stats.raw_questions,
            "kept_questions": stats.kept_questions,
            "excluded": {reason.value: stats.excluded[reason] for reason in ExclusionReason},
            "offsets_repaired": stats.offsets_repaired,
            "answers_merged": stats.answers_merged,
            "documents": stats.documents,
        },
        "corpus": {
            "documents": len(corpus.documents),
            "sha256": canonical_hash([document_record(doc) for doc in corpus.documents]),
        },
        "questions": {
            "count": len(corpus.questions),
            "sha256": canonical_hash([question_record(q) for q in corpus.questions]),
        },
        "splits": {name: _split_entry(getattr(splits, name)) for name in SPLIT_NAMES},
    }
    return body | {"sha256": canonical_hash(body)}


def render_manifest(manifest: Manifest) -> str:
    """Stable text form: fixed key order, two-space indent, UTF-8 characters, final newline."""
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 with LF line endings on every platform (Windows would write CRLF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def read_manifest(path: Path) -> Manifest | None:
    """Return the manifest at `path`, or None if there is none.

    Raises:
        DataError: the file exists but is not a JSON object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataError(f"{path}: cannot read manifest: {error}") from error
    if not isinstance(data, dict):
        raise DataError(f"{path}: manifest must be a JSON object")
    return data


def diff_manifests(old: Any, new: Any, prefix: str = "") -> list[str]:
    """Dotted paths of the fields that differ (lists are compared as a whole)."""
    if isinstance(old, dict) and isinstance(new, dict):
        differences: list[str] = []
        for key in [*old, *(key for key in new if key not in old)]:
            differences += diff_manifests(old.get(key), new.get(key), f"{prefix}{key}.")
        return differences
    return [] if old == new else [prefix.rstrip(".")]


def verify_manifest(manifest: Manifest, expected_sizes: dict[str, int]) -> list[str]:
    """Check a manifest's internal consistency; return the problems found (empty if none).

    Checks the self-hash, each split's hash, size and uniqueness, `ci` ⊂ `test`, one question
    per document within each split, and no document shared by `dev` and `test`.
    """
    problems: list[str] = []
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != canonical_hash(body):
        problems.append("sha256 does not match the manifest content")

    splits = manifest.get("splits", {})
    for name in SPLIT_NAMES:
        entry = splits.get(name, {})
        ids: Sequence[str] = entry.get("question_ids", [])
        doc_ids: Sequence[str] = entry.get("doc_ids", [])
        if entry.get("sha256") != canonical_hash(list(ids)):
            problems.append(f"splits.{name}.sha256 does not match its question IDs")
        if list(ids) != sorted(set(ids)):
            problems.append(f"splits.{name}.question_ids must be sorted and unique")
        if not (entry.get("size") == len(ids) == expected_sizes.get(name)):
            problems.append(f"splits.{name}.size must be {expected_sizes.get(name)}")
        if len(set(doc_ids)) != len(ids) or len(doc_ids) != len(ids):
            problems.append(f"splits.{name} must have exactly one question per document")

    def ids_of(name: str, field: str) -> set[str]:
        return set(splits.get(name, {}).get(field, []))

    if not ids_of("ci", "question_ids") <= ids_of("test", "question_ids"):
        problems.append("splits.ci must be a subset of splits.test")
    if ids_of("dev", "doc_ids") & ids_of("test", "doc_ids"):
        problems.append("splits.dev and splits.test must not share documents")
    return problems
