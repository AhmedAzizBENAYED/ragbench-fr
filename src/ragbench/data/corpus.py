"""Corpus construction: answer validation, normalisation, deduplication and stable IDs.

One PIAF context is one document and one chunk (spec §5.2). Answer offsets are checked on the
raw text, before any normalisation. Lower-casing and accent removal are deliberately absent:
they belong to the EM metric (T13), not to the stored data.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from ragbench.data import DataError
from ragbench.data.piaf import RawRecord
from ragbench.hashing import sha256_hex

# Bump whenever normalisation, deduplication or cleaning rules change: it changes the manifest.
NORMALIZATION_VERSION = 1

# PIAF has answers whose `answer_start` is off by one character; they are kept and counted.
OFFSET_TOLERANCE = 1

DOC_ID_PREFIX = "doc_"
_DOC_ID_HEX_LENGTH = 16
_WHITESPACE = re.compile(r"\s+")


class ExclusionReason(StrEnum):
    """Why a raw question is left out of the dataset."""

    EMPTY_QUESTION = "empty_question"
    EMPTY_ANSWER = "empty_answer"
    ANSWER_NOT_FOUND = "answer_not_found"
    ANSWER_OFFSET_MISMATCH = "answer_offset_mismatch"
    DUPLICATE_QUESTION = "duplicate_question"


@dataclass(frozen=True)
class Document:
    """A retrievable passage."""

    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Question:
    """A test question; `doc_id` is the retrieval ground truth (its original context)."""

    question_id: str
    question: str
    answers: tuple[str, ...]
    doc_id: str


@dataclass(frozen=True)
class CorpusStats:
    """Counts reported in the manifest."""

    raw_questions: int
    kept_questions: int
    excluded: Mapping[ExclusionReason, int]
    offsets_repaired: int
    answers_merged: int
    documents: int


@dataclass(frozen=True)
class Corpus:
    """Documents sorted by `doc_id`, questions sorted by `question_id`."""

    documents: tuple[Document, ...]
    questions: tuple[Question, ...]
    stats: CorpusStats


def normalize_text(text: str) -> str:
    """Stored form of every text: Unicode NFC, without leading or trailing whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def dedup_key(text: str) -> str:
    """Deduplication key: NFC with whitespace runs collapsed (case and accents are kept)."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def make_doc_id(text: str) -> str:
    """Content-derived, stable document ID."""
    return DOC_ID_PREFIX + sha256_hex(dedup_key(text))[:_DOC_ID_HEX_LENGTH]


_AnswerStatus = Literal["exact", "repaired", "elsewhere", "absent"]


def locate_answer(context: str, text: str, start: int) -> _AnswerStatus:
    """Check an extractive answer against the raw context."""
    if context[start : start + len(text)] == text:
        return "exact"
    for delta in range(1, OFFSET_TOLERANCE + 1):
        for candidate in (start - delta, start + delta):
            if candidate >= 0 and context[candidate : candidate + len(text)] == text:
                return "repaired"
    return "elsewhere" if text in context else "absent"


@dataclass
class _Accumulator:
    documents: dict[str, Document]
    questions: dict[tuple[str, str], Question]
    excluded: dict[ExclusionReason, int]
    offsets_repaired: int = 0
    answers_merged: int = 0


def _register_document(acc: _Accumulator, record: RawRecord) -> str:
    doc_id = make_doc_id(record.context)
    title = normalize_text(record.title)
    existing = acc.documents.get(doc_id)
    if existing is None:
        acc.documents[doc_id] = Document(doc_id, title, normalize_text(record.context))
    elif existing.title != title:
        raise DataError(
            f"context {doc_id} has two titles: {existing.title!r} and {title!r} "
            f"(question {record.id})"
        )
    return doc_id


def _valid_answers(acc: _Accumulator, record: RawRecord) -> list[str] | ExclusionReason:
    valid: list[str] = []
    non_empty = misplaced = False
    for text, start in zip(record.answer_texts, record.answer_starts, strict=True):
        if not text.strip():
            continue
        non_empty = True
        status = locate_answer(record.context, text, start)
        if status in ("exact", "repaired"):
            acc.offsets_repaired += status == "repaired"
            answer = normalize_text(text)
            if answer not in valid:
                valid.append(answer)
        else:
            misplaced = misplaced or status == "elsewhere"
    if valid:
        return valid
    if not non_empty:
        return ExclusionReason.EMPTY_ANSWER
    return ExclusionReason.ANSWER_OFFSET_MISMATCH if misplaced else ExclusionReason.ANSWER_NOT_FOUND


def _add_question(acc: _Accumulator, record: RawRecord, doc_id: str, answers: list[str]) -> None:
    question = normalize_text(record.question)
    group = (dedup_key(question), doc_id)
    kept = acc.questions.get(group)
    if kept is None:
        acc.questions[group] = Question(record.id, question, tuple(answers), doc_id)
        return
    # Same question on the same document: keep the smallest ID, merge the reference answers.
    acc.excluded[ExclusionReason.DUPLICATE_QUESTION] += 1
    new_answers = [answer for answer in answers if answer not in kept.answers]
    acc.answers_merged += len(new_answers)
    acc.questions[group] = Question(
        kept.question_id, kept.question, kept.answers + tuple(new_answers), doc_id
    )


def build_corpus(records: Iterable[RawRecord]) -> Corpus:
    """Build the deduplicated corpus and the clean question set.

    The result does not depend on the order of `records`. Every unique context becomes a
    document, including contexts whose questions are all excluded.

    Raises:
        DataError: duplicate question IDs, or one context with two different titles.
    """
    ordered = sorted(records, key=lambda record: record.id)
    counts = Counter(record.id for record in ordered)
    duplicates = sorted(qid for qid, count in counts.items() if count > 1)
    if duplicates:
        raise DataError(f"duplicate question IDs: {', '.join(duplicates)}")

    acc = _Accumulator({}, {}, dict.fromkeys(ExclusionReason, 0))
    for record in ordered:
        doc_id = _register_document(acc, record)
        if not normalize_text(record.question):
            acc.excluded[ExclusionReason.EMPTY_QUESTION] += 1
            continue
        answers = _valid_answers(acc, record)
        if isinstance(answers, ExclusionReason):
            acc.excluded[answers] += 1
            continue
        _add_question(acc, record, doc_id, answers)

    documents = tuple(sorted(acc.documents.values(), key=lambda doc: doc.doc_id))
    questions = tuple(sorted(acc.questions.values(), key=lambda q: q.question_id))
    stats = CorpusStats(
        raw_questions=len(ordered),
        kept_questions=len(questions),
        excluded=dict(acc.excluded),
        offsets_repaired=acc.offsets_repaired,
        answers_merged=acc.answers_merged,
        documents=len(documents),
    )
    return Corpus(documents, questions, stats)
