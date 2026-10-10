"""Tests of corpus construction: cleaning, normalisation, deduplication and IDs."""

import random
import unicodedata
from dataclasses import replace

import pytest

from ragbench.data import DataError
from ragbench.data.corpus import (
    Corpus,
    ExclusionReason,
    Question,
    build_corpus,
    dedup_key,
    locate_answer,
    make_doc_id,
    normalize_text,
)
from ragbench.data.piaf import RawRecord

EIFFEL = "La Tour Eiffel a été construite par Gustave Eiffel pour l'Exposition universelle de 1889."


def _question(corpus: Corpus, question_id: str) -> Question:
    [question] = [q for q in corpus.questions if q.question_id == question_id]
    return question


def test_counts(corpus: Corpus) -> None:
    stats = corpus.stats

    assert stats.raw_questions == 16
    assert stats.kept_questions == len(corpus.questions) == 9
    assert stats.documents == len(corpus.documents) == 8
    assert stats.offsets_repaired == 1
    assert stats.answers_merged == 1
    assert dict(stats.excluded) == {
        ExclusionReason.EMPTY_QUESTION: 1,
        ExclusionReason.EMPTY_ANSWER: 1,
        ExclusionReason.ANSWER_NOT_FOUND: 2,
        ExclusionReason.ANSWER_OFFSET_MISMATCH: 1,
        ExclusionReason.DUPLICATE_QUESTION: 2,
    }
    assert stats.raw_questions == stats.kept_questions + sum(stats.excluded.values())


def test_kept_question_ids(corpus: Corpus) -> None:
    assert [q.question_id for q in corpus.questions] == [
        "p001", "p002", "p003", "p004", "p007", "p010", "p012", "p014", "p015",
    ]  # fmt: skip


def test_duplicated_context_is_one_document(corpus: Corpus) -> None:
    eiffel_id = make_doc_id(EIFFEL)

    assert _question(corpus, "p003").doc_id == _question(corpus, "p001").doc_id == eiffel_id
    [document] = [doc for doc in corpus.documents if doc.doc_id == eiffel_id]
    assert document.text == EIFFEL  # first occurrence by question ID, stored in NFC
    assert document.title == "Tour Eiffel"


def test_context_without_kept_question_is_still_a_document(corpus: Corpus) -> None:
    louvre = make_doc_id("Le Louvre est un musée situé à Paris.")

    assert louvre in {doc.doc_id for doc in corpus.documents}
    assert louvre not in {q.doc_id for q in corpus.questions}


def test_offset_off_by_one_is_repaired(corpus: Corpus) -> None:
    assert _question(corpus, "p002").answers == ("l'Exposition universelle de 1889",)


def test_duplicate_questions_keep_smallest_id_and_merge_answers(corpus: Corpus) -> None:
    question = _question(corpus, "p004")

    assert question.question == "Où le Rhône prend-il sa source ?"
    assert question.answers == ("en Suisse", "Suisse")


def test_documents_are_sorted_with_prefixed_ids(corpus: Corpus) -> None:
    doc_ids = [doc.doc_id for doc in corpus.documents]

    assert doc_ids == sorted(doc_ids)
    assert all(doc_id.startswith("doc_") and len(doc_id) == 20 for doc_id in doc_ids)


def test_result_does_not_depend_on_input_order(piaf_records: list[RawRecord]) -> None:
    shuffled = list(piaf_records)
    random.Random(3).shuffle(shuffled)

    assert build_corpus(shuffled) == build_corpus(piaf_records)


def test_doc_id_is_stable_and_content_derived() -> None:
    # Golden value: changing it changes every doc_id, hence the manifest and the ground truth.
    assert make_doc_id(EIFFEL) == "doc_6148af11715eabe2"
    assert make_doc_id(unicodedata.normalize("NFD", EIFFEL) + "  ") == make_doc_id(EIFFEL)
    assert make_doc_id(EIFFEL.lower()) != make_doc_id(EIFFEL)


def test_normalisation_keeps_case_and_accents() -> None:
    nfd = unicodedata.normalize("NFD", "  Été  à   Paris ")

    assert normalize_text(nfd) == "Été  à   Paris"
    assert dedup_key(nfd) == "Été à Paris"


@pytest.mark.parametrize(
    ("text", "start", "status"),
    [
        ("Eiffel", 8, "exact"),
        ("Eiffel", 9, "repaired"),
        ("Eiffel", 7, "repaired"),
        ("Eiffel", 0, "elsewhere"),
        ("Lyon", 0, "absent"),
        ("La", 1, "repaired"),
    ],
)
def test_locate_answer(text: str, start: int, status: str) -> None:
    assert locate_answer(EIFFEL, text, start) == status


def test_duplicate_question_ids_are_rejected(piaf_records: list[RawRecord]) -> None:
    with pytest.raises(DataError, match="duplicate question IDs: p001"):
        build_corpus([*piaf_records, piaf_records[0]])


def test_same_context_with_two_titles_is_rejected(piaf_records: list[RawRecord]) -> None:
    conflicting = replace(piaf_records[0], id="p999", title="Autre titre")

    with pytest.raises(DataError, match="two titles"):
        build_corpus([*piaf_records, conflicting])


def test_question_with_one_valid_answer_among_several_is_kept() -> None:
    record = RawRecord(
        "p1", "T", EIFFEL, "Qui ?", ("Lyon", "Gustave Eiffel", "Gustave Eiffel"), (0, 36, 36)
    )

    [question] = build_corpus([record]).questions

    assert question.answers == ("Gustave Eiffel",)
