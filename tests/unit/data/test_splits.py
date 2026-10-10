"""Tests of the split draw by hash ranking."""

import random

import pytest

from ragbench.data import DataError
from ragbench.data.build_config import SplitSizes
from ragbench.data.corpus import Corpus, Question
from ragbench.data.splits import Splits, make_splits, rank


def _questions(n_docs: int, per_doc: int = 3) -> list[Question]:
    return [
        Question(f"q{doc:03d}-{i}", f"Question {i} ?", ("réponse",), f"doc_{doc:03d}")
        for doc in range(n_docs)
        for i in range(per_doc)
    ]


SIZES = SplitSizes(dev=10, test=30, ci=5)


def test_sizes_are_respected() -> None:
    splits = make_splits(_questions(60), SIZES, seed=1)

    assert (len(splits.dev.question_ids), len(splits.test.question_ids)) == (10, 30)
    assert len(splits.ci.question_ids) == 5


def test_dev_and_test_share_no_document() -> None:
    splits = make_splits(_questions(60), SIZES, seed=1)

    assert not set(splits.dev.doc_ids) & set(splits.test.doc_ids)


def test_ci_is_a_subset_of_test() -> None:
    splits = make_splits(_questions(60), SIZES, seed=1)

    assert set(splits.ci.question_ids) <= set(splits.test.question_ids)
    assert set(splits.ci.doc_ids) <= set(splits.test.doc_ids)


def test_at_most_one_question_per_document() -> None:
    splits = make_splits(_questions(60), SIZES, seed=1)

    for split in (splits.dev, splits.test, splits.ci):
        assert len(set(split.doc_ids)) == len(split.doc_ids) == len(split.question_ids)
        assert {qid.split("-")[0].replace("q", "doc_") for qid in split.question_ids} == set(
            split.doc_ids
        )


def test_ids_are_sorted() -> None:
    splits = make_splits(_questions(60), SIZES, seed=1)

    for split in (splits.dev, splits.test, splits.ci):
        assert list(split.question_ids) == sorted(split.question_ids)
        assert list(split.doc_ids) == sorted(split.doc_ids)


def test_same_seed_gives_same_splits() -> None:
    assert make_splits(_questions(60), SIZES, seed=1) == make_splits(_questions(60), SIZES, seed=1)


def test_shuffled_input_gives_same_splits() -> None:
    questions = _questions(60)
    shuffled = list(questions)
    random.Random(5).shuffle(shuffled)

    assert make_splits(shuffled, SIZES, seed=1) == make_splits(questions, SIZES, seed=1)


def test_different_seed_gives_different_splits() -> None:
    first = make_splits(_questions(60), SIZES, seed=1)

    assert make_splits(_questions(60), SIZES, seed=2).test != first.test


def test_draw_follows_the_documented_hash_ranking() -> None:
    questions = _questions(60)
    docs = sorted({q.doc_id for q in questions}, key=lambda d: rank(1, "doc", d))
    first_test_doc = docs[0]
    expected_question = min(
        (q for q in questions if q.doc_id == first_test_doc),
        key=lambda q: rank(1, "question", q.question_id),
    )

    splits = make_splits(questions, SIZES, seed=1)

    assert expected_question.question_id in splits.test.question_ids
    assert set(splits.dev.doc_ids) == set(docs[30:40])
    assert list(splits.ci.question_ids) == sorted(
        sorted(splits.test.question_ids, key=lambda qid: rank(1, "ci", qid))[:5]
    )


def test_rank_is_a_sha256_of_seed_purpose_and_id() -> None:
    # sha256("1:doc:doc_001"), computed independently with sha256sum.
    assert rank(1, "doc", "doc_001") == (
        "0af7475142e139cb4fb674ddf3608bb94354205fce90009b621133e18e761403"
    )
    assert rank(1, "doc", "x") != rank(2, "doc", "x") != rank(1, "ci", "x")


def test_not_enough_documents_is_an_error() -> None:
    with pytest.raises(DataError, match="39 documents with questions, 40 needed"):
        make_splits(_questions(39), SIZES, seed=1)


def test_fixture_corpus_splits(corpus: Corpus, sizes: SplitSizes) -> None:
    splits: Splits = make_splits(corpus.questions, sizes, seed=7)

    assert not set(splits.dev.doc_ids) & set(splits.test.doc_ids)
    assert set(splits.ci.question_ids) <= set(splits.test.question_ids)
    assert len(set(splits.test.doc_ids)) == 3
