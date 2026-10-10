"""Frozen splits drawn by hash ranking: deterministic on every Python version and platform.

`random` is not used: only `Random.random()` is guaranteed stable across Python versions, while
`shuffle`, `choice` and `sample` may change algorithm. Every choice is instead the smallest
`sha256(f"{seed}:{purpose}:{id}")`, which also makes the result independent of input order.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ragbench.data import DataError
from ragbench.data.build_config import SplitSizes
from ragbench.data.corpus import Question
from ragbench.hashing import sha256_hex


@dataclass(frozen=True)
class Split:
    """Sorted question IDs and the sorted IDs of their documents."""

    question_ids: tuple[str, ...]
    doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class Splits:
    """`dev` and `test` share no document; each split has at most one question per document."""

    dev: Split
    test: Split
    ci: Split


def rank(seed: int, purpose: str, identifier: str) -> str:
    """Hash rank of an identifier for one purpose (`doc`, `question` or `ci`)."""
    return sha256_hex(f"{seed}:{purpose}:{identifier}")


def _split(questions: Iterable[Question]) -> Split:
    chosen = list(questions)
    return Split(
        question_ids=tuple(sorted(q.question_id for q in chosen)),
        doc_ids=tuple(sorted(q.doc_id for q in chosen)),
    )


def make_splits(questions: Iterable[Question], sizes: SplitSizes, seed: int) -> Splits:
    """Draw `test`, `dev` (disjoint documents) and `ci` (subset of `test`).

    Documents are ranked by hash: the first `sizes.test` form `test`, the next `sizes.dev` form
    `dev`. Each selected document contributes its question with the smallest hash rank. `ci` is
    the `sizes.ci` questions of `test` with the smallest `ci` rank.

    Raises:
        DataError: fewer documents with questions than `sizes.test + sizes.dev`.
    """
    by_doc: defaultdict[str, list[Question]] = defaultdict(list)
    for question in questions:
        by_doc[question.doc_id].append(question)

    needed = sizes.test + sizes.dev
    if len(by_doc) < needed:
        raise DataError(f"{len(by_doc)} documents with questions, {needed} needed (test + dev)")

    docs = sorted(by_doc, key=lambda doc_id: (rank(seed, "doc", doc_id), doc_id))

    def pick(doc_id: str) -> Question:
        return min(
            by_doc[doc_id], key=lambda q: (rank(seed, "question", q.question_id), q.question_id)
        )

    test = [pick(doc_id) for doc_id in docs[: sizes.test]]
    dev = [pick(doc_id) for doc_id in docs[sizes.test : needed]]
    ci = sorted(test, key=lambda q: (rank(seed, "ci", q.question_id), q.question_id))[: sizes.ci]
    return Splits(dev=_split(dev), test=_split(test), ci=_split(ci))
