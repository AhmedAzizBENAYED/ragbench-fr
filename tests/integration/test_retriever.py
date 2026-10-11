"""The Retriever reproduces T04's dev figures with the real e5 model (network, CPU).

Consistency check between tickets: T04 measured top-1 = 69/100 and recall@5 = 90/100 on the
dev split with raw embeddings; the same model behind build_index and the Retriever (float64
scores, ties by doc_id) must give exactly the same counts. Official metrics arrive with T13.
"""

import json
from pathlib import Path

from ragbench.config import load_run_config
from ragbench.data.build import build_dataset
from ragbench.data.build_config import load_corpus_build_config
from ragbench.index.build import build_index
from tests.constants import EXAMPLE_CONFIG, REPO_ROOT

T04_TOP1 = 69
T04_RECALL_AT_5 = 90


def test_retriever_reproduces_t04_dev_figures(tmp_path: Path) -> None:
    data_config = load_corpus_build_config(REPO_ROOT / "configs" / "data.yaml")
    isolated = data_config.model_copy(update={"manifest_path": tmp_path / "manifest.json"})
    built = build_dataset(isolated, tmp_path / "data")
    config = load_run_config(EXAMPLE_CONFIG)

    result = build_index(config, isolated.manifest_path, tmp_path / "data", tmp_path / "cache")

    questions = {
        record["question_id"]: record
        for record in map(
            json.loads,
            (tmp_path / "data" / "processed" / "questions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    dev = [questions[qid] for qid in built.manifest["splits"]["dev"]["question_ids"]]
    top1 = recall_at_5 = 0
    for question in dev:
        retrieved = [doc.doc_id for doc in result.retriever.retrieve(question["question"], 5)]
        top1 += retrieved[0] == question["doc_id"]
        recall_at_5 += question["doc_id"] in retrieved

    assert len(dev) == 100
    assert (top1, recall_at_5) == (T04_TOP1, T04_RECALL_AT_5)
