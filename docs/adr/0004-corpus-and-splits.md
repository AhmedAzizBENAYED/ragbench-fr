# 0004. Corpus, document IDs and frozen splits

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

The benchmark needs a corpus to retrieve from, a retrieval ground truth, and question splits
that never change between runs (FR-1, NFR-1, spec §5). PIAF, pinned at commit `bda8c06`, ships a
single parquet file: 3,835 questions on 761 distinct Wikipedia paragraphs, about five questions
per paragraph. Questions on the same paragraph are correlated, which would make bootstrap
confidence intervals too narrow. Python's `random.shuffle`, `choice` and `sample` are not
guaranteed to give the same result across Python versions.

## Decision

- **One context = one document = one chunk.** PIAF paragraphs are already short; chunking would
  add an uncontrolled variable. The corpus holds **every** unique context, including those whose
  questions are not in any split: retrieval runs on the full corpus.
- **Content-derived IDs.** Contexts are deduplicated on NFC with collapsed whitespace (case and
  accents kept: those transformations belong to the EM metric). `doc_id` is `doc_` followed by
  the first 16 hex characters of the SHA-256 of that key, so it is stable across builds and
  input orders. `question_id` is PIAF's own ID, checked for uniqueness. The ground truth of a
  question is the `doc_id` of its original context.
- **Cleaning.** Answer offsets are checked on the raw text before normalisation. An offset off
  by one character (33 cases in PIAF) is repaired and counted; empty questions, missing answers
  and duplicates (same normalised question on the same document) are excluded and counted. A
  duplicate keeps the smallest `question_id` and its group's answers are merged (stable order),
  since different annotators' answers are all valid references.
- **Document-level splits by hash ranking.** With `h(x) = sha256(f"{seed}:{x}")`:
  - documents are sorted by `h("doc:" + doc_id)`; the first 300 form `test`, the next 100 `dev`,
    so `dev` and `test` share no document;
  - each selected document contributes one question: the smallest `h("question:" + id)`;
  - `ci` is the 30 questions of `test` with the smallest `h("ci:" + id)`, so `ci ⊂ test`.
- **Committed manifest.** `data/manifest.json` records the source (dataset, revision, raw file
  SHA-256), build parameters, normalisation version, counts, corpus and question hashes, each
  split's sorted IDs and hash, and its own hash. Its location comes from `configs/data.yaml`,
  never from the environment (ADR 0003). Hashes use canonical in-memory JSON, and files are
  written in UTF-8 with LF, so a rebuild is byte-identical on Windows and Linux.
  `build-corpus` refuses to change an existing manifest unless `--force` is given.

## Consequences

- Retrieval metrics are deterministic and need no LLM.
- The draw is identical on any Python version and independent of input order; changing the
  seed, sizes or cleaning rules visibly changes the manifest (bump `NORMALIZATION_VERSION` when
  rules change).
- Only 400 of 761 documents carry a split question; the others act as retrieval distractors.
- A real rebuild needs network access, so it runs as an integration test; unit tests only check
  the committed manifest's internal consistency.
