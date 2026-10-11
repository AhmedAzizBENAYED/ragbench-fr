# 0006. Vector store: in-memory and pgvector, exact search, behind a Retriever

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

Every pipeline retrieves from the same 761-passage corpus with the same embedder and `k`
(spec §6). Decision D1 weighed pgvector, an in-memory index and LanceDB. Results must be
deterministic (NFR-1) and identical on Windows and Linux; pipelines should not depend on vectors
or on a storage backend.

## Decision

- **D1, option A:** two `VectorStore` implementations: in memory (this ticket: tests, demo) and
  pgvector (T06: benchmark). T06 adds an equivalence test between the two backends.
- **Exact search only:** no HNSW or IVFFlat. With 761 documents a full scan is fast, and it is
  the only deterministic mode.
- **Read-only protocol:** `dim`, `index_key`, `__len__`, `search(query, k) -> [(doc_id, score)]`.
  Filling is backend-specific and happens once (`InMemoryVectorStore.from_vectors`; a persistent,
  costly table with T06); `search` may raise `VectorStoreError` (e.g. a lost connection).
- **`index_key` = hash of the embedder identity and the corpus hash.** The `Retriever` refuses a
  store filled for another embedder or corpus, which matters for a persistent SQL table.
- **Deterministic ranking:** scores are dot products of L2-normalised vectors computed in
  **float64** (float32 BLAS summation order differs between platforms, and near-ties could swap);
  ties are broken by increasing `doc_id`, i.e. sorted by `(-score, doc_id)`. pgvector must follow
  the same rule.
- **The backend is part of the run** (`retrieval.backend`, default `memory`, in the fingerprint):
  pgvector probably computes distances in float32, so it could swap near-tied ranks. If T06's
  equivalence test shows strictly identical rankings, this ADR will say so; the backend stays in
  the fingerprint regardless. Connection settings (PostgreSQL URL, possibly with a password) go
  in `Settings`, never in the run config.
- **No separate index file for the in-memory store:** passage vectors already persist in the
  embedding cache (ADR 0005); building the index is loading the corpus (its hash checked against
  the manifest), embedding passages (cache hits) and filling the store.
- **`Retriever` is the only retrieval interface pipelines see:** `retrieve(question, k)` returns
  `RetrievedDoc(doc_id, title, text, score, rank)` with ranks from 1. Run artefacts (T15) keep
  only `doc_id`, `score` and `rank`.
- `build-index` checks that the run's `data.dataset` and `data.revision` match the manifest.

## Consequences

- Retrieval is reproducible across machines: through the Retriever, dev gives exactly T04's
  counts (top-1 69/100, recall@5 90/100), with float64 scoring.
- Rebuilding the in-memory index with a warm cache takes about 0.02 s (1.4 s for the whole
  command), without loading torch.
- Adding pgvector means registering a factory in `STORE_BACKENDS` and a `VectorBackend` value;
  the protocol and the pipelines do not change.
