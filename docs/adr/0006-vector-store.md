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

## pgvector backend (T06)

- **Image** `pgvector/pgvector:0.8.7-pg18-trixie`, pinned identically in `docker-compose.yml` and
  the CI service (a unit test compares them). Host port 5433 by default.
- **PostgreSQL 18 volume:** the official images keep their data under
  `/var/lib/postgresql/18/docker`; the named volume is mounted at `/var/lib/postgresql`, not at
  the former `/var/lib/postgresql/data`.
- **Schema** (`SCHEMA_VERSION = 1`, created with `IF NOT EXISTS`, no migration tool): a metadata
  table `indexes` (`index_key`, `identity`, `corpus_sha256`, `dim`, `count`, `schema_version`)
  and `vectors` (`index_key`, `doc_id`, `embedding vector`, primary key `(index_key, doc_id)`).
  The `vector` column has no fixed dimension so several models can coexist. The only index is
  the primary key's btree; a test checks that no `hnsw` or `ivfflat` index exists.
- **Atomic filling:** metadata and every vector are written in one transaction; an index is
  complete or absent. Opening checks key, identity, corpus, dimension, count and schema version,
  and names the command to run (`build-index`, or `build-index --rebuild`) otherwise.
- **Lazy backends:** `StoreBackend.open_or_create(spec, vectors, rebuild)` receives the passage
  vectors as a callable, so `build-index` on an existing pgvector index embeds nothing.
- **Ranking:** `ORDER BY embedding <#> query, doc_id COLLATE "C"` (score = −distance). Without
  `COLLATE "C"`, the database collation (`en_US.utf8` in the image) sorts `doc_a` before `doc_B`
  whereas Python sorts `doc_B` first: ties would be broken differently from the in-memory store.
- **Concurrency:** one connection per backend, shared by its stores behind a lock; a pool only if
  the runner (T15) measures contention. Connection errors become readable `VectorStoreError`s
  without the password.
- **Settings:** `RAGBENCH_PG_DSN` (`SecretStr`, no default) in the environment, never in the run
  config.

### Equivalence with the in-memory store

With the real e5 model on the 100 `dev` questions, the top-10 rankings of both backends are
**strictly identical** (1,000 ranks, no difference). The largest score gap is **1.8e-7**
(pgvector computes the inner product in float32, the in-memory store in float64). The backend
stays part of the run fingerprint anyway: it costs nothing and remains honest should a future
corpus or model produce a near-tie.

**Benchmark backend:** official runs use `pgvector`, as D1 intends; the in-memory store remains
the default of `configs/example.yaml` and serves tests, the demo and any context without a
database. Since rankings are identical, results do not depend on this choice today.

## Consequences

- Retrieval is reproducible across machines: through the Retriever, dev gives exactly T04's
  counts (top-1 69/100, recall@5 90/100), with float64 scoring.
- Rebuilding the in-memory index with a warm cache takes about 0.02 s (1.4 s for the whole
  command), without loading torch.
- Adding pgvector means registering a factory in `STORE_BACKENDS` and a `VectorBackend` value;
  the protocol and the pipelines do not change.
