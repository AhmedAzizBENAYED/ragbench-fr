# 0005. Embedder: e5-base, pinned, two methods, normalised, cached by composition

- **Status:** Accepted
- **Date:** 2026-10-11

## Context

Every pipeline shares one retriever (spec §6), so the embedding model is a controlled variable.
It must run on a laptop CPU and in CI (NFR-4), give identical results across runs (NFR-1) and
never be recomputed for a known text. Unit tests must not load torch or the network (NFR-6).

## Decision

- **Model: `intfloat/multilingual-e5-base`** (768 dimensions, 512 tokens), pinned at revision
  `d128750597153bb5987e10b1c3493a34e5a4502a` in `RunConfig.embedding.revision` (required).
  - `BAAI/bge-m3` is stronger but about twice the size: too slow for CPU CI.
  - `nomic-embed-text` is trained mostly on English.
  - sentence-transformers downloads only `model.safetensors`, the tokenizer and the configs:
    **1.1 GB** (the repository, with ONNX, OpenVINO and `pytorch_model.bin`, is ~5.5 GB).
  - No PIAF context exceeds 512 tokens (max 304 with the prefix): nothing is truncated, and
    "one context = one chunk" (ADR 0004) holds.
- **Two methods, not a parameter:** `embed_queries` and `embed_passages`. The e5 prefixes
  (`query: ` / `passage: `) are an implementation detail callers never see, so they cannot be
  forgotten or swapped.
- **L2-normalised float32 vectors:** cosine similarity is a dot product. Empty text raises
  `ValueError`.
- **Lazy loading:** the model loads on the first text missing from the cache; a fully cached run
  never imports torch.
- **Cache by composition:** `CachedEmbedder(inner, store)` implements the same protocol on top of
  a thread-safe SQLite key-value store (`cache.py`, reused by the LLM cache). Key:
  `sha256([identity, kind, text])`; `identity` holds model, revision and an implementation
  version that **must be bumped on any logic change** (prefixes, normalisation, pooling).
- **CPU-only torch:** an explicit `pytorch-cpu` uv index for torch on every platform, so Linux
  never resolves the CUDA build and its `nvidia-*` libraries.

## Evidence on the prefixes

Measured on the 100 `dev` questions against the full corpus (761 passages):

| Variant | top-1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| **e5 prefixes** (`query:` / `passage:`) | **0.69** | 0.90 | 0.95 | 0.779 |
| No prefixes | 0.70 | 0.88 | 0.96 | 0.785 |
| `query:` only | 0.68 | 0.90 | 0.95 | 0.774 |
| Swapped prefixes | 0.60 | 0.85 | 0.92 | 0.723 |

The hypothesis "without prefixes, retrieval is worse" is **not confirmed** on `dev` (n = 100, a
difference of one question, not significant). Swapping the prefixes, however, costs 9 points of
top-1: the model does distinguish the two roles. The prefixes are kept, as the model card
requires; the integration test asserts that correct prefixes beat swapped ones, not that they
beat no prefixes.

## Consequences

- With k = 5, recall@5 on `dev` is 0.90: about 10 % of questions lack their document in the
  context. This caps every RAG pipeline and the report must say so.
- A run needs the 1.1 GB model once (Hugging Face cache) and about 70 s of CPU to encode the
  corpus; later runs hit `embeddings.sqlite3`.
- mypy now analyses torch and transformers (33 s cold, 2 s warm): CI caches `.mypy_cache`.
