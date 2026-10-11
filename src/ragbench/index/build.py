"""Build the retrieval index of a run: verified corpus, cached passage embeddings, filled store.

The in-memory store keeps no separate index file: passage vectors already persist in the
embedding cache, so rebuilding is loading the corpus and hitting the cache.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ragbench.config import EmbeddingConfig, RunConfig
from ragbench.data.processed import PROCESSED_DIR, check_run_data, load_corpus, require_manifest
from ragbench.index import factory
from ragbench.index.cached import CachedEmbedder, CacheStats
from ragbench.index.retriever import Retriever
from ragbench.log import get_logger

logger = get_logger(__name__)

EmbedderFactory = Callable[[EmbeddingConfig, Path], CachedEmbedder]


@dataclass(frozen=True)
class IndexBuildResult:
    """A ready retriever and what it took to build it."""

    retriever: Retriever
    documents: int
    dim: int
    embedder_identity: str
    corpus_sha256: str
    seconds: float
    cache: CacheStats


def build_index(
    config: RunConfig,
    manifest_path: Path,
    data_dir: Path,
    cache_dir: Path,
    *,
    embedder_factory: EmbedderFactory | None = None,
) -> IndexBuildResult:
    """Check the run against the manifest, load the corpus and fill the configured store.

    Raises:
        RunDataMismatchError: the run's dataset or revision differs from the manifest.
        DataError: missing manifest, missing or stale corpus.
        EmbedderError, CacheError, VectorStoreError: embedding or store failure.
    """
    manifest = require_manifest(manifest_path)
    check_run_data(config.data.dataset, config.data.revision, manifest)
    documents = load_corpus(data_dir / PROCESSED_DIR, manifest)
    corpus_sha256: str = manifest["corpus"]["sha256"]

    make = embedder_factory or factory.make_embedder
    embedder = make(config.embedding, cache_dir)
    start = time.perf_counter()
    retriever = Retriever.build(embedder, documents, corpus_sha256, config.retrieval.backend)
    seconds = time.perf_counter() - start

    result = IndexBuildResult(
        retriever=retriever,
        documents=len(retriever),
        dim=embedder.dim,
        embedder_identity=embedder.identity,
        corpus_sha256=corpus_sha256,
        seconds=seconds,
        cache=embedder.stats,
    )
    logger.info(
        "index.built",
        backend=config.retrieval.backend.value,
        documents=result.documents,
        seconds=round(seconds, 3),
        cache_hits=result.cache.hits,
        cache_misses=result.cache.misses,
    )
    return result
