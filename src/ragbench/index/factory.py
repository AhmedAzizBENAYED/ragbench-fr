"""Build the run's embedder from its configuration."""

from pathlib import Path

from ragbench.cache import SqliteStore
from ragbench.config import EmbeddingConfig
from ragbench.index.cached import CachedEmbedder
from ragbench.index.e5 import E5Embedder

EMBEDDING_CACHE_FILE = "embeddings.sqlite3"


def make_embedder(config: EmbeddingConfig, cache_dir: Path) -> CachedEmbedder:
    """e5 embedder at the pinned revision, behind the disk cache in `cache_dir`.

    Nothing is loaded until a text is missing from the cache. The caller closes it (it is a
    context manager).
    """
    store = SqliteStore(cache_dir / EMBEDDING_CACHE_FILE)
    return CachedEmbedder(E5Embedder(config.model, config.revision), store)
