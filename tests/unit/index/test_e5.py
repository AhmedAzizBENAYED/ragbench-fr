"""Tests of the e5 embedder with a fake backend (no torch, no network)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ragbench.index.e5 import E5Embedder, EncoderBackend, count_truncated
from ragbench.index.embedder import EmbedderError
from ragbench.log import configure_logging
from tests.constants import E5_REVISION

MODEL = "intfloat/multilingual-e5-base"


class FakeBackend:
    """Records encoded texts; returns unnormalised vectors derived from text length."""

    def __init__(self, dim: int = 4, max_length: int = 512) -> None:
        self._dim = dim
        self._max_length = max_length
        self.encoded: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_length(self) -> int:
        return self._max_length

    def encode(self, texts: list[str], batch_size: int) -> NDArray[np.floating]:
        self.encoded.append(texts)
        return np.array([[len(t), 1.0, 2.0, 3.0][: self._dim] for t in texts], dtype=np.float64)

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]


class Loader:
    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model: str, revision: str) -> EncoderBackend:
        self.calls.append((model, revision))
        return self.backend


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def loader(backend: FakeBackend) -> Loader:
    return Loader(backend)


@pytest.fixture
def embedder(loader: Loader) -> E5Embedder:
    return E5Embedder(MODEL, E5_REVISION, dim=4, loader=loader)


def test_queries_get_query_prefix(embedder: E5Embedder, backend: FakeBackend) -> None:
    embedder.embed_queries(["Qui a construit la tour ?"])

    assert backend.encoded == [["query: Qui a construit la tour ?"]]


def test_passages_get_passage_prefix(embedder: E5Embedder, backend: FakeBackend) -> None:
    embedder.embed_passages(["La tour Eiffel.", "Le Rhône."])

    assert backend.encoded == [["passage: La tour Eiffel.", "passage: Le Rhône."]]


def test_vectors_are_l2_normalised_float32(embedder: E5Embedder) -> None:
    vectors = embedder.embed_passages(["a", "bb bb"])

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("texts", [[""], ["ok", "   "]])
def test_empty_text_is_rejected_before_loading(
    embedder: E5Embedder, loader: Loader, texts: list[str]
) -> None:
    with pytest.raises(ValueError, match="empty text"):
        embedder.embed_queries(texts)
    assert loader.calls == []


def test_empty_list_returns_no_vectors_without_loading(
    embedder: E5Embedder, loader: Loader
) -> None:
    assert embedder.embed_passages([]).shape == (0, 4)
    assert loader.calls == []


def test_model_is_loaded_lazily_and_once(embedder: E5Embedder, loader: Loader) -> None:
    assert embedder.identity == f"e5:{MODEL}@{E5_REVISION}:v1"
    assert embedder.dim == 4
    assert loader.calls == []

    embedder.embed_queries(["a"])
    embedder.embed_passages(["b"])

    assert loader.calls == [(MODEL, E5_REVISION)]


def test_dimension_mismatch_is_an_error(loader: Loader) -> None:
    embedder = E5Embedder(MODEL, E5_REVISION, dim=768, loader=loader)

    with pytest.raises(EmbedderError, match="dimension 4, expected 768"):
        embedder.embed_queries(["a"])


def test_unexpected_vector_shape_is_an_error(embedder: E5Embedder, backend: FakeBackend) -> None:
    backend.encode = lambda texts, batch_size: np.ones((len(texts) + 1, 4))  # type: ignore[method-assign]

    with pytest.raises(EmbedderError, match="expected vectors of shape"):
        embedder.embed_passages(["a"])


def test_zero_vector_is_an_error(embedder: E5Embedder, backend: FakeBackend) -> None:
    backend.encode = lambda texts, batch_size: np.zeros((len(texts), 4))  # type: ignore[method-assign]

    with pytest.raises(EmbedderError, match="zero vector"):
        embedder.embed_passages(["a"])


def test_count_truncated() -> None:
    assert count_truncated([10, 512, 513, 600], max_length=512) == 2
    assert count_truncated([], max_length=512) == 0


def test_truncation_is_logged_as_warning(
    loader: Loader, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("WARNING", "json")
    embedder = E5Embedder(MODEL, E5_REVISION, dim=4, loader=Loader(FakeBackend(max_length=3)))

    embedder.embed_passages(["un deux", "un deux trois quatre", "un"])

    [line] = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert line["event"] == "embedder.truncated"
    assert line["level"] == "warning"
    assert (line["truncated"], line["total"], line["max_length"]) == (1, 3, 3)
    assert line["kind"] == "passage"


def test_no_truncation_no_warning(embedder: E5Embedder, capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", "json")

    embedder.embed_passages(["court"])

    assert capsys.readouterr().err == ""


_NO_TORCH_SCRIPT = """
import sys
from pathlib import Path
import numpy as np
from ragbench.config import EmbeddingConfig
from ragbench.index.e5 import E5Embedder
from ragbench.index.factory import make_embedder

class Backend:
    dim = 2
    max_length = 512
    def encode(self, texts, batch_size):
        return np.ones((len(texts), 2))
    def token_lengths(self, texts):
        return [1] * len(texts)

E5Embedder("m", "r", dim=2, loader=lambda m, r: Backend()).embed_queries(["a"])
make_embedder(EmbeddingConfig(revision="{revision}"), Path("cache"))
heavy = sorted(m for m in ("torch", "sentence_transformers", "transformers") if m in sys.modules)
assert not heavy, heavy
"""


def test_embedder_never_imports_torch_unless_the_model_is_loaded(tmp_path: Path) -> None:
    # In a fresh interpreter: the pytest process's sys.modules depends on test order.
    subprocess.run(
        [sys.executable, "-c", _NO_TORCH_SCRIPT.replace("{revision}", E5_REVISION)],
        check=True,
        cwd=tmp_path,
        timeout=120,
    )
