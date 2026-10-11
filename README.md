# ragbench-fr

A reproducible benchmark comparing RAG architectures (closed-book, naive RAG, Adaptive RAG, CRAG,
Self-RAG) on French data, with the same data, retriever and generator for every pipeline.

> **Status:** work in progress. See the [specification](docs/specification.md) (in French).

## Installation

Requires [uv](https://docs.astral.sh/uv/) (>= 0.13) and GNU make. uv installs Python 3.12 if needed.

```bash
uv sync
uv run pre-commit install
uv run ragbench --version
```

Copy `.env.example` to `.env` for API keys, paths and logging options (never experiment
parameters). Check a run configuration and print its fingerprint with
`uv run ragbench validate-config configs/example.yaml`.

## Data

The benchmark uses [PIAF](https://huggingface.co/datasets/AgentPublic/piaf), French extractive
question answering on Wikipedia ([piaf.etalab.studio](https://piaf.etalab.studio); Keraron et al.,
*Project PIAF: Building a Native French Question-Answering Dataset*, LREC 2020), released under
the MIT licence. Build the corpus and
the frozen splits (downloads the pinned revision; checks `data/manifest.json` is unchanged):

```bash
uv run ragbench build-corpus
uv run ragbench build-index    # embeds the corpus (e5, cached) and builds the vector store
```

### PostgreSQL backend (pgvector)

Set `retrieval.backend: pgvector` in the run config and `RAGBENCH_PG_DSN` in `.env` (see
`.env.example`), then:

```bash
make db-up                 # docker compose up -d --wait (pgvector, host port 5433)
uv run ragbench build-index --config <your-config.yaml>   # fills the index once
uv run ragbench build-index --config <your-config.yaml>   # "existing": nothing to do
```

`--rebuild` drops and recreates the index of the current embedder and corpus only.

## Development

| Command          | What it does                                      |
|------------------|---------------------------------------------------|
| `make install`   | Install dependencies (`uv sync`)                  |
| `make lint`      | Lint with ruff                                    |
| `make format`    | Format with ruff                                  |
| `make typecheck` | Type-check with `mypy --strict`                   |
| `make test`      | Run unit tests with coverage (>= 80 %)            |
| `make check`     | All of the above checks (format in check mode)    |

Integration tests are excluded by default: `uv run pytest -m integration`.
Database tests (`db` marker) need PostgreSQL: run `make db-up` first, otherwise they are skipped.

## License

[MIT](LICENSE)
