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

## License

[MIT](LICENSE)
