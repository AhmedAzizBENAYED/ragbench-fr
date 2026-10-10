# 0001. Python tooling

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

The project must be reproducible (NFR-1), strictly typed and linted with a blocking CI (NFR-6),
and developed on Windows while CI runs on Linux. Decision D7 of the
[specification](../specification.md#d7--outillage-) lists the tooling.

## Decision

- **uv** manages Python, dependencies and the lockfile (`uv.lock`, committed). A minimum uv
  version is set in `pyproject.toml` (`[tool.uv] required-version`); CI pins the exact version.
- **Python 3.12** only (`requires-python = ">=3.12,<3.13"`, `.python-version`).
- **hatchling** as build backend, `src/` layout, distribution `ragbench-fr`, package `ragbench`.
- **ruff** for lint and formatting, **mypy --strict** on `src/` and `tests/`.
- **pytest** with **pytest-cov** (branch coverage, >= 80 %, provider adapters excluded) and
  **pytest-socket** to forbid network access in unit tests.
- **pre-commit** runs ruff and mypy from the project environment (`uv run`) so hook versions
  match `uv.lock`, plus gitleaks against leaked secrets (NFR-7).
- **typer** for the CLI.
- A **Makefile** defines the checks once; CI calls the same targets.
- Libraries listed in D7 for later needs (pydantic, structlog, matplotlib…) are added by the
  ticket that first uses them.

## Consequences

- One command (`make check`) gives the same verdict locally and in CI.
- Contributors need uv and GNU make (on Windows: `winget install ezwinports.make`).
- Every dependency is introduced and justified by the ticket that needs it.
