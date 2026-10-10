# CLAUDE.md

Reproducible French benchmark of RAG architectures. Package `ragbench` (`src/` layout),
distribution `ragbench-fr`.

## Source of truth

`docs/specification.md` (in French) is the source of truth. Any deviation from it must update
the spec **in the same ticket** and be listed in the final report.

## Commands

```bash
uv sync                          # install (Python 3.12 is fetched by uv if needed)
make check                       # lint + format check + mypy --strict + tests with coverage
make lint | format | typecheck | test
uv run pytest -m integration     # integration tests (excluded by default)
uv run pre-commit run --all-files
```

## Environment

- Dev machine: Windows + Git Bash. CI: Linux (ubuntu-latest). Code must be cross-platform.
- Use `pathlib`, never hard-coded separators.
- Always pass `encoding="utf-8"` explicitly to `open()`, `read_text()`, `write_text()`: the
  Windows default (cp1252) breaks on French data.

## Language

Code, docstrings, comments, README, ADRs, commit messages: English. Data, prompts and the results
report: French.

## Architecture rules

- Anything that can change a result goes in `RunConfig` (versioned YAML); `Settings` (env) only
  holds secrets, paths and infrastructure knobs (ADR 0003).
- No business module calls a provider SDK directly: everything goes through `LLMClient`.
- Prompts live in `prompts/`, never hard-coded in Python.
- No network in unit tests (enforced by pytest-socket in `tests/unit/conftest.py`); use
  deterministic fake `LLMClient` and `Embedder`.
- Tests run in a temporary working directory: read repository files via `tests.constants.REPO_ROOT`,
  never relative paths. Test modules never import from other test modules.
- No secrets in code or history; configuration via environment / `.env` (git-ignored).
- Structured logs (JSON), no `print`.
- Explicit error handling: never `except Exception: pass`.

## Definition of Done (spec §13)

1. Code follows the spec interfaces (or the spec is updated in the same ticket).
2. `ruff`, `mypy --strict` and `pytest` pass; module coverage >= 80 %.
3. No network calls in unit tests.
4. Errors handled explicitly, logs structured.
5. Docs up to date (README if visible behaviour changes, ADR if a decision is made).
6. Atomic, conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`); CI green.

## Ticket workflow

1. Read the ticket and spec §15.
2. Propose a short plan and **wait for validation**.
3. Implement on branch `tNN-<slug>` with atomic conventional commits; each commit must pass the
   checks (tests ship with the code they test).
4. Run `make check` and `uv run pre-commit run --all-files`.
5. Write the final report. Never push.

## Dependencies

Add only what the current ticket needs, and justify every new dependency in the report.

## Final report (in French)

1. Summary in 3 lines.
2. Files created or modified.
3. Summarised output of `make check` and pre-commit.
4. Deviations from the spec or the ticket, and why.
5. Dependencies added and justification.
6. Open questions and points to verify.
7. Commands to push the branch and open the PR.
