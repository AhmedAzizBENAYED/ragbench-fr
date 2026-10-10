# 0002. Repository language

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

The project targets an international audience (GitHub, recruiters) but its subject is French
RAG evaluation. Decision D8 of the [specification](../specification.md#d8--langue-du-repo-)
asks which language each artefact uses.

## Decision

- **English:** code, identifiers, docstrings, comments, README, ADRs, `CLAUDE.md`, commit
  messages.
- **French:** data, prompts sent to the models, and the published results report.
- The specification (`docs/specification.md`) stays in French.

## Consequences

- The codebase is readable by any contributor.
- Prompts and data are tested in the language actually being evaluated.
- Files in French (prompts, data, report) must always be read and written as UTF-8 explicitly.
