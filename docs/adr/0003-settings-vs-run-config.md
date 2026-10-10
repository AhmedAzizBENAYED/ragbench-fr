# 0003. Settings versus run configuration, and the run fingerprint

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

Results must be reproducible (NFR-1) and comparable: only the architecture may vary within a run
(spec §6). If an environment variable could change a model, `k` or a temperature, two runs with
the same versioned config could silently differ. Secrets must never be committed (NFR-7).

## Decision

- **Guiding rule:** anything that can change a result belongs to `RunConfig`, a versioned YAML
  file (`configs/*.yaml`). No environment variable may change a result.
- **`Settings`** (pydantic-settings, environment and optional `.env`) only holds secrets (API
  keys as `SecretStr`, under their providers' standard names), paths and infrastructure knobs
  (Ollama URL, log level and format), prefixed `RAGBENCH_`.
- **`RunConfig`** (pydantic, `frozen=True`, `extra="forbid"`) holds the seed, the dataset pinned to
  a full commit SHA, the split, the embedding model, `k`, the generator and judge (`LLMSpec`) and
  the pipelines with their caps. A typo is an error, not an ignored key.
- **Fingerprint:** SHA-256 of the canonical JSON (sorted keys, separators `,` and `:`) of the
  whole resolved config except `name` and `description`:
  - defaults are included, so omitting `k` or writing `k: 5` gives the same fingerprint;
  - pipelines are sorted by name, since their order does not change results.
- `dump_run_config` writes the fully resolved config; it becomes the run's `config.yaml` (§9.3).

## Consequences

- Two configs with the same fingerprint describe the same experiment; the fingerprint can feed
  the run identity (T15) and cache keys.
- Changing a default in code changes the fingerprint of configs that rely on it. This is
  intended: the result may change. A golden test on `configs/example.yaml` makes it visible.
- Prompt hashes are not part of the fingerprint yet; they join the run identity with T09.
