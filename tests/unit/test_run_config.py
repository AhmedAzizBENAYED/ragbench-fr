"""Tests of the run configuration: validation, loading, dumping and fingerprint."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from ragbench.config import (
    ConfigError,
    CragPipeline,
    PipelineName,
    Provider,
    RunConfig,
    Split,
    dump_run_config,
    load_run_config,
)
from tests.constants import E5_REVISION, EXAMPLE_CONFIG, EXAMPLE_FINGERPRINT, PIAF_REVISION


def _minimal() -> dict[str, Any]:
    return {
        "name": "minimal",
        "seed": 1,
        "data": {"revision": PIAF_REVISION, "split": "ci"},
        "embedding": {"revision": E5_REVISION},
        "generator": {"provider": "ollama", "model": "some-model", "max_tokens": 256},
        "judge": {"provider": "gemini", "model": "other-model", "max_tokens": 512},
        "pipelines": [{"name": "naive-rag"}, {"name": "crag"}],
    }


def _write(data: Any, path: Path = Path("config.yaml")) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _config_error(data: Any) -> ConfigError:
    with pytest.raises(ConfigError) as excinfo:
        load_run_config(_write(data))
    return excinfo.value


# --- Loading and defaults ------------------------------------------------------------------------


def test_example_config_is_valid() -> None:
    config = load_run_config(EXAMPLE_CONFIG)

    assert config.data.split is Split.DEV
    assert config.generator.provider is Provider.GROQ
    assert [p.name for p in config.pipelines] == list(PipelineName)


def test_defaults_follow_the_specification() -> None:
    config = RunConfig.model_validate(_minimal())

    assert config.data.dataset == "AgentPublic/piaf"
    assert config.embedding.model == "intfloat/multilingual-e5-base"
    assert config.retrieval.k == 5
    assert config.generator.temperature == 0.0
    assert config.description == ""
    assert config.pipelines[1] == CragPipeline(name=PipelineName.CRAG, max_reformulations=1)


def test_default_caps_match_section_7() -> None:
    data = _minimal()
    data["pipelines"] = [{"name": name} for name in ["adaptive-rag", "crag", "self-rag"]]

    adaptive, crag, self_rag = RunConfig.model_validate(data).model_dump()["pipelines"]

    assert adaptive["max_llm_calls"] == 3
    assert crag["max_reformulations"] == 1
    assert self_rag["max_corrections"] == 2


def test_config_is_immutable() -> None:
    config = RunConfig.model_validate(_minimal())

    with pytest.raises(ValidationError):
        config.seed = 2  # type: ignore[misc]


# --- Validation errors ---------------------------------------------------------------------------


def test_unknown_top_level_field_is_rejected() -> None:
    data = _minimal() | {"temperatur": 0.5}

    error = _config_error(data)

    assert str(error) == "config.yaml: temperatur: Extra inputs are not permitted"


def test_unknown_nested_field_is_rejected() -> None:
    data = _minimal() | {"retrieval": {"kk": 3}}

    assert "config.yaml: retrieval.kk: Extra inputs are not permitted" in str(_config_error(data))


def test_k_zero_is_rejected() -> None:
    data = _minimal() | {"retrieval": {"k": 0}}

    assert str(_config_error(data)) == (
        "config.yaml: retrieval.k: Input should be greater than or equal to 1"
    )


@pytest.mark.parametrize(
    "revision", ["main", PIAF_REVISION[:7], PIAF_REVISION.upper(), PIAF_REVISION + "0", "refs/pr/3"]
)
def test_revision_must_be_a_full_commit_sha(revision: str) -> None:
    data = _minimal()
    data["data"]["revision"] = revision

    message = str(_config_error(data))

    assert message.startswith("config.yaml: data.revision: ")
    assert "commit SHA, not a branch or tag name" in message


@pytest.mark.parametrize(
    ("section", "field", "value", "loc"),
    [
        ("generator", "temperature", 2.5, "generator.temperature"),
        ("generator", "temperature", -0.1, "generator.temperature"),
        ("judge", "max_tokens", 0, "judge.max_tokens"),
        ("generator", "provider", "openai", "generator.provider"),
        ("data", "split", "train", "data.split"),
    ],
)
def test_out_of_range_values_are_rejected(section: str, field: str, value: Any, loc: str) -> None:
    data = _minimal()
    data[section][field] = value

    assert f"config.yaml: {loc}: " in str(_config_error(data))


def test_embedding_revision_is_required_and_must_be_a_commit_sha() -> None:
    missing = _minimal() | {"embedding": {"model": "intfloat/multilingual-e5-base"}}
    branch = _minimal() | {"embedding": {"revision": "main"}}

    assert "config.yaml: embedding.revision: Field required" in str(_config_error(missing))
    assert "embedding.revision: Value error, must be a full" in str(_config_error(branch))


def test_negative_cap_is_rejected() -> None:
    data = _minimal() | {"pipelines": [{"name": "self-rag", "max_corrections": -1}]}

    assert "pipelines[0].self-rag.max_corrections: Input should be greater than or equal to 0" in (
        str(_config_error(data))
    )


def test_cap_of_another_pipeline_is_rejected() -> None:
    data = _minimal() | {"pipelines": [{"name": "closed-book", "max_corrections": 2}]}

    assert "pipelines[0].closed-book.max_corrections: Extra inputs are not permitted" in str(
        _config_error(data)
    )


def test_unknown_pipeline_is_rejected() -> None:
    data = _minimal() | {"pipelines": [{"name": "graph-rag"}]}

    message = str(_config_error(data))

    assert message.startswith("config.yaml: pipelines[0]: ")
    assert "graph-rag" in message


def test_duplicate_pipelines_are_rejected() -> None:
    data = _minimal() | {"pipelines": [{"name": "crag"}, {"name": "naive-rag"}, {"name": "crag"}]}

    assert "config.yaml: pipelines: Value error, duplicate pipelines: crag" in str(
        _config_error(data)
    )


def test_empty_pipeline_list_is_rejected() -> None:
    data = _minimal() | {"pipelines": []}

    assert "config.yaml: pipelines: " in str(_config_error(data))


def test_every_error_is_reported_on_its_own_line() -> None:
    data = _minimal()
    del data["seed"]
    data["retrieval"] = {"k": 0}

    error = _config_error(data)

    assert error.errors == (
        "seed: Field required",
        "retrieval.k: Input should be greater than or equal to 1",
    )
    assert str(error).splitlines() == [f"config.yaml: {line}" for line in error.errors]


def test_message_contains_no_raw_pydantic_trace() -> None:
    message = str(_config_error(_minimal() | {"retrieval": {"k": 0}}))

    assert "pydantic" not in message
    assert "validation error" not in message
    assert "input_value" not in message


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match=r"missing\.yaml: file not found"):
        load_run_config("missing.yaml")


def test_invalid_yaml() -> None:
    path = Path("broken.yaml")
    path.write_text("name: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"broken\.yaml: invalid YAML"):
        load_run_config(path)


def test_top_level_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_run_config(_write(["a", "list"]))


def test_file_must_be_utf8() -> None:
    path = Path("latin1.yaml")
    path.write_bytes("description: réponse\n".encode("latin-1"))

    with pytest.raises(ConfigError, match=r"latin1\.yaml: cannot read file"):
        load_run_config(path)


def test_utf8_content_is_read_correctly() -> None:
    data = _minimal() | {"description": "Évaluation des réponses générées"}

    config = load_run_config(_write(data))

    assert config.description == "Évaluation des réponses générées"


# --- Fingerprint ---------------------------------------------------------------------------------


def test_example_fingerprint_golden() -> None:
    assert load_run_config(EXAMPLE_CONFIG).fingerprint() == EXAMPLE_FINGERPRINT


def test_fingerprint_is_a_stable_sha256() -> None:
    first = RunConfig.model_validate(_minimal()).fingerprint()

    assert first == RunConfig.model_validate(_minimal()).fingerprint()
    assert len(first) == 64
    assert int(first, 16) >= 0


def test_fingerprint_ignores_key_order() -> None:
    data = _minimal()
    reordered = {key: data[key] for key in reversed(data)}
    reordered["generator"] = dict(reversed(data["generator"].items()))

    assert (
        load_run_config(_write(reordered, Path("b.yaml"))).fingerprint()
        == load_run_config(_write(data, Path("a.yaml"))).fingerprint()
    )


def test_fingerprint_ignores_name_and_description() -> None:
    base = RunConfig.model_validate(_minimal())
    renamed = RunConfig.model_validate(_minimal() | {"name": "other", "description": "Autre."})

    assert renamed.fingerprint() == base.fingerprint()


def test_fingerprint_ignores_pipeline_order() -> None:
    data = _minimal()
    swapped = data | {"pipelines": list(reversed(data["pipelines"]))}

    assert (
        RunConfig.model_validate(swapped).fingerprint()
        == RunConfig.model_validate(data).fingerprint()
    )


def test_omitted_default_and_explicit_default_have_same_fingerprint() -> None:
    explicit = _minimal() | {"retrieval": {"k": 5}}

    assert (
        RunConfig.model_validate(explicit).fingerprint()
        == RunConfig.model_validate(_minimal()).fingerprint()
    )


def test_fingerprint_changes_with_k() -> None:
    changed = _minimal() | {"retrieval": {"k": 6}}

    assert (
        RunConfig.model_validate(changed).fingerprint()
        != RunConfig.model_validate(_minimal()).fingerprint()
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("seed",), 2),
        (("data", "revision"), "0" * 40),
        (("embedding", "revision"), "1" * 40),
        (("data", "split"), "test"),
        (("generator", "temperature"), 0.7),
        (("judge", "model"), "another-judge"),
        (("pipelines",), [{"name": "naive-rag"}, {"name": "crag", "max_reformulations": 0}]),
    ],
)
def test_fingerprint_changes_with_any_result_affecting_field(
    path: tuple[str, ...], value: Any
) -> None:
    data = _minimal()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    assert (
        RunConfig.model_validate(data).fingerprint()
        != RunConfig.model_validate(_minimal()).fingerprint()
    )


# --- Dump ----------------------------------------------------------------------------------------


def test_dump_then_load_round_trip_loses_nothing() -> None:
    original = load_run_config(EXAMPLE_CONFIG)

    dump_run_config(original, "resolved.yaml")
    reloaded = load_run_config("resolved.yaml")

    assert reloaded == original
    assert reloaded.fingerprint() == original.fingerprint()
    assert reloaded.name == original.name
    assert reloaded.description == original.description


def test_dump_writes_resolved_defaults() -> None:
    dump_run_config(RunConfig.model_validate(_minimal()), "resolved.yaml")

    dumped = yaml.safe_load(Path("resolved.yaml").read_text(encoding="utf-8"))

    assert dumped["retrieval"] == {"k": 5}
    assert dumped["embedding"] == {
        "model": "intfloat/multilingual-e5-base",
        "revision": E5_REVISION,
    }
    assert dumped["pipelines"][1] == {"name": "crag", "max_reformulations": 1}


def test_dump_is_canonical_utf8_with_lf() -> None:
    config = RunConfig.model_validate(_minimal() | {"description": "Réponses générées"})

    dump_run_config(config, "first.yaml")
    dump_run_config(load_run_config("first.yaml"), "second.yaml")

    raw = Path("first.yaml").read_bytes()
    assert raw == Path("second.yaml").read_bytes()
    assert b"\r\n" not in raw
    assert "Réponses générées".encode() in raw
    assert list(yaml.safe_load(raw.decode("utf-8"))) == list(RunConfig.model_fields)
