"""Command-line interface of ragbench.

Command output goes to stdout; logs and error messages go to stderr.
Commands (build-corpus, build-index, run, report, gate) are added by their own tickets.
"""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from ragbench import __version__
from ragbench.config import ConfigError, get_settings, load_run_config
from ragbench.data import DataError
from ragbench.data.build import BuildResult, build_dataset
from ragbench.data.build_config import load_corpus_build_config
from ragbench.log import configure_logging, get_logger

EXIT_INVALID_CONFIG = 1
EXIT_INVALID_ENVIRONMENT = 2
EXIT_DATA_ERROR = 3

app = typer.Typer(
    name="ragbench",
    help="Reproducible French benchmark of RAG architectures.",
    no_args_is_help=True,
)
logger = get_logger(__name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ragbench {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Reproducible French benchmark of RAG architectures."""
    try:
        settings = get_settings()
    except ValidationError as error:
        for detail in error.errors():
            field = ".".join(str(part) for part in detail["loc"])
            typer.echo(f"invalid environment setting {field}: {detail['msg']}", err=True)
        raise typer.Exit(EXIT_INVALID_ENVIRONMENT) from None
    configure_logging(settings.log_level, settings.log_format)


@app.command("validate-config")
def validate_config(
    path: Annotated[Path, typer.Argument(help="Run configuration YAML file.")],
) -> None:
    """Validate a run configuration and print its fingerprint."""
    try:
        config = load_run_config(path)
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(EXIT_INVALID_CONFIG) from None
    fingerprint = config.fingerprint()
    logger.info("config.valid", path=str(path), name=config.name, fingerprint=fingerprint)
    typer.echo(fingerprint)


def _summary(result: BuildResult) -> str:
    manifest = result.manifest
    counts = manifest["counts"]
    excluded = ", ".join(f"{reason}={n}" for reason, n in counts["excluded"].items())
    splits = ", ".join(
        f"{name} {entry['size']} (sha256 {entry['sha256'][:12]})"
        for name, entry in manifest["splits"].items()
    )
    source = manifest["source"]
    return "\n".join(
        [
            f"manifest:  {result.manifest_path} ({result.status}, sha256 {manifest['sha256']})",
            f"source:    {source['dataset']}@{source['revision']} (sha256 {source['sha256']})",
            f"questions: raw {counts['raw_questions']}, kept {counts['kept_questions']}",
            f"excluded:  {excluded}",
            f"repaired:  offsets {counts['offsets_repaired']}, "
            f"merged answers {counts['answers_merged']}",
            f"documents: {counts['documents']} (sha256 {manifest['corpus']['sha256']})",
            f"splits:    {splits}",
        ]
    )


@app.command("build-corpus")
def build_corpus(
    config_path: Annotated[Path, typer.Option("--config", help="Corpus build YAML file.")] = Path(
        "configs/data.yaml"
    ),
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite a manifest that differs from the rebuild.")
    ] = False,
) -> None:
    """Download PIAF, build the corpus and the frozen splits, and write the manifest."""
    try:
        config = load_corpus_build_config(config_path)
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(EXIT_INVALID_CONFIG) from None
    try:
        result = build_dataset(config, get_settings().data_dir, force=force)
    except DataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(EXIT_DATA_ERROR) from None
    typer.echo(_summary(result))
