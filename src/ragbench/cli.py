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
from ragbench.log import configure_logging, get_logger

EXIT_INVALID_CONFIG = 1
EXIT_INVALID_ENVIRONMENT = 2

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
