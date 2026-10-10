"""Command-line interface of ragbench.

Commands (build-corpus, build-index, run, report, gate) are added by their own tickets.
"""

from typing import Annotated

import typer

from ragbench import __version__

app = typer.Typer(
    name="ragbench",
    help="Reproducible French benchmark of RAG architectures.",
    no_args_is_help=True,
)


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
