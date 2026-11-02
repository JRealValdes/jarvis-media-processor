"""CLI entry point for Jarvis Media Processor (``jmp``)."""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from jarvis_media_processor import __version__
from jarvis_media_processor.config import get_settings
from jarvis_media_processor.logging_config import configure_logging

app = typer.Typer(
    name="jmp",
    help=(
        "Jarvis Media Processor — graft donor audio onto a base media timeline. "
        "Base keeps video and synced audio; donor provides the audio track to add."
    ),
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def global_options(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option(
            "--log-level",
            help="Override JMP_LOG_LEVEL for this invocation.",
        ),
    ] = None,
) -> None:
    """Global options applied before any subcommand."""
    settings = get_settings()
    configure_logging(log_level or settings.log_level)
    logger.debug("jmp %s starting", __version__)


@app.command("version")
def version_cmd() -> None:
    """Print the package version."""
    typer.echo(__version__)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
