"""``jmp inspect`` command."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.config import get_settings
from jarvis_media_processor.media import (
    FFprobeError,
    FFprobeNotFoundError,
    MediaFileNotFoundError,
    inspect_media_files,
)

logger = logging.getLogger(__name__)


def inspect_command(
    paths: Annotated[
        list[Path],
        typer.Argument(
            help="One or more media files to inspect (e.g. base and donor).",
            exists=False,
            readable=False,
        ),
    ],
    pretty: Annotated[
        bool,
        typer.Option("--pretty/--compact", help="Pretty-print JSON (default: pretty)."),
    ] = True,
) -> None:
    """Inspect media files with FFprobe and print a structured JSON report."""
    if not paths:
        typer.echo("At least one media path is required.", err=True)
        raise typer.Exit(code=2)

    settings = get_settings()
    try:
        report = inspect_media_files(paths, settings=settings)
    except MediaFileNotFoundError as exc:
        logger.error("%s", exc)
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FFprobeNotFoundError as exc:
        logger.error("%s", exc)
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FFprobeError as exc:
        logger.error("%s", exc)
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if pretty:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(report.model_dump_json())
