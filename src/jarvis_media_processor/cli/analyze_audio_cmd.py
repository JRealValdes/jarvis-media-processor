"""``jmp analyze-audio`` command."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.audio import FFmpegNotFoundError, analyze_media_audio
from jarvis_media_processor.config import get_settings
from jarvis_media_processor.media import MediaFileNotFoundError

logger = logging.getLogger(__name__)


def analyze_audio_command(
    path: Annotated[
        Path,
        typer.Argument(help="Media file to analyze (base or donor)."),
    ],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help="Workspace for WAV/artifacts (default: JMP_WORK_DIR/audio-analysis).",
        ),
    ] = None,
    role: Annotated[
        str | None,
        typer.Option("--role", help="Optional role label: base or donor."),
    ] = None,
) -> None:
    """Extract analysis audio and print acoustic landmark JSON."""
    settings = get_settings()
    work = workspace or (settings.work_dir / "audio-analysis")
    try:
        artifact = analyze_media_audio(path, work, role=role, settings=settings)
    except MediaFileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FFmpegNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(artifact.model_dump_json(indent=2))
