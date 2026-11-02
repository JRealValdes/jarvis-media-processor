"""``jmp reconstruct`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.alignment.models import AlignmentDocument
from jarvis_media_processor.audio import FFmpegNotFoundError
from jarvis_media_processor.config import get_settings
from jarvis_media_processor.media import MediaFileNotFoundError
from jarvis_media_processor.reconstruct import reconstruct_donor_audio


def reconstruct_command(
    donor: Annotated[Path, typer.Argument(help="Donor media path.")],
    alignment_json: Annotated[
        Path,
        typer.Argument(help="Alignment JSON from ``jmp align``."),
    ],
    base_duration: Annotated[
        float,
        typer.Option("--base-duration", help="Base timeline duration in seconds."),
    ],
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="Workspace for WAV/JSON artifacts."),
    ] = None,
) -> None:
    """Reconstruct donor audio onto the base timeline as PCM WAV."""
    settings = get_settings()
    work = workspace or (settings.work_dir / "reconstruct")
    try:
        alignment = AlignmentDocument.model_validate_json(
            alignment_json.read_text(encoding="utf-8")
        )
        result = reconstruct_donor_audio(
            donor,
            alignment,
            work,
            base_duration=base_duration,
            settings=settings,
        )
    except MediaFileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FFmpegNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.model_dump_json(indent=2))
