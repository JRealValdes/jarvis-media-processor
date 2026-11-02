"""``jmp align`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.alignment import detect_alignment
from jarvis_media_processor.audio import FFmpegNotFoundError
from jarvis_media_processor.config import get_settings
from jarvis_media_processor.media import MediaFileNotFoundError


def align_command(
    base: Annotated[Path, typer.Argument(help="Base media path (timeline authority).")],
    donor: Annotated[Path, typer.Argument(help="Donor media path (audio to add).")],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help="Workspace for artifacts (default: JMP_WORK_DIR/alignment).",
        ),
    ] = None,
    no_video: Annotated[
        bool,
        typer.Option("--no-video", help="Skip visual matching evidence."),
    ] = False,
) -> None:
    """Detect escalating temporal alignment between base and donor."""
    settings = get_settings()
    work = workspace or (settings.work_dir / "alignment")
    try:
        doc = detect_alignment(
            base,
            donor,
            work,
            settings=settings,
            use_video=not no_video,
        )
    except MediaFileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FFmpegNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(doc.model_dump_json(indent=2))
