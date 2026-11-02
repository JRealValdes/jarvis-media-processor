"""``jmp match-visual`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.config import get_settings
from jarvis_media_processor.media import MediaFileNotFoundError
from jarvis_media_processor.video import FFmpegNotFoundError, match_media_visuals


def match_visual_command(
    base: Annotated[Path, typer.Argument(help="Base media path (timeline authority).")],
    donor: Annotated[Path, typer.Argument(help="Donor media path.")],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help="Workspace for frames/report (default: JMP_WORK_DIR/visual-match).",
        ),
    ] = None,
) -> None:
    """Sample frames from base and donor and print visual correspondence JSON."""
    settings = get_settings()
    work = workspace or (settings.work_dir / "visual-match")
    try:
        report = match_media_visuals(base, donor, work, settings=settings)
    except MediaFileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FFmpegNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(report.model_dump_json(indent=2))
