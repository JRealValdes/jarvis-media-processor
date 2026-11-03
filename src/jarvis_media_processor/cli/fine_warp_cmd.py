"""``jmp fine-warp`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.audio import FFmpegNotFoundError
from jarvis_media_processor.config import get_settings
from jarvis_media_processor.media import MediaFileNotFoundError
from jarvis_media_processor.reconstruct.fine_warp_pipeline import (
    fine_warp_reconstructed_audio,
)


def fine_warp_command(
    base: Annotated[Path, typer.Argument(help="Base media path (audio stream 0).")],
    reconstructed_wav: Annotated[
        Path,
        typer.Argument(help="Reconstructed donor WAV from ``jmp reconstruct``."),
    ],
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="Workspace for plan JSON and output WAV."),
    ] = None,
    output_name: Annotated[
        str,
        typer.Option(
            "--output-name",
            help="Output WAV filename inside the workspace.",
        ),
    ] = "donor_on_base_fine.wav",
) -> None:
    """Fine-warp reconstructed audio against base track 0 (onset landmarks)."""
    settings = get_settings()
    work = workspace or (settings.work_dir / "fine-warp")
    try:
        result = fine_warp_reconstructed_audio(
            base,
            reconstructed_wav,
            work,
            settings=settings,
            output_name=output_name,
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
