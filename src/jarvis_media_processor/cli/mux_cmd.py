"""``jmp mux`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.audio import FFmpegNotFoundError
from jarvis_media_processor.config import get_settings
from jarvis_media_processor.media import MediaFileNotFoundError
from jarvis_media_processor.output import OutputPathError, mux_mkv


def mux_command(
    base: Annotated[Path, typer.Argument(help="Base media path.")],
    donor_audio: Annotated[
        Path,
        typer.Argument(help="Reconstructed donor audio WAV (from ``jmp reconstruct``)."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output MKV path (must not overwrite inputs)."),
    ],
    donor_language: Annotated[
        str,
        typer.Option("--donor-language", help="Language tag for donor track."),
    ] = "spa",
    donor_title: Annotated[
        str,
        typer.Option("--donor-title", help="Track title for donor audio."),
    ] = "Spanish (reconstructed)",
) -> None:
    """Mux base A/V (stream copy) with reconstructed donor audio into an MKV."""
    settings = get_settings()
    try:
        out = mux_mkv(
            base,
            donor_audio,
            output,
            settings=settings,
            donor_language=donor_language,
            donor_title=donor_title,
        )
    except (MediaFileNotFoundError, OutputPathError, FFmpegNotFoundError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(str(out))
