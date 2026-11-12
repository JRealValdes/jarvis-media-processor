"""``jmp optimize`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jarvis_media_processor.audio import FFmpegNotFoundError
from jarvis_media_processor.config import get_settings
from jarvis_media_processor.media import MediaFileNotFoundError
from jarvis_media_processor.output import (
    OptimizeProfile,
    OptimizeSpecError,
    OutputPathError,
    optimize_mkv,
)


def optimize_command(
    source: Annotated[Path, typer.Argument(help="Input media path (never modified).")],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output MKV path (must not overwrite input)."),
    ],
    profile: Annotated[
        OptimizeProfile,
        typer.Option(
            "--profile",
            "-p",
            help=(
                "Quality/size profile: archive (HEVC CRF~18, default), "
                "balanced (HEVC CRF~21), audio-slim (video copy + audio re-encode), "
                "remux (stream copy + drop)."
            ),
        ),
    ] = OptimizeProfile.ARCHIVE,
    keep_audio: Annotated[
        str | None,
        typer.Option(
            "--keep-audio",
            help=(
                "Comma-separated audio language tags to keep (e.g. spa,eng). "
                "Omit to keep all audio streams."
            ),
        ),
    ] = None,
    keep_subs: Annotated[
        str | None,
        typer.Option(
            "--keep-subs",
            help=(
                "Comma-separated subtitle language tags to keep (e.g. spa,eng). "
                "Omit to keep all subtitle streams."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the FFmpeg command and exit without writing output.",
        ),
    ] = False,
) -> None:
    """Remux or re-encode a multi-track MKV to reduce size while keeping soft tracks."""
    settings = get_settings()
    try:
        result = optimize_mkv(
            source,
            output,
            profile=profile,
            keep_audio=keep_audio,
            keep_subs=keep_subs,
            settings=settings,
            dry_run=dry_run,
        )
    except (
        MediaFileNotFoundError,
        OutputPathError,
        OptimizeSpecError,
        FFmpegNotFoundError,
        ValueError,
        RuntimeError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if dry_run:
        assert isinstance(result, list)
        typer.echo(" ".join(result))
        return

    typer.echo(str(result))
