"""High-level media inspection API."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.media.errors import MediaFileNotFoundError
from jarvis_media_processor.media.ffprobe import run_ffprobe_json
from jarvis_media_processor.media.models import InspectionReport, MediaInfo, media_info_from_ffprobe


def inspect_media(path: Path | str, *, settings: Settings | None = None) -> MediaInfo:
    """Inspect a single media file via FFprobe.

    Args:
        path: Path to the media file (must exist).
        settings: Optional settings override; defaults to process settings.

    Returns:
        Normalized ``MediaInfo``.

    Raises:
        MediaFileNotFoundError: If ``path`` does not exist or is not a file.
        FFprobeNotFoundError: If FFprobe is unavailable.
        FFprobeError: If probing fails.
    """
    cfg = settings or get_settings()
    media_path = Path(path)
    if not media_path.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {media_path}")

    payload = run_ffprobe_json(media_path, ffprobe_path=cfg.ffprobe_path)
    return media_info_from_ffprobe(media_path.resolve(), payload)


def inspect_media_files(
    paths: Sequence[Path | str],
    *,
    settings: Settings | None = None,
) -> InspectionReport:
    """Inspect one or more media files and build a structured report.

    Args:
        paths: Media file paths (order preserved).
        settings: Optional settings override.

    Returns:
        ``InspectionReport`` with one ``MediaInfo`` per path.

    Raises:
        MediaFileNotFoundError: If any path is missing.
        FFprobeNotFoundError: If FFprobe is unavailable.
        FFprobeError: If any probe fails.
    """
    files = [inspect_media(p, settings=settings) for p in paths]
    return InspectionReport(files=files)
