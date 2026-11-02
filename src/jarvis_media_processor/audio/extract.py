"""Extract analysis audio from media files via FFmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.media.errors import MediaFileNotFoundError

logger = logging.getLogger(__name__)


class FFmpegNotFoundError(RuntimeError):
    """FFmpeg executable is missing or not runnable."""


def resolve_ffmpeg(ffmpeg_path: str) -> str:
    """Resolve an FFmpeg executable path or name on ``PATH``."""
    candidate = Path(ffmpeg_path)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(ffmpeg_path)
    if found:
        return found
    raise FFmpegNotFoundError(
        f"FFmpeg not found: {ffmpeg_path!r}. Install FFmpeg and ensure it is on PATH, "
        "or set JMP_FFMPEG_PATH."
    )


def build_extract_command(
    ffmpeg: str,
    source: Path,
    destination: Path,
    *,
    sample_rate: int,
    channels: int,
) -> list[str]:
    """Build an FFmpeg command that extracts PCM WAV for analysis.

    Args:
        ffmpeg: FFmpeg executable.
        source: Input media path.
        destination: Output WAV path.
        sample_rate: Target sample rate.
        channels: Target channel count (typically 1 for analysis).

    Returns:
        Argument list for ``subprocess``.
    """
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]


def extract_analysis_wav(
    source: Path | str,
    destination: Path | str,
    *,
    settings: Settings | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> Path:
    """Extract mono/stereo PCM WAV from ``source`` into ``destination``.

    Does not modify the original media file.

    Args:
        source: Input media path.
        destination: Output WAV path (parent dirs created).
        settings: Optional settings override.
        sample_rate: Override ``JMP_AUDIO_ANALYSIS_SAMPLE_RATE``.
        channels: Override ``JMP_AUDIO_ANALYSIS_CHANNELS``.

    Returns:
        Resolved destination path.

    Raises:
        MediaFileNotFoundError: If ``source`` is missing.
        FFmpegNotFoundError: If FFmpeg is unavailable.
        RuntimeError: If FFmpeg fails.
    """
    cfg = settings or get_settings()
    src = Path(source)
    if not src.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {src}")

    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)

    rate = sample_rate if sample_rate is not None else cfg.audio_analysis_sample_rate
    ch = channels if channels is not None else cfg.audio_analysis_channels
    ffmpeg = resolve_ffmpeg(cfg.ffmpeg_path)
    cmd = build_extract_command(ffmpeg, src, dst, sample_rate=rate, channels=ch)
    logger.debug("Extracting analysis audio: %s", " ".join(cmd))

    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"FFmpeg audio extraction failed for {src}: {stderr}")

    return dst.resolve()
