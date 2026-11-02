"""High-quality donor audio extraction for reconstruction (not analysis)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from jarvis_media_processor.audio.extract import resolve_ffmpeg
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.media.errors import MediaFileNotFoundError

logger = logging.getLogger(__name__)


def extract_donor_wav(
    source: Path | str,
    destination: Path | str,
    *,
    settings: Settings | None = None,
    sample_rate: int = 48_000,
    channels: int = 2,
    audio_stream_index: int | None = None,
) -> Path:
    """Extract PCM WAV from donor media without modifying the source.

    Uses stream copy avoidance: always decodes to PCM for editing. Prefer
    high sample rate / channel count for quality preservation before later mux.

    Args:
        source: Donor media path.
        destination: Output WAV path.
        settings: Optional settings.
        sample_rate: Target sample rate (default 48 kHz).
        channels: Target channels (default stereo).
        audio_stream_index: Optional ``0``-based audio stream index among audio streams.

    Returns:
        Resolved destination path.
    """
    cfg = settings or get_settings()
    src = Path(source)
    if not src.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {src}")
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg(cfg.ffmpeg_path)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
    ]
    if audio_stream_index is not None:
        cmd.extend(["-map", f"0:a:{audio_stream_index}"])
    cmd.extend(
        [
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    logger.debug("Extracting donor audio: %s", " ".join(cmd))
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
        raise RuntimeError(f"FFmpeg donor extraction failed for {src}: {stderr}")
    return dst.resolve()
