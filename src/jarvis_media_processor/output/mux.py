"""MKV mux: base video/audio (stream copy) + reconstructed donor audio."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from jarvis_media_processor.audio.extract import resolve_ffmpeg
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.media.errors import MediaFileNotFoundError
from jarvis_media_processor.media.inspect import inspect_media

logger = logging.getLogger(__name__)


class OutputPathError(ValueError):
    """Output path is unsafe (missing parent, clobbers inputs, etc.)."""


def validate_output_path(output: Path, *inputs: Path) -> Path:
    """Ensure ``output`` does not overwrite inputs; create parent dirs."""
    out = Path(output).resolve()
    for inp in inputs:
        if Path(inp).resolve() == out:
            raise OutputPathError(f"Refusing to overwrite input file: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def build_mux_mkv_command(
    ffmpeg: str,
    base: Path,
    donor_audio: Path,
    output: Path,
    *,
    donor_language: str,
    donor_title: str,
    donor_audio_codec: str,
    base_audio_count: int = 1,
) -> list[str]:
    """Build FFmpeg mux args.

    Copies base video and base audio streams; encodes/appends donor audio as
    output audio index ``base_audio_count`` with language/title metadata.
    """
    donor_a_index = max(0, base_audio_count)
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(base),
        "-i",
        str(donor_audio),
        "-map",
        "0:v?",
        "-map",
        "0:a?",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        f"-c:a:{donor_a_index}",
        donor_audio_codec,
        f"-metadata:s:a:{donor_a_index}",
        f"language={donor_language}",
        f"-metadata:s:a:{donor_a_index}",
        f"title={donor_title}",
        str(output),
    ]


def mux_mkv(
    base_path: Path | str,
    donor_audio_path: Path | str,
    output_path: Path | str,
    *,
    settings: Settings | None = None,
    donor_language: str = "spa",
    donor_title: str = "Spanish (reconstructed)",
    donor_audio_codec: str = "flac",
) -> Path:
    """Create an MKV with base streams copied and donor audio appended.

    Args:
        base_path: Base media (video + existing audio). Never modified.
        donor_audio_path: Reconstructed donor audio (WAV from #8).
        output_path: Destination MKV (must not equal any input).
        settings: Optional settings.
        donor_language: Language tag for the new track (e.g. ``spa``).
        donor_title: Track title metadata.
        donor_audio_codec: Encoder for donor track (``flac`` lossless default).

    Returns:
        Resolved output path.
    """
    cfg = settings or get_settings()
    base = Path(base_path)
    donor_audio = Path(donor_audio_path)
    if not base.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {base}")
    if not donor_audio.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {donor_audio}")

    out = validate_output_path(Path(output_path), base, donor_audio)
    ffmpeg = resolve_ffmpeg(cfg.ffmpeg_path)

    info = inspect_media(base, settings=cfg)
    base_audio_count = len(info.audio_streams)
    if base_audio_count == 0:
        raise RuntimeError(f"Base media has no audio streams: {base}")

    cmd = build_mux_mkv_command(
        ffmpeg,
        base,
        donor_audio,
        out,
        donor_language=donor_language,
        donor_title=donor_title,
        donor_audio_codec=donor_audio_codec,
        base_audio_count=base_audio_count,
    )
    logger.debug("Muxing MKV: %s", " ".join(cmd))
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
        raise RuntimeError(f"FFmpeg mux failed: {stderr}")
    return out
