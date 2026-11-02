"""Extract representative frames from media via FFmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.media.errors import MediaFileNotFoundError
from jarvis_media_processor.video.hashing import compute_phash
from jarvis_media_processor.video.models import SampledFrame

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


def build_frame_extract_command(
    ffmpeg: str,
    source: Path,
    pattern: Path,
    *,
    fps: float,
) -> list[str]:
    """Build FFmpeg args to sample frames at ``fps`` into an image pattern."""
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps={fps}",
        str(pattern),
    ]


def extract_sampled_frames(
    source: Path | str,
    output_dir: Path | str,
    *,
    role: str | None = None,
    settings: Settings | None = None,
    fps: float | None = None,
) -> list[SampledFrame]:
    """Sample frames from ``source`` into ``output_dir`` and compute pHashes.

    Does not modify the original media file.

    Args:
        source: Input media path.
        output_dir: Directory for ``frame_XXXX.png`` files.
        role: Optional ``base`` / ``donor`` label.
        settings: Optional settings override.
        fps: Override sampling rate (frames per second).

    Returns:
        List of ``SampledFrame`` with times and hashes.

    Raises:
        MediaFileNotFoundError: If ``source`` is missing.
        FFmpegNotFoundError: If FFmpeg is unavailable.
        RuntimeError: If FFmpeg fails.
    """
    cfg = settings or get_settings()
    src = Path(source)
    if not src.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {src}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample_fps = fps if fps is not None else cfg.video_sample_fps
    pattern = out / "frame_%04d.png"

    ffmpeg = resolve_ffmpeg(cfg.ffmpeg_path)
    cmd = build_frame_extract_command(ffmpeg, src, pattern, fps=sample_fps)
    logger.debug("Extracting frames: %s", " ".join(cmd))
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
        raise RuntimeError(f"FFmpeg frame extraction failed for {src}: {stderr}")

    frames: list[SampledFrame] = []
    paths = sorted(out.glob("frame_*.png"))
    for index, path in enumerate(paths):
        time_seconds = index / sample_fps if sample_fps > 0 else float(index)
        with Image.open(path) as img:
            phash = str(
                compute_phash(
                    img,
                    hash_size=cfg.video_phash_size,
                    target_size=cfg.video_normalize_size,
                    crop_bars=cfg.video_crop_letterbox,
                )
            )
        frames.append(
            SampledFrame(
                time_seconds=time_seconds,
                path=path.resolve(),
                phash=phash,
                role=role,
            )
        )
    return frames
