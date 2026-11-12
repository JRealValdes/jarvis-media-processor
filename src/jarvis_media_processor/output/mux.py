"""MKV mux: base video/audio (stream copy) + reconstructed donor audio + softsubs."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jarvis_media_processor.audio.extract import resolve_ffmpeg
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.media.errors import MediaFileNotFoundError
from jarvis_media_processor.media.inspect import inspect_media
from jarvis_media_processor.media.models import StreamKind

logger = logging.getLogger(__name__)

# Text / bitmap containers we remux as selectable soft tracks (never burn-in).
_SUBTITLE_EXTENSIONS = frozenset({".srt", ".ass", ".ssa", ".vtt", ".sup", ".sub"})


class OutputPathError(ValueError):
    """Output path is unsafe (missing parent, clobbers inputs, etc.)."""


class SubtitleSpecError(ValueError):
    """Subtitle CLI/API specification is invalid."""


@dataclass(frozen=True)
class SubtitleTrack:
    """One soft subtitle track to mux into the output MKV (never burned in)."""

    path: Path
    language: str
    title: str = ""


def validate_output_path(output: Path, *inputs: Path) -> Path:
    """Ensure ``output`` does not overwrite inputs; create parent dirs."""
    out = Path(output).resolve()
    for inp in inputs:
        if Path(inp).resolve() == out:
            raise OutputPathError(f"Refusing to overwrite input file: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def parse_subtitle_spec(spec: str) -> SubtitleTrack:
    """Parse ``LANG=PATH`` or ``LANG:TITLE=PATH`` into a ``SubtitleTrack``.

    Examples:
        ``spa=subs/es.srt``
        ``spa:Spanish=subs/es.srt``
        ``eng:English (SDH)=C:\\subs\\en.srt``
    """
    raw = spec.strip()
    if not raw:
        raise SubtitleSpecError("Empty subtitle specification")
    if "=" not in raw:
        raise SubtitleSpecError(
            f"Subtitle spec must be LANG[:TITLE]=PATH (got {spec!r})"
        )
    left, path_s = raw.split("=", 1)
    path_s = path_s.strip()
    left = left.strip()
    if not left or not path_s:
        raise SubtitleSpecError(
            f"Subtitle spec must be LANG[:TITLE]=PATH (got {spec!r})"
        )

    if ":" in left:
        language, title = left.split(":", 1)
        language = language.strip()
        title = title.strip()
    else:
        language = left
        title = ""

    if not language:
        raise SubtitleSpecError(f"Subtitle language is required (got {spec!r})")
    if not re.fullmatch(r"[A-Za-z]{2,3}", language):
        raise SubtitleSpecError(
            f"Subtitle language must be a 2–3 letter tag (got {language!r} in {spec!r})"
        )

    path = Path(path_s)
    _validate_subtitle_path(path)
    return SubtitleTrack(path=path, language=language.lower(), title=title)


def _validate_subtitle_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in _SUBTITLE_EXTENSIONS:
        allowed = ", ".join(sorted(_SUBTITLE_EXTENSIONS))
        raise SubtitleSpecError(
            f"Unsupported subtitle extension {suffix!r} for {path}; "
            f"expected one of: {allowed}"
        )


def _append_subtitle_metadata(
    cmd: list[str],
    tracks: list[SubtitleTrack],
    *,
    subtitle_index_offset: int,
) -> None:
    """Append language/title metadata for external soft subtitle tracks."""
    for i, track in enumerate(tracks):
        out_index = subtitle_index_offset + i
        cmd.extend(
            [
                f"-metadata:s:s:{out_index}",
                f"language={track.language}",
            ]
        )
        if track.title:
            cmd.extend(
                [
                    f"-metadata:s:s:{out_index}",
                    f"title={track.title}",
                ]
            )


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
    subtitles: list[SubtitleTrack] | None = None,
    copy_base_subtitles: bool = False,
    base_subtitle_count: int = 0,
) -> list[str]:
    """Build FFmpeg mux args.

    Copies base video and base audio streams; encodes/appends donor audio as
    output audio index ``base_audio_count`` with language/title metadata.
    Optional soft subtitle tracks are mapped as selectable streams (never
    burned into the picture via filters).
    """
    tracks = list(subtitles or [])
    donor_a_index = max(0, base_audio_count)
    # When not copying base subs, ignore any probed count for metadata offsets.
    sub_offset = base_subtitle_count if copy_base_subtitles else 0

    cmd: list[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(base),
        "-i",
        str(donor_audio),
    ]
    for track in tracks:
        cmd.extend(["-i", str(track.path)])

    cmd.extend(
        [
            "-map",
            "0:v?",
            "-map",
            "0:a?",
            "-map",
            "1:a:0",
        ]
    )
    if copy_base_subtitles:
        cmd.extend(["-map", "0:s?"])
    for i, _track in enumerate(tracks):
        # Inputs: 0=base, 1=donor audio, 2..=external subtitle files
        cmd.extend(["-map", f"{i + 2}:0"])

    cmd.extend(
        [
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            f"-c:a:{donor_a_index}",
            donor_audio_codec,
            "-c:s",
            "copy",
            f"-metadata:s:a:{donor_a_index}",
            f"language={donor_language}",
            f"-metadata:s:a:{donor_a_index}",
            f"title={donor_title}",
        ]
    )
    _append_subtitle_metadata(cmd, tracks, subtitle_index_offset=sub_offset)

    # Guard against accidental burn-in if this builder is extended poorly.
    if "-vf" in cmd or "-filter_complex" in cmd:
        raise RuntimeError("Refusing mux command that would filter/burn video")
    joined = " ".join(cmd)
    if "subtitles=" in joined or "ass=" in joined:
        raise RuntimeError("Refusing mux command that would burn subtitles into video")

    cmd.append(str(output))
    return cmd


def mux_mkv(
    base_path: Path | str,
    donor_audio_path: Path | str,
    output_path: Path | str,
    *,
    settings: Settings | None = None,
    donor_language: str = "spa",
    donor_title: str = "Spanish (reconstructed)",
    donor_audio_codec: str = "flac",
    subtitles: list[SubtitleTrack] | None = None,
    copy_base_subtitles: bool = False,
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
        subtitles: Optional soft subtitle tracks (external files). Never burned in.
        copy_base_subtitles: When True, also map subtitle streams from ``base``.

    Returns:
        Resolved output path.
    """
    cfg = settings or get_settings()
    base = Path(base_path)
    donor_audio = Path(donor_audio_path)
    tracks = list(subtitles or [])

    if not base.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {base}")
    if not donor_audio.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {donor_audio}")
    for track in tracks:
        _validate_subtitle_path(track.path)
        if not track.path.is_file():
            raise MediaFileNotFoundError(f"Subtitle file not found: {track.path}")
        if not track.language:
            raise SubtitleSpecError(f"Subtitle language is required for {track.path}")

    out = validate_output_path(
        Path(output_path),
        base,
        donor_audio,
        *[t.path for t in tracks],
    )
    ffmpeg = resolve_ffmpeg(cfg.ffmpeg_path)

    info = inspect_media(base, settings=cfg)
    base_audio_count = len(info.audio_streams)
    if base_audio_count == 0:
        raise RuntimeError(f"Base media has no audio streams: {base}")
    base_subtitle_count = sum(1 for s in info.streams if s.kind == StreamKind.SUBTITLE)

    cmd = build_mux_mkv_command(
        ffmpeg,
        base,
        donor_audio,
        out,
        donor_language=donor_language,
        donor_title=donor_title,
        donor_audio_codec=donor_audio_codec,
        base_audio_count=base_audio_count,
        subtitles=tracks,
        copy_base_subtitles=copy_base_subtitles,
        base_subtitle_count=base_subtitle_count,
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
