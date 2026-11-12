"""Remux / re-encode oversized multi-track MKVs while keeping soft tracks."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from jarvis_media_processor.audio.extract import resolve_ffmpeg
from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.media.errors import MediaFileNotFoundError
from jarvis_media_processor.media.inspect import inspect_media
from jarvis_media_processor.media.models import MediaInfo, StreamKind
from jarvis_media_processor.output.mux import validate_output_path

logger = logging.getLogger(__name__)


class OptimizeProfile(StrEnum):
    """Named quality / size trade-offs for ``jmp optimize``."""

    ARCHIVE = "archive"
    BALANCED = "balanced"
    AUDIO_SLIM = "audio-slim"
    REMUX = "remux"


class OptimizeSpecError(ValueError):
    """Invalid optimize CLI/API specification."""


# Modern codecs where a second lossy pass only pays off at high bitrates.
_EFFICIENT_VIDEO_CODECS = frozenset({"hevc", "h265", "av1", "vp9"})


@dataclass(frozen=True)
class OptimizeDecision:
    """Whether a post-mux optimize pass should run, with a human-readable reason."""

    should_run: bool
    reason: str


@dataclass(frozen=True)
class SelectedStreams:
    """Stream indices (absolute FFprobe indices) to map into the output."""

    video: tuple[int, ...]
    audio: tuple[int, ...]
    subtitle: tuple[int, ...]


def parse_language_list(spec: str | None) -> frozenset[str] | None:
    """Parse a comma-separated language list.

    Args:
        spec: ``None`` means keep all languages. An empty / whitespace-only
            string is rejected. Otherwise returns lowercased 2–3 letter tags.

    Returns:
        ``None`` (keep all) or a frozenset of language tags.

    Raises:
        OptimizeSpecError: Empty list or invalid language tags.
    """
    if spec is None:
        return None
    raw = spec.strip()
    if not raw:
        raise OptimizeSpecError(
            "Language list must be comma-separated tags (e.g. spa,eng); "
            "omit the option to keep all"
        )
    tags: list[str] = []
    for part in raw.split(","):
        tag = part.strip().lower()
        if not tag:
            raise OptimizeSpecError(f"Empty language tag in {spec!r}")
        if not (2 <= len(tag) <= 3 and tag.isalpha()):
            raise OptimizeSpecError(
                f"Language must be a 2–3 letter tag (got {tag!r} in {spec!r})"
            )
        tags.append(tag)
    return frozenset(tags)


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    tag = language.strip().lower()
    return tag or None


def _language_matches(stream_language: str | None, keep: frozenset[str]) -> bool:
    """Return True when the stream language is in ``keep``."""
    lang = _normalize_language(stream_language)
    if lang is None:
        return False
    return lang in keep


def estimate_video_bitrate_bps(info: MediaInfo, file_size_bytes: int) -> int | None:
    """Best-effort video bitrate in bits/second.

    Prefers the first video stream's tagged bitrate; otherwise derives an
    average from file size and duration (includes audio/subs — slightly high).
    """
    if info.video_streams:
        tagged = info.video_streams[0].bitrate
        if tagged is not None and tagged > 0:
            return int(tagged)
    duration = info.duration_seconds
    if duration is not None and duration > 0 and file_size_bytes > 0:
        return int((file_size_bytes * 8) / duration)
    if info.format.bitrate is not None and info.format.bitrate > 0:
        return int(info.format.bitrate)
    return None


def should_optimize_output(
    path: Path,
    info: MediaInfo,
    *,
    enabled: bool,
    size_threshold_bytes: int,
    skip_efficient_max_bitrate_bps: int,
) -> OptimizeDecision:
    """Decide whether an archive-quality optimize pass is worth running.

    Runs only when all of the following hold:

    1. ``enabled`` is True (job/settings toggle).
    2. File size is strictly greater than ``size_threshold_bytes``.
    3. There is a video stream.
    4. Either the codec is not already modern (HEVC/AV1/VP9), or it is modern
       but still above ``skip_efficient_max_bitrate_bps`` (fat remux / 4K).

    This avoids a quality-losing second HEVC pass on files that are already
    efficiently encoded near the size target.
    """
    if not enabled:
        return OptimizeDecision(False, "optimize_after_mux disabled")
    if not path.is_file():
        return OptimizeDecision(False, f"output missing: {path}")

    size = path.stat().st_size
    if size <= size_threshold_bytes:
        return OptimizeDecision(
            False,
            f"size {size} bytes <= threshold {size_threshold_bytes} bytes",
        )
    if not info.video_streams:
        return OptimizeDecision(False, "no video stream")

    raw_codec = (info.video_streams[0].codec_name or "unknown").lower()
    codec = "hevc" if raw_codec == "h265" else raw_codec
    bitrate = estimate_video_bitrate_bps(info, size)

    if codec in _EFFICIENT_VIDEO_CODECS:
        if bitrate is not None and bitrate <= skip_efficient_max_bitrate_bps:
            return OptimizeDecision(
                False,
                (
                    f"already efficient {codec} at {bitrate} bps "
                    f"(<= {skip_efficient_max_bitrate_bps}); skip re-encode"
                ),
            )
        br_label = str(bitrate) if bitrate is not None else "unknown"
        return OptimizeDecision(
            True,
            (
                f"size {size} over threshold; {codec} bitrate {br_label} bps "
                f"warrants archive re-encode"
            ),
        )

    return OptimizeDecision(
        True,
        f"size {size} over threshold; codec {codec} benefits from HEVC archive",
    )


def select_streams(
    info: MediaInfo,
    *,
    keep_audio: frozenset[str] | None = None,
    keep_subs: frozenset[str] | None = None,
) -> SelectedStreams:
    """Pick video / audio / subtitle streams for the optimize pass.

    When a keep filter is set, only streams whose language tag matches are
    retained (unlabeled streams are dropped). When unset, all streams of that
    kind are kept.
    """
    videos = [s for s in info.streams if s.kind == StreamKind.VIDEO]
    audios = [s for s in info.streams if s.kind == StreamKind.AUDIO]
    subs = [s for s in info.streams if s.kind == StreamKind.SUBTITLE]

    if keep_audio is not None:
        audios = [s for s in audios if _language_matches(s.language, keep_audio)]
    if keep_subs is not None:
        subs = [s for s in subs if _language_matches(s.language, keep_subs)]

    if not videos:
        raise OptimizeSpecError(f"No video streams in {info.path}")
    if not audios:
        raise OptimizeSpecError(
            f"No audio streams selected for {info.path}"
            + (
                f" (keep-audio={sorted(keep_audio)})"
                if keep_audio is not None
                else ""
            )
        )

    return SelectedStreams(
        video=tuple(s.index for s in videos),
        audio=tuple(s.index for s in audios),
        subtitle=tuple(s.index for s in subs),
    )


def _video_encode_args(profile: OptimizeProfile, settings: Settings) -> list[str]:
    if profile is OptimizeProfile.REMUX or profile is OptimizeProfile.AUDIO_SLIM:
        return ["-c:v", "copy"]
    crf = (
        settings.optimize_archive_crf
        if profile is OptimizeProfile.ARCHIVE
        else settings.optimize_balanced_crf
    )
    return [
        "-c:v",
        "libx265",
        "-crf",
        str(crf),
        "-preset",
        settings.optimize_x265_preset,
    ]


def _audio_encode_args(profile: OptimizeProfile, settings: Settings) -> list[str]:
    if profile is OptimizeProfile.REMUX:
        return ["-c:a", "copy"]
    args = [
        "-c:a",
        settings.optimize_audio_codec,
        "-b:a",
        settings.optimize_audio_bitrate,
    ]
    # libopus rejects layouts like 5.1(side); rematrix + mapping_family fix it.
    if settings.optimize_audio_codec in {"libopus", "opus"}:
        args.extend(
            [
                "-af",
                "aformat=channel_layouts=7.1|5.1|stereo|mono",
                "-mapping_family",
                "1",
            ]
        )
    return args


def build_optimize_command(
    ffmpeg: str,
    source: Path,
    output: Path,
    selected: SelectedStreams,
    *,
    profile: OptimizeProfile,
    settings: Settings,
) -> list[str]:
    """Build FFmpeg argv for an optimize pass (never burns subtitles)."""
    cmd: list[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
    ]
    for index in selected.video:
        cmd.extend(["-map", f"0:{index}"])
    for index in selected.audio:
        cmd.extend(["-map", f"0:{index}"])
    for index in selected.subtitle:
        cmd.extend(["-map", f"0:{index}"])

    cmd.extend(_video_encode_args(profile, settings))
    cmd.extend(_audio_encode_args(profile, settings))
    cmd.extend(["-c:s", "copy"])
    cmd.append(str(output))

    if "-vf" in cmd or "-filter_complex" in cmd:
        raise RuntimeError("Refusing optimize command that would filter/burn video")
    joined = " ".join(cmd)
    if "subtitles=" in joined or "ass=" in joined:
        raise RuntimeError("Refusing optimize command that would burn subtitles into video")

    return cmd


def optimize_mkv(
    source_path: Path | str,
    output_path: Path | str,
    *,
    profile: OptimizeProfile | str = OptimizeProfile.ARCHIVE,
    keep_audio: frozenset[str] | str | None = None,
    keep_subs: frozenset[str] | str | None = None,
    settings: Settings | None = None,
    dry_run: bool = False,
) -> Path | list[str]:
    """Remux / re-encode a multi-track MKV with the given profile.

    Args:
        source_path: Input media (never modified).
        output_path: Destination MKV (must not equal the input).
        profile: Quality/size profile name or enum.
        keep_audio: Optional language filter (``None`` = all). String form is
            comma-separated tags.
        keep_subs: Optional subtitle language filter (``None`` = all).
        settings: Optional settings override.
        dry_run: When True, return the FFmpeg argv without running it.

    Returns:
        Resolved output path, or the FFmpeg argv when ``dry_run`` is True.
    """
    cfg = settings or get_settings()
    source = Path(source_path)
    if not source.is_file():
        raise MediaFileNotFoundError(f"Media file not found: {source}")

    profile_enum = (
        profile
        if isinstance(profile, OptimizeProfile)
        else OptimizeProfile(str(profile).strip().lower())
    )

    audio_filter: frozenset[str] | None
    if isinstance(keep_audio, str):
        audio_filter = parse_language_list(keep_audio)
    else:
        audio_filter = keep_audio

    subs_filter: frozenset[str] | None
    if isinstance(keep_subs, str):
        subs_filter = parse_language_list(keep_subs)
    else:
        subs_filter = keep_subs

    out = validate_output_path(Path(output_path), source)
    ffmpeg = resolve_ffmpeg(cfg.ffmpeg_path)
    info = inspect_media(source, settings=cfg)
    selected = select_streams(info, keep_audio=audio_filter, keep_subs=subs_filter)
    cmd = build_optimize_command(
        ffmpeg,
        source,
        out,
        selected,
        profile=profile_enum,
        settings=cfg,
    )
    logger.debug("Optimize MKV (%s): %s", profile_enum, " ".join(cmd))

    if dry_run:
        return cmd

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
        raise RuntimeError(f"FFmpeg optimize failed: {stderr}")
    return out
