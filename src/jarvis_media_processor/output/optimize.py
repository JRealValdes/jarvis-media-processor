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
    return [
        "-c:a",
        settings.optimize_audio_codec,
        "-b:a",
        settings.optimize_audio_bitrate,
    ]


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
