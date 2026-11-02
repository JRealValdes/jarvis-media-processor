"""Normalized media metadata models derived from FFprobe."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StreamKind(StrEnum):
    """High-level stream category."""

    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"
    DATA = "data"
    ATTACHMENT = "attachment"
    UNKNOWN = "unknown"


class StreamInfo(BaseModel):
    """One media stream (video, audio, subtitle, etc.)."""

    index: int
    kind: StreamKind
    codec_name: str | None = None
    codec_long_name: str | None = None
    profile: str | None = None
    language: str | None = None
    title: str | None = None
    duration_seconds: float | None = None
    bitrate: int | None = None
    # Video
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = None
    avg_frame_rate: str | None = None
    avg_frame_rate_fps: float | None = None
    # Audio
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None
    sample_fmt: str | None = None
    # Raw tags for forward compatibility
    tags: dict[str, str] = Field(default_factory=dict)


class FormatInfo(BaseModel):
    """Container / format-level metadata."""

    filename: str | None = None
    format_name: str | None = None
    format_long_name: str | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None
    bitrate: int | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class MediaInfo(BaseModel):
    """Normalized inspection result for a single media file."""

    path: Path
    format: FormatInfo
    streams: list[StreamInfo] = Field(default_factory=list)
    video_streams: list[StreamInfo] = Field(default_factory=list)
    audio_streams: list[StreamInfo] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Best-effort duration from format or the longest stream."""
        if self.format.duration_seconds is not None:
            return self.format.duration_seconds
        durations = [s.duration_seconds for s in self.streams if s.duration_seconds is not None]
        return max(durations) if durations else None


class InspectionReport(BaseModel):
    """Structured report for one or more inspected files (CLI / API)."""

    files: list[MediaInfo]
    warnings: list[str] = Field(default_factory=list)


def _parse_float(value: Any) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_frame_rate(rate: str | None) -> float | None:
    if not rate or rate in {"0/0", "N/A"}:
        return None
    if "/" in rate:
        num_s, den_s = rate.split("/", 1)
        num = _parse_float(num_s)
        den = _parse_float(den_s)
        if num is None or den is None or den == 0:
            return None
        return num / den
    return _parse_float(rate)


def _stream_kind(codec_type: str | None) -> StreamKind:
    mapping = {
        "video": StreamKind.VIDEO,
        "audio": StreamKind.AUDIO,
        "subtitle": StreamKind.SUBTITLE,
        "data": StreamKind.DATA,
        "attachment": StreamKind.ATTACHMENT,
    }
    if codec_type is None:
        return StreamKind.UNKNOWN
    return mapping.get(codec_type.lower(), StreamKind.UNKNOWN)


def stream_info_from_ffprobe(raw: dict[str, Any]) -> StreamInfo:
    """Build ``StreamInfo`` from one FFprobe ``streams[]`` entry.

    Args:
        raw: Raw stream object from FFprobe JSON.

    Returns:
        Normalized stream metadata.
    """
    tags_raw = raw.get("tags") or {}
    tags = {str(k): str(v) for k, v in tags_raw.items()}
    kind = _stream_kind(raw.get("codec_type"))
    avg_rate = raw.get("avg_frame_rate") or raw.get("r_frame_rate")
    return StreamInfo(
        index=int(raw.get("index", 0)),
        kind=kind,
        codec_name=raw.get("codec_name"),
        codec_long_name=raw.get("codec_long_name"),
        profile=raw.get("profile"),
        language=tags.get("language") or tags.get("LANGUAGE"),
        title=tags.get("title") or tags.get("TITLE"),
        duration_seconds=_parse_float(raw.get("duration")),
        bitrate=_parse_int(raw.get("bit_rate")),
        width=_parse_int(raw.get("width")),
        height=_parse_int(raw.get("height")),
        pix_fmt=raw.get("pix_fmt"),
        avg_frame_rate=avg_rate,
        avg_frame_rate_fps=_parse_frame_rate(avg_rate),
        sample_rate=_parse_int(raw.get("sample_rate")),
        channels=_parse_int(raw.get("channels")),
        channel_layout=raw.get("channel_layout"),
        sample_fmt=raw.get("sample_fmt"),
        tags=tags,
    )


def format_info_from_ffprobe(raw: dict[str, Any] | None) -> FormatInfo:
    """Build ``FormatInfo`` from FFprobe ``format`` object.

    Args:
        raw: Raw format object from FFprobe JSON, if present.

    Returns:
        Normalized format metadata.
    """
    if not raw:
        return FormatInfo()
    tags_raw = raw.get("tags") or {}
    tags = {str(k): str(v) for k, v in tags_raw.items()}
    return FormatInfo(
        filename=raw.get("filename"),
        format_name=raw.get("format_name"),
        format_long_name=raw.get("format_long_name"),
        duration_seconds=_parse_float(raw.get("duration")),
        size_bytes=_parse_int(raw.get("size")),
        bitrate=_parse_int(raw.get("bit_rate")),
        tags=tags,
    )


def media_info_from_ffprobe(path: Path, payload: dict[str, Any]) -> MediaInfo:
    """Build ``MediaInfo`` from a full FFprobe JSON payload.

    Args:
        path: Path that was probed.
        payload: Parsed FFprobe JSON (``streams`` + ``format``).

    Returns:
        Normalized media metadata.
    """
    streams = [stream_info_from_ffprobe(s) for s in payload.get("streams") or []]
    return MediaInfo(
        path=path,
        format=format_info_from_ffprobe(payload.get("format")),
        streams=streams,
        video_streams=[s for s in streams if s.kind == StreamKind.VIDEO],
        audio_streams=[s for s in streams if s.kind == StreamKind.AUDIO],
    )
