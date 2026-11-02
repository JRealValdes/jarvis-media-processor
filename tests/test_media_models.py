"""Tests for FFprobe JSON normalization."""

from __future__ import annotations

import json
from pathlib import Path

from jarvis_media_processor.media.models import (
    StreamKind,
    media_info_from_ffprobe,
    stream_info_from_ffprobe,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ffprobe_sample.json"


def test_parse_sample_ffprobe_fixture() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    info = media_info_from_ffprobe(Path("sample.mkv"), payload)

    assert info.format.format_name == "matroska,webm"
    assert info.format.duration_seconds == 120.12
    assert info.format.size_bytes == 80_000_000
    assert len(info.streams) == 3
    assert len(info.video_streams) == 1
    assert len(info.audio_streams) == 2

    video = info.video_streams[0]
    assert video.kind == StreamKind.VIDEO
    assert video.codec_name == "h264"
    assert video.width == 1920
    assert video.height == 1080
    assert video.avg_frame_rate_fps is not None
    assert abs(video.avg_frame_rate_fps - (24000 / 1001)) < 1e-6

    eng = info.audio_streams[0]
    assert eng.language == "eng"
    assert eng.title == "English"
    assert eng.sample_rate == 48000
    assert eng.channels == 2

    spa = info.audio_streams[1]
    assert spa.language == "spa"
    assert spa.channels == 6
    assert info.duration_seconds == 120.12


def test_frame_rate_edge_cases() -> None:
    stream = stream_info_from_ffprobe(
        {
            "index": 0,
            "codec_type": "video",
            "avg_frame_rate": "0/0",
        }
    )
    assert stream.avg_frame_rate_fps is None

    stream2 = stream_info_from_ffprobe(
        {
            "index": 0,
            "codec_type": "audio",
            "sample_rate": "N/A",
            "channels": "N/A",
        }
    )
    assert stream2.sample_rate is None
    assert stream2.channels is None
    assert stream2.kind == StreamKind.AUDIO
