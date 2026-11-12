"""Tests for optimize / compress command construction and stream selection."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from jarvis_media_processor.config import Settings
from jarvis_media_processor.media.models import FormatInfo, MediaInfo, StreamInfo, StreamKind
from jarvis_media_processor.output import (
    OptimizeProfile,
    OptimizeSpecError,
    OutputPathError,
    build_optimize_command,
    optimize_mkv,
    parse_language_list,
    select_streams,
)
from jarvis_media_processor.testing import requires_ffmpeg


def _stream(
    index: int,
    kind: StreamKind,
    *,
    language: str | None = None,
    codec_name: str | None = "aac",
) -> StreamInfo:
    return StreamInfo(
        index=index,
        kind=kind,
        codec_name=codec_name,
        language=language,
    )


def _media(*streams: StreamInfo) -> MediaInfo:
    return MediaInfo(
        path=Path("sample.mkv"),
        format=FormatInfo(),
        streams=list(streams),
        video_streams=[s for s in streams if s.kind == StreamKind.VIDEO],
        audio_streams=[s for s in streams if s.kind == StreamKind.AUDIO],
    )


def test_parse_language_list_none_keeps_all() -> None:
    assert parse_language_list(None) is None


def test_parse_language_list_tags() -> None:
    assert parse_language_list("spa, eng") == frozenset({"spa", "eng"})


@pytest.mark.parametrize("spec", ["", "  ", "english", "spa,", ",eng"])
def test_parse_language_list_rejects_invalid(spec: str) -> None:
    with pytest.raises(OptimizeSpecError):
        parse_language_list(spec)


def test_select_streams_keeps_all_by_default() -> None:
    info = _media(
        _stream(0, StreamKind.VIDEO, codec_name="h264"),
        _stream(1, StreamKind.AUDIO, language="eng"),
        _stream(2, StreamKind.AUDIO, language="spa"),
        _stream(3, StreamKind.SUBTITLE, language="eng", codec_name="subrip"),
        _stream(4, StreamKind.SUBTITLE, language="spa", codec_name="subrip"),
    )
    selected = select_streams(info)
    assert selected.video == (0,)
    assert selected.audio == (1, 2)
    assert selected.subtitle == (3, 4)


def test_select_streams_filters_by_language() -> None:
    info = _media(
        _stream(0, StreamKind.VIDEO, codec_name="h264"),
        _stream(1, StreamKind.AUDIO, language="eng"),
        _stream(2, StreamKind.AUDIO, language="spa"),
        _stream(3, StreamKind.AUDIO, language=None),
        _stream(4, StreamKind.SUBTITLE, language="eng", codec_name="subrip"),
        _stream(5, StreamKind.SUBTITLE, language="fre", codec_name="subrip"),
    )
    selected = select_streams(
        info,
        keep_audio=frozenset({"spa"}),
        keep_subs=frozenset({"eng"}),
    )
    assert selected.audio == (2,)
    assert selected.subtitle == (4,)


def test_select_streams_rejects_empty_audio_after_filter() -> None:
    info = _media(
        _stream(0, StreamKind.VIDEO, codec_name="h264"),
        _stream(1, StreamKind.AUDIO, language="eng"),
    )
    with pytest.raises(OptimizeSpecError, match="No audio streams selected"):
        select_streams(info, keep_audio=frozenset({"spa"}))


def test_build_optimize_command_remux_copies() -> None:
    settings = Settings(_env_file=None)
    selected = select_streams(
        _media(
            _stream(0, StreamKind.VIDEO, codec_name="h264"),
            _stream(1, StreamKind.AUDIO, language="spa"),
            _stream(2, StreamKind.SUBTITLE, language="spa", codec_name="subrip"),
        )
    )
    cmd = build_optimize_command(
        "ffmpeg",
        Path("in.mkv"),
        Path("out.mkv"),
        selected,
        profile=OptimizeProfile.REMUX,
        settings=settings,
    )
    assert cmd[0] == "ffmpeg"
    assert "-map" in cmd and "0:0" in cmd and "0:1" in cmd and "0:2" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert cmd[cmd.index("-c:s") + 1] == "copy"
    assert "-vf" not in cmd
    assert "subtitles=" not in " ".join(cmd)


def test_build_optimize_command_balanced_uses_x265() -> None:
    settings = Settings(
        _env_file=None,
        optimize_balanced_crf=22,
        optimize_x265_preset="fast",
        optimize_audio_codec="libopus",
        optimize_audio_bitrate="160k",
    )
    selected = select_streams(
        _media(
            _stream(0, StreamKind.VIDEO, codec_name="h264"),
            _stream(1, StreamKind.AUDIO, language="eng"),
        )
    )
    cmd = build_optimize_command(
        "ffmpeg",
        Path("in.mkv"),
        Path("out.mkv"),
        selected,
        profile=OptimizeProfile.BALANCED,
        settings=settings,
    )
    assert cmd[cmd.index("-c:v") + 1] == "libx265"
    assert cmd[cmd.index("-crf") + 1] == "22"
    assert cmd[cmd.index("-preset") + 1] == "fast"
    assert cmd[cmd.index("-c:a") + 1] == "libopus"
    assert cmd[cmd.index("-b:a") + 1] == "160k"


def test_build_optimize_command_archive_crf() -> None:
    settings = Settings(_env_file=None, optimize_archive_crf=18)
    selected = select_streams(
        _media(
            _stream(0, StreamKind.VIDEO, codec_name="h264"),
            _stream(1, StreamKind.AUDIO, language="eng"),
        )
    )
    cmd = build_optimize_command(
        "ffmpeg",
        Path("in.mkv"),
        Path("out.mkv"),
        selected,
        profile=OptimizeProfile.ARCHIVE,
        settings=settings,
    )
    assert cmd[cmd.index("-c:v") + 1] == "libx265"
    assert cmd[cmd.index("-crf") + 1] == "18"


def test_build_optimize_command_audio_slim() -> None:
    settings = Settings(_env_file=None)
    selected = select_streams(
        _media(
            _stream(0, StreamKind.VIDEO, codec_name="h264"),
            _stream(1, StreamKind.AUDIO, language="eng"),
        )
    )
    cmd = build_optimize_command(
        "ffmpeg",
        Path("in.mkv"),
        Path("out.mkv"),
        selected,
        profile=OptimizeProfile.AUDIO_SLIM,
        settings=settings,
    )
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "libopus"


def _write_tiny_srt(path: Path, text: str = "Hello") -> None:
    path.write_text(
        "1\n00:00:00,000 --> 00:00:00,400\n" + text + "\n",
        encoding="utf-8",
    )


def _build_multi_track_fixture(tmp_path: Path) -> Path:
    """Create a short MKV with eng+spa audio and eng+spa softsubs."""
    source = tmp_path / "source.mkv"
    eng_srt = tmp_path / "en.srt"
    spa_srt = tmp_path / "es.srt"
    _write_tiny_srt(eng_srt, "Hello")
    _write_tiny_srt(spa_srt, "Hola")

    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=0.5",
            "-f",
            "lavfi",
            "-i",
            "sine=f=880:d=0.5",
            "-i",
            str(eng_srt),
            "-i",
            str(spa_srt),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-map",
            "2:a:0",
            "-map",
            "3:0",
            "-map",
            "4:0",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-c:s",
            "srt",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=spa",
            "-metadata:s:s:0",
            "language=eng",
            "-metadata:s:s:1",
            "language=spa",
            "-shortest",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg multi-track fixture failed: {completed.stderr}")
    return source


@requires_ffmpeg
def test_optimize_remux_keep_filters_integration(tmp_path: Path) -> None:
    from jarvis_media_processor.media import inspect_media

    source = _build_multi_track_fixture(tmp_path)
    out = tmp_path / "out.mkv"
    settings = Settings(_env_file=None)

    result = optimize_mkv(
        source,
        out,
        profile=OptimizeProfile.REMUX,
        keep_audio="spa",
        keep_subs="spa",
        settings=settings,
    )
    assert result == out
    assert out.is_file()
    info = inspect_media(out, settings=settings)
    assert len(info.video_streams) == 1
    assert len(info.audio_streams) == 1
    assert info.audio_streams[0].language == "spa"
    subs = [s for s in info.streams if s.kind == StreamKind.SUBTITLE]
    assert len(subs) == 1
    assert subs[0].language == "spa"


@requires_ffmpeg
def test_optimize_audio_slim_reencodes_audio(tmp_path: Path) -> None:
    from jarvis_media_processor.media import inspect_media

    source = _build_multi_track_fixture(tmp_path)
    out = tmp_path / "slim.mkv"
    # Use AAC in the integration fixture — widely available; Opus is covered
    # by unit tests of the command builder / settings defaults.
    settings = Settings(
        _env_file=None,
        optimize_audio_codec="aac",
        optimize_audio_bitrate="128k",
    )

    optimize_mkv(
        source,
        out,
        profile=OptimizeProfile.AUDIO_SLIM,
        keep_audio="eng,spa",
        settings=settings,
    )
    info = inspect_media(out, settings=settings)
    assert info.video_streams[0].codec_name in {"h264", "libx264"}
    assert all(a.codec_name == "aac" for a in info.audio_streams)
    assert {a.language for a in info.audio_streams} == {"eng", "spa"}


@requires_ffmpeg
def test_optimize_dry_run_returns_command(tmp_path: Path) -> None:
    source = _build_multi_track_fixture(tmp_path)
    out = tmp_path / "dry.mkv"
    settings = Settings(_env_file=None)
    cmd = optimize_mkv(
        source,
        out,
        profile=OptimizeProfile.REMUX,
        settings=settings,
        dry_run=True,
    )
    assert isinstance(cmd, list)
    assert Path(cmd[0]).name.lower() in {"ffmpeg", "ffmpeg.exe"}
    assert not out.exists()


@requires_ffmpeg
def test_optimize_refuses_overwrite_input(tmp_path: Path) -> None:
    source = _build_multi_track_fixture(tmp_path)
    with pytest.raises(OutputPathError):
        optimize_mkv(
            source,
            source,
            profile=OptimizeProfile.REMUX,
            settings=Settings(_env_file=None),
        )
