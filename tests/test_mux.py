"""Tests for MKV mux command construction and safety checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_media_processor.media.models import StreamKind
from jarvis_media_processor.output import (
    OutputPathError,
    SubtitleSpecError,
    SubtitleTrack,
    build_mux_mkv_command,
    parse_subtitle_spec,
    validate_output_path,
)
from jarvis_media_processor.testing import requires_ffmpeg


def test_validate_output_refuses_clobber(tmp_path: Path) -> None:
    base = tmp_path / "base.mkv"
    base.write_bytes(b"x")
    with pytest.raises(OutputPathError):
        validate_output_path(base, base)


def test_build_mux_command_copies_base_and_tags_donor() -> None:
    cmd = build_mux_mkv_command(
        "ffmpeg",
        Path("base.mkv"),
        Path("donor.wav"),
        Path("out.mkv"),
        donor_language="spa",
        donor_title="Spanish (reconstructed)",
        donor_audio_codec="flac",
        base_audio_count=1,
    )
    assert cmd[0] == "ffmpeg"
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-c:a:1" in cmd
    assert "flac" in cmd
    assert "language=spa" in cmd
    assert "title=Spanish (reconstructed)" in cmd
    assert cmd[-1].endswith("out.mkv")
    assert "0:s?" not in cmd
    assert "-vf" not in cmd
    assert "-filter_complex" not in cmd


def test_parse_subtitle_spec_language_and_title() -> None:
    track = parse_subtitle_spec(r"spa:Spanish=C:\subs\es.srt")
    assert track.language == "spa"
    assert track.title == "Spanish"
    assert track.path == Path(r"C:\subs\es.srt")

    track2 = parse_subtitle_spec("eng=subs/en.srt")
    assert track2.language == "eng"
    assert track2.title == ""
    assert track2.path == Path("subs/en.srt")


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "subs/en.srt",
        "=subs/en.srt",
        "english=subs/en.srt",
        "eng=subs/en.mp4",
    ],
)
def test_parse_subtitle_spec_rejects_invalid(spec: str) -> None:
    with pytest.raises(SubtitleSpecError):
        parse_subtitle_spec(spec)


def test_build_mux_command_maps_soft_subtitles() -> None:
    subs = [
        SubtitleTrack(Path("en.srt"), "eng", "English"),
        SubtitleTrack(Path("es.srt"), "spa", "Spanish"),
    ]
    cmd = build_mux_mkv_command(
        "ffmpeg",
        Path("base.mkv"),
        Path("donor.wav"),
        Path("out.mkv"),
        donor_language="spa",
        donor_title="Spanish (reconstructed)",
        donor_audio_codec="flac",
        base_audio_count=1,
        subtitles=subs,
    )
    assert "-i" in cmd
    assert str(Path("en.srt")) in cmd
    assert str(Path("es.srt")) in cmd
    assert "-map" in cmd and "2:0" in cmd and "3:0" in cmd
    assert "-c:s" in cmd and cmd[cmd.index("-c:s") + 1] == "copy"
    assert "-metadata:s:s:0" in cmd
    assert "language=eng" in cmd
    assert "title=English" in cmd
    assert "-metadata:s:s:1" in cmd
    assert "language=spa" in cmd
    assert "title=Spanish" in cmd
    # Soft tracks only — no burn-in filters.
    assert "-vf" not in cmd
    assert "-filter_complex" not in cmd
    assert "subtitles=" not in " ".join(cmd)
    assert "ass=" not in " ".join(cmd)


def test_build_mux_command_copy_base_subtitles_offsets_metadata() -> None:
    cmd = build_mux_mkv_command(
        "ffmpeg",
        Path("base.mkv"),
        Path("donor.wav"),
        Path("out.mkv"),
        donor_language="spa",
        donor_title="Spanish (reconstructed)",
        donor_audio_codec="flac",
        base_audio_count=1,
        subtitles=[SubtitleTrack(Path("en.srt"), "eng", "English")],
        copy_base_subtitles=True,
        base_subtitle_count=2,
    )
    assert "0:s?" in cmd
    assert "-metadata:s:s:2" in cmd
    assert "language=eng" in cmd


def _write_tiny_srt(path: Path, text: str = "Hello") -> None:
    path.write_text(
        "1\n00:00:00,000 --> 00:00:00,400\n" + text + "\n",
        encoding="utf-8",
    )


@requires_ffmpeg
def test_mux_mkv_integration(tmp_path: Path) -> None:
    import subprocess

    from jarvis_media_processor.config import Settings
    from jarvis_media_processor.media import inspect_media
    from jarvis_media_processor.output import mux_mkv

    base = tmp_path / "base.mkv"
    donor = tmp_path / "donor.wav"
    out = tmp_path / "out.mkv"

    # Tiny base with video+audio
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
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(base),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg base fixture failed: {completed.stderr}")

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
            "sine=f=880:d=0.5",
            str(donor),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg donor fixture failed: {completed.stderr}")

    result = mux_mkv(base, donor, out, settings=Settings(_env_file=None))
    assert result.is_file()
    assert base.stat().st_size > 0  # inputs unchanged / still present
    info = inspect_media(result, settings=Settings(_env_file=None))
    assert info.video_streams
    assert len(info.audio_streams) >= 2
    donor_track = info.audio_streams[-1]
    assert donor_track.language == "spa"
    assert donor_track.title == "Spanish (reconstructed)"
    assert donor_track.codec_name == "flac"


@requires_ffmpeg
def test_mux_mkv_soft_subtitles_integration(tmp_path: Path) -> None:
    import subprocess

    from jarvis_media_processor.config import Settings
    from jarvis_media_processor.media import inspect_media
    from jarvis_media_processor.output import mux_mkv

    base = tmp_path / "base.mkv"
    donor = tmp_path / "donor.wav"
    out = tmp_path / "out.mkv"
    eng = tmp_path / "en.srt"
    spa = tmp_path / "es.srt"
    _write_tiny_srt(eng, "Hello")
    _write_tiny_srt(spa, "Hola")

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
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(base),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg base fixture failed: {completed.stderr}")

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
            "sine=f=880:d=0.5",
            str(donor),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg donor fixture failed: {completed.stderr}")

    result = mux_mkv(
        base,
        donor,
        out,
        settings=Settings(_env_file=None),
        subtitles=[
            SubtitleTrack(eng, "eng", "English"),
            SubtitleTrack(spa, "spa", "Spanish"),
        ],
    )
    assert result.is_file()
    info = inspect_media(result, settings=Settings(_env_file=None))
    assert info.video_streams
    assert len(info.audio_streams) >= 2
    subs = [s for s in info.streams if s.kind == StreamKind.SUBTITLE]
    assert len(subs) == 2
    assert {s.language for s in subs} == {"eng", "spa"}
    titles = {s.language: s.title for s in subs}
    assert titles["eng"] == "English"
    assert titles["spa"] == "Spanish"
    # Softsubs: video remains a normal coded stream (not burned text frames).
    assert info.video_streams[0].codec_name in {"h264", "libx264", "mpeg4", "hevc"}


def test_mux_mkv_missing_subtitle_file(tmp_path: Path) -> None:
    from jarvis_media_processor.config import Settings
    from jarvis_media_processor.media import MediaFileNotFoundError
    from jarvis_media_processor.output import mux_mkv

    base = tmp_path / "base.mkv"
    donor = tmp_path / "donor.wav"
    # Placeholders: missing-sub check runs before media inspect / FFmpeg.
    base.write_bytes(b"not-a-real-media")
    donor.write_bytes(b"not-a-real-media")
    missing = tmp_path / "missing.srt"
    with pytest.raises(MediaFileNotFoundError, match="Subtitle file not found"):
        mux_mkv(
            base,
            donor,
            tmp_path / "out.mkv",
            settings=Settings(_env_file=None),
            subtitles=[SubtitleTrack(missing, "eng", "English")],
        )
