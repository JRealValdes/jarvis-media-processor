"""Tests for MKV mux command construction and safety checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis_media_processor.output import OutputPathError, build_mux_mkv_command, validate_output_path
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
