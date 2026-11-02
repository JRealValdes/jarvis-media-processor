"""Tests for media inspection API and FFprobe runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from jarvis_media_processor.config import Settings
from jarvis_media_processor.media import (
    FFprobeError,
    FFprobeNotFoundError,
    MediaFileNotFoundError,
    inspect_media,
    inspect_media_files,
)
from jarvis_media_processor.media.ffprobe import resolve_ffprobe, run_ffprobe_json

FIXTURE = Path(__file__).parent / "fixtures" / "ffprobe_sample.json"


def test_inspect_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mkv"
    with pytest.raises(MediaFileNotFoundError, match="not found"):
        inspect_media(missing, settings=Settings(_env_file=None))


def test_resolve_ffprobe_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(FFprobeNotFoundError, match="not found"):
        resolve_ffprobe("ffprobe-does-not-exist-xyz")


def test_run_ffprobe_json_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    media = tmp_path / "clip.mkv"
    media.write_bytes(b"fake")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    class Result:
        returncode = 0
        stdout = json.dumps(payload)
        stderr = ""

    def fake_run(cmd: list[str], **_kwargs: Any) -> Result:
        assert cmd[-1] == str(media)
        return Result()

    monkeypatch.setattr(
        "jarvis_media_processor.media.ffprobe.resolve_ffprobe",
        lambda _p: "ffprobe",
    )
    monkeypatch.setattr("jarvis_media_processor.media.ffprobe.subprocess.run", fake_run)

    parsed = run_ffprobe_json(media, ffprobe_path="ffprobe")
    assert parsed["format"]["format_name"] == "matroska,webm"


def test_run_ffprobe_nonzero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    media = tmp_path / "bad.mkv"
    media.write_bytes(b"x")

    class Result:
        returncode = 1
        stdout = ""
        stderr = "Invalid data found"

    monkeypatch.setattr(
        "jarvis_media_processor.media.ffprobe.resolve_ffprobe",
        lambda _p: "ffprobe",
    )
    monkeypatch.setattr(
        "jarvis_media_processor.media.ffprobe.subprocess.run",
        lambda *_a, **_k: Result(),
    )

    with pytest.raises(FFprobeError, match="Invalid data") as exc_info:
        run_ffprobe_json(media)
    assert exc_info.value.returncode == 1


def test_inspect_media_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    media = tmp_path / "base.mkv"
    media.write_bytes(b"fake")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        "jarvis_media_processor.media.inspect.run_ffprobe_json",
        lambda _path, ffprobe_path="ffprobe": payload,
    )

    info = inspect_media(media, settings=Settings(_env_file=None))
    assert info.path == media.resolve()
    assert len(info.audio_streams) == 2
    assert info.video_streams[0].width == 1920


def test_inspect_media_files_two_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "base.mkv"
    donor = tmp_path / "donor.mkv"
    base.write_bytes(b"a")
    donor.write_bytes(b"b")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        "jarvis_media_processor.media.inspect.run_ffprobe_json",
        lambda _path, ffprobe_path="ffprobe": payload,
    )

    report = inspect_media_files([base, donor], settings=Settings(_env_file=None))
    assert len(report.files) == 2
    assert report.files[0].path.name == "base.mkv"
    assert report.files[1].path.name == "donor.mkv"


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_inspect_real_ffprobe_integration(tmp_path: Path) -> None:
    """Optional smoke: generate a tiny media file with ffmpeg and inspect it."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    out = tmp_path / "tiny.mkv"
    import subprocess

    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=0.2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"ffmpeg failed to create fixture: {completed.stderr}")

    info = inspect_media(out, settings=Settings(_env_file=None))
    assert info.video_streams
    assert info.audio_streams
    assert info.duration_seconds is not None
    assert info.duration_seconds > 0
