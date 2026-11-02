"""Tests for FFmpeg analysis-audio extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis_media_processor.audio.extract import extract_analysis_wav
from jarvis_media_processor.config import Settings
from jarvis_media_processor.media import MediaFileNotFoundError


def test_extract_missing_source(tmp_path: Path) -> None:
    with pytest.raises(MediaFileNotFoundError):
        extract_analysis_wav(
            tmp_path / "missing.mkv",
            tmp_path / "out.wav",
            settings=Settings(_env_file=None),
        )


def test_extract_analysis_wav_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "base.mkv"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.wav"

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd: list[str], **_kwargs: Any) -> Result:
        assert "pcm_s16le" in cmd
        Path(cmd[-1]).write_bytes(b"RIFF")
        return Result()

    monkeypatch.setattr(
        "jarvis_media_processor.audio.extract.resolve_ffmpeg",
        lambda _p: "ffmpeg",
    )
    monkeypatch.setattr("jarvis_media_processor.audio.extract.subprocess.run", fake_run)

    out = extract_analysis_wav(src, dst, settings=Settings(_env_file=None))
    assert out == dst.resolve()
    assert dst.is_file()
