"""Pytest helpers for optional FFmpeg-backed tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from jarvis_media_processor.config import get_settings


def ffmpeg_available() -> bool:
    """Return True if the configured FFmpeg executable can be resolved."""
    path = get_settings().ffmpeg_path
    if Path(path).is_file():
        return True
    return shutil.which(path) is not None


requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(),
    reason="FFmpeg not available (install and/or set JMP_FFMPEG_PATH)",
)
