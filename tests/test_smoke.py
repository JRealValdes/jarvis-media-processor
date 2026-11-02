"""Smoke tests for project foundation."""

import pytest
from typer.testing import CliRunner

from jarvis_media_processor import __version__
from jarvis_media_processor.cli.main import app
from jarvis_media_processor.config import Settings, get_settings

runner = CliRunner()


def test_version_constant() -> None:
    assert __version__ == "0.1.0"


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    help_text = result.stdout.lower()
    assert "base" in help_text or "donor" in help_text


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.delenv("JMP_LOG_LEVEL", raising=False)
    monkeypatch.delenv("JMP_WORK_DIR", raising=False)
    monkeypatch.delenv("JMP_FFMPEG_PATH", raising=False)
    monkeypatch.delenv("JMP_FFPROBE_PATH", raising=False)
    settings = Settings(_env_file=None)
    assert settings.log_level == "INFO"
    assert settings.ffmpeg_path == "ffmpeg"
    assert settings.ffprobe_path == "ffprobe"
    get_settings.cache_clear()
