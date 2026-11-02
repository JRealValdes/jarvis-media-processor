"""CLI tests for ``jmp inspect``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis_media_processor.cli.main import app
from jarvis_media_processor.media.models import InspectionReport, media_info_from_ffprobe

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "ffprobe_sample.json"


def test_cli_inspect_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "base.mkv"
    donor = tmp_path / "donor.mkv"
    base.write_bytes(b"a")
    donor.write_bytes(b"b")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def fake_inspect_files(paths: list[Path], settings: object = None) -> InspectionReport:
        files = [media_info_from_ffprobe(Path(p).resolve(), payload) for p in paths]
        return InspectionReport(files=files)

    monkeypatch.setattr(
        "jarvis_media_processor.cli.inspect_cmd.inspect_media_files",
        fake_inspect_files,
    )

    result = runner.invoke(app, ["inspect", str(base), str(donor)])
    assert result.exit_code == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert len(data["files"]) == 2
    assert data["files"][0]["video_streams"][0]["codec_name"] == "h264"
    assert data["files"][0]["audio_streams"][0]["language"] == "eng"


def test_cli_inspect_missing_file() -> None:
    result = runner.invoke(app, ["inspect", "definitely-missing-file-xyz.mkv"])
    assert result.exit_code == 1
    assert "not found" in (result.stdout + result.stderr).lower()
