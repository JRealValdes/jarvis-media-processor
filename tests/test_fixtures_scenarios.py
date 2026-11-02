"""Tests for synthetic fixture catalog and loaders."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis_media_processor.cli.main import app
from jarvis_media_processor.fixtures import (
    GroundTruth,
    SyncStrategyLevel,
    get_scenario,
    list_scenario_ids,
    load_scenario_truth,
)
from jarvis_media_processor.fixtures.generate import resolve_ffmpeg
from jarvis_media_processor.testing import ffmpeg_available, requires_ffmpeg

runner = CliRunner()


def test_catalog_contains_required_scenarios() -> None:
    ids = list_scenario_ids()
    for required in (
        "already_synced",
        "constant_offset",
        "unrelated_block",
        "constant_speed",
        "local_length_drift",
    ):
        assert required in ids


def test_scenarios_round_trip_json() -> None:
    for scenario_id in list_scenario_ids():
        truth = get_scenario(scenario_id)
        restored = GroundTruth.model_validate_json(truth.model_dump_json())
        assert restored == truth
        assert restored.segments
        assert restored.base.duration_seconds > 0
        assert restored.donor.duration_seconds > 0


def test_local_length_drift_is_piecewise() -> None:
    truth = get_scenario("local_length_drift")
    assert truth.expected_strategy == SyncStrategyLevel.PIECEWISE
    assert len(truth.segments) == 3
    # After the elongated middle scene, donor times diverge from base.
    assert truth.segments[2].donor_start > truth.segments[2].base_start


def test_constant_offset_metadata() -> None:
    truth = get_scenario("constant_offset")
    assert truth.expected_strategy == SyncStrategyLevel.CONSTANT_OFFSET
    assert truth.global_offset_seconds == 1.0
    for lm in truth.landmarks:
        assert lm.donor_time == pytest.approx(lm.base_time + 1.0)


def test_load_scenario_truth(tmp_path: Path) -> None:
    truth = get_scenario("already_synced")
    scenario_path = tmp_path / "already_synced"
    scenario_path.mkdir()
    (scenario_path / "ground_truth.json").write_text(truth.model_dump_json(), encoding="utf-8")
    loaded = load_scenario_truth(scenario_path)
    assert loaded.scenario_id == "already_synced"


def test_cli_fixtures_list() -> None:
    result = runner.invoke(app, ["fixtures", "list"])
    assert result.exit_code == 0
    assert "local_length_drift" in result.stdout


def test_resolve_ffmpeg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(Exception, match="not found"):
        resolve_ffmpeg("ffmpeg-missing-xyz")


@requires_ffmpeg
def test_generate_already_synced(tmp_path: Path) -> None:
    from jarvis_media_processor.config import Settings
    from jarvis_media_processor.fixtures import generate_scenario

    out = tmp_path / "already_synced"
    truth = generate_scenario("already_synced", out, settings=Settings(_env_file=None))
    assert (out / "base.mkv").is_file()
    assert (out / "donor.mkv").is_file()
    assert (out / "ground_truth.json").is_file()
    assert load_scenario_truth(out).scenario_id == truth.scenario_id


def test_ffmpeg_helper_reports_boolean() -> None:
    assert isinstance(ffmpeg_available(), bool)
