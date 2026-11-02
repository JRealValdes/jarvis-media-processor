"""Synthetic base/donor fixture scenarios and generators."""

from jarvis_media_processor.fixtures.generate import (
    FFmpegNotFoundError,
    generate_scenario,
    generate_scenarios,
)
from jarvis_media_processor.fixtures.loader import load_scenario_truth, scenario_dir
from jarvis_media_processor.fixtures.models import (
    GroundTruth,
    Landmark,
    SegmentMapping,
    SyncStrategyLevel,
)
from jarvis_media_processor.fixtures.scenarios import SCENARIOS, get_scenario, list_scenario_ids

__all__ = [
    "SCENARIOS",
    "FFmpegNotFoundError",
    "GroundTruth",
    "Landmark",
    "SegmentMapping",
    "SyncStrategyLevel",
    "generate_scenario",
    "generate_scenarios",
    "get_scenario",
    "list_scenario_ids",
    "load_scenario_truth",
    "scenario_dir",
]
