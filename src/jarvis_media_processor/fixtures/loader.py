"""Load generated synthetic fixture ground truth from disk."""

from __future__ import annotations

from pathlib import Path

from jarvis_media_processor.fixtures.models import GroundTruth


def scenario_dir(root: Path, scenario_id: str) -> Path:
    """Return ``root / scenario_id``."""
    return Path(root) / scenario_id


def load_scenario_truth(scenario_path: Path) -> GroundTruth:
    """Load ``ground_truth.json`` from a generated scenario directory.

    Args:
        scenario_path: Directory containing ``ground_truth.json`` (and media files).

    Returns:
        Parsed ``GroundTruth``.

    Raises:
        FileNotFoundError: If the JSON file is missing.
        ValidationError: If the JSON does not match the schema.
    """
    path = Path(scenario_path) / "ground_truth.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ground truth: {path}")
    return GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))
