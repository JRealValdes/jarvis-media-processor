"""High-level visual matching for a base/donor pair."""

from __future__ import annotations

import logging
from pathlib import Path

from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.video.extract import extract_sampled_frames
from jarvis_media_processor.video.match import match_frames
from jarvis_media_processor.video.models import VisualMatchReport
from jarvis_media_processor.video.shots import shot_boundary_times

logger = logging.getLogger(__name__)


def match_media_visuals(
    base_path: Path | str,
    donor_path: Path | str,
    workspace: Path | str,
    *,
    settings: Settings | None = None,
) -> VisualMatchReport:
    """Sample frames from base and donor and return ranked visual matches.

    Detects abrupt pHash jumps (shot cuts) and boosts matches that involve
    those frames so they act as stronger temporal anchors for alignment.

    Args:
        base_path: Base media path (timeline authority).
        donor_path: Donor media path.
        workspace: Directory for extracted frames and report JSON.
        settings: Optional settings override.

    Returns:
        ``VisualMatchReport`` (also written under ``workspace``).
    """
    cfg = settings or get_settings()
    base = Path(base_path)
    donor = Path(donor_path)
    work = Path(workspace)
    work.mkdir(parents=True, exist_ok=True)

    base_frames = extract_sampled_frames(
        base,
        work / "base_frames",
        role="base",
        settings=cfg,
    )
    donor_frames = extract_sampled_frames(
        donor,
        work / "donor_frames",
        role="donor",
        settings=cfg,
    )
    base_shots = shot_boundary_times(
        base_frames, min_distance=cfg.video_shot_min_distance
    )
    donor_shots = shot_boundary_times(
        donor_frames, min_distance=cfg.video_shot_min_distance
    )
    matches = match_frames(
        base_frames,
        donor_frames,
        max_distance=cfg.video_match_max_distance,
        max_matches_per_base=cfg.video_match_max_per_base,
        base_shot_times=base_shots,
        donor_shot_times=donor_shots,
    )

    report = VisualMatchReport(
        base_path=base.resolve(),
        donor_path=donor.resolve(),
        base_frames=base_frames,
        donor_frames=donor_frames,
        matches=matches,
        base_shot_times=sorted(base_shots),
        donor_shot_times=sorted(donor_shots),
        params={
            "sample_fps": cfg.video_sample_fps,
            "phash_size": cfg.video_phash_size,
            "normalize_size": cfg.video_normalize_size,
            "crop_letterbox": int(cfg.video_crop_letterbox),
            "max_distance": cfg.video_match_max_distance,
            "max_per_base": cfg.video_match_max_per_base,
            "shot_min_distance": cfg.video_shot_min_distance,
            "base_shot_count": len(base_shots),
            "donor_shot_count": len(donor_shots),
        },
    )
    out = work / "visual_match_report.json"
    out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Visual match %s vs %s: %d matches -> %s",
        base.name,
        donor.name,
        len(matches),
        out,
    )
    return report
