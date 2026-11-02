"""High-level visual matching for a base/donor pair."""

from __future__ import annotations

import logging
from pathlib import Path

from jarvis_media_processor.config import Settings, get_settings
from jarvis_media_processor.video.extract import extract_sampled_frames
from jarvis_media_processor.video.match import match_frames
from jarvis_media_processor.video.models import VisualMatchReport

logger = logging.getLogger(__name__)


def match_media_visuals(
    base_path: Path | str,
    donor_path: Path | str,
    workspace: Path | str,
    *,
    settings: Settings | None = None,
) -> VisualMatchReport:
    """Sample frames from base and donor and return ranked visual matches.

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
    matches = match_frames(
        base_frames,
        donor_frames,
        max_distance=cfg.video_match_max_distance,
        max_matches_per_base=cfg.video_match_max_per_base,
    )

    report = VisualMatchReport(
        base_path=base.resolve(),
        donor_path=donor.resolve(),
        base_frames=base_frames,
        donor_frames=donor_frames,
        matches=matches,
        params={
            "sample_fps": cfg.video_sample_fps,
            "phash_size": cfg.video_phash_size,
            "normalize_size": cfg.video_normalize_size,
            "crop_letterbox": int(cfg.video_crop_letterbox),
            "max_distance": cfg.video_match_max_distance,
            "max_per_base": cfg.video_match_max_per_base,
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
