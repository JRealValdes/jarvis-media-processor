"""Match sampled frames between base and donor via perceptual hashes."""

from __future__ import annotations

from jarvis_media_processor.video.hashing import distance_to_confidence, phash_distance
from jarvis_media_processor.video.models import FrameCorrespondence, SampledFrame


def match_frames(
    base_frames: list[SampledFrame],
    donor_frames: list[SampledFrame],
    *,
    max_distance: int = 12,
    max_matches_per_base: int = 3,
    base_shot_times: set[float] | None = None,
    donor_shot_times: set[float] | None = None,
    shot_confidence_boost: float = 0.15,
) -> list[FrameCorrespondence]:
    """Find ranked base↔donor frame correspondences.

    For each base frame, keep up to ``max_matches_per_base`` donor frames with
    Hamming distance ``<= max_distance``, sorted by distance then time.
    Matches involving detected shot-boundary times are flagged and slightly
    boosted so they act as stronger temporal anchors downstream.

    Args:
        base_frames: Frames from the base media.
        donor_frames: Frames from the donor media.
        max_distance: Maximum accepted pHash distance.
        max_matches_per_base: Cap of donor candidates per base frame.
        base_shot_times: Optional shot-start times on the base timeline.
        donor_shot_times: Optional shot-start times on the donor timeline.
        shot_confidence_boost: Added confidence when either side is a shot cut.

    Returns:
        Flat list of correspondences sorted by confidence descending.
    """
    base_shots = base_shot_times or set()
    donor_shots = donor_shot_times or set()
    matches: list[FrameCorrespondence] = []
    for base in base_frames:
        candidates: list[FrameCorrespondence] = []
        for donor in donor_frames:
            distance = phash_distance(base.phash, donor.phash)
            if distance > max_distance:
                continue
            confidence = distance_to_confidence(distance, max_distance=max_distance)
            shot_anchor = (
                round(base.time_seconds, 3) in base_shots
                or round(donor.time_seconds, 3) in donor_shots
            )
            if shot_anchor:
                confidence = min(1.0, confidence + shot_confidence_boost)
            candidates.append(
                FrameCorrespondence(
                    base_time=base.time_seconds,
                    donor_time=donor.time_seconds,
                    distance=distance,
                    confidence=confidence,
                    base_frame=base.path,
                    donor_frame=donor.path,
                    shot_anchor=shot_anchor,
                )
            )
        candidates.sort(key=lambda m: (m.distance, m.donor_time))
        matches.extend(candidates[:max_matches_per_base])

    matches.sort(key=lambda m: (-m.confidence, m.base_time, m.donor_time))
    return matches
