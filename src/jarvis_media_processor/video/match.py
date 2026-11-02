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
) -> list[FrameCorrespondence]:
    """Find ranked base↔donor frame correspondences.

    For each base frame, keep up to ``max_matches_per_base`` donor frames with
    Hamming distance ``<= max_distance``, sorted by distance then time.

    Args:
        base_frames: Frames from the base media.
        donor_frames: Frames from the donor media.
        max_distance: Maximum accepted pHash distance.
        max_matches_per_base: Cap of donor candidates per base frame.

    Returns:
        Flat list of correspondences sorted by confidence descending.
    """
    matches: list[FrameCorrespondence] = []
    for base in base_frames:
        candidates: list[FrameCorrespondence] = []
        for donor in donor_frames:
            distance = phash_distance(base.phash, donor.phash)
            if distance > max_distance:
                continue
            confidence = distance_to_confidence(distance, max_distance=max_distance)
            candidates.append(
                FrameCorrespondence(
                    base_time=base.time_seconds,
                    donor_time=donor.time_seconds,
                    distance=distance,
                    confidence=confidence,
                    base_frame=base.path,
                    donor_frame=donor.path,
                )
            )
        candidates.sort(key=lambda m: (m.distance, m.donor_time))
        matches.extend(candidates[:max_matches_per_base])

    matches.sort(key=lambda m: (-m.confidence, m.base_time, m.donor_time))
    return matches
