"""Shot-boundary (abrupt frame change) detection from sampled frames."""

from __future__ import annotations

from jarvis_media_processor.video.hashing import phash_distance
from jarvis_media_processor.video.models import SampledFrame


def detect_shot_boundaries(
    frames: list[SampledFrame],
    *,
    min_distance: int = 20,
) -> list[SampledFrame]:
    """Return frames that start a new shot after an abrupt pHash jump.

    Compares consecutive frames in timeline order. When Hamming distance
    ``>= min_distance``, the later frame is treated as a shot start / visual
    landmark.

    Args:
        frames: Sampled frames (any order; sorted by time internally).
        min_distance: Minimum consecutive-frame pHash distance to count as a cut.

    Returns:
        Shot-start frames (excluding the very first frame of the stream).
    """
    if len(frames) < 2:
        return []
    ordered = sorted(frames, key=lambda f: f.time_seconds)
    cuts: list[SampledFrame] = []
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if phash_distance(prev.phash, cur.phash) >= min_distance:
            cuts.append(cur)
    return cuts


def shot_boundary_times(
    frames: list[SampledFrame],
    *,
    min_distance: int = 20,
) -> set[float]:
    """Return rounded times (ms) of detected shot-start frames."""
    return {round(f.time_seconds, 3) for f in detect_shot_boundaries(frames, min_distance=min_distance)}
