"""Confirm hypothesized timeline pairs by comparing actual frame content."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from jarvis_media_processor.video.hashing import phash_distance
from jarvis_media_processor.video.models import SampledFrame


def nearest_frame(
    frames: Sequence[SampledFrame],
    time_seconds: float,
) -> SampledFrame | None:
    """Return the sampled frame closest to ``time_seconds``."""
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f.time_seconds - time_seconds))


def frame_distance_at(
    base_frames: Sequence[SampledFrame],
    donor_frames: Sequence[SampledFrame],
    *,
    base_time: float,
    donor_time: float,
) -> int | None:
    """pHash Hamming distance between nearest frames at the given times."""
    base = nearest_frame(base_frames, base_time)
    donor = nearest_frame(donor_frames, donor_time)
    if base is None or donor is None:
        return None
    return phash_distance(base.phash, donor.phash)


def frames_match_at(
    base_frames: Sequence[SampledFrame],
    donor_frames: Sequence[SampledFrame],
    *,
    base_time: float,
    donor_time: float,
    max_distance: int = 12,
    max_time_delta: float = 2.0,
) -> bool:
    """True when nearest frames look alike and are close to the requested times."""
    base = nearest_frame(base_frames, base_time)
    donor = nearest_frame(donor_frames, donor_time)
    if base is None or donor is None:
        return False
    if abs(base.time_seconds - base_time) > max_time_delta:
        return False
    if abs(donor.time_seconds - donor_time) > max_time_delta:
        return False
    return phash_distance(base.phash, donor.phash) <= max_distance


def make_frame_confirmer(
    base_frames: Sequence[SampledFrame],
    donor_frames: Sequence[SampledFrame],
    *,
    max_distance: int = 12,
    max_time_delta: float = 2.0,
) -> Callable[[float, float], bool]:
    """Build ``(base_time, donor_time) -> bool`` using sampled frame pHashes."""

    def _confirm(base_time: float, donor_time: float) -> bool:
        return frames_match_at(
            base_frames,
            donor_frames,
            base_time=base_time,
            donor_time=donor_time,
            max_distance=max_distance,
            max_time_delta=max_time_delta,
        )

    return _confirm
