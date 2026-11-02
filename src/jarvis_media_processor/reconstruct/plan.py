"""Build placement plans from an alignment document."""

from __future__ import annotations

from jarvis_media_processor.alignment.models import AlignmentDocument, AlignmentSegment
from jarvis_media_processor.reconstruct.models import Placement


def _usable_segments(
    doc: AlignmentDocument,
    *,
    min_duration: float,
    require_reliable: bool,
) -> tuple[list[AlignmentSegment], list[AlignmentSegment]]:
    used: list[AlignmentSegment] = []
    skipped: list[AlignmentSegment] = []
    for segment in doc.segments:
        duration = segment.base_end - segment.base_start
        donor_duration = segment.donor_end - segment.donor_start
        if duration < min_duration or donor_duration < min_duration:
            skipped.append(segment)
            continue
        if require_reliable and not segment.reliable:
            skipped.append(segment)
            continue
        used.append(segment)
    return used, skipped


def plan_placements(
    doc: AlignmentDocument,
    *,
    min_segment_seconds: float = 0.05,
    require_reliable: bool = False,
    crossfade_seconds: float = 0.01,
) -> tuple[list[Placement], list[AlignmentSegment]]:
    """Convert alignment segments into concrete audio placements.

    Missing/unaligned base regions are intentionally omitted (silence later).

    Args:
        doc: Alignment document.
        min_segment_seconds: Drop tiny/point segments.
        require_reliable: If True, skip ``reliable=False`` segments.
        crossfade_seconds: Soft boundary overlap hint for the renderer.

    Returns:
        ``(placements, skipped_segments)``.
    """
    used, skipped = _usable_segments(
        doc,
        min_duration=min_segment_seconds,
        require_reliable=require_reliable,
    )
    placements = [
        Placement(
            base_start=s.base_start,
            base_end=s.base_end,
            donor_start=s.donor_start,
            donor_end=s.donor_end,
            speed=s.speed if s.speed > 0 else 1.0,
            crossfade_seconds=crossfade_seconds,
            reliable=s.reliable,
        )
        for s in sorted(used, key=lambda s: s.base_start)
    ]
    return placements, skipped
