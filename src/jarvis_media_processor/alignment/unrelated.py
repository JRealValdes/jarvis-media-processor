"""Detect and carve unrelated donor spans; expose curve-walk based detection."""

from __future__ import annotations

from jarvis_media_processor.alignment.activity_sync import ActivitySyncResult
from jarvis_media_processor.alignment.curve_walk import walk_activity_curves
from jarvis_media_processor.alignment.models import (
    AlignmentDocument,
    AlignmentSegment,
    TimelineInterval,
)
from jarvis_media_processor.video.activity import ActivityEnvelope


def detect_unrelated_donor_spans(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    sync: ActivitySyncResult | None = None,
    window_seconds: float = 12.0,
    hop_seconds: float = 2.0,
    search_seconds: float = 1200.0,
    min_score: float = 0.3,
    lag_jump_seconds: float = 2.0,
    min_span_seconds: float = 8.0,
    score_ratio: float = 0.4,
    max_surplus_seconds: float = 1200.0,
    missing_base_fill_seconds: float = 3.0,
    confirm_seconds: float = 6.0,
    frame_confirm=None,
) -> list[TimelineInterval]:
    """Find surplus donor intervals via activity-curve walking.

    ``sync`` / ``score_ratio`` / ``lag_jump_seconds`` are kept for call-site
    compatibility; the walker expands offset search itself and does not need a
    precomputed global sync. Optional ``frame_confirm(base_t, donor_t)`` rejects
    derivative lookalikes using actual frame content.
    """
    _ = (sync, score_ratio, lag_jump_seconds, search_seconds)
    result = walk_activity_curves(
        base,
        donor,
        half_window=window_seconds,
        hop_seconds=hop_seconds,
        min_score=min_score,
        confirm_seconds=confirm_seconds,
        min_surplus_seconds=min_span_seconds,
        max_surplus_seconds=max_surplus_seconds,
        missing_base_fill_seconds=missing_base_fill_seconds,
        max_offset_seconds=max_surplus_seconds,
        frame_confirm=frame_confirm,
    )
    return result.unrelated_donor_intervals


def detect_curve_gaps(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    window_seconds: float = 12.0,
    hop_seconds: float = 2.0,
    min_score: float = 0.3,
    min_span_seconds: float = 8.0,
    max_surplus_seconds: float = 1200.0,
    missing_base_fill_seconds: float = 3.0,
    confirm_seconds: float = 6.0,
    frame_confirm=None,
):
    """Return full curve-walk result (surplus donor + missing base)."""
    return walk_activity_curves(
        base,
        donor,
        half_window=window_seconds,
        hop_seconds=hop_seconds,
        min_score=min_score,
        confirm_seconds=confirm_seconds,
        min_surplus_seconds=min_span_seconds,
        max_surplus_seconds=max_surplus_seconds,
        missing_base_fill_seconds=missing_base_fill_seconds,
        max_offset_seconds=max_surplus_seconds,
        frame_confirm=frame_confirm,
    )


def merge_timeline_intervals(
    intervals: list[TimelineInterval],
) -> list[TimelineInterval]:
    """Merge overlapping/adjacent intervals that share timeline + reason family."""
    if not intervals:
        return []
    by_key: dict[str, list[TimelineInterval]] = {}
    for item in intervals:
        by_key.setdefault(item.timeline, []).append(item)
    merged: list[TimelineInterval] = []
    for timeline, group in by_key.items():
        ordered = sorted(group, key=lambda x: (x.start, x.end))
        cur_start = ordered[0].start
        cur_end = ordered[0].end
        cur_reason = ordered[0].reason
        for item in ordered[1:]:
            if item.start <= cur_end + 1e-6:
                cur_end = max(cur_end, item.end)
                if item.reason != cur_reason and (
                    "curve" in item.reason or "activity" in item.reason
                ):
                    cur_reason = item.reason
            else:
                merged.append(
                    TimelineInterval(
                        start=cur_start,
                        end=cur_end,
                        timeline=timeline,
                        reason=cur_reason,
                    )
                )
                cur_start, cur_end, cur_reason = item.start, item.end, item.reason
        merged.append(
            TimelineInterval(
                start=cur_start,
                end=cur_end,
                timeline=timeline,
                reason=cur_reason,
            )
        )
    return sorted(merged, key=lambda x: (x.timeline, x.start))


def donor_time_in_intervals(donor_time: float, intervals: list[TimelineInterval]) -> bool:
    """True if ``donor_time`` falls inside any donor-timeline interval."""
    for item in intervals:
        if item.timeline != "donor":
            continue
        if item.start - 1e-9 <= donor_time <= item.end + 1e-9:
            return True
    return False


def complementary_donor_intervals(
    donor_duration: float,
    unrelated: list[TimelineInterval],
) -> list[tuple[float, float]]:
    """Return donor keep-ranges outside unrelated spans."""
    excludes = sorted(
        (item.start, item.end)
        for item in unrelated
        if item.timeline == "donor" and item.end > item.start
    )
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in excludes:
        if start > cursor:
            keep.append((cursor, min(start, donor_duration)))
        cursor = max(cursor, end)
    if donor_duration > cursor:
        keep.append((cursor, donor_duration))
    return [(a, b) for a, b in keep if b - a > 1e-6]


def slice_envelope(
    envelope: ActivityEnvelope,
    start: float,
    end: float,
) -> ActivityEnvelope | None:
    """Extract the envelope samples inside ``[start, end]``."""
    import numpy as np

    if envelope.times.size == 0 or end <= start:
        return None
    mask = (envelope.times >= start) & (envelope.times <= end)
    times = envelope.times[mask]
    values = envelope.values[mask]
    if times.size < 4:
        return None
    return ActivityEnvelope(
        times=np.asarray(times, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        sample_period=envelope.sample_period,
    )


def split_segment_excluding_donor(
    segment: AlignmentSegment,
    exclude: list[TimelineInterval],
    *,
    min_duration: float = 0.05,
) -> list[AlignmentSegment]:
    """Split ``segment`` so donor ranges overlapping ``exclude`` are dropped."""
    donor_excludes = [
        (item.start, item.end)
        for item in exclude
        if item.timeline == "donor" and item.end > item.start
    ]
    if not donor_excludes:
        return [segment]

    speed = segment.speed if segment.speed > 0 else 1.0
    offset = segment.offset_seconds

    def base_of(donor_t: float) -> float:
        return (donor_t - offset) / speed

    keep: list[tuple[float, float]] = []
    cursor = segment.donor_start
    for ex0, ex1 in sorted(donor_excludes):
        if ex1 <= segment.donor_start or ex0 >= segment.donor_end:
            continue
        left = min(ex0, segment.donor_end)
        if left > cursor + 1e-9:
            keep.append((cursor, min(left, segment.donor_end)))
        cursor = max(cursor, min(ex1, segment.donor_end))
    if segment.donor_end > cursor + 1e-9:
        keep.append((cursor, segment.donor_end))

    out: list[AlignmentSegment] = []
    for d0, d1 in keep:
        if d1 - d0 < min_duration:
            continue
        b0 = base_of(d0)
        b1 = base_of(d1)
        if b1 < b0:
            b0, b1 = b1, b0
        b0 = max(segment.base_start, min(segment.base_end, b0))
        b1 = max(segment.base_start, min(segment.base_end, b1))
        if b1 - b0 < min_duration:
            continue
        out.append(
            AlignmentSegment(
                base_start=float(b0),
                base_end=float(b1),
                donor_start=float(d0),
                donor_end=float(d1),
                speed=speed,
                offset_seconds=offset,
                confidence=segment.confidence,
                reliable=segment.reliable,
            )
        )
    return out


def carve_segments_excluding_donor(
    segments: list[AlignmentSegment],
    exclude: list[TimelineInterval],
    *,
    min_duration: float = 0.05,
) -> list[AlignmentSegment]:
    """Apply unrelated-donor carving to every segment."""
    result: list[AlignmentSegment] = []
    for segment in segments:
        result.extend(
            split_segment_excluding_donor(
                segment, exclude, min_duration=min_duration
            )
        )
    return sorted(result, key=lambda s: s.base_start)


def apply_unrelated_to_document(
    doc: AlignmentDocument,
    intervals: list[TimelineInterval],
    *,
    missing_base: list[TimelineInterval] | None = None,
    min_duration: float = 0.05,
) -> AlignmentDocument:
    """Attach unrelated donor (and optional missing-base) intervals; carve segments."""
    if not intervals and not missing_base:
        return doc
    merged = merge_timeline_intervals(
        list(doc.unrelated_donor_intervals) + list(intervals or [])
    )
    doc.unrelated_donor_intervals = merged
    if missing_base:
        doc.missing_base_intervals = merge_timeline_intervals(
            list(doc.missing_base_intervals) + list(missing_base)
        )
    if merged:
        doc.segments = carve_segments_excluding_donor(
            doc.segments, merged, min_duration=min_duration
        )
        doc.notes = list(doc.notes) + [
            f"Excluded {len(merged)} unrelated donor span(s) via curve walk."
        ]
    doc.evidence = {
        **doc.evidence,
        "unrelated_donor_spans": len(merged),
        "unrelated_donor_seconds": round(
            sum(i.end - i.start for i in merged), 3
        ),
        "missing_base_spans": len(doc.missing_base_intervals),
    }
    return doc
