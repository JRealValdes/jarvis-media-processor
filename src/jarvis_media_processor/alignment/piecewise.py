"""Piecewise temporal mapping and discontinuity detection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from jarvis_media_processor.alignment.evidence import TimePair
from jarvis_media_processor.alignment.models import (
    AlignmentDocument,
    AlignmentSegment,
    SyncStrategyLevel,
    TimelineInterval,
)


def select_monotonic_correspondences(pairs: list[TimePair]) -> list[TimePair]:
    """Reduce noisy candidate pairs to a monotonic base→donor chain.

    For each distinct base time, pick a donor time that keeps donor times
    non-decreasing and prefers offset continuity with the previous choice.
    """
    by_base: dict[float, list[TimePair]] = defaultdict(list)
    for pair in pairs:
        by_base[round(pair.base_time, 3)].append(pair)

    selected: list[TimePair] = []
    last_donor = -1.0
    last_offset: float | None = None
    for base_time in sorted(by_base):
        candidates = by_base[base_time]
        valid = [c for c in candidates if c.donor_time + 1e-9 >= last_donor]
        pool = valid or candidates
        if last_offset is None:
            choice = max(pool, key=lambda c: c.weight)
        else:
            target_offset = float(last_offset)
            choice = min(
                pool,
                key=lambda c: (
                    abs((c.donor_time - c.base_time) - target_offset),
                    -c.weight,
                ),
            )
        selected.append(choice)
        last_donor = choice.donor_time
        last_offset = choice.donor_time - choice.base_time
    return selected


def _cluster_by_offset(
    chain: list[TimePair],
    *,
    offset_tolerance: float,
) -> list[list[TimePair]]:
    if not chain:
        return []
    clusters: list[list[TimePair]] = []
    current = [chain[0]]
    current_offset = chain[0].donor_time - chain[0].base_time
    for pair in chain[1:]:
        offset = pair.donor_time - pair.base_time
        if abs(offset - current_offset) <= offset_tolerance:
            current.append(pair)
            current_offset = float(np.mean([p.donor_time - p.base_time for p in current]))
        else:
            clusters.append(current)
            current = [pair]
            current_offset = offset
    clusters.append(current)
    return clusters


def _segment_from_cluster(cluster: list[TimePair]) -> AlignmentSegment:
    base_times = [p.base_time for p in cluster]
    donor_times = [p.donor_time for p in cluster]
    offset = float(np.mean([d - b for b, d in zip(base_times, donor_times, strict=True)]))
    speed = 1.0
    if len(cluster) >= 2:
        b0, b1 = base_times[0], base_times[-1]
        d0, d1 = donor_times[0], donor_times[-1]
        if abs(b1 - b0) > 1e-6:
            speed = float((d1 - d0) / (b1 - b0))
            if speed <= 0:
                speed = 1.0
            offset = float(d0 - speed * b0)
    conf = min(1.0, 0.45 + 0.2 * len(cluster))
    return AlignmentSegment(
        base_start=min(base_times),
        base_end=max(base_times) + 1e-3,
        donor_start=min(donor_times),
        donor_end=max(donor_times) + 1e-3,
        speed=speed,
        offset_seconds=offset,
        confidence=conf,
        reliable=len(cluster) >= 2 or conf >= 0.6,
    )


def _gaps_on_timeline(
    covered: list[tuple[float, float]],
    duration: float | None,
    *,
    timeline: str,
    min_gap: float,
    reason_prefix: str,
) -> list[TimelineInterval]:
    if duration is None or duration <= 0:
        return []
    intervals: list[TimelineInterval] = []
    ordered = sorted(covered, key=lambda x: x[0])
    # Point samples only cover instants; treat each as a zero-width mark and
    # report gaps between distinct marks / edges.
    marks: list[float] = []
    for start, end in ordered:
        marks.append(start)
        if end - start > 1e-6:
            marks.append(end)
    marks = sorted(set(round(m, 6) for m in marks))
    cursor = 0.0
    for mark in marks:
        if mark - cursor >= min_gap:
            intervals.append(
                TimelineInterval(
                    start=cursor,
                    end=mark,
                    timeline=timeline,
                    reason=f"{reason_prefix}_gap",
                )
            )
        cursor = max(cursor, mark)
    if duration - cursor >= min_gap:
        intervals.append(
            TimelineInterval(
                start=cursor,
                end=duration,
                timeline=timeline,
                reason=f"{reason_prefix}_tail",
            )
        )
    return intervals


def detect_unrelated_donor_intervals(
    segments: list[AlignmentSegment],
    *,
    donor_duration: float | None,
    min_gap: float = 0.25,
) -> list[TimelineInterval]:
    """Flag donor spans skipped between consecutive base-ordered segments."""
    if len(segments) < 2:
        return _gaps_on_timeline(
            [(s.donor_start, s.donor_end) for s in segments],
            donor_duration,
            timeline="donor",
            min_gap=min_gap,
            reason_prefix="unrelated_donor",
        )

    ordered = sorted(segments, key=lambda s: s.base_start)
    unrelated: list[TimelineInterval] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        gap = right.donor_start - left.donor_end
        base_gap = right.base_start - left.base_end
        # Donor jumped ahead more than the base timeline ⇒ unrelated/extra material.
        if gap >= min_gap and gap > base_gap + min_gap:
            unrelated.append(
                TimelineInterval(
                    start=left.donor_end,
                    end=right.donor_start,
                    timeline="donor",
                    reason="unrelated_or_extra_donor_block",
                )
            )
    # Also include uncovered donor head/tail not used by any segment.
    unrelated.extend(
        _gaps_on_timeline(
            [(s.donor_start, s.donor_end) for s in ordered],
            donor_duration,
            timeline="donor",
            min_gap=min_gap,
            reason_prefix="unrelated_donor",
        )
    )
    # Deduplicate overlapping intervals roughly by (start,end,reason)
    uniq: dict[tuple[float, float, str], TimelineInterval] = {}
    for item in unrelated:
        key = (round(item.start, 3), round(item.end, 3), item.reason)
        uniq[key] = item
    return list(uniq.values())


def detect_duplicate_donor_usage(
    segments: list[AlignmentSegment],
    *,
    overlap_tolerance: float = 0.05,
) -> list[TimelineInterval]:
    """Flag overlapping donor intervals claimed by multiple segments."""
    ordered = sorted(segments, key=lambda s: s.donor_start)
    dupes: list[TimelineInterval] = []
    for left, right in zip(ordered, ordered[1:], strict=False):
        overlap = min(left.donor_end, right.donor_end) - max(left.donor_start, right.donor_start)
        if overlap > overlap_tolerance:
            dupes.append(
                TimelineInterval(
                    start=max(left.donor_start, right.donor_start),
                    end=min(left.donor_end, right.donor_end),
                    timeline="donor",
                    reason="duplicate_donor_region",
                )
            )
    return dupes


def build_piecewise_alignment(
    pairs: list[TimePair],
    *,
    base_duration: float | None = None,
    donor_duration: float | None = None,
    offset_tolerance: float = 0.25,
    min_gap: float = 0.25,
    notes: list[str] | None = None,
) -> AlignmentDocument:
    """Build a piecewise alignment with gaps/unrelated/duplicate flags.

    Does not invent coverage across large uncertain regions.
    """
    chain = select_monotonic_correspondences(pairs)
    notes = list(notes or [])
    notes.append("Piecewise mapping from monotonic landmark/video correspondences.")

    if not chain:
        return AlignmentDocument(
            strategy=SyncStrategyLevel.PIECEWISE,
            confidence=0.0,
            uncertain_intervals=[
                TimelineInterval(
                    start=0.0,
                    end=base_duration or 0.0,
                    timeline="base",
                    reason="no_monotonic_correspondences",
                )
            ],
            notes=notes + ["No monotonic correspondences available."],
            evidence={"pair_count": len(pairs), "chain_len": 0},
        )

    clusters = _cluster_by_offset(chain, offset_tolerance=offset_tolerance)
    segments = [_segment_from_cluster(cluster) for cluster in clusters]

    missing_base = _gaps_on_timeline(
        [(s.base_start, s.base_end) for s in segments],
        base_duration,
        timeline="base",
        min_gap=min_gap,
        reason_prefix="missing_base",
    )
    uncertain = [
        TimelineInterval(
            start=item.start,
            end=item.end,
            timeline="base",
            reason="uncertain_uncovered_base",
        )
        for item in missing_base
    ]
    unrelated = detect_unrelated_donor_intervals(
        segments,
        donor_duration=donor_duration,
        min_gap=min_gap,
    )
    duplicates = detect_duplicate_donor_usage(segments)

    # Offset discontinuities between adjacent segments
    discontinuities: list[TimelineInterval] = []
    ordered = sorted(segments, key=lambda s: s.base_start)
    for left, right in zip(ordered, ordered[1:], strict=False):
        delta = abs(right.offset_seconds - left.offset_seconds)
        if delta > offset_tolerance:
            discontinuities.append(
                TimelineInterval(
                    start=left.base_end,
                    end=right.base_start,
                    timeline="base",
                    reason=f"offset_discontinuity_{delta:.3f}s",
                )
            )

    conf = float(np.mean([s.confidence for s in segments])) if segments else 0.0
    return AlignmentDocument(
        strategy=SyncStrategyLevel.PIECEWISE,
        confidence=conf,
        segments=segments,
        uncertain_intervals=uncertain + discontinuities,
        missing_base_intervals=missing_base,
        unrelated_donor_intervals=unrelated,
        notes=notes
        + [
            f"segments={len(segments)}",
            f"unrelated_donor={len(unrelated)}",
            f"missing_base={len(missing_base)}",
            f"duplicates={len(duplicates)}",
        ],
        evidence={
            "pair_count": len(pairs),
            "chain_len": len(chain),
            "segment_count": len(segments),
            "duplicate_count": len(duplicates),
        },
    )
