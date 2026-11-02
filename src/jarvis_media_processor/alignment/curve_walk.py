"""Walk activity envelopes like a human matching derivative curves.

Lock characteristic shapes on the base, find them on the donor with an
expanding offset search, follow while they track (optional mild speed), and
when they diverge locate either surplus donor material (ads) or missing-base
stretches (content only on the base timeline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jarvis_media_processor.alignment.activity_sync import _resample_to_period
from jarvis_media_processor.alignment.models import TimelineInterval
from jarvis_media_processor.video.activity import (
    ActivityEnvelope,
    activity_peak_times,
    normalize_activity,
)


@dataclass(frozen=True)
class CurveLock:
    """Locked linear map ``donor ≈ speed * base + offset`` at a base time."""

    speed: float
    offset_seconds: float
    base_time: float
    donor_time: float
    score: float


@dataclass(frozen=True)
class CurveWalkResult:
    """Spans discovered by walking the activity curves."""

    unrelated_donor_intervals: list[TimelineInterval]
    missing_base_intervals: list[TimelineInterval]
    locks: list[CurveLock]
    notes: list[str]


def _window_values(
    times: NDArray[np.float64],
    values: NDArray[np.float64],
    center: float,
    half: float,
    period: float,
) -> NDArray[np.float64]:
    if times.size == 0 or half <= 0:
        return np.zeros(0, dtype=np.float64)
    grid = np.arange(center - half, center + half + period * 0.5, period, dtype=np.float64)
    return np.interp(grid, times, values, left=0.0, right=0.0)


def local_curve_score(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    donor_time: float,
    speed: float,
    half_window: float,
    period: float,
) -> float:
    """Normalized correlation of local activity shapes around a hypothesized lock."""
    bw = _window_values(base.times, base.values, base_time, half_window, period)
    # Donor window covers the same base-duration span, warped by speed.
    donor_half = half_window * max(speed, 1e-6)
    dw = _window_values(donor.times, donor.values, donor_time, donor_half, period)
    n = min(bw.size, dw.size)
    if n < 4:
        return 0.0
    a = normalize_activity(bw[:n])
    b = normalize_activity(dw[:n])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))


def _expanding_offsets(max_offset: float) -> list[float]:
    """Offsets to try, starting near zero then expanding (both signs)."""
    steps = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 600.0, 1200.0]
    out: list[float] = []
    for step in steps:
        if step > max_offset + 1e-9:
            break
        out.append(step)
        if step > 0:
            out.append(-step)
    if max_offset > 0 and max_offset not in out and -max_offset not in out:
        out.extend([max_offset, -max_offset])
    return out


def _score_lock(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    donor_time: float,
    speed: float,
    half_window: float,
    period: float,
    min_score: float,
) -> CurveLock | None:
    if donor.times.size == 0:
        return None
    if donor_time < float(donor.times[0]) or donor_time > float(donor.times[-1]):
        return None
    if base_time < float(base.times[0]) or base_time > float(base.times[-1]):
        return None
    score = local_curve_score(
        base,
        donor,
        base_time=base_time,
        donor_time=donor_time,
        speed=speed,
        half_window=half_window,
        period=period,
    )
    if score < min_score:
        return None
    return CurveLock(
        speed=float(speed),
        offset_seconds=float(donor_time - speed * base_time),
        base_time=float(base_time),
        donor_time=float(donor_time),
        score=float(score),
    )


def _backwards_refine_lock(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    lock: CurveLock,
    *,
    half_window: float,
    period: float,
    hop: float,
    min_score: float,
) -> CurveLock:
    """Walk backwards from a lock to the earliest still-matching coincidence."""
    speed = lock.speed
    offset = lock.offset_seconds
    base_t = lock.base_time
    best = lock
    t = base_t - hop
    floor = float(base.times[0]) + half_window
    while t >= floor:
        donor_t = speed * t + offset
        scored = _score_lock(
            base,
            donor,
            base_time=t,
            donor_time=donor_t,
            speed=speed,
            half_window=half_window,
            period=period,
            min_score=min_score,
        )
        if scored is None:
            break
        best = scored
        t -= hop
    return best


def find_initial_lock(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    half_window: float = 12.0,
    min_score: float = 0.35,
    max_offset_seconds: float = 1200.0,
    speed_candidates: tuple[float, ...] = (1.0, 0.96, 1.04, 0.92, 1.08),
) -> CurveLock | None:
    """Find the first base↔donor coincidence via chronological curve search.

    One expanding search covers the "happy" alignments as early hits:
    start↔start, donor-head↔later-base (missing open credits), and
    base-start↔later-donor (leading ads). On the first confirmed hit,
    backwards-walk to the exact coincidence start.
    """
    period = float(np.median([base.sample_period, donor.sample_period])) or 1.0
    hop = max(period, 2.0)
    base_start = float(base.times[0]) + half_window
    donor_start = float(donor.times[0]) + half_window
    base_end = float(base.times[-1]) - half_window
    donor_end = float(donor.times[-1]) - half_window
    if base_end <= base_start or donor_end <= donor_start:
        return None

    def _try_and_confirm(lock: CurveLock | None) -> CurveLock | None:
        if lock is None:
            return None
        if not _confirm_lock(
            base,
            donor,
            lock,
            half_window=half_window,
            period=period,
            hop=hop,
            min_score=min_score * 0.85,
            confirm_seconds=max(half_window, 8.0),
        ):
            return None
        return _backwards_refine_lock(
            base,
            donor,
            lock,
            half_window=half_window,
            period=period,
            hop=hop,
            min_score=min_score * 0.8,
        )

    def _best_of(candidates: list[CurveLock]) -> CurveLock | None:
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c.score, -abs(c.speed - 1.0)))

    peaks = activity_peak_times(base, min_prominence=0.12)
    if not peaks:
        bt, bv = _resample_to_period(base, period)
        if bt.size == 0:
            return None
        order = np.argsort(bv)[::-1][:16]
        peaks = [float(bt[i]) for i in order]

    # Dense chronological base samples over the searchable head, then peaks.
    # Density recovers start-aligned / missing-credits locks without a
    # separate special-case pass.
    head_limit = min(base_end, base_start + max_offset_seconds)
    anchors: list[float] = []
    for t in np.arange(base_start, head_limit + hop * 0.5, hop):
        anchors.append(float(t))
    for t in sorted(peaks):
        if head_limit < t <= base_end and (
            not anchors or abs(t - anchors[-1]) >= hop
        ):
            anchors.append(float(t))
    if len(anchors) > 400:
        # Keep the head dense; thin only the far tail.
        keep_head = int(min(base_start + 600.0, head_limit) - base_start) // int(hop) + 1
        keep_head = max(keep_head, 60)
        head = anchors[:keep_head]
        tail = anchors[keep_head :: max(1, (len(anchors) - keep_head) // 80)]
        anchors = head + tail

    radii: list[float] = []
    for step in _expanding_offsets(max_offset_seconds):
        r = abs(step)
        if r not in radii:
            radii.append(r)

    # Ring 0 — donor head across chronological base times (start↔start and
    # missing open credits). Best-per-speed avoids weak early false locks.
    for speed in speed_candidates:
        best: CurveLock | None = None
        for base_t in anchors:
            cand = _score_lock(
                base,
                donor,
                base_time=base_t,
                donor_time=donor_start,
                speed=speed,
                half_window=half_window,
                period=period,
                min_score=min_score,
            )
            if cand is None:
                continue
            if best is None or cand.score > best.score:
                best = cand
            if cand.score >= min_score + 0.1:
                break
        confirmed = _try_and_confirm(best)
        if confirmed is not None:
            return confirmed

    # Later rings — chronological base × expanding offset (leading ads when
    # base_t is early; mid-file coincidences otherwise).
    for base_t in anchors:
        for radius in radii:
            offsets = [0.0] if radius == 0 else [radius, -radius]
            if radius >= 10:
                offsets.extend([radius * 0.5, -radius * 0.5])
            if radius >= 40:
                offsets.extend(
                    [radius * 0.75, -radius * 0.75, radius * 0.25, -radius * 0.25]
                )
            ring_hits: list[CurveLock] = []
            for speed in speed_candidates:
                for offset in offsets:
                    donor_t = speed * base_t + offset
                    if abs(donor_t - donor_start) < hop * 0.5:
                        continue
                    cand = _score_lock(
                        base,
                        donor,
                        base_time=base_t,
                        donor_time=donor_t,
                        speed=speed,
                        half_window=half_window,
                        period=period,
                        min_score=min_score,
                    )
                    if cand is not None:
                        ring_hits.append(cand)
            confirmed = _try_and_confirm(_best_of(ring_hits))
            if confirmed is not None:
                return confirmed
    return None


def _confirm_lock(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    lock: CurveLock,
    *,
    half_window: float,
    period: float,
    hop: float,
    min_score: float,
    confirm_seconds: float,
) -> bool:
    """Require the lock to keep matching while walking a short stretch."""
    t = lock.base_time
    end = min(float(base.times[-1]) - half_window, t + confirm_seconds)
    good = 0.0
    while t <= end:
        donor_t = lock.speed * t + lock.offset_seconds
        if donor_t < float(donor.times[0]) or donor_t > float(donor.times[-1]):
            return False
        score = local_curve_score(
            base,
            donor,
            base_time=t,
            donor_time=donor_t,
            speed=lock.speed,
            half_window=half_window,
            period=period,
        )
        if score < min_score:
            return False
        good += hop
        t += hop
    return good >= confirm_seconds * 0.5


def walk_activity_curves(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    half_window: float = 12.0,
    hop_seconds: float = 2.0,
    min_score: float = 0.3,
    break_score: float = 0.22,
    confirm_seconds: float = 6.0,
    min_surplus_seconds: float = 8.0,
    max_surplus_seconds: float = 1200.0,
    missing_base_fill_seconds: float = 3.0,
    max_offset_seconds: float = 1200.0,
    speed_candidates: tuple[float, ...] = (1.0, 0.96, 1.04, 0.92, 1.08),
) -> CurveWalkResult:
    """Follow base→donor activity curves; emit surplus-donor and missing-base spans.

    Human-inspired loop:
    1. Lock a characteristic base shape on the donor (expanding offset).
    2. Walk forward while local curves keep matching.
    3. On a clear break, search forward on the donor for where the base curve
       resumes → unrelated donor interval (ad / surplus).
    4. If the donor does not resume but a later base time rematches at the same
       donor point → missing-base interval (fill from base audio later).
    """
    notes: list[str] = []
    if base.times.size < 8 or donor.times.size < 8:
        return CurveWalkResult([], [], [], ["Curve walk skipped: envelopes too short."])

    period = float(np.median([base.sample_period, donor.sample_period])) or 1.0
    lock = find_initial_lock(
        base,
        donor,
        half_window=half_window,
        min_score=min_score,
        max_offset_seconds=max_offset_seconds,
        speed_candidates=speed_candidates,
    )
    if lock is None:
        return CurveWalkResult(
            [],
            [],
            [],
            ["Curve walk: no initial lock found."],
        )

    notes.append(
        f"Curve lock speed={lock.speed:.4f} offset={lock.offset_seconds:.3f}s "
        f"at base={lock.base_time:.1f}s donor={lock.donor_time:.1f}s "
        f"score={lock.score:.3f}."
    )
    locks = [lock]
    unrelated: list[TimelineInterval] = []
    missing_base: list[TimelineInterval] = []

    # Head gaps implied by the first coincidence (after backwards refine).
    head_base = float(base.times[0])
    head_donor = float(donor.times[0])
    if lock.base_time - head_base >= missing_base_fill_seconds:
        missing_base.append(
            TimelineInterval(
                start=head_base,
                end=float(lock.base_time),
                timeline="base",
                reason="leading_missing_base",
            )
        )
        notes.append(
            f"Leading missing base 0-{lock.base_time:.1f}s "
            "(donor starts matching later; fill from base track 0)."
        )
    if lock.donor_time - head_donor >= min_surplus_seconds:
        unrelated.append(
            TimelineInterval(
                start=head_donor,
                end=float(lock.donor_time),
                timeline="donor",
                reason="leading_surplus_donor",
            )
        )
        notes.append(
            f"Leading surplus donor 0-{lock.donor_time:.1f}s "
            "(ads / junk before first match)."
        )

    speed = lock.speed
    offset = lock.offset_seconds
    base_t = lock.base_time
    base_end = float(base.times[-1]) - half_window
    donor_end = float(donor.times[-1])
    donor_start = float(donor.times[0])
    bad_run = 0.0
    hop = max(hop_seconds, period)

    while base_t <= base_end:
        donor_t = speed * base_t + offset
        if donor_t < donor_start - 1 or donor_t > donor_end + 1:
            # Ran off the donor — treat remaining base as missing if long enough.
            if base_end - base_t >= missing_base_fill_seconds:
                missing_base.append(
                    TimelineInterval(
                        start=float(base_t),
                        end=float(base_end),
                        timeline="base",
                        reason="missing_base_beyond_donor",
                    )
                )
            break

        score = local_curve_score(
            base,
            donor,
            base_time=base_t,
            donor_time=donor_t,
            speed=speed,
            half_window=half_window,
            period=period,
        )
        if score >= break_score:
            bad_run = 0.0
            # Mild local offset refine.
            if score >= min_score:
                best_off = offset
                best_sc = score
                for delta in (-hop, 0.0, hop):
                    trial_off = offset + delta
                    trial_d = speed * base_t + trial_off
                    sc = local_curve_score(
                        base,
                        donor,
                        base_time=base_t,
                        donor_time=trial_d,
                        speed=speed,
                        half_window=half_window,
                        period=period,
                    )
                    if sc > best_sc:
                        best_sc = sc
                        best_off = trial_off
                offset = best_off
            base_t += hop
            continue

        bad_run += hop
        if bad_run < confirm_seconds:
            base_t += hop
            continue

        # Confirmed break.
        break_base = base_t - bad_run
        break_donor = speed * break_base + offset
        break_donor = max(donor_start, min(donor_end, break_donor))

        # Candidate A — surplus on donor: same base time, donor jumps ahead.
        surplus = None
        surplus_anchor = break_base
        for anchor in (break_base, max(half_window, break_base - hop), break_base + hop):
            if anchor < half_window or anchor > base_end:
                continue
            cand = _search_donor_resume(
                base,
                donor,
                base_time=float(anchor),
                donor_search_start=break_donor + min_surplus_seconds,
                speed=speed,
                half_window=half_window,
                period=period,
                min_score=min_score,
                max_ahead=max_surplus_seconds,
                hop=max(hop, 1.0),
            )
            if cand is None:
                continue
            surplus_end, resume_score = cand
            if surplus_end - break_donor < min_surplus_seconds:
                continue
            surplus = cand
            surplus_anchor = float(anchor)
            break

        # Candidate B — missing on base: same donor time, base jumps ahead.
        missing = _search_base_resume(
            base,
            donor,
            donor_time=break_donor,
            base_search_start=break_base + missing_base_fill_seconds,
            speed=speed,
            offset=offset,
            half_window=half_window,
            period=period,
            min_score=min_score,
            max_ahead=max_surplus_seconds,
            hop=hop,
        )
        if missing is not None and missing[0] - break_base < missing_base_fill_seconds:
            missing = None

        choose_missing = False
        if missing is not None and surplus is not None:
            miss_len = missing[0] - break_base
            sur_len = surplus[0] - break_donor
            # Ads jump the donor forward while base waits → prefer surplus unless
            # missing-base clearly fits better (stronger score and shorter gap).
            if missing[1] > surplus[1] + 0.1 and miss_len < sur_len * 0.75:
                choose_missing = True
            else:
                choose_missing = False
        elif missing is not None:
            choose_missing = True

        if choose_missing and missing is not None:
            resume_base, resume_score = missing
            missing_base.append(
                TimelineInterval(
                    start=float(break_base),
                    end=float(resume_base),
                    timeline="base",
                    reason="curve_break_missing_base",
                )
            )
            notes.append(
                f"Missing base {break_base:.1f}-{resume_base:.1f}s "
                f"(resume score={resume_score:.3f})."
            )
            bad_run = 0.0
            base_t = resume_base + hop
            continue

        if surplus is not None:
            surplus_end, resume_score = surplus
            unrelated.append(
                TimelineInterval(
                    start=float(break_donor),
                    end=float(surplus_end),
                    timeline="donor",
                    reason="curve_break_surplus_donor",
                )
            )
            offset = surplus_end - speed * surplus_anchor
            locks.append(
                CurveLock(
                    speed=speed,
                    offset_seconds=offset,
                    base_time=surplus_anchor,
                    donor_time=surplus_end,
                    score=resume_score,
                )
            )
            notes.append(
                f"Surplus donor {break_donor:.1f}-{surplus_end:.1f}s "
                f"(resume score={resume_score:.3f})."
            )
            bad_run = 0.0
            base_t = surplus_anchor + hop
            continue

        # Neither resume found — stop walking to avoid inventing mappings.
        notes.append(
            f"Curve walk stopped at base={break_base:.1f}s donor≈{break_donor:.1f}s "
            "(no resume found)."
        )
        break

    return CurveWalkResult(
        unrelated_donor_intervals=unrelated,
        missing_base_intervals=missing_base,
        locks=locks,
        notes=notes,
    )


def _search_donor_resume(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    donor_search_start: float,
    speed: float,
    half_window: float,
    period: float,
    min_score: float,
    max_ahead: float,
    hop: float,
) -> tuple[float, float] | None:
    """Search forward on the donor for the base curve at ``base_time``."""
    donor_end = float(donor.times[-1])
    t = max(float(donor_search_start), float(donor.times[0]))
    limit = min(donor_end, t + max_ahead)
    best_t: float | None = None
    best_score = 0.0
    while t <= limit:
        score = local_curve_score(
            base,
            donor,
            base_time=base_time,
            donor_time=t,
            speed=speed,
            half_window=half_window,
            period=period,
        )
        if score > best_score:
            best_score = score
            best_t = t
        if score >= min_score + 0.05:
            return t, score
        t += hop
    if best_t is not None and best_score >= min_score:
        return best_t, best_score
    return None


def _search_base_resume(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    donor_time: float,
    base_search_start: float,
    speed: float,
    offset: float,
    half_window: float,
    period: float,
    min_score: float,
    max_ahead: float,
    hop: float,
) -> tuple[float, float] | None:
    """Advance on the base until it matches again near ``donor_time``."""
    base_end = float(base.times[-1])
    t = max(float(base_search_start), float(base.times[0]))
    limit = min(base_end, t + max_ahead)
    # Allow a small donor nudge while searching.
    for base_t in np.arange(t, limit + hop * 0.5, hop):
        for delta in (-2 * hop, 0.0, 2 * hop):
            d_t = donor_time + delta
            score = local_curve_score(
                base,
                donor,
                base_time=float(base_t),
                donor_time=d_t,
                speed=speed,
                half_window=half_window,
                period=period,
            )
            if score >= min_score:
                return float(base_t), float(score)
        # Also try the predicted donor from the old offset at this new base time
        # — if it suddenly scores well, the gap was noise not a true miss.
        pred = speed * float(base_t) + offset
        score_pred = local_curve_score(
            base,
            donor,
            base_time=float(base_t),
            donor_time=pred,
            speed=speed,
            half_window=half_window,
            period=period,
        )
        if score_pred >= min_score:
            return float(base_t), float(score_pred)
    return None
