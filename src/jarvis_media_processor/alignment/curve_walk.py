"""Walk activity envelopes like a human matching derivative curves.

Lock characteristic shapes on the base, find them on the donor with an
expanding offset search, follow while they track (optional mild speed), and
when they diverge locate either surplus donor material (ads) or missing-base
stretches (content only on the base timeline).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from jarvis_media_processor.alignment.activity_sync import _resample_to_period
from jarvis_media_processor.alignment.models import TimelineInterval
from jarvis_media_processor.video.activity import (
    ActivityEnvelope,
    activity_peak_times,
    local_activity_score,
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
    _ = period  # kept for call-site compatibility; period comes from envelopes
    return local_activity_score(
        base,
        donor,
        base_time=base_time,
        donor_time=donor_time,
        speed=speed,
        half_window=half_window,
    )


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
    frame_confirm: Callable[[float, float], bool] | None = None,
) -> CurveLock | None:
    """Find the first base↔donor coincidence via chronological curve search.

    Locks on activity derivatives (cross-encode safe). Frame confirmation for
    picture identity lives in ``match_frames_two_phase``; ``frame_confirm`` is
    accepted for API compatibility but not required to accept a curve lock —
    requiring pHash here preferred late false locks on grade-mismatched pairs.
    """
    _ = frame_confirm
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
            frame_confirm=None,
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

    # Ring 0 — donor head, chronological per speed. Take the *first* base
    # time that confirms (not the global max score): a later high-score false
    # peak (e.g. base=88) must not hide an earlier true lock (base=42→32).
    # Among speeds that confirm, prefer near-1.0 to avoid cascade false ads.
    ring0_hits: list[CurveLock] = []
    for speed in speed_candidates:
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
            confirmed = _try_and_confirm(cand)
            if confirmed is not None:
                ring0_hits.append(confirmed)
                break
    if ring0_hits:
        # Earliest coincidence wins; among ties, prefer speed≈1 and higher score.
        return min(
            ring0_hits,
            key=lambda c: (c.base_time, abs(c.speed - 1.0), -c.score),
        )

    # Later rings — chronological base × expanding offset.
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
    frame_confirm: Callable[[float, float], bool] | None = None,
) -> bool:
    """Require the lock to keep matching while walking a short stretch.

    Uses mean score with a soft floor so a single noisy hop cannot reject a
    otherwise solid near-1.0 speed lock (cross-encode activity is jittery).

    When ``frame_confirm`` is set, also require actual frames to agree on a
    majority of the confirm hops (derivative propose → frame confirm).
    """
    t = lock.base_time
    end = min(float(base.times[-1]) - half_window, t + confirm_seconds)
    scores: list[float] = []
    frame_checks = 0
    frame_hits = 0
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
        # Hard reject only total collapse; mild dips are OK if the mean holds.
        if score < min_score * 0.55:
            return False
        scores.append(score)
        if frame_confirm is not None:
            frame_checks += 1
            if frame_confirm(t, donor_t):
                frame_hits += 1
        t += hop
    if not scores:
        return False
    covered = len(scores) * hop
    if covered < confirm_seconds * 0.5:
        return False
    mean_sc = float(np.mean(scores))
    if mean_sc < min_score or min(scores) < min_score * 0.7:
        return False
    if frame_confirm is not None and frame_checks > 0:
        return frame_hits >= max(1, (frame_checks + 1) // 2)
    return True


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
    frame_confirm: Callable[[float, float], bool] | None = None,
) -> CurveWalkResult:
    """Follow base→donor activity curves; emit surplus-donor and missing-base spans.

    Human-inspired loop:
    1. Lock a characteristic base shape on the donor (expanding offset).
    2. Optionally confirm the lock with actual frames (``frame_confirm``).
    3. Walk forward while local curves keep matching.
    4. On a clear break, search forward on the donor for where the base curve
       resumes → unrelated donor interval (ad / surplus).
    5. If the donor does not resume but a later base time rematches at the same
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
        frame_confirm=frame_confirm,
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
    # Only when the lock itself sits near a timeline head — a mid-file lock
    # must not invent a multi-minute "leading" gap.
    head_base = float(base.times[0])
    head_donor = float(donor.times[0])
    base_floor = head_base + half_window
    donor_floor = head_donor + half_window
    lock_near_base_head = lock.base_time - base_floor <= max_offset_seconds * 0.25
    lock_near_donor_head = lock.donor_time - donor_floor <= half_window * 2
    if (
        lock_near_donor_head
        and lock.base_time - base_floor >= missing_base_fill_seconds
    ):
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
    if (
        lock_near_base_head
        and lock.donor_time - donor_floor >= min_surplus_seconds
    ):
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
    anchor_offset = lock.offset_seconds
    base_t = lock.base_time
    base_end = float(base.times[-1]) - half_window
    donor_end = float(donor.times[-1])
    donor_start = float(donor.times[0])
    bad_run = 0.0
    hop = max(hop_seconds, period)
    max_offset_drift = hop  # keep the lock map; do not accumulate phase slip
    # Adaptive floor: break when score falls well below recent good tracking.
    # Seed from the confirm stretch, not the single lock peak (often low).
    good_ema = max(lock.score, min_score, break_score + 0.15)

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
        # Mild ±hop only, clamped to the last solid lock offset so refine
        # cannot slowly walk away from a good map and invent surplus.
        best_off = offset
        best_sc = score
        for delta in (-hop, 0.0, hop):
            trial_off = offset + delta
            if abs(trial_off - anchor_offset) > max_offset_drift:
                continue
            trial_d = speed * base_t + trial_off
            if trial_d < donor_start - 1 or trial_d > donor_end + 1:
                continue
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
        score = best_sc

        track_floor = max(break_score, good_ema * 0.55)
        if score >= track_floor:
            offset = best_off
            good_ema = 0.85 * good_ema + 0.15 * max(score, good_ema)
            bad_run = 0.0
            base_t += hop
            continue

        bad_run += hop
        if bad_run < confirm_seconds:
            base_t += hop
            continue

        # Confirmed degradation — search surplus from the break origin, not the
        # end of the bad run (keeps gap checks on the pre-jump map).
        break_base = base_t - bad_run
        resume_floor = max(min_score + 0.08, good_ema * 0.75)
        surplus_hit = _find_forward_surplus(
            base,
            donor,
            base_time=break_base,
            offset=offset,
            speed=speed,
            half_window=half_window,
            period=period,
            min_score=resume_floor,
            min_surplus_seconds=min_surplus_seconds,
            max_surplus_seconds=max_surplus_seconds,
            break_score=break_score,
            donor_start=donor_start,
            donor_end=donor_end,
            frame_confirm=frame_confirm,
            confirm_seconds=confirm_seconds,
            hop_seconds=hop,
        )
        # Long weak resumes (~0.45 over minutes) invent false ads; short real
        # black ads can still resume with a moderate score. Multi-minute jumps
        # also need frame agreement when available (film-fragment false resumes).
        if surplus_hit is not None:
            _gs, _ge, _no, _rs = surplus_hit
            _slip = _ge - _gs
            if _slip > 45.0 and _rs < 0.55:
                surplus_hit = None
            elif (
                surplus_hit is not None
                and _slip > 90.0
                and frame_confirm is not None
            ):
                _resume_base = break_base
                _resume_donor = speed * _resume_base + _no
                if not frame_confirm(_resume_base, _resume_donor):
                    surplus_hit = None
        if surplus_hit is not None:
            gap_start, gap_end, new_off, resume_score = surplus_hit
            unrelated.append(
                TimelineInterval(
                    start=float(gap_start),
                    end=float(gap_end),
                    timeline="donor",
                    reason="curve_break_surplus_donor",
                )
            )
            offset = new_off
            anchor_offset = new_off
            locks.append(
                CurveLock(
                    speed=speed,
                    offset_seconds=offset,
                    base_time=break_base,
                    donor_time=speed * break_base + offset,
                    score=resume_score,
                )
            )
            notes.append(
                f"Surplus donor {gap_start:.1f}-{gap_end:.1f}s "
                f"(resume score={resume_score:.3f})."
            )
            good_ema = max(resume_score, min_score)
            bad_run = 0.0
            base_t = break_base + hop
            continue

        # Confirmed break — try missing-base resume, else surplus search.
        break_donor = speed * break_base + offset
        break_donor = max(donor_start, min(donor_end, break_donor))
        resume_min = min_score + 0.05

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
                min_score=resume_min,
                max_ahead=max_surplus_seconds,
                hop=max(hop, 1.0),
            )
            if cand is None:
                continue
            surplus_end, resume_score = cand
            if surplus_end - break_donor < min_surplus_seconds:
                continue
            if not _donor_gap_is_unrelated(
                base,
                donor,
                base_time=float(anchor),
                donor_gap_start=break_donor,
                donor_gap_end=surplus_end,
                speed=speed,
                half_window=half_window,
                period=period,
                break_score=break_score,
            ):
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
            min_score=resume_min,
            max_ahead=max_surplus_seconds,
            hop=hop,
        )
        min_missing = max(missing_base_fill_seconds, confirm_seconds)
        if missing is not None and missing[0] - break_base < min_missing:
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
            good_ema = max(resume_score, min_score)
            bad_run = 0.0
            base_t = resume_base + hop
            continue

        if surplus is not None:
            surplus_end, resume_score = surplus
            _slip2 = surplus_end - break_donor
            if _slip2 > 45.0 and resume_score < 0.55:
                surplus = None
            elif surplus is not None and _slip2 > 90.0 and frame_confirm is not None:
                if not frame_confirm(surplus_anchor, surplus_end):
                    surplus = None
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
            good_ema = max(resume_score, min_score)
            bad_run = 0.0
            base_t = surplus_anchor + hop
            continue

        # Neither resume found — keep walking rather than inventing a span.
        notes.append(
            f"Curve break without resume at base={break_base:.1f}s "
            f"donor≈{break_donor:.1f}s; continuing."
        )
        bad_run = 0.0
        base_t += hop
        continue

    return CurveWalkResult(
        unrelated_donor_intervals=_merge_same_timeline(unrelated),
        missing_base_intervals=_merge_same_timeline(missing_base),
        locks=locks,
        notes=notes,
    )


def _merge_same_timeline(intervals: list[TimelineInterval]) -> list[TimelineInterval]:
    """Merge overlapping/adjacent intervals on the same timeline."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x.timeline, x.start, x.end))
    out: list[TimelineInterval] = []
    cur = ordered[0]
    for item in ordered[1:]:
        if (
            item.timeline == cur.timeline
            and item.start <= cur.end + 1e-6
        ):
            cur = TimelineInterval(
                start=cur.start,
                end=max(cur.end, item.end),
                timeline=cur.timeline,
                reason=cur.reason if cur.reason == item.reason else item.reason,
            )
        else:
            out.append(cur)
            cur = item
    out.append(cur)
    return out


def _find_forward_surplus(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    offset: float,
    speed: float,
    half_window: float,
    period: float,
    min_score: float,
    min_surplus_seconds: float,
    max_surplus_seconds: float,
    break_score: float,
    donor_start: float,
    donor_end: float,
    frame_confirm: Callable[[float, float], bool] | None = None,
    confirm_seconds: float = 6.0,
    hop_seconds: float = 2.0,
) -> tuple[float, float, float, float] | None:
    """If a forward donor jump restores the curve, return surplus span + new offset.

    Returns ``(gap_start, gap_end, new_offset, resume_score)`` or ``None``.

    Mid-roll *film* inserts can produce short false activity peaks; require a
    sustained resume and prefer the longest adequate jump (not the loudest).
    When ``frame_confirm`` is available, prefer candidates that also pass it,
    but fall back to activity-only so cross-encode black ads still resume.
    """
    hop = max(hop_seconds, period)
    confirmed: list[tuple[float, float, float]] = []  # slip, score, trial_off
    for step in _expanding_offsets(max_surplus_seconds):
        if step < min_surplus_seconds:
            continue
        trial_off = offset + step
        trial_d = speed * base_time + trial_off
        if trial_d < donor_start - 1 or trial_d > donor_end + 1:
            continue
        sc = local_curve_score(
            base,
            donor,
            base_time=base_time,
            donor_time=trial_d,
            speed=speed,
            half_window=half_window,
            period=period,
        )
        if sc < min_score:
            continue
        trial_lock = CurveLock(
            speed=speed,
            offset_seconds=float(trial_off),
            base_time=float(base_time),
            donor_time=float(trial_d),
            score=float(sc),
        )
        if not _confirm_lock(
            base,
            donor,
            trial_lock,
            half_window=half_window,
            period=period,
            hop=hop,
            min_score=min_score,
            confirm_seconds=confirm_seconds,
            frame_confirm=None,  # activity sustained first; frames below
        ):
            continue
        confirmed.append((float(step), float(sc), float(trial_off)))
    if not confirmed:
        return None

    pool = confirmed
    if frame_confirm is not None:
        framed: list[tuple[float, float, float]] = []
        for slip, sc, trial_off in confirmed:
            trial_d = speed * base_time + trial_off
            trial_lock = CurveLock(
                speed=speed,
                offset_seconds=trial_off,
                base_time=float(base_time),
                donor_time=float(trial_d),
                score=sc,
            )
            if _confirm_lock(
                base,
                donor,
                trial_lock,
                half_window=half_window,
                period=period,
                hop=hop,
                min_score=min_score,
                confirm_seconds=confirm_seconds,
                frame_confirm=frame_confirm,
            ):
                framed.append((slip, sc, trial_off))
        if framed:
            pool = framed

    # Prefer higher resume score; on ties, the longer jump (real ads outrank
    # brief mid-insert coincidences). Fall back if the gap check rejects one.
    ordered = sorted(pool, key=lambda item: (item[1], item[0]), reverse=True)
    for slip, best_sc, best_off in ordered:
        gap_start = max(donor_start, speed * base_time + offset)
        gap_end = min(donor_end, speed * base_time + best_off)
        if gap_end - gap_start < min_surplus_seconds:
            continue
        if not _donor_gap_is_unrelated(
            base,
            donor,
            base_time=base_time,
            donor_gap_start=gap_start,
            donor_gap_end=gap_end,
            speed=speed,
            half_window=half_window,
            period=period,
            break_score=break_score,
        ):
            continue
        return gap_start, gap_end, best_off, best_sc
    return None


def _donor_gap_is_unrelated(
    base: ActivityEnvelope,
    donor: ActivityEnvelope,
    *,
    base_time: float,
    donor_gap_start: float,
    donor_gap_end: float,
    speed: float,
    half_window: float,
    period: float,
    break_score: float,
) -> bool:
    """Reject surplus spans that still track under the pre-jump offset.

    A false surplus skips real continuing content: the *old* map
    ``donor ≈ speed·base + offset`` still correlates through the interval.
    A real ad / junk block does not.
    """
    span = donor_gap_end - donor_gap_start
    if span <= 0:
        return False
    offset_old = donor_gap_start - speed * base_time
    # Probe the *interior* only — windows that still overlap pre/post content at
    # the gap edges look related even when the middle is a real ad.
    inner0 = donor_gap_start + half_window
    inner1 = donor_gap_end - half_window
    if inner1 <= inner0 + period:
        return True
    inner_span = inner1 - inner0
    # Dense probes — a single lucky zero must not green-light a false surplus.
    # Also try a small offset nudge: mild refine drift must not fake a gap.
    n_probes = max(5, int(inner_span / max(period, 2.0)))
    n_probes = min(n_probes, 24)
    base_lo = float(base.times[0]) + half_window
    base_hi = float(base.times[-1]) - half_window
    hop = max(period, 2.0)

    def _gap_stats(trial_offset: float) -> tuple[float, float] | None:
        scores: list[float] = []
        for i in range(1, n_probes + 1):
            donor_t = inner0 + inner_span * (i / (n_probes + 1))
            pred_base = (float(donor_t) - trial_offset) / max(speed, 1e-6)
            if pred_base < base_lo or pred_base > base_hi:
                continue
            scores.append(
                local_curve_score(
                    base,
                    donor,
                    base_time=pred_base,
                    donor_time=float(donor_t),
                    speed=speed,
                    half_window=half_window,
                    period=period,
                )
            )
        if not scores:
            return None
        mean_sc = float(np.mean(scores))
        hit_frac = sum(1 for s in scores if s >= break_score) / len(scores)
        return mean_sc, hit_frac

    best_mean = 0.0
    best_hits = 0.0
    for trial_off in (offset_old - hop, offset_old, offset_old + hop):
        stats = _gap_stats(trial_off)
        if stats is None:
            continue
        mean_sc, hit_frac = stats
        if mean_sc > best_mean:
            best_mean, best_hits = mean_sc, hit_frac
    if best_mean <= 0.0 and best_hits <= 0.0:
        return True
    # Continued tracking under the old map (±hop) → not unrelated surplus.
    return best_mean < break_score * 0.75 and best_hits <= 0.2


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
    """Search forward on the donor for the base curve at ``base_time``.

    Collects all adequate hits and prefers the *farthest* near-best resume so a
    short mid-ad coincidence cannot beat the real post-ad re-lock.
    """
    donor_end = float(donor.times[-1])
    t = max(float(donor_search_start), float(donor.times[0]))
    limit = min(donor_end, t + max_ahead)
    hits: list[tuple[float, float]] = []
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
        if score >= min_score:
            hits.append((float(t), float(score)))
        t += hop
    if not hits:
        return None
    best_score = max(score for _, score in hits)
    # Near-best scores only; among those prefer the farthest resume so a brief
    # mid-ad coincidence cannot beat the real post-ad re-lock.
    near = [
        (donor_t, score)
        for donor_t, score in hits
        if score >= best_score - 0.08 and score >= min_score
    ]
    if not near:
        near = hits
    donor_t, score = max(near, key=lambda item: (item[0], item[1]))
    return donor_t, score


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
