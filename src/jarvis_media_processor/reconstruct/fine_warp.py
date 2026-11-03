"""Piecewise fine audio warp: reconstructed donor vs base track 0.

After coarse reconstruct, residual lag (tens of ms) can remain. This pass
matches strong energy onsets and applies only short, mild local warps.
Uncertain matches are skipped — never stretched across long spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jarvis_media_processor.audio.landmarks import detect_energy_landmarks, rms_envelope
from jarvis_media_processor.video.activity import normalize_activity

# Block size when remapping long PCM (samples per channel).
_WARP_BLOCK = 1_000_000


@dataclass(frozen=True)
class AnchorMatch:
    """A base onset paired with a nearby reconstructed onset."""

    base_time: float
    recon_time: float
    lag_seconds: float
    score: float


@dataclass(frozen=True)
class FineWarpSegment:
    """Map reconstructed ``[recon_start, recon_end]`` onto ``[base_start, base_end]``."""

    base_start: float
    base_end: float
    recon_start: float
    recon_end: float
    speed: float
    score: float


@dataclass
class FineWarpPlan:
    """Planned fine warps plus diagnostics."""

    anchors: list[AnchorMatch] = field(default_factory=list)
    segments: list[FineWarpSegment] = field(default_factory=list)
    skipped_notes: list[str] = field(default_factory=list)


def _local_lag(
    base_env: NDArray[np.floating],
    recon_env: NDArray[np.floating],
    *,
    hop: float,
    center_seconds: float,
    window_seconds: float,
    max_lag_seconds: float,
) -> tuple[float, float, float]:
    """Return ``(best_lag, best_score, zero_lag_score)``.

    ``lag`` is recon_time − base_time at the best match (positive ⇒ recon late).
    """
    if hop <= 0 or base_env.size < 4 or recon_env.size < 4:
        return 0.0, 0.0, 0.0
    half_w = max(1, int(round((window_seconds * 0.5) / hop)))
    center_i = int(round(center_seconds / hop))
    b0 = max(0, center_i - half_w)
    b1 = min(int(base_env.size), center_i + half_w + 1)
    if b1 - b0 < 4:
        return 0.0, 0.0, 0.0
    template = normalize_activity(np.asarray(base_env[b0:b1], dtype=np.float64))
    if float(np.max(np.abs(template))) < 1e-12:
        return 0.0, 0.0, 0.0

    max_lag_hops = max(1, int(round(max_lag_seconds / hop)))
    best_lag = 0.0
    best_score = -1.0
    zero_score = -1.0
    for lag_h in range(-max_lag_hops, max_lag_hops + 1):
        r_center = center_i + lag_h
        r0 = r_center - half_w
        r1 = r_center + half_w + 1
        if r0 < 0 or r1 > int(recon_env.size):
            continue
        cand = normalize_activity(np.asarray(recon_env[r0:r1], dtype=np.float64))
        if cand.size != template.size:
            continue
        # Both sides are z-scored → mean product ≈ Pearson correlation.
        score = float(np.mean(template * cand))
        if lag_h == 0:
            zero_score = score
        if score > best_score:
            best_score = score
            best_lag = lag_h * hop
    if best_score < 0:
        return 0.0, 0.0, 0.0
    if zero_score < 0:
        # Zero lag was out of bounds (shouldn't happen mid-timeline).
        zero_score = 0.0
    return best_lag, best_score, zero_score


def match_onset_anchors(
    base_samples: NDArray[np.floating],
    recon_samples: NDArray[np.floating],
    sample_rate: int,
    *,
    max_lag_seconds: float = 0.12,
    window_seconds: float = 0.45,
    min_score: float = 0.55,
    min_score_margin: float = 0.08,
    hop_seconds: float = 0.01,
    frame_seconds: float = 0.04,
    min_distance_seconds: float = 0.75,
    prominence: float = 0.28,
    max_landmarks: int = 300,
) -> list[AnchorMatch]:
    """Pair strong base energy landmarks with local lags on the reconstructed track.

    Rejects weak correlations, search-edge saturations, and peaks that do not beat
    lag=0 by ``min_score_margin`` (avoids picking the wrong nearby dialogue line).
    """
    base = np.asarray(base_samples, dtype=np.float64).reshape(-1)
    recon = np.asarray(recon_samples, dtype=np.float64).reshape(-1)
    landmarks, _, _ = detect_energy_landmarks(
        base,
        sample_rate,
        hop_seconds=hop_seconds,
        frame_seconds=frame_seconds,
        min_distance_seconds=min_distance_seconds,
        prominence=prominence,
        max_landmarks=max_landmarks,
    )
    base_env, hop = rms_envelope(
        base, sample_rate, hop_seconds=hop_seconds, frame_seconds=frame_seconds
    )
    recon_env, _ = rms_envelope(
        recon, sample_rate, hop_seconds=hop_seconds, frame_seconds=frame_seconds
    )
    hop = float(hop) if hop > 0 else hop_seconds
    # Treat lags within one hop of the search limit as saturated / untrusted.
    edge_eps = max(hop * 0.75, 1e-3)

    matches: list[AnchorMatch] = []
    for lm in landmarks:
        lag, score, zero_score = _local_lag(
            base_env,
            recon_env,
            hop=hop,
            center_seconds=lm.time_seconds,
            window_seconds=window_seconds,
            max_lag_seconds=max_lag_seconds,
        )
        if score < min_score:
            continue
        if score - zero_score < min_score_margin:
            continue
        if abs(lag) > max_lag_seconds - edge_eps:
            continue
        matches.append(
            AnchorMatch(
                base_time=float(lm.time_seconds),
                recon_time=float(lm.time_seconds + lag),
                lag_seconds=float(lag),
                score=float(score),
            )
        )
    matches.sort(key=lambda m: m.base_time)
    return matches


def filter_coherent_anchors(
    anchors: list[AnchorMatch],
    *,
    max_lag_jump_seconds: float = 0.05,
) -> tuple[list[AnchorMatch], list[str]]:
    """Drop interior lag spikes that disagree with agreeing neighbors.

    Endpoints are kept; pairwise ``max_lag_jump_seconds`` still gates spans.
    """
    if len(anchors) < 3:
        return list(anchors), []
    kept: list[AnchorMatch] = []
    notes: list[str] = []
    for i, anchor in enumerate(anchors):
        if i == 0 or i + 1 == len(anchors):
            kept.append(anchor)
            continue
        left = anchors[i - 1].lag_seconds
        right = anchors[i + 1].lag_seconds
        neighbors_agree = abs(left - right) <= max_lag_jump_seconds
        spike = (
            abs(anchor.lag_seconds - left) > max_lag_jump_seconds
            and abs(anchor.lag_seconds - right) > max_lag_jump_seconds
        )
        if neighbors_agree and spike:
            notes.append(
                f"Drop incoherent anchor at {anchor.base_time:.2f}s "
                f"(lag={anchor.lag_seconds:+.3f}s vs neighbors "
                f"{left:+.3f}/{right:+.3f}s)."
            )
            continue
        kept.append(anchor)
    return kept, notes


def plan_fine_warps(
    anchors: list[AnchorMatch],
    *,
    min_speed: float = 0.97,
    max_speed: float = 1.03,
    min_span_seconds: float = 0.4,
    max_span_seconds: float = 4.0,
    max_lag_seconds: float = 0.12,
    max_lag_jump_seconds: float = 0.05,
) -> FineWarpPlan:
    """Turn nearby coherent anchors into short mild stretch/compress segments.

    Spans longer than ``max_span_seconds`` are never warped (avoids multi-minute
    rubber-banding from a pair of dubious endpoints).
    """
    coherent, drop_notes = filter_coherent_anchors(
        anchors, max_lag_jump_seconds=max_lag_jump_seconds
    )
    plan = FineWarpPlan(anchors=list(coherent), skipped_notes=list(drop_notes))
    if len(coherent) < 2:
        if coherent:
            plan.skipped_notes.append(
                "Fewer than 2 coherent onset anchors; no piecewise fine-warp segments."
            )
        else:
            plan.skipped_notes.append("No coherent onset anchors for fine-warp.")
        return plan

    for left, right in zip(coherent, coherent[1:], strict=False):
        base_span = right.base_time - left.base_time
        recon_span = right.recon_time - left.recon_time
        if base_span < min_span_seconds:
            plan.skipped_notes.append(
                f"Skip short span {left.base_time:.2f}–{right.base_time:.2f}s "
                f"(<{min_span_seconds:.2f}s)."
            )
            continue
        if base_span > max_span_seconds:
            plan.skipped_notes.append(
                f"Skip long span {left.base_time:.2f}–{right.base_time:.2f}s "
                f"(>{max_span_seconds:.2f}s; refuse multi-minute rubber-band)."
            )
            continue
        if recon_span <= 1e-6:
            plan.skipped_notes.append(
                f"Skip non-positive recon span at base {left.base_time:.2f}–"
                f"{right.base_time:.2f}s."
            )
            continue
        if abs(left.lag_seconds) > max_lag_seconds + 1e-9:
            plan.skipped_notes.append(
                f"Skip span {left.base_time:.2f}–{right.base_time:.2f}s "
                f"(left lag {left.lag_seconds:+.3f}s exceeds ±{max_lag_seconds:.3f}s)."
            )
            continue
        if abs(right.lag_seconds) > max_lag_seconds + 1e-9:
            plan.skipped_notes.append(
                f"Skip span {left.base_time:.2f}–{right.base_time:.2f}s "
                f"(right lag {right.lag_seconds:+.3f}s exceeds ±{max_lag_seconds:.3f}s)."
            )
            continue
        if abs(right.lag_seconds - left.lag_seconds) > max_lag_jump_seconds + 1e-9:
            plan.skipped_notes.append(
                f"Skip lag-jump span {left.base_time:.2f}–{right.base_time:.2f}s "
                f"(Δlag={right.lag_seconds - left.lag_seconds:+.3f}s)."
            )
            continue
        speed = recon_span / base_span
        if speed < min_speed or speed > max_speed:
            plan.skipped_notes.append(
                f"Skip content-diff span {left.base_time:.2f}–{right.base_time:.2f}s "
                f"(implied speed={speed:.4f} outside [{min_speed:.3f},{max_speed:.3f}])."
            )
            continue
        # Near-identity: skip to avoid needless resampling noise.
        if (
            abs(speed - 1.0) < 1e-4
            and abs(left.lag_seconds) < 1e-4
            and abs(right.lag_seconds) < 1e-4
        ):
            continue
        plan.segments.append(
            FineWarpSegment(
                base_start=left.base_time,
                base_end=right.base_time,
                recon_start=left.recon_time,
                recon_end=right.recon_time,
                speed=float(speed),
                score=float(min(left.score, right.score)),
            )
        )
    if not plan.segments and not plan.skipped_notes:
        plan.skipped_notes.append("Anchors matched but no non-trivial fine warps needed.")
    return plan


def _ensure_2d(samples: NDArray[np.floating]) -> NDArray[np.float64]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return arr


def apply_fine_warps(
    recon_samples: NDArray[np.floating],
    sample_rate: int,
    segments: list[FineWarpSegment],
) -> NDArray[np.floating]:
    """Remap reconstructed audio onto the base timeline with piecewise linear warps.

    Outside planned segments the mapping is identity. Multi-channel input keeps
    channel count; mono may be 1-D or ``(1, n)``.
    """
    if not segments:
        return np.asarray(recon_samples)

    recon = _ensure_2d(recon_samples)
    channels, n = int(recon.shape[0]), int(recon.shape[-1])
    out = np.empty((channels, n), dtype=np.float64)
    # Default identity map: src index == dst index.
    src_index = np.arange(n, dtype=np.float64)

    for seg in segments:
        start_i = max(0, int(round(seg.base_start * sample_rate)))
        end_i = min(n, int(round(seg.base_end * sample_rate)))
        if end_i - start_i < 2:
            continue
        base_span = max(1e-9, seg.base_end - seg.base_start)
        # Linear map base_time → recon_time within the segment.
        t_frac = (
            np.arange(start_i, end_i, dtype=np.float64) / sample_rate - seg.base_start
        ) / base_span
        t_frac = np.clip(t_frac, 0.0, 1.0)
        recon_t = seg.recon_start + t_frac * (seg.recon_end - seg.recon_start)
        src_index[start_i:end_i] = recon_t * sample_rate

    np.clip(src_index, 0.0, float(n - 1), out=src_index)

    for start in range(0, n, _WARP_BLOCK):
        end = min(n, start + _WARP_BLOCK)
        pos = src_index[start:end]
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        frac = pos - i0.astype(np.float64)
        for c in range(channels):
            out[c, start:end] = recon[c, i0] * (1.0 - frac) + recon[c, i1] * frac

    if np.asarray(recon_samples).ndim == 1:
        return out[0]
    return out


def plan_and_apply_fine_warp(
    base_samples: NDArray[np.floating],
    recon_samples: NDArray[np.floating],
    sample_rate: int,
    *,
    max_lag_seconds: float = 0.12,
    window_seconds: float = 0.45,
    min_score: float = 0.55,
    min_score_margin: float = 0.08,
    min_speed: float = 0.97,
    max_speed: float = 1.03,
    min_span_seconds: float = 0.4,
    max_span_seconds: float = 4.0,
    max_lag_jump_seconds: float = 0.05,
    prominence: float = 0.28,
    min_distance_seconds: float = 0.75,
) -> tuple[NDArray[np.floating], FineWarpPlan]:
    """Match strong onsets, plan short mild warps, and return warped audio."""
    base_mono = np.asarray(base_samples, dtype=np.float64)
    if base_mono.ndim == 2:
        base_mono = base_mono.mean(axis=0)
    recon_arr = np.asarray(recon_samples, dtype=np.float64)
    recon_mono = recon_arr.mean(axis=0) if recon_arr.ndim == 2 else recon_arr

    n = min(base_mono.size, recon_mono.size)
    base_mono = base_mono[:n]
    if recon_arr.ndim == 2:
        recon_work = recon_arr[:, :n]
        recon_mono = recon_mono[:n]
    else:
        recon_work = recon_arr[:n]
        recon_mono = recon_work

    anchors = match_onset_anchors(
        base_mono,
        recon_mono,
        sample_rate,
        max_lag_seconds=max_lag_seconds,
        window_seconds=window_seconds,
        min_score=min_score,
        min_score_margin=min_score_margin,
        prominence=prominence,
        min_distance_seconds=min_distance_seconds,
    )
    plan = plan_fine_warps(
        anchors,
        min_speed=min_speed,
        max_speed=max_speed,
        min_span_seconds=min_span_seconds,
        max_span_seconds=max_span_seconds,
        max_lag_seconds=max_lag_seconds,
        max_lag_jump_seconds=max_lag_jump_seconds,
    )
    warped = apply_fine_warps(recon_work, sample_rate, plan.segments)
    return warped, plan
