"""Unit tests for post-reconstruct piecewise audio fine-warp."""

from __future__ import annotations

import numpy as np

from jarvis_media_processor.reconstruct.fine_warp import (
    AnchorMatch,
    FineWarpSegment,
    apply_fine_warps,
    filter_coherent_anchors,
    match_onset_anchors,
    plan_and_apply_fine_warp,
    plan_fine_warps,
)


def _bursts(
    sample_rate: int,
    times: list[float],
    *,
    duration: float,
    width: float = 0.03,
    amp: float = 0.9,
) -> np.ndarray:
    n = int(round(duration * sample_rate))
    out = np.zeros(n, dtype=np.float64)
    half = max(1, int(round(width * sample_rate)))
    for t in times:
        center = int(round(t * sample_rate))
        a = max(0, center - half // 2)
        b = min(n, a + half)
        if b > a:
            out[a:b] = amp
    return out


def test_plan_accepts_mild_speed_skips_coarse() -> None:
    anchors = [
        AnchorMatch(1.0, 1.02, 0.02, 0.9),
        AnchorMatch(2.0, 2.03, 0.03, 0.9),  # speed ≈ 1.01
        AnchorMatch(3.0, 3.10, 0.10, 0.9),  # speed from 2→3 ≈ 1.07 (coarse)
        AnchorMatch(4.0, 4.11, 0.11, 0.9),
    ]
    plan = plan_fine_warps(
        anchors,
        min_speed=0.97,
        max_speed=1.03,
        min_span_seconds=0.4,
        max_span_seconds=5.0,
        max_lag_seconds=0.15,
        max_lag_jump_seconds=0.12,
    )
    assert any(abs(s.base_start - 1.0) < 1e-9 for s in plan.segments)
    assert any("content-diff" in n for n in plan.skipped_notes)
    assert not any(abs(s.base_start - 2.0) < 1e-9 for s in plan.segments)


def test_plan_skips_long_spans() -> None:
    anchors = [
        AnchorMatch(10.0, 10.04, 0.04, 0.9),
        AnchorMatch(20.0, 20.04, 0.04, 0.9),  # 10s apart
    ]
    plan = plan_fine_warps(
        anchors,
        max_span_seconds=4.0,
        max_lag_jump_seconds=0.05,
    )
    assert plan.segments == []
    assert any("long span" in n for n in plan.skipped_notes)


def test_filter_drops_incoherent_lag_outlier() -> None:
    anchors = [
        AnchorMatch(1.0, 1.03, 0.03, 0.9),
        AnchorMatch(2.0, 2.14, 0.14, 0.9),  # outlier
        AnchorMatch(3.0, 3.04, 0.04, 0.9),
    ]
    kept, notes = filter_coherent_anchors(anchors, max_lag_jump_seconds=0.05)
    assert len(kept) == 2
    assert all(abs(a.base_time - 2.0) > 1e-9 for a in kept)
    assert any("incoherent" in n for n in notes)


def test_apply_constant_lag_pulls_onset_earlier() -> None:
    sr = 8_000
    recon = _bursts(sr, [1.05], duration=2.5)
    segments = [
        FineWarpSegment(
            base_start=0.5,
            base_end=1.5,
            recon_start=0.55,
            recon_end=1.55,
            speed=1.0,
            score=1.0,
        )
    ]
    warped = apply_fine_warps(recon, sr, segments)
    peak_i = int(np.argmax(warped))
    peak_t = peak_i / sr
    assert abs(peak_t - 1.0) < 0.03


def test_match_anchors_recovers_known_lag() -> None:
    sr = 8_000
    times = [0.8, 1.6, 2.4, 3.2]
    base = _bursts(sr, times, duration=4.0)
    lag = 0.06
    recon = _bursts(sr, [t + lag for t in times], duration=4.0)
    matches = match_onset_anchors(
        base,
        recon,
        sr,
        max_lag_seconds=0.12,
        window_seconds=0.4,
        min_score=0.25,
        min_score_margin=0.02,
        min_distance_seconds=0.4,
        prominence=0.2,
        max_landmarks=20,
    )
    assert len(matches) >= 3
    median_lag = float(np.median([m.lag_seconds for m in matches]))
    assert abs(median_lag - lag) < 0.025


def test_plan_and_apply_reduces_residual_lag() -> None:
    sr = 8_000
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    base = _bursts(sr, times, duration=6.0)
    lag = 0.08
    recon = _bursts(sr, [t + lag for t in times], duration=6.0)
    warped, plan = plan_and_apply_fine_warp(
        base,
        recon,
        sr,
        max_lag_seconds=0.15,
        window_seconds=0.45,
        min_score=0.25,
        min_score_margin=0.02,
        min_speed=0.97,
        max_speed=1.03,
        min_span_seconds=0.5,
        max_span_seconds=5.0,
        max_lag_jump_seconds=0.08,
        prominence=0.2,
        min_distance_seconds=0.5,
    )
    assert plan.anchors
    assert plan.segments
    assert max(s.base_end - s.base_start for s in plan.segments) <= 5.0 + 1e-9

    def _centroid_near(signal: np.ndarray, t: float, radius: float = 0.2) -> float:
        lo = max(0, int((t - radius) * sr))
        hi = min(signal.size, int((t + radius) * sr))
        seg = np.abs(np.asarray(signal[lo:hi], dtype=np.float64))
        if float(seg.sum()) < 1e-9:
            return t
        idx = lo + int(round(float(np.average(np.arange(seg.size), weights=seg))))
        return idx / sr

    err_before = [abs(_centroid_near(recon, t) - t) for t in times]
    err_after = [abs(_centroid_near(np.asarray(warped), t) - t) for t in times]
    assert float(np.mean(err_after)) < float(np.mean(err_before)) * 0.5
    assert float(np.mean(err_after)) < 0.03


def test_apply_identity_when_no_segments() -> None:
    sr = 4_000
    recon = np.linspace(-0.2, 0.2, sr, dtype=np.float64)
    out = apply_fine_warps(recon, sr, [])
    assert np.allclose(out, recon)
