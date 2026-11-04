"""Render placements onto a silence canvas matching the base duration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from jarvis_media_processor.reconstruct.models import Placement

# FFT-based ``scipy.signal.resample`` OOMs on multi-minute film chunks
# (needs ~O(n) complex buffers). Linear interp in blocks stays O(block).
_STRETCH_BLOCK = 1_000_000


def _slice_samples(
    samples: NDArray[np.floating],
    sample_rate: int,
    start: float,
    end: float,
) -> NDArray[np.float64]:
    start_i = max(0, int(round(start * sample_rate)))
    end_i = min(samples.shape[-1], int(round(end * sample_rate)))
    if end_i <= start_i:
        return np.zeros(0, dtype=np.float64)
    if samples.ndim == 1:
        return np.asarray(samples[start_i:end_i], dtype=np.float64)
    return np.asarray(samples[:, start_i:end_i], dtype=np.float64)


def _ensure_2d(samples: NDArray[np.floating]) -> NDArray[np.float64]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return arr


def _time_stretch(
    chunk: NDArray[np.float64],
    *,
    source_duration: float,
    target_duration: float,
    sample_rate: int,
) -> NDArray[np.float64]:
    """Stretch/compress audio to ``target_duration`` via linear resampling.

    Mild speed changes (e.g. 0.96×) sound fine for reconstruction; avoids the
    memory spike of an FFT resample on hour-scale placements.
    """
    _ = source_duration  # duration implied by chunk length / sample_rate
    target_n = max(1, int(round(target_duration * sample_rate)))
    channels = int(chunk.shape[0])
    source_n = int(chunk.shape[-1])
    if source_n == 0:
        return np.zeros((channels, target_n), dtype=np.float64)
    if source_n == target_n:
        return chunk
    out = np.empty((channels, target_n), dtype=np.float64)
    # pos = index in source for each output sample (linear map).
    scale = float(source_n) / float(target_n)
    for start in range(0, target_n, _STRETCH_BLOCK):
        end = min(target_n, start + _STRETCH_BLOCK)
        pos = (np.arange(start, end, dtype=np.float64) + 0.5) * scale - 0.5
        np.clip(pos, 0.0, float(source_n - 1), out=pos)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, source_n - 1)
        frac = pos - i0.astype(np.float64)
        for c in range(channels):
            out[c, start:end] = chunk[c, i0] * (1.0 - frac) + chunk[c, i1] * frac
    return out


def _apply_edges(
    chunk: NDArray[np.float64],
    *,
    fade_samples: int,
) -> NDArray[np.float64]:
    if fade_samples <= 0 or chunk.shape[-1] == 0:
        return chunk
    fade = min(fade_samples, chunk.shape[-1] // 2)
    if fade <= 0:
        return chunk
    out = chunk.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
    out[:, :fade] *= ramp
    out[:, -fade:] *= ramp[::-1]
    return out


def render_placements(
    donor_samples: NDArray[np.floating],
    sample_rate: int,
    placements: list[Placement],
    *,
    base_duration: float,
) -> NDArray[np.floating]:
    """Mix placements onto a zero canvas of ``base_duration``.

    Missing intervals remain silence and do not shift later content.
    """
    donor = _ensure_2d(donor_samples)
    channels = donor.shape[0]
    total = max(1, int(round(base_duration * sample_rate)))
    # float32 canvas: a 2h stereo 48 kHz float64 buffer is ~5.6 GiB alone.
    canvas = np.zeros((channels, total), dtype=np.float32)

    for placement in placements:
        donor_chunk = _slice_samples(
            donor,
            sample_rate,
            placement.donor_start,
            placement.donor_end,
        )
        donor_chunk = _ensure_2d(donor_chunk)
        source_duration = max(1e-6, placement.donor_end - placement.donor_start)
        target_duration = max(1e-6, placement.base_end - placement.base_start)
        stretched = _time_stretch(
            donor_chunk,
            source_duration=source_duration,
            target_duration=target_duration,
            sample_rate=sample_rate,
        )
        fade = int(round(placement.crossfade_seconds * sample_rate))
        stretched = _apply_edges(stretched, fade_samples=fade)

        start_i = max(0, int(round(placement.base_start * sample_rate)))
        end_i = min(total, start_i + stretched.shape[-1])
        length = end_i - start_i
        if length <= 0:
            continue
        canvas[:, start_i:end_i] += stretched[:, :length].astype(
            np.float32, copy=False
        )

    # Soft clip to [-1, 1]
    peak = float(np.max(np.abs(canvas))) if canvas.size else 0.0
    if peak > 1.0:
        canvas = canvas / np.float32(peak)
    return canvas


def fill_missing_base_audio(
    canvas: NDArray[np.floating],
    base_samples: NDArray[np.floating],
    sample_rate: int,
    intervals: list,
    *,
    min_seconds: float = 3.0,
    crossfade_seconds: float = 0.01,
) -> NDArray[np.float64]:
    """Copy base track audio into long missing-base gaps on the canvas.

    Used when the donor lacks scenes that exist on the base timeline (deleted /
    undubbed / missing credits). Gaps shorter than ``min_seconds`` stay silent.
    """
    out = _ensure_2d(canvas).copy()
    base = _ensure_2d(base_samples)
    # Match channel count (duplicate mono → stereo if needed).
    if base.shape[0] == 1 and out.shape[0] > 1:
        base = np.repeat(base, out.shape[0], axis=0)
    elif base.shape[0] > out.shape[0]:
        base = base[: out.shape[0], :]
    fade = int(round(crossfade_seconds * sample_rate))
    total = out.shape[-1]
    for item in intervals:
        timeline = getattr(item, "timeline", "base")
        if timeline != "base":
            continue
        start = float(getattr(item, "start", 0.0))
        end = float(getattr(item, "end", 0.0))
        if end - start < min_seconds - 1e-9:
            continue
        chunk = _slice_samples(base, sample_rate, start, end)
        chunk = _ensure_2d(chunk)
        if chunk.shape[-1] == 0:
            continue
        chunk = _apply_edges(chunk, fade_samples=fade)
        start_i = max(0, int(round(start * sample_rate)))
        end_i = min(total, start_i + chunk.shape[-1])
        length = end_i - start_i
        if length <= 0:
            continue
        # Never overwrite existing Spanish placements: only fill near-silent
        # samples (false missing residuals used to stomp donor audio with EN).
        dest = out[:, start_i:end_i]
        occupied = np.max(np.abs(dest), axis=0) > 1e-4
        if bool(np.all(occupied)):
            continue
        write = chunk[:, :length].copy()
        if bool(np.any(occupied)):
            write[:, occupied] = dest[:, occupied]
        out[:, start_i:end_i] = write
    peak = np.max(np.abs(out)) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out
