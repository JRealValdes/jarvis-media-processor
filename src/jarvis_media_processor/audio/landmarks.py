"""Acoustic landmark detection from PCM samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from jarvis_media_processor.audio.models import AcousticLandmark


def rms_envelope(
    samples: NDArray[np.floating],
    sample_rate: int,
    *,
    hop_seconds: float = 0.01,
    frame_seconds: float = 0.04,
) -> tuple[NDArray[np.float64], float]:
    """Compute a hop-wise RMS energy envelope.

    Args:
        samples: Mono PCM samples.
        sample_rate: Sample rate in Hz.
        hop_seconds: Hop size in seconds.
        frame_seconds: Frame size in seconds.

    Returns:
        ``(envelope, hop_seconds)`` where envelope is non-negative floats.
    """
    if samples.size == 0:
        return np.zeros(0, dtype=np.float64), hop_seconds

    hop = max(1, int(round(sample_rate * hop_seconds)))
    frame = max(hop, int(round(sample_rate * frame_seconds)))
    if samples.size < frame:
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        return np.array([rms], dtype=np.float64), hop_seconds

    window = np.hanning(frame)
    n_hops = 1 + (samples.size - frame) // hop
    env = np.empty(n_hops, dtype=np.float64)
    for i in range(n_hops):
        start = i * hop
        frame_samples = samples[start : start + frame] * window
        env[i] = float(np.sqrt(np.mean(np.square(frame_samples, dtype=np.float64))))
    return env, hop / sample_rate


def detect_energy_landmarks(
    samples: NDArray[np.floating],
    sample_rate: int,
    *,
    hop_seconds: float = 0.01,
    frame_seconds: float = 0.04,
    min_distance_seconds: float = 0.25,
    prominence: float = 0.15,
    max_landmarks: int = 200,
) -> tuple[list[AcousticLandmark], NDArray[np.float64], float]:
    """Detect sparse energy-peak landmarks suitable as sync anchors.

    Args:
        samples: Mono PCM samples in roughly [-1, 1].
        sample_rate: Sample rate in Hz.
        hop_seconds: Envelope hop size.
        frame_seconds: Envelope frame size.
        min_distance_seconds: Minimum spacing between peaks.
        prominence: Minimum peak prominence on the normalized envelope.
        max_landmarks: Cap on returned peaks (strongest kept).

    Returns:
        ``(landmarks, normalized_envelope, hop_seconds_used)``.
    """
    env, hop = rms_envelope(
        samples,
        sample_rate,
        hop_seconds=hop_seconds,
        frame_seconds=frame_seconds,
    )
    if env.size == 0:
        return [], env, hop

    peak = float(np.max(env))
    if peak <= 0:
        return [], env, hop

    norm = env / peak
    min_dist = max(1, int(round(min_distance_seconds / hop)))
    indices, props = find_peaks(norm, distance=min_dist, prominence=prominence)
    if indices.size == 0:
        return [], norm, hop

    prominences = props.get("prominences", np.ones(indices.size))
    order = np.argsort(prominences)[::-1][:max_landmarks]
    chosen = sorted(indices[order])

    landmarks = [
        AcousticLandmark(
            time_seconds=float(i * hop),
            strength=float(norm[i]),
            kind="energy_peak",
        )
        for i in chosen
    ]
    return landmarks, norm, hop
