"""Minimal WAV helpers (PCM 16-bit) without extra audio I/O dependencies."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def write_pcm16_wav(
    path: Path,
    samples: NDArray[np.floating],
    *,
    sample_rate: int,
) -> None:
    """Write float samples in [-1, 1] as 16-bit PCM WAV.

    Args:
        path: Destination path.
        samples: 1-D mono array, or 2-D shaped ``(channels, n_samples)``.
        sample_rate: Sample rate in Hz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        channels = 1
        frames = np.clip(arr, -1.0, 1.0)
        interleaved = (frames * 32767.0).astype(np.int16)
    elif arr.ndim == 2:
        channels = int(arr.shape[0])
        frames = np.clip(arr, -1.0, 1.0)
        interleaved = (frames.T.reshape(-1) * 32767.0).astype(np.int16)
    else:
        raise ValueError("samples must be 1-D or 2-D (channels, n)")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())


def read_pcm16_wav(
    path: Path,
    *,
    mix_to_mono: bool = True,
) -> tuple[NDArray[np.float64], int]:
    """Read a PCM WAV into float64 samples.

    Args:
        path: WAV file path.
        mix_to_mono: If True (default), return 1-D mono. If False, return
            ``(channels, n_samples)`` for multi-channel files.

    Returns:
        ``(samples, sample_rate)``.

    Raises:
        ValueError: If the WAV is not 16-bit PCM.
    """
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV supported, got sample_width={sample_width}")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if channels == 1:
        return data, sample_rate
    shaped = data.reshape(-1, channels).T
    if mix_to_mono:
        return shaped.mean(axis=0), sample_rate
    return shaped, sample_rate
