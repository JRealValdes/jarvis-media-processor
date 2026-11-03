"""Structural (encode-robust) activity envelope tests."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from jarvis_media_processor.video.activity import (
    aligned_activity_correlation,
    build_activity_envelope,
    build_fused_activity_envelope,
    build_structural_activity_envelope,
)
from jarvis_media_processor.video.hashing import compute_phash
from jarvis_media_processor.video.models import SampledFrame
from jarvis_media_processor.video.preprocess import structure_change, structure_map


def _solid(color: tuple[int, int, int], size: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), color)


def _stripes(phase: int = 0, *, contrast: int = 220) -> Image.Image:
    img = Image.new("RGB", (128, 128), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    for x in range(phase % 8, 128, 16):
        draw.rectangle([x, 0, x + 7, 127], fill=(contrast, contrast, contrast))
    return img


def _grade(img: Image.Image, *, gain: float, bias: int) -> Image.Image:
    """Crude grade shift: scale luma-ish channels and add bias (clip)."""
    arr = np.asarray(img, dtype=np.float64)
    out = np.clip(arr * gain + bias, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def test_structure_map_stable_under_grade() -> None:
    base = _stripes(0)
    graded = _grade(base, gain=1.35, bias=25)
    a = structure_map(base, target_size=48)
    b = structure_map(graded, target_size=48)
    # Same edge layout → small structural distance after equalize+Sobel.
    assert structure_change(a, b) < 0.15


def test_structure_change_spikes_on_cut() -> None:
    vertical = _stripes(0)
    # Orthogonal structure (horizontal bars) — not a periodic alias of vertical.
    horizontal = Image.new("RGB", (128, 128), (20, 20, 20))
    draw = ImageDraw.Draw(horizontal)
    for y in range(0, 128, 16):
        draw.rectangle([0, y, 127, y + 7], fill=(220, 220, 220))
    a = structure_map(vertical, target_size=48)
    b = structure_map(horizontal, target_size=48)
    c = structure_map(_solid((10, 180, 10)), target_size=48)
    assert structure_change(a, b) > 0.1
    assert structure_change(a, c) > structure_change(a, a)


def test_structural_activity_flat_on_static() -> None:
    frames = [_solid((40, 40, 40)) for _ in range(8)]
    env = build_structural_activity_envelope(frames, times=list(range(8)))
    assert float(np.max(env.values)) < 0.05


def test_structural_activity_spike_at_hard_cut() -> None:
    frames = [_stripes(0) for _ in range(5)] + [_solid((200, 30, 30)) for _ in range(5)]
    env = build_structural_activity_envelope(frames, times=list(range(10)))
    assert int(np.argmax(env.values)) == 5
    assert float(env.values[5]) > 0.2


def test_structural_beats_phash_under_grade_shift(tmp_path) -> None:
    """Same motion pattern; one side grade-shifted — structural shape matches better."""
    sequence = [_stripes(i) for i in range(0, 24, 2)] + [
        _solid((180, 20, 20)) for _ in range(4)
    ]
    graded = [_grade(img, gain=1.4, bias=30) for img in sequence]
    structural_a = build_structural_activity_envelope(sequence, times=list(range(len(sequence))))
    structural_b = build_structural_activity_envelope(graded, times=list(range(len(graded))))
    struct_corr = aligned_activity_correlation(structural_a, structural_b)

    base_frames: list[SampledFrame] = []
    donor_frames: list[SampledFrame] = []
    for i, (img_a, img_b) in enumerate(zip(sequence, graded, strict=True)):
        pa = tmp_path / f"a_{i}.png"
        pb = tmp_path / f"b_{i}.png"
        img_a.save(pa)
        img_b.save(pb)
        base_frames.append(
            SampledFrame(
                time_seconds=float(i),
                path=pa,
                phash=str(compute_phash(img_a)),
                role="base",
            )
        )
        donor_frames.append(
            SampledFrame(
                time_seconds=float(i),
                path=pb,
                phash=str(compute_phash(img_b)),
                role="donor",
            )
        )
    phash_corr = aligned_activity_correlation(
        build_activity_envelope(base_frames),
        build_activity_envelope(donor_frames),
    )
    assert struct_corr >= 0.85
    assert struct_corr >= phash_corr - 0.02


def test_fused_activity_keeps_hard_cut_spike(tmp_path) -> None:
    images = [_stripes(0) for _ in range(5)] + [_solid((200, 30, 30)) for _ in range(5)]
    frames: list[SampledFrame] = []
    for i, img in enumerate(images):
        path = tmp_path / f"f_{i}.png"
        img.save(path)
        frames.append(
            SampledFrame(
                time_seconds=float(i),
                path=path,
                phash=str(compute_phash(img)),
                role="base",
            )
        )
    fused = build_fused_activity_envelope(frames)
    assert int(np.argmax(fused.values)) == 5
    assert float(fused.values[5]) > 0.5
