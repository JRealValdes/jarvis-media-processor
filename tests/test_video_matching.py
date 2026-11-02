"""Tests for visual preprocessing, hashing, and frame matching."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from jarvis_media_processor.video import (
    SampledFrame,
    build_frame_extract_command,
    compute_phash,
    crop_letterbox,
    distance_to_confidence,
    match_frames,
    phash_distance,
)


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (160, 90)) -> Image.Image:
    return Image.new("RGB", size, color)


def _pattern(seed: int = 0) -> Image.Image:
    img = Image.new("RGB", (160, 90), (20, 20, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20 + seed, 20, 80 + seed, 70), fill=(200, 80, 40))
    draw.ellipse((90, 15, 140, 65), fill=(40, 160, 220))
    return img


def test_build_frame_extract_command() -> None:
    cmd = build_frame_extract_command(
        "ffmpeg",
        Path("base.mkv"),
        Path("out/frame_%04d.png"),
        fps=0.5,
    )
    assert "fps=0.5" in cmd
    assert "base.mkv" in cmd


def test_letterbox_crop_removes_bars() -> None:
    inner = _pattern()
    framed = Image.new("RGB", (160, 140), (0, 0, 0))
    framed.paste(inner, (0, 25))
    cropped = crop_letterbox(framed)
    assert cropped.size[1] < framed.size[1]
    assert cropped.size[1] >= inner.size[1] - 2


def test_phash_similar_under_scale() -> None:
    img = _pattern()
    small = img.resize((80, 45))
    d = phash_distance(compute_phash(img), compute_phash(small))
    assert d <= 8


def test_phash_similar_with_letterbox() -> None:
    img = _pattern()
    framed = Image.new("RGB", (160, 140), (0, 0, 0))
    framed.paste(img, (0, 25))
    d = phash_distance(compute_phash(img), compute_phash(framed))
    assert d <= 10


def test_different_scenes_are_farther() -> None:
    a = compute_phash(_pattern(0))
    b = compute_phash(_solid((10, 180, 10)))
    assert phash_distance(a, b) >= 8


def test_match_frames_finds_offset_pair(tmp_path: Path) -> None:
    base_img = _pattern(0)
    donor_img = _pattern(0)
    base_path = tmp_path / "b.png"
    donor_path = tmp_path / "d.png"
    base_img.save(base_path)
    donor_img.save(donor_path)

    base_hash = str(compute_phash(base_img))
    donor_hash = str(compute_phash(donor_img))
    base_frames = [
        SampledFrame(time_seconds=1.0, path=base_path, phash=base_hash, role="base"),
        SampledFrame(
            time_seconds=2.0,
            path=base_path,
            phash=str(compute_phash(_solid((200, 10, 10)))),
            role="base",
        ),
    ]
    donor_frames = [
        SampledFrame(time_seconds=3.0, path=donor_path, phash=donor_hash, role="donor"),
    ]

    matches = match_frames(base_frames, donor_frames, max_distance=12)
    assert matches
    best = matches[0]
    assert best.base_time == 1.0
    assert best.donor_time == 3.0
    assert best.confidence == distance_to_confidence(best.distance, max_distance=12)
