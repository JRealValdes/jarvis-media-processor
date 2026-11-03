"""Visual analyze uses structural activity envelopes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from jarvis_media_processor.config import Settings
from jarvis_media_processor.video.analyze import match_media_visuals
from jarvis_media_processor.video.hashing import compute_phash
from jarvis_media_processor.video.models import SampledFrame


def _stripe(phase: int) -> Image.Image:
    img = Image.new("RGB", (64, 64), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    for x in range(phase % 4, 64, 8):
        draw.rectangle([x, 0, x + 3, 63], fill=(200, 200, 200))
    return img


def test_match_media_visuals_writes_structural_activity(tmp_path: Path) -> None:
    base_media = tmp_path / "base.mkv"
    donor_media = tmp_path / "donor.mkv"
    base_media.write_bytes(b"fake")
    donor_media.write_bytes(b"fake")
    work = tmp_path / "visual"

    def fake_extract(source, output_dir, *, role=None, settings=None, fps=None):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        frames: list[SampledFrame] = []
        for i in range(12):
            # Hard cut mid-timeline so structural activity must spike.
            if i < 6:
                img = _stripe(0)
            else:
                img = Image.new("RGB", (64, 64), (200, 40, 40))
            path = out / f"frame_{i:04d}.png"
            img.save(path)
            frames.append(
                SampledFrame(
                    time_seconds=float(i),
                    path=path.resolve(),
                    phash=str(compute_phash(img)),
                    role=role,
                )
            )
        return frames

    settings = Settings(_env_file=None)
    with patch(
        "jarvis_media_processor.video.analyze.extract_sampled_frames",
        side_effect=fake_extract,
    ):
        report = match_media_visuals(
            base_media, donor_media, work, settings=settings
        )

    assert report.params.get("activity_method") == "fused_max_phash_structural"
    assert len(report.base_activity_values) == 12
    assert report.base_activity_values[0] == 0.0
    assert max(report.base_activity_values) > 0.0
