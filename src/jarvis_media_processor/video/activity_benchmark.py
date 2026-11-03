"""Compare pHash-diff vs structural activity on sampled frame directories.

Used to validate encode-robustness before switching the production envelope.
Not a curve-walk smoke — only shape correlation under a known linear map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from jarvis_media_processor.video.activity import (
    aligned_activity_correlation,
    build_activity_envelope,
    build_structural_activity_envelope,
)
from jarvis_media_processor.video.hashing import compute_phash
from jarvis_media_processor.video.models import SampledFrame


def sampled_frames_from_dir(
    directory: Path,
    *,
    role: str,
    fps: float = 1.0,
) -> list[SampledFrame]:
    """Load ``frame_*.png`` (or any images) as ``SampledFrame`` with pHashes."""
    paths = sorted(directory.glob("frame_*.png"))
    if not paths:
        paths = sorted(directory.glob("*.png"))
    frames: list[SampledFrame] = []
    for index, path in enumerate(paths):
        with Image.open(path) as img:
            phash = str(compute_phash(img))
        time_seconds = index / fps if fps > 0 else float(index)
        frames.append(
            SampledFrame(
                time_seconds=time_seconds,
                path=path.resolve(),
                phash=phash,
                role=role,
            )
        )
    return frames


def compare_envelopes(
    base_frames: list[SampledFrame],
    donor_frames: list[SampledFrame],
    *,
    speed: float,
    offset_seconds: float,
    base_start: float,
    base_end: float,
) -> dict[str, float]:
    """Return shape correlations for pHash-diff and structural activity."""
    phash_base = build_activity_envelope(base_frames)
    phash_donor = build_activity_envelope(donor_frames)
    struct_base = build_structural_activity_envelope(base_frames)
    struct_donor = build_structural_activity_envelope(donor_frames)
    return {
        "phash_corr": aligned_activity_correlation(
            phash_base,
            phash_donor,
            speed=speed,
            offset_seconds=offset_seconds,
            base_start=base_start,
            base_end=base_end,
        ),
        "structural_corr": aligned_activity_correlation(
            struct_base,
            struct_donor,
            speed=speed,
            offset_seconds=offset_seconds,
            base_start=base_start,
            base_end=base_end,
        ),
        "phash_peak": float(phash_base.values.max()) if phash_base.values.size else 0.0,
        "structural_peak": (
            float(struct_base.values.max()) if struct_base.values.size else 0.0
        ),
    }


def _resolve_visual_dir(smoke_root: Path) -> Path:
    for name in ("align-sr/visual", "align/visual"):
        cand = smoke_root / name
        if (cand / "base_frames").is_dir() and (cand / "donor_frames").is_dir():
            return cand
    raise FileNotFoundError(f"No visual frame dirs under {smoke_root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "smoke_root",
        type=Path,
        help="Smoke scenario dir containing align/visual/{{base,donor}}_frames",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--base-start", type=float, required=True)
    parser.add_argument("--base-end", type=float, required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args(argv)

    visual = _resolve_visual_dir(args.smoke_root)
    base_frames = sampled_frames_from_dir(
        visual / "base_frames", role="base", fps=args.fps
    )
    donor_frames = sampled_frames_from_dir(
        visual / "donor_frames", role="donor", fps=args.fps
    )
    stats = compare_envelopes(
        base_frames,
        donor_frames,
        speed=args.speed,
        offset_seconds=args.offset,
        base_start=args.base_start,
        base_end=args.base_end,
    )
    print(f"smoke={args.smoke_root.name}")
    print(
        f"span base=[{args.base_start:.1f},{args.base_end:.1f}] "
        f"speed={args.speed:.4f} offset={args.offset:.3f}"
    )
    print(f"frames base={len(base_frames)} donor={len(donor_frames)}")
    print(f"phash_corr={stats['phash_corr']:.4f}")
    print(f"structural_corr={stats['structural_corr']:.4f}")
    delta = stats["structural_corr"] - stats["phash_corr"]
    print(f"delta_structural_minus_phash={delta:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
