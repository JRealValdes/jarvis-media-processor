"""Perceptual hashing helpers."""

from __future__ import annotations

import imagehash
from PIL import Image

from jarvis_media_processor.video.preprocess import normalize_frame


def compute_phash(
    image: Image.Image,
    *,
    hash_size: int = 8,
    target_size: int = 256,
    crop_bars: bool = True,
) -> imagehash.ImageHash:
    """Compute a perceptual hash after normalization.

    Args:
        image: Source frame.
        hash_size: pHash size parameter.
        target_size: Resize edge before hashing.
        crop_bars: Whether to crop letterbox/pillarbox bars.

    Returns:
        ``imagehash.ImageHash`` instance.
    """
    normalized = normalize_frame(image, target_size=target_size, crop_bars=crop_bars)
    return imagehash.phash(normalized, hash_size=hash_size)


def phash_distance(a: imagehash.ImageHash | str, b: imagehash.ImageHash | str) -> int:
    """Hamming distance between two perceptual hashes."""
    ha = imagehash.hex_to_hash(a) if isinstance(a, str) else a
    hb = imagehash.hex_to_hash(b) if isinstance(b, str) else b
    return int(ha - hb)


def distance_to_confidence(distance: int, *, max_distance: int = 16) -> float:
    """Map Hamming distance to a [0, 1] confidence score."""
    if max_distance <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (distance / max_distance)))
