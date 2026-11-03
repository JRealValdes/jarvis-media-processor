"""Frame preprocessing for robust perceptual hashing and structural activity."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps
from scipy import ndimage


def crop_letterbox(image: Image.Image, *, luminance_threshold: int = 16) -> Image.Image:
    """Crop near-black letterbox/pillarbox bars when present.

    Args:
        image: Input RGB/L image.
        luminance_threshold: Pixels at or below this luma count as bar.

    Returns:
        Cropped image, or the original if no clear bars are found.
    """
    gray = ImageOps.grayscale(image)
    width, height = gray.size
    pixels = gray.load()
    assert pixels is not None

    def row_is_bar(y: int) -> bool:
        dark = sum(1 for x in range(width) if pixels[x, y] <= luminance_threshold)
        return dark >= int(0.98 * width)

    def col_is_bar(x: int) -> bool:
        dark = sum(1 for y in range(height) if pixels[x, y] <= luminance_threshold)
        return dark >= int(0.98 * height)

    top = 0
    while top < height and row_is_bar(top):
        top += 1
    bottom = height - 1
    while bottom >= top and row_is_bar(bottom):
        bottom -= 1
    left = 0
    while left < width and col_is_bar(left):
        left += 1
    right = width - 1
    while right >= left and col_is_bar(right):
        right -= 1

    if top == 0 and left == 0 and bottom == height - 1 and right == width - 1:
        return image
    if bottom <= top or right <= left:
        return image
    return image.crop((left, top, right + 1, bottom + 1))


def normalize_frame(
    image: Image.Image,
    *,
    target_size: int = 256,
    crop_bars: bool = True,
) -> Image.Image:
    """Convert to RGB, optionally crop letterbox, and downscale for hashing."""
    rgb = image.convert("RGB")
    if crop_bars:
        rgb = crop_letterbox(rgb)
    return ImageOps.fit(rgb, (target_size, target_size), method=Image.Resampling.LANCZOS)


def structure_map(
    image: Image.Image,
    *,
    target_size: int = 64,
    crop_bars: bool = True,
) -> NDArray[np.float64]:
    """Encode-robust structural fingerprint of a frame (edge energy on equalized luma).

    Pipeline: letterbox crop → grayscale → global histogram equalization →
    downscale → Sobel magnitude in ``[0, 1]``. Grade and mild compression
    shift color/luma but leave edge layout largely intact, so consecutive-frame
    change on this map tracks real cuts/motion better than raw pHash distance.
    """
    rgb = image.convert("RGB")
    if crop_bars:
        rgb = crop_letterbox(rgb)
    gray = ImageOps.grayscale(rgb)
    gray = ImageOps.equalize(gray)
    gray = ImageOps.fit(gray, (target_size, target_size), method=Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float64) / 255.0
    # Separable Sobel kernels.
    kx = np.asarray([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = np.asarray([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    gx = ndimage.convolve(arr, kx, mode="nearest")
    gy = ndimage.convolve(arr, ky, mode="nearest")
    mag = np.sqrt(gx * gx + gy * gy)
    peak = float(np.max(mag))
    if peak < 1e-12:
        return np.zeros_like(mag)
    return (mag / peak).astype(np.float64, copy=False)


def structure_change(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Mean absolute difference between two structure maps (0 = identical)."""
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
