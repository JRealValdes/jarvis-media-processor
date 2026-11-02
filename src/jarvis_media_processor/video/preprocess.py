"""Frame preprocessing for robust perceptual hashing."""

from __future__ import annotations

from PIL import Image, ImageOps


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
