"""
Alpha mask utilities for applying segmentation masks to images.

Provides functions to combine a mask with an image as its alpha channel,
producing a transparent PNG where only the masked region is visible.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def save_with_alpha(
    image: str | Path | np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
    Save an image with a mask applied as its alpha channel.

    The output is a PNG file where pixels inside the mask are fully opaque
    and pixels outside the mask are fully transparent.

    Args:
        image: Path to the source image, or an RGB numpy array (HxWx3 uint8).
        mask: Boolean mask array (HxW). True = keep (opaque), False = remove (transparent).
        output_path: Where to save the resulting RGBA PNG.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)

    if isinstance(image, (str, Path)):
        pil_image = Image.open(image).convert("RGBA")
    else:
        pil_image = Image.fromarray(image).convert("RGBA")

    alpha = np.where(mask, 255, 0).astype(np.uint8)
    pil_image.putalpha(Image.fromarray(alpha))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(str(output_path))
    return output_path


def apply_alpha_mask(
    image: str | Path | np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Apply a mask as the alpha channel of an image and return the RGBA array.

    Args:
        image: Path to the source image, or an RGB numpy array (HxWx3 uint8).
        mask: Boolean mask array (HxW). True = keep (opaque), False = remove (transparent).

    Returns:
        RGBA numpy array (HxWx4 uint8).
    """
    if isinstance(image, (str, Path)):
        img = np.array(Image.open(image).convert("RGB"))
    else:
        img = image

    h, w = img.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img
    rgba[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
    return rgba
