"""Where a picture sits on the tagger's canvas (NumPy and PIL only, for the program's CPU nodes
and the model alike)."""

import numpy as np
from PIL import Image

SIZE = 1008


def fit(array: np.ndarray, size: int = SIZE) -> np.ndarray:
    """The picture scaled (bilinear, either way) so its longer side is ``size``."""
    h, w = array.shape[:2]
    r = min(size / h, size / w)
    new = (max(1, int(w * r)), max(1, int(h * r)))
    img = Image.fromarray(array).convert("RGB")
    return np.asarray(img.resize(new, Image.BILINEAR))


def pad_geometry(height: int, width: int, size: int = SIZE) -> tuple[int, int, int, int]:
    """``(top, left, height, width)`` of a fitted picture on the centre-padded canvas."""
    return (size - height) // 2, (size - width) // 2, height, width
