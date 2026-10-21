"""Images and crops.

An image value is ``{"key", "path", "size"}``: light enough to pass between nodes and
processes. Pixels are decoded where a crop needs them, once per process for the images in use
(a small LRU), as ml-gbc's ``ImageCache`` (``Image.open(path).convert("RGB")``) decodes them.
"""

import io
import os
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image

from kohakugbc.nodes import boxes as bx
from kohakugbc.nodes.registry import node

CACHE_IMAGES = 8
_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_LOCK = threading.Lock()


def pixels(path: str) -> np.ndarray:
    with _LOCK:
        if path in _CACHE:
            _CACHE.move_to_end(path)
            return _CACHE[path]
    with Image.open(path) as img:
        array = np.array(img.convert("RGB"))
    with _LOCK:
        _CACHE[path] = array
        while len(_CACHE) > CACHE_IMAGES:
            _CACHE.popitem(last=False)
    return array


def cut(image: dict, box) -> np.ndarray:
    """The pixels of ``box`` (None: the whole image), as ml-gbc's ``crop_bbox`` cuts them."""
    array = pixels(image["path"])
    if box is None:
        return array
    left, top, right, bottom = bx.pixels(box, array.shape[1], array.shape[0])
    return array[top:bottom, left:right]


@node(outputs=["image"])
def open_image(image) -> dict:
    """The image value of a job input: a path, or ``{"path", "key"?}``."""
    if isinstance(image, str):
        image = {"path": image}
    path = image["path"]
    with Image.open(path) as img:
        size = [img.width, img.height]
    return {"key": image.get("key") or os.path.basename(path), "path": path, "size": size}


def _where(where):
    return where.get("box") if isinstance(where, dict) else where


@node(outputs=["patch"])
def crop(image: dict, where) -> np.ndarray:
    """The pixels of ``where``: None (the whole image), a box, or a region or record's box."""
    return cut(image, _where(where))


def encode(array: np.ndarray, max_side: int) -> dict | None:
    """A picture as the VLM is sent it (the PoC's ``encode_image``): RGB, the longer side scaled
    down to ``max_side`` (Lanczos), JPEG quality 95. None for an empty crop."""
    if array.size == 0:
        return None
    img = Image.fromarray(array).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(size, Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    return {"jpeg": buffer.getvalue(), "size": [img.width, img.height]}


@node(outputs=["picture"])
def picture(image: dict, where, controls: dict) -> dict | None:
    """The picture of ``where`` the VLM is sent (``encode``)."""
    return encode(cut(image, _where(where)), controls["max_side"])


@node(outputs=["pictures"])
def pictures(image: dict, regions: list, controls: dict) -> list:
    return [encode(cut(image, r["box"]), controls["max_side"]) for r in regions]
