"""Images and crops.

An image value is ``{"key", "path", "size"}``: light enough to pass between nodes and
processes. Pixels are decoded where a crop needs them, once per process for the images in use
(a small LRU), as ml-gbc's ``ImageCache`` (``Image.open(path).convert("RGB")``) decodes them.
"""

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


@node(outputs=["patch"])
def crop(image: dict, where) -> np.ndarray:
    """The pixels of ``where``: None (the whole image), a box, or a region or record's box."""
    return cut(image, where.get("box") if isinstance(where, dict) else where)


@node(outputs=["patches"])
def crop_many(image: dict, regions: list) -> list:
    return [cut(image, r["box"]) for r in regions]
