"""Images and crops.

An image value is ``{"key", "path", "size"}``: light enough to pass between nodes and
processes. Pixels are decoded where a crop needs them, as ml-gbc's ``ImageCache``
(``Image.open(path).convert("RGB")``) decodes them, once per machine: the decoded array is
written to the shared pixel cache (``KOHAKUGBC_PIXEL_CACHE``, default
``/dev/shm/kohakugbc-pixels`` where ``/dev/shm`` exists; empty: none) as one ``.npy`` per image
file (keyed by its path, size and modification time), and every process memory-maps it,
read-only. The cache keeps at most ``KOHAKUGBC_PIXEL_CACHE_IMAGES`` files (2048), the least
recently written going first. Each process also keeps its last ``CACHE_IMAGES`` arrays.
"""

import hashlib
import io
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from kohakugbc.nodes import boxes as bx
from kohakugbc.nodes.registry import node

CACHE_IMAGES = 64
_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_LOCK = threading.Lock()
_DEFAULT_SHARED = "/dev/shm/kohakugbc-pixels" if os.path.isdir("/dev/shm") else ""
_SHARED_ROOT = os.environ.get("KOHAKUGBC_PIXEL_CACHE", _DEFAULT_SHARED)
SHARED = Path(_SHARED_ROOT) if _SHARED_ROOT else None
SHARED_IMAGES = int(os.environ.get("KOHAKUGBC_PIXEL_CACHE_IMAGES", "2048"))
# Writes between two evictions of one process
EVICT_EVERY = 64
_WRITES = [0]


def decode(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _shared(path: str) -> np.ndarray:
    """The image's decoded array from the shared cache, decoding and writing it when absent."""
    st = os.stat(path)
    ident = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    file = SHARED / f"{hashlib.sha1(ident.encode()).hexdigest()}.npy"
    try:
        return np.load(file, mmap_mode="r")
    except (OSError, ValueError):
        pass
    array = decode(path)
    SHARED.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f"{file.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, file)
    _WRITES[0] += 1
    if _WRITES[0] % EVICT_EVERY == 0:
        _evict()
    return array


def _evict() -> None:
    """The oldest files past ``SHARED_IMAGES`` leave (a process mapping one keeps its view)."""
    files = []
    with os.scandir(SHARED) as it:
        for e in it:
            if e.name.endswith(".npy"):
                try:
                    files.append((e.stat().st_mtime, e.path))
                except OSError:
                    continue
    files.sort()
    for _, path in files[: max(0, len(files) - SHARED_IMAGES)]:
        try:
            os.unlink(path)
        except OSError:
            continue


def pixels(path: str) -> np.ndarray:
    with _LOCK:
        if path in _CACHE:
            _CACHE.move_to_end(path)
            return _CACHE[path]
    array = _shared(path) if SHARED is not None else decode(path)
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
