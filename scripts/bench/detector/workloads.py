"""Detector workloads: the (crop, texts) requests a GBC run sends, and the process's memory peak.

Needs only numpy and Pillow, so the benches run in interpreters without GBC installed.

- ``graph_requests(graph, image)``: the requests one GBC graph asked: the root's object names on
  the whole image, then each entity's part names on its crop (texts in first-seen order).
- ``trace_requests(trace_dir)``: every detector request of a recorded trace (``detect.jsonl``).

A request is ``{"name", "box" (image fractions or None), "texts", "patch" (HxWx3 uint8)}``.
"""

import ctypes
import json
import sys
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import resource
except ImportError:  # Windows: the peak working set instead
    resource = None


class Counters(ctypes.Structure):
    _fields_ = [
        (n, ctypes.c_size_t)
        for n in (
            "cb",
            "PageFaultCount",
            "PeakWorkingSetSize",
            "WorkingSetSize",
            "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage",
            "PagefileUsage",
            "PeakPagefileUsage",
        )
    ]


def peak_gib() -> float | None:
    """The process's peak working set (Windows), or its max RSS elsewhere."""
    if sys.platform == "win32":
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get = ctypes.windll.psapi.GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if get(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / 2**30
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def pixels(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def cut(image: np.ndarray, box) -> np.ndarray:
    """The pixels of ``box`` (image fractions; None: the whole image), truncated as GBC cuts."""
    if box is None:
        return image
    h, w = image.shape[:2]
    return image[int(box[1] * h) : int(box[3] * h), int(box[0] * w) : int(box[2] * w)]


def graph_requests(graph: dict, image: str | Path) -> list[dict]:
    by = {v["vertex_id"]: v for v in graph["vertices"]}
    array = pixels(image)
    out = []
    for v in graph["vertices"]:
        texts = []
        for e in v["out_edges"]:
            t = by[e["target"]]
            if (
                e["target"] != v["vertex_id"]
                and t["label"] != "relation"
                and e["text"] not in texts
            ):
                texts.append(e["text"])
        if not texts or v["label"] not in ("image", "entity"):
            continue
        b = v["bbox"]
        box = None if v["label"] == "image" else [b["left"], b["top"], b["right"], b["bottom"]]
        out.append(
            {
                "name": v["vertex_id"] or "image",
                "box": box,
                "texts": texts,
                "patch": cut(array, box),
            }
        )
    return out


def graph_file_requests(graphs: str | Path, image: str | Path) -> list[dict]:
    """The requests of the first graph of a ``graphs.jsonl``."""
    graph = json.loads(Path(graphs).read_text(encoding="utf-8").splitlines()[0])
    return graph_requests(graph, image)


def trace_requests(trace_dir: str | Path) -> list[dict]:
    root = Path(trace_dir)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    images: dict[str, np.ndarray] = {}
    out = []
    for line in (root / "detect.jsonl").read_text(encoding="utf-8").splitlines():
        key = json.loads(json.loads(line)["key"])
        if key["image"] not in images:
            images[key["image"]] = pixels(root / meta["image_root"] / key["image"])
        name = f"{key['image']}:{len(out)}"
        out.append(
            {
                "name": name,
                "box": key["box"],
                "texts": key["texts"],
                "patch": cut(images[key["image"]], key["box"]),
            }
        )
    return out
