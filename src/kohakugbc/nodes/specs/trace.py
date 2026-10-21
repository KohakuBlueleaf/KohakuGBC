"""The trace format: every model call of a run, keyed by what was asked.

A trace is a folder: ``meta.json`` (the run: source, settings, ``image_root``) and one JSONL
file per kind of line:

- ``jobs``: per image job, ``key``, ``image``, ``inputs`` (captions, tags, controls), the
  ``records`` and ``graph`` the source produced, its ``rounds``.
- ``vlm``: per VLM request, ``key``, ``request`` (system, query, image sizes, schema, sampling,
  max tokens), ``reply``, timings, usage.
- ``detect``: per detector call, ``key``, ``size`` (crop pixels), ``raw`` (boxes, scores, text
  index), ``selected`` (after post-processing), timings.
- ``embed``: per text embedded, ``text``, ``vector`` (float32, base64).
- ``same``: per same-object decision, ``a``, ``b``, ``same``, ``score``.

Keys are canonical JSON of what the request means, never of its bytes: the node, the prompt
set (``spec``), the image job, and the entity, box or texts asked about. The program's nodes
compute the same keys from their inputs, so replay finds a reply exactly when the program asks
what the source asked.
"""

import base64
import json
import threading
from pathlib import Path

import numpy as np

FILES = ("jobs", "vlm", "detect", "embed", "same")


def canonical(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def box_key(box) -> list | None:
    """A box as it enters a key: the four coordinates, never the confidence."""
    return None if box is None else [float(b) for b in box[:4]]


def vlm_key(node: str, spec: str, image: str, **asked) -> str:
    return canonical({"node": node, "spec": spec, "image": image, **asked})


def detect_key(image: str, box, texts: list[str]) -> str:
    return canonical({"node": "detect", "image": image, "box": box_key(box), "texts": list(texts)})


def pack(vector) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def unpack(data: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype=np.float32)


class TraceWriter:
    """Appends to a trace folder from any thread."""

    def __init__(self, root: str | Path, meta: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.lock = threading.Lock()

    def write(self, file: str, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.lock, open(self.root / f"{file}.jsonl", "a", encoding="utf-8") as f:
            f.write(line)


class Trace:
    """A trace folder, loaded: lookups by key; repeated keys keep their first record."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.jobs = {r["key"]: r for r in self._lines("jobs")}
        # A key asked twice with different replies means the key leaves out part of the input
        self.vlm: dict[str, dict] = {}
        self.conflicts: list[str] = []
        for r in self._lines("vlm"):
            first = self.vlm.setdefault(r["key"], r)
            if first is not r and first["reply"] != r["reply"] and r["key"] is not None:
                self.conflicts.append(r["key"])
        self.detect: dict[str, dict] = {}
        for r in self._lines("detect"):
            self.detect.setdefault(r["key"], r)
        self.embed = {r["text"]: unpack(r["vector"]) for r in self._lines("embed")}
        self.same = list(self._lines("same"))

    def _lines(self, name: str):
        path = self.root / f"{name}.jsonl"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def image_path(self, job: dict) -> Path:
        """The job's image; a relative ``image_root`` is relative to the trace folder."""
        return self.root / self.meta.get("image_root", ".") / job["image"]
