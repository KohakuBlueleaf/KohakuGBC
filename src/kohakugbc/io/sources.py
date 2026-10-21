"""Sources of image jobs: ``{"image", "captions"?, "tags"?, "controls"?}`` keyed by image.

Each source takes the configuration's ``controls`` as defaults under each job's own.
"""

import json
from pathlib import Path

from togeari import SourceItem

from kohakugbc.nodes.specs.trace import Trace

IMAGES = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


class _Source:
    def __init__(self, controls: dict | None = None, rank: int = 0, world: int = 1):
        self.controls = dict(controls or {})
        self.rank, self.world = rank, world

    def shard(self, rank: int, world: int):
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)
        out.rank, out.world = rank, world
        return out

    def items(self):
        raise NotImplementedError

    def __iter__(self):
        for n, (key, inputs) in enumerate(self.items()):
            if n % self.world == self.rank:
                inputs = {**inputs, "controls": {**self.controls, **(inputs.get("controls") or {})}}
                yield SourceItem(key, inputs)


class Folder(_Source):
    """Every image under ``root`` (recursively), keyed by its path relative to ``root``."""

    def __init__(self, root: str, controls: dict | None = None, patterns=IMAGES):
        super().__init__(controls)
        self.root = Path(root)
        self.patterns = tuple(patterns)

    def items(self):
        paths = sorted({p for pat in self.patterns for p in self.root.rglob(pat)})
        for p in paths:
            key = p.relative_to(self.root).as_posix()
            yield key, {"image": {"path": str(p), "key": key}}


class Jsonl(_Source):
    """One job per line: ``image`` (path, relative to the file's folder or absolute), and
    optionally ``key``, ``captions``, ``tags``, ``controls``."""

    def __init__(self, path: str, controls: dict | None = None):
        super().__init__(controls)
        self.path = Path(path)

    def items(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                job = json.loads(line)
                path = Path(job["image"])
                path = path if path.is_absolute() else self.path.parent / path
                key = job.get("key") or path.name
                inputs = {k: job[k] for k in ("captions", "tags", "controls") if k in job}
                yield key, {**inputs, "image": {"path": str(path), "key": key}}


# The PoC's control names and the root each recorded arm used
POC_CONTROLS = {"target_min_nodes": "min_nodes", "target_max_nodes": "max_nodes"}
POC_ROOTS = {"text": "text", "image_text": "image_text", "tags": "tags", "query": "query"}


class TraceJobs(_Source):
    """The jobs of a recorded trace, with the inputs and controls the recorded run gave them;
    ``repeat`` above 1 gives each job that many times (keys ``<key>#<n>``, the same image)."""

    def __init__(self, path: str, controls: dict | None = None, repeat: int = 1):
        super().__init__(controls)
        self.trace = Trace(path)
        self.repeat = repeat

    def items(self):
        for n in range(self.repeat):
            for key, inputs in self._jobs():
                yield (key if self.repeat == 1 else f"{key}#{n}"), inputs

    def _jobs(self):
        arm = self.trace.meta.get("arm")
        for key, job in sorted(self.trace.jobs.items()):
            controls = {"root": POC_ROOTS.get(arm, "text")}
            for name, value in job["inputs"].get("controls", {}).items():
                controls[POC_CONTROLS.get(name, name)] = int(value) if value.isdigit() else value
            seeded = controls["root"] != "query"
            yield key, {
                "image": {"path": str(self.trace.image_path(job)), "key": key},
                "captions": job["inputs"].get("captions") if seeded else None,
                "tags": job["inputs"].get("tags"),
                "controls": controls,
            }


def build(spec: dict | None, controls: dict):
    match (spec or {}).get("kind"):
        case "folder":
            return Folder(spec["path"], controls)
        case "jsonl":
            return Jsonl(spec["path"], controls)
        case "trace":
            return TraceJobs(spec["path"], controls, int(spec.get("repeat", 1)))
        case None:
            return None
        case kind:
            raise ValueError(f"source kind {kind!r}: one of folder, jsonl, trace")
