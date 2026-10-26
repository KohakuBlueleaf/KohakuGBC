"""GBC jobs over ``togeari serve``'s HTTP stream (``gbc serve``).

A posted job is an image and optionally its ``captions``, ``tags`` and ``controls``:

- a raw image body (``image/*``), ``?key=`` optional;
- an NDJSON / JSON job with ``image`` (a path on the serving host) or ``image_b64``.

Posted pictures are written into ``inbox`` as ``<key><ext>``; a job without a key is keyed by
its bytes' SHA-1 (16 hex digits), or by the path's name. The answer line is
``{"key", "ok", "graph", "stats", "failure"}``, ``graph`` the GBC graph as JSON.
"""

import base64
import hashlib
import json
from pathlib import Path

EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
}
PASSED = ("captions", "tags")


class Jobs:
    """``prepare`` and ``present`` for ``togeari.serve.Service``; ``controls`` are the
    configuration's, under each job's own."""

    def __init__(self, inbox: str, controls: dict):
        self.inbox = Path(inbox)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.controls = dict(controls)

    def _save(self, data: bytes, key: str | None, content_type: str) -> tuple[str, Path]:
        key = key or hashlib.sha1(data).hexdigest()[:16]
        path = self.inbox / f"{key}{EXTENSIONS.get(content_type, '.img')}"
        path.write_bytes(data)
        return key, path

    def prepare(self, job: dict) -> dict:
        key = job.get("key")
        if "body" in job:
            if not job["content_type"].startswith("image/"):
                raise ValueError(f"a raw body is an image, not {job['content_type'] or 'untyped'}")
            key, path = self._save(job["body"], key, job["content_type"])
        elif "image_b64" in job:
            data = base64.b64decode(job["image_b64"])
            key, path = self._save(data, key, job.get("content_type", ""))
        elif "image" in job:
            path = Path(job["image"])
            if not path.is_file():
                raise FileNotFoundError(f"no image at {path} on the serving host")
            key = key or path.name
        else:
            raise ValueError("a job carries a raw image body, image_b64 or image (a path)")
        inputs = {k: job[k] for k in PASSED if job.get(k) is not None}
        controls = job.get("controls") or {}
        if isinstance(controls, str):
            controls = json.loads(controls)
        return {
            "key": str(key),
            "inputs": {
                **inputs,
                "image": {"path": str(path), "key": str(key)},
                "controls": {**self.controls, **controls},
            },
        }

    def present(self, record) -> dict:
        graph = record.outputs.get("graph") if record.ok else None
        return {
            "key": record.key,
            "ok": record.ok and graph is not None,
            "graph": None if graph is None else json.loads(graph),
            "stats": record.outputs.get("stats") or record.stats,
            "failure": record.failure,
        }
