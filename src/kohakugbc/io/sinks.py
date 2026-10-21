"""Sinks of GBC records.

``GraphJsonl(path)``: one GBC graph per line (the graph's JSON text as resolution wrote it);
failed jobs, one per line with key and reason, in ``<path stem>.failures.jsonl``; each job's
counts in ``<path stem>.stats.jsonl``.
"""

import json
import threading
from pathlib import Path

from togeari.io import Memory


class GraphJsonl:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stem = self.path.with_suffix("")
        self.files = {
            "graphs": self.path,
            "failures": stem.with_name(stem.name + ".failures.jsonl"),
            "stats": stem.with_name(stem.name + ".stats.jsonl"),
        }
        self.lock = threading.Lock()
        self.counts = {"ok": 0, "failed": 0}

    def _append(self, name: str, line: str) -> None:
        with open(self.files[name], "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write(self, record) -> None:
        with self.lock:
            if record.ok and record.outputs.get("graph") is not None:
                self._append("graphs", record.outputs["graph"])
                stats = {"key": record.key, **(record.outputs.get("stats") or {})}
                self._append("stats", json.dumps(stats, ensure_ascii=False))
                self.counts["ok"] += 1
            else:
                failure = record.failure or {"kind": "empty", "message": "no described record"}
                self._append(
                    "failures", json.dumps({"key": record.key, **failure}, ensure_ascii=False)
                )
                self.counts["failed"] += 1

    def close(self) -> None:
        return None


def build(spec: dict | None):
    match (spec or {"kind": "memory"})["kind"]:
        case "memory":
            return Memory()
        case "graphs":
            return GraphJsonl(spec["path"])
        case kind:
            raise ValueError(f"sink kind {kind!r}: one of memory, graphs")
