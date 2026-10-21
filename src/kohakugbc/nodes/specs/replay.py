"""Models that answer from a trace: the program's model nodes replayed, never a model run.

A request the trace does not hold is a miss: the program asked something the source did not.
A request whose text differs from the recorded one is a mismatch: the program asked the same
thing in other words. Both are counted per key, so a replayed run says where it left the
source's path.
"""

import numpy as np

from kohakugbc.nodes.specs.trace import Trace


class ReplayMiss(KeyError):
    """The trace holds no answer for this request."""


class ReplayModels:
    def __init__(self, trace: Trace):
        self.trace = trace
        self.misses: list[str] = []
        self.mismatches: list[dict] = []
        self.calls = {"chat": 0, "detector": 0, "embedder": 0}

    def chat(self, request: dict) -> str:
        self.calls["chat"] += 1
        rec = self.trace.vlm.get(request["key"])
        if rec is None:
            self.misses.append(request["key"])
            raise ReplayMiss(request["key"])
        recorded = rec["request"]
        for field in ("query", "system"):
            if field in recorded and recorded[field] != request[field]:
                self.mismatches.append(
                    {
                        "key": request["key"],
                        "field": field,
                        "recorded": recorded[field],
                        "asked": request[field],
                    }
                )
        return rec["reply"] or ""

    def detector(self, patch, request: dict) -> dict:
        self.calls["detector"] += 1
        rec = self.trace.detect.get(request["key"])
        if rec is None:
            self.misses.append(request["key"])
            raise ReplayMiss(request["key"])
        if rec["size"] != [int(patch.shape[1]), int(patch.shape[0])]:
            self.mismatches.append({"key": request["key"], "field": "size"})
        return rec["raw"]

    def embedder(self, text: str) -> np.ndarray | None:
        """The recorded vector; None for a name the source never compared (it embeds lazily)."""
        self.calls["embedder"] += 1
        return self.trace.embed.get(text)
