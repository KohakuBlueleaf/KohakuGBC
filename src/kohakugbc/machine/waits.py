"""Why jobs wait: Togeari's ``waits`` snapshot every ``every_s`` seconds, counted by reason and
by the lane a wait points at, one JSON line per snapshot in ``path``.

A line is ``{"t", "cost_ms", "entities", "relations", "truncated", "reasons": {reason: n},
"lanes": {lane: {reason: n}}}``: ``cost_ms`` is the snapshot's time on the dispatcher thread,
``reasons`` counts every relation (``input``, ``control``, ``effect``, ``window``,
``lane_capacity``, ``batch_wait``, ``retry_backoff``, ``resource_budget``, ``checkpoint``,
``unknown``), ``lanes`` the relations whose target is a lane. A snapshot holds at most ``limit``
entities and relations together (Togeari's cap, 10000); ``truncated`` marks one that hit it,
whose counts are then a part of the jobs' waits. Most relations are ``control`` (strands waiting
on a branch's operand). A snapshot runs on the dispatcher thread: 3-30 ms on a 16-job fixture;
on the flagship machine of one RTX 5090 (384 VLM lanes) 180 ms (max 310), truncated at the cap,
and sampling every 5 s cost 2.6% img/s, so it is a diagnosis tool at long periods.
"""

import json
import threading
import time
from collections import Counter
from pathlib import Path

from togeari import BasePlugin, PluginContext


def histogram(snapshot: dict) -> dict:
    """Relations by reason, and by lane name for the waits that point at a lane."""
    lanes = {e["id"]: e.get("name") or "?" for e in snapshot["entities"] if e["kind"] == "lane"}
    reasons, by_lane = Counter(), {}
    for rel in snapshot["relations"]:
        reasons[rel["reason"]] += 1
        lane = lanes.get(rel["to"])
        if lane is not None:
            by_lane.setdefault(lane, Counter())[rel["reason"]] += 1
    return {
        "entities": len(snapshot["entities"]),
        "relations": len(snapshot["relations"]),
        "truncated": snapshot["truncated"],
        "reasons": dict(reasons),
        "lanes": {k: dict(v) for k, v in by_lane.items()},
    }


class WaitsReport(BasePlugin):
    name = "gbc_waits"

    def __init__(self, path: str | Path, every_s: float = 5.0, limit: int = 10_000):
        self.path, self.every_s, self.limit = Path(path), float(every_s), int(limit)
        self.ctx: PluginContext | None = None
        self.halt = threading.Event()
        self.ticker: threading.Thread | None = None
        self.out = None

    def on_load(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.out = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 - closed in on_unload

    def on_machine_start(self) -> None:
        self.halt.clear()
        self.ticker = threading.Thread(target=self.tick, name="gbc-waits", daemon=True)
        self.ticker.start()

    def tick(self) -> None:
        while not self.halt.wait(self.every_s):
            self.ctx.post(self.sample)

    def sample(self) -> None:
        """On the dispatcher thread, between cycles."""
        t0 = time.perf_counter()
        snapshot = self.ctx.query_now("waits", limit=self.limit)
        cost = time.perf_counter() - t0
        row = {"t": round(time.time(), 2), "cost_ms": round(cost * 1e3, 3), **histogram(snapshot)}
        self.out.write(json.dumps(row) + "\n")
        self.out.flush()

    def on_machine_stop(self) -> None:
        self.halt.set()
        if self.ticker is not None:
            self.ticker.join()
            self.ticker = None

    def on_unload(self) -> None:
        if self.out is not None:
            self.out.close()
            self.out = None
