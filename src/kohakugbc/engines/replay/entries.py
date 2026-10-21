"""Replay entries of the ``vlm``, ``detect`` and ``embed`` slots.

Settings: ``trace`` (folder), ``units`` (1), ``capacity`` (requests in flight per unit; batch
slots per unit for batch lanes), ``max_batch`` (batch lanes), ``wait_ms`` (batch former),
``latency_ms`` (a fixed latency; default: the trace's recorded latencies), ``speed`` (latencies
divided by it; ``speeds``, one per unit, override it), ``seed``, ``item_error_rate``,
``outages`` (per unit, ``[[start_s, end_s], ...]``); ``unknown`` (embed slot: ``"hash"`` gives
names the trace never embedded a vector from their hash, for runs that leave the recorded path).

The slot's node runs as a ``ReplayCall``: the trace folder, the call and the node's inputs, a
payload that pickles, so a replay unit also runs behind an engine host. Each process answers
from its own ``ReplayModels`` per trace folder.
"""

import threading
from dataclasses import dataclass
from pathlib import Path

from togeari.policies import SizeOrWait
from togeari.testing import SimBatch, SimStream

from kohakugbc.engines.registry import Slot, entry
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace

_MODELS: dict[str, ReplayModels] = {}
_LOCK = threading.Lock()


def models_of(trace: str | Path) -> ReplayModels:
    """This process's ``ReplayModels`` of a trace folder."""
    key = str(Path(trace).resolve())
    with _LOCK:
        if key not in _MODELS:
            _MODELS[key] = ReplayModels(Trace(key))
        return _MODELS[key]


@dataclass
class ReplayCall:
    trace: str
    call: str
    kwargs: dict

    def run(self):
        return getattr(models_of(self.trace), self.call)(**self.kwargs)


class ReplayAdapter:
    """The slot's node inputs as a ``ReplayCall``; the reply is the node's result. ``extra``
    keyword arguments go to the call (the embedder's ``unknown``)."""

    protocols = {"python"}
    light = True

    def __init__(self, trace: str, call: str, extra: dict | None = None):
        self.trace, self.call = str(Path(trace).resolve()), call
        self.extra = dict(extra or {})

    def check(self, signature) -> list[str]:
        return []

    def prepare(self, inputs: dict, protocol: str) -> ReplayCall:
        return ReplayCall(self.trace, self.call, {**inputs, **self.extra})

    def decode(self, raw, protocol: str):
        return raw


def _latencies(models: ReplayModels, slot: str) -> list[float] | None:
    records = models.trace.vlm.values() if slot == "vlm" else models.trace.detect.values()
    samples = [r["latency"] * 1000.0 for r in records if r.get("latency")]
    return samples or None


def _units(slot: str, settings: dict, models: ReplayModels) -> list:
    latency = settings.get("latency_ms")
    samples = None if latency is not None or slot == "embed" else _latencies(models, slot)
    outages = settings.get("outages") or []
    speeds = settings.get("speeds") or []
    units = []
    for i in range(int(settings.get("units", 1))):
        speed = speeds[i] if i < len(speeds) else settings.get("speed", 1.0)
        common = {
            "name": f"replay-{slot}-{i}",
            "latency_ms": float(latency if latency is not None else 2.0),
            "tail": 0.0,
            "seed": int(settings.get("seed", 0)) + i,
            "item_error_rate": float(settings.get("item_error_rate", 0.0)),
            "outages": [tuple(o) for o in outages[i]] if i < len(outages) else None,
            "speed": float(speed),
            "samples_ms": samples,
        }
        if slot == "vlm":
            units.append(SimStream(capacity=int(settings.get("capacity", 64)), **common))
        else:
            units.append(
                SimBatch(
                    capacity=int(settings.get("capacity", 2)),
                    max_batch=int(settings.get("max_batch", 32)),
                    **common,
                )
            )
    return units


def _slot(slot: str, call: str):
    def factory(settings: dict, shared: dict) -> Slot:
        models = models_of(settings["trace"])
        former = None
        if slot != "vlm":
            former = SizeOrWait(
                int(settings.get("max_batch", 32)), float(settings.get("wait_ms", 2.0))
            )
        return Slot(
            units=_units(slot, settings, models),
            former=former,
            adapter=ReplayAdapter(
                settings["trace"],
                call,
                {"unknown": settings["unknown"]} if "unknown" in settings else None,
            ),
            lane=dict(settings.get("lane") or {}),
        )

    return factory


entry("vlm", "replay")(_slot("vlm", "chat"))
entry("detect", "replay")(_slot("detect", "detector"))
entry("embed", "replay")(_slot("embed", "embedder"))
