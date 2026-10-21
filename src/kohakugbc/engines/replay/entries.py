"""Replay entries of the ``vlm``, ``detect`` and ``embed`` slots.

Settings: ``trace`` (folder), ``units`` (1), ``capacity`` (requests in flight per unit; batch
slots per unit for batch lanes), ``max_batch`` (batch lanes), ``wait_ms`` (batch former),
``latency_ms`` (a fixed latency; default: the trace's recorded latencies), ``speed`` (latencies
divided by it), ``seed``, ``item_error_rate``, ``outages`` (``[[start_s, end_s], ...]`` per
unit, as a list per unit index). Units run the slot's node function, which answers from the
trace (``ReplayModels``).
"""

from pathlib import Path

from togeari.policies import SizeOrWait
from togeari.testing import SimBatch, SimStream

from kohakugbc.engines.registry import Slot, entry
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace


def models_of(trace: str | Path, shared: dict) -> ReplayModels:
    """One ``ReplayModels`` per trace folder per machine build."""
    key = ("replay", str(Path(trace).resolve()))
    if key not in shared:
        shared[key] = ReplayModels(Trace(trace))
    return shared[key]


def _latencies(models: ReplayModels, slot: str) -> list[float] | None:
    records = models.trace.vlm.values() if slot == "vlm" else models.trace.detect.values()
    samples = [r["latency"] * 1000.0 for r in records if r.get("latency")]
    return samples or None


def _units(slot: str, settings: dict, models: ReplayModels) -> list:
    n = int(settings.get("units", 1))
    latency = settings.get("latency_ms")
    samples = None if latency is not None or slot == "embed" else _latencies(models, slot)
    outages = settings.get("outages") or []
    units = []
    for i in range(n):
        common = {
            "name": f"replay-{slot}-{i}",
            "latency_ms": float(latency if latency is not None else 2.0),
            "tail": 0.0,
            "seed": int(settings.get("seed", 0)) + i,
            "item_error_rate": float(settings.get("item_error_rate", 0.0)),
            "outages": [tuple(o) for o in outages[i]] if i < len(outages) else None,
            "speed": float(settings.get("speed", 1.0)),
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


def _slot(slot: str, call_name: str):
    def factory(settings: dict, shared: dict) -> Slot:
        models = models_of(settings["trace"], shared)
        former = None
        if slot != "vlm":
            former = SizeOrWait(
                int(settings.get("max_batch", 32)), float(settings.get("wait_ms", 2.0))
            )
        return Slot(
            units=_units(slot, settings, models),
            former=former,
            call=getattr(models, call_name),
            lane=dict(settings.get("lane") or {}),
        )

    return factory


entry("vlm", "replay")(_slot("vlm", "chat"))
entry("detect", "replay")(_slot("detect", "detector"))
entry("embed", "replay")(_slot("embed", "embedder"))
