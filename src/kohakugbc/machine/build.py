"""The GBC machine: the GBC program on Togeari.

``build(*layers)`` returns ``(machine, source, sink)`` (Togeari's machine-module contract): one
lane per engine slot from the configuration's entries, the bindings, admission that keeps the
VLM lane full and every limited lane's backlog bounded, oldest-job-first issue.
"""

from dataclasses import dataclass
from pathlib import Path

from togeari import Inline, Lane, Machine, Use
from togeari.plugins import RunReport
from togeari.policies import All, OldestJobFirst, TargetLoad
from togeari.stores import MemoryStore

from kohakugbc.engines import builtin, placement, registry
from kohakugbc.io import sinks, sources
from kohakugbc.machine import config as cfg_layers
from kohakugbc.machine.waits import WaitsReport
from kohakugbc.nodes import library
from kohakugbc.program import ENTRY, PROGRAM

ENTRY_MODULES = builtin.MODULES

# Plain nodes heavier than the dispatcher should run: pixels, parsing, selection, resolution
CPU_NODES = (
    "open_image",
    "crop",
    "picture",
    "pictures",
    "image_root",
    "seeded_root",
    "detected",
    "entity_record",
    "batch_replies",
    "coverage_root",
    "relational_queries",
    "relational_request",
    "relational_record",
    "to_gbc",
    "tag_request",
    "place_tags",
)


class SlotModels:
    """The program's ``Models`` from the slots: each model node calls its slot's ``call``."""

    def __init__(self, slots: dict):
        self.slots = slots

    def _call(self, slot: str):
        fn = self.slots[slot].call
        if fn is None:
            raise RuntimeError(f"the {slot} slot's units run {registry.NODES[slot]} themselves")
        return fn

    def chat(self, request: dict) -> str:
        return self._call("vlm")(request)

    def detector(self, patch, request: dict) -> dict:
        return self._call("detect")(patch, request)

    def embedder(self, text: str):
        return self._call("embed")(text)

    def tagger(self, request: dict) -> dict:
        return self._call("tag")(request)


def lanes_of(slots: dict) -> dict[str, Lane]:
    """One lane per slot; the slot's ``lane`` options (former, select, ...) win over the
    entry's own former and selector."""
    out = {}
    for name, s in slots.items():
        options = {"former": s.former, "select": s.select, **s.lane}
        out[name] = Lane(s.units, name=name, **{k: v for k, v in options.items() if v is not None})
    return out


def bindings(lanes: dict, slots: dict, store: bool) -> dict:
    """Model nodes to their slots' lanes (an optional slot left out: its node inline, never
    reached), heavy plain nodes to the CPU lane, the rest inline."""
    bind = {
        node: Use(lanes[slot], adapter=slots[slot].adapter)
        for slot, node in registry.NODES.items()
        if slot in slots
    }
    if store:
        # A repeated name costs no embedding, across jobs and reruns
        bind["embedder"] = Use(
            lanes["embed"], adapter=slots["embed"].adapter, store="content", deterministic=True
        )
    bind.update({node: Use(lanes["cpu"]) for node in CPU_NODES})
    bind["*"] = Inline()
    bind["kir.std.*"] = Inline()
    return bind


def store_of(spec):
    """The machine's store: ``"memory"``, a store object (``togeari.stores``), or None."""
    match spec:
        case None:
            return None
        case "memory":
            return MemoryStore()
        case str() | dict():
            raise ValueError(f"machine.store {spec!r}: 'memory' or a store object")
    return spec


def admission_of(lanes: dict, m: dict) -> All:
    """Admit while the VLM lane holds fewer than ``vlm_waiting`` waiting (its queue and its
    units' own) and every lane of ``waiting`` ({lane: items}) fewer than its limit, up to
    ``max_jobs`` resident.

    The limits gate new jobs only: jobs already resident still send their later requests, so a
    lane's backlog can pass its limit by what they have yet to send. What bounds a job's time in
    the machine is ``max_jobs``: about ``max_jobs`` / throughput, however long the source. A
    torch lane's waiting is its queue only (its units report none)."""
    extra = dict(m.get("waiting") or {})
    if "vlm" in extra:
        raise ValueError("machine.waiting: the VLM lane's limit is machine.vlm_waiting")
    waiting = {"vlm": m["vlm_waiting"], **extra}
    unknown = sorted(set(waiting) - set(lanes))
    if unknown:
        raise ValueError(f"machine.waiting: no lane {unknown} (lanes: {sorted(lanes)})")
    small = sorted(name for name, n in waiting.items() if int(n) < 1)
    if small or m["max_jobs"] < 1:
        raise ValueError(f"admission limits must be at least 1 ({small or 'max_jobs'})")
    return All(
        *(TargetLoad(lanes[name], waiting=n, max_jobs=m["max_jobs"]) for name, n in waiting.items())
    )


@dataclass
class Parts:
    machine: Machine
    source: object
    sink: object
    lanes: dict
    slots: dict
    controls: dict


def build(*layers) -> tuple:
    """(machine, source, sink) from configuration layers (``kohakugbc.machine.config``)."""
    parts = build_parts(*layers)
    return parts.machine, parts.source, parts.sink


def build_parts(*layers) -> Parts:
    """The machine with its source and sink, lanes (by slot) and slots."""
    cfg = cfg_layers.load(*layers)
    m = cfg["machine"]

    # A GBC job under run-ahead always holds speculative work, so it never reaches a snapshot
    # boundary (an empty window, one strand)
    knobs = m.get("knobs", {})
    if knobs.get("snapshot_every") and knobs.get("run_ahead", True):
        raise ValueError(
            "machine.knobs: snapshot_every needs run_ahead=False; under run-ahead a GBC job "
            "never reaches a snapshot boundary"
        )
    slots = registry.build(cfg["engines"])

    # Every engine's memory on every device, before any unit loads a weight
    placement.check(
        [n for s in slots.values() for n in s.needs], headroom=m.get("placement_headroom", 0.05)
    )
    lanes = lanes_of(slots)
    plugins = [RunReport(Path(m["report"]))] if m.get("report") else []
    if m.get("waits") is not None:
        if not m.get("report"):
            raise ValueError("machine.waits: needs machine.report (it writes beside it)")
        plugins.append(WaitsReport(Path(m["report"]) / "waits.jsonl", **m["waits"]))
    store = store_of(m.get("store"))
    machine = Machine(
        program=PROGRAM,
        entry=ENTRY,
        libraries=[library.build(SlotModels(slots))],
        bind=bindings(lanes, slots, store is not None),
        admission=admission_of(lanes, m),
        issue=OldestJobFirst(),
        store=store,
        plugins=plugins,
        record=m.get("record"),
        start_timeout_s=m["start_timeout_s"],
        window=m["window"],
        **m.get("knobs", {}),
    )
    source = sources.build(cfg["source"], cfg["controls"])
    return Parts(machine, source, sinks.build(cfg["sink"]), lanes, slots, cfg["controls"])
