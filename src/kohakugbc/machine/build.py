"""The GBC machine: the GBC program on Togeari.

``build(*layers)`` returns ``(machine, source, sink)`` (Togeari's machine-module contract): one
lane per engine slot from the configuration's entries, the bindings, admission that keeps the
VLM lane full, oldest-job-first issue.
"""

from pathlib import Path

from togeari import Inline, Lane, Machine, Use
from togeari.plugins import RunReport
from togeari.policies import All, OldestJobFirst, TargetLoad
from togeari.stores import MemoryStore

from kohakugbc.engines import builtin, registry
from kohakugbc.io import sinks, sources
from kohakugbc.machine import config as cfg_layers
from kohakugbc.nodes import library
from kohakugbc.program import ENTRY, PROGRAM

ENTRY_MODULES = builtin.MODULES

# Plain nodes heavier than the dispatcher should run: pixels, parsing, selection, resolution
CPU_NODES = (
    "open_image",
    "crop",
    "crop_many",
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


def lanes_of(slots: dict) -> dict[str, Lane]:
    return {
        name: Lane(s.units, former=s.former, select=s.select, name=name, **s.lane)
        for name, s in slots.items()
    }


def bindings(lanes: dict, slots: dict, store: bool) -> dict:
    """Model nodes to their slots' lanes, heavy plain nodes to the CPU lane, the rest inline."""
    bind = {
        node: Use(lanes[slot], adapter=slots[slot].adapter) for slot, node in registry.NODES.items()
    }
    if store:
        # A repeated name costs no embedding, across jobs and reruns
        bind["embedder"] = Use(lanes["embed"], adapter=slots["embed"].adapter, store="content")
    bind.update({node: Use(lanes["cpu"]) for node in CPU_NODES})
    bind["*"] = Inline()
    bind["kir.std.*"] = Inline()
    return bind


def build(*layers) -> tuple:
    """(machine, source, sink) from configuration layers (``kohakugbc.machine.config``)."""
    cfg = cfg_layers.load(*layers)
    m = cfg["machine"]
    slots = registry.build(cfg["engines"])
    lanes = lanes_of(slots)
    plugins = [RunReport(Path(m["report"]))] if m.get("report") else []
    store = MemoryStore() if m.get("store") == "memory" else None
    machine = Machine(
        program=PROGRAM,
        entry=ENTRY,
        libraries=[library.build(SlotModels(slots))],
        bind=bindings(lanes, slots, store is not None),
        admission=All(TargetLoad(lanes["vlm"], waiting=m["vlm_waiting"], max_jobs=m["max_jobs"])),
        issue=OldestJobFirst(),
        store=store,
        plugins=plugins,
        record=m.get("record"),
        window=m["window"],
        **m.get("knobs", {}),
    )
    return machine, sources.build(cfg["source"], cfg["controls"]), sinks.build(cfg["sink"])
