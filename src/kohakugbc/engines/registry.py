"""Engine slots and their entries, selected once when the machine is built.

A slot is one kind of work the program sends out: ``vlm`` (``chat``), ``detect``
(``detector``), ``embed`` (``embedder``) and ``cpu`` (plain nodes heavier than the dispatcher
should run). An entry fills a slot: ``entry(slot, name)`` registers a factory, and the machine's
engine configuration names one entry per slot with its settings,
``{"vlm": {"entry": "replay", "trace": "..."}, ...}``. Nothing dispatches on the entry after
the build.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

SLOTS = ("vlm", "detect", "embed", "cpu")
NODES = {"vlm": "chat", "detect": "detector", "embed": "embedder"}

ENTRIES: dict[tuple[str, str], Callable[[dict, dict], "Slot"]] = {}


@dataclass
class Slot:
    """What an entry gives the machine for one slot."""

    # The lane's units, its former (batch lanes) and unit selector
    units: list
    former: Any = None
    select: Any = None
    # The Togeari adapter between the slot's node and its units; None runs the node's own
    # Python function on them (``PythonAdapter``), which then calls ``call``
    adapter: Any = None
    call: Callable | None = None
    # The lane's options (``max_inflight``, ``timeout_s``, ...)
    lane: dict = field(default_factory=dict)


def entry(slot: str, name: str):
    """Register a factory ``(settings, shared) -> Slot`` as entry ``name`` of ``slot``;
    ``shared`` is one dict per machine build, for state entries of several slots share."""
    if slot not in SLOTS:
        raise ValueError(f"unknown slot {slot!r}")

    def register(factory):
        ENTRIES[(slot, name)] = factory
        return factory

    return register


def build(engines: dict) -> dict[str, Slot]:
    """Every slot's ``Slot`` from the engine configuration; every problem in one error."""
    shared: dict = {}
    problems, out = [], {}
    for slot in SLOTS:
        settings = dict(engines.get(slot) or {})
        name = settings.pop("entry", None)
        if name is None:
            problems.append(f"engines.{slot}: no entry")
            continue
        factory = ENTRIES.get((slot, name))
        if factory is None:
            known = sorted(n for s, n in ENTRIES if s == slot)
            problems.append(f"engines.{slot}: unknown entry {name!r} (known: {known})")
            continue
        try:
            out[slot] = factory(settings, shared)
        except (TypeError, ValueError, KeyError, OSError) as exc:
            problems.append(f"engines.{slot}.{name}: {exc}")
    if problems:
        raise ValueError("engine configuration:\n  " + "\n  ".join(problems))
    return out
