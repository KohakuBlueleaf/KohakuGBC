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

from kohakugbc.engines import spawn

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
    # What the units will hold on which device (``kohakugbc.engines.placement.Need``)
    needs: list = field(default_factory=list)


def entry(slot: str, name: str):
    """Register a factory ``(settings, shared) -> Slot`` as entry ``name`` of ``slot``;
    ``shared`` is one dict per machine build, for state entries of several slots share."""
    if slot not in SLOTS:
        raise ValueError(f"unknown slot {slot!r}")

    def register(factory):
        ENTRIES[(slot, name)] = factory
        return factory

    return register


def build_slot(slot: str, settings: dict | None, shared: dict) -> Slot:
    """One slot's ``Slot`` from its settings (``entry`` and the entry's own)."""
    settings = dict(settings or {})
    name = settings.pop("entry", None)
    if name is None:
        raise ValueError(f"engines.{slot}: no entry")
    factory = ENTRIES.get((slot, name))
    if factory is None:
        known = sorted(n for s, n in ENTRIES if s == slot)
        raise ValueError(f"engines.{slot}: unknown entry {name!r} (known: {known})")
    try:
        return factory(settings, shared)
    except (TypeError, ValueError, KeyError, OSError) as exc:
        raise ValueError(f"engines.{slot}.{name}: {exc}") from exc


def build(engines: dict) -> dict[str, Slot]:
    """Every slot's ``Slot`` from the engine configuration; every problem in one error.
    ``engines["hosts"]`` (name -> ``{"env", "engines"}``): engine host processes started
    first, all at once, for the slots' ``remote`` entries to name."""
    shared: dict = {}
    if engines.get("hosts"):
        shared["hosts"] = spawn.launch_all(engines["hosts"])
    problems, out = [], {}
    for slot in SLOTS:
        try:
            out[slot] = build_slot(slot, engines.get(slot), shared)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError("engine configuration:\n  " + "\n  ".join(problems))
    return out
