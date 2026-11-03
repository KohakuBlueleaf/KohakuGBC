"""Engine slots and their entries, selected once when the machine is built.

A slot is one kind of work the program sends out: ``vlm`` (``chat``), ``detect``
(``detector``), ``embed`` (``embedder``), ``tag`` (``tagger``) and ``cpu`` (plain nodes heavier
than the dispatcher should run). An entry fills a slot: ``entry(slot, name)`` registers a
factory, and the machine's engine configuration names one entry per slot with its settings,
``{"vlm": {"entry": "replay", "trace": "..."}, ...}``. Nothing dispatches on the entry after
the build. An ``OPTIONAL`` slot the configuration leaves out has no lane (its node is only
reached under controls that ask for it: ``detect`` and ``embed`` below the root, ``tag`` under
``tagging``).

No engine runs in the machine's process: an entry registered with ``hosted=True`` loads one, and
a configuration that names it directly gets it in one spawned host (``local``) with the slot on
``remote``. ``hosts`` place them per card.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kohakugbc.engines import spawn

SLOTS = ("vlm", "detect", "embed", "tag", "cpu")
NODES = {"vlm": "chat", "detect": "detector", "embed": "embedder", "tag": "tagger"}
OPTIONAL = ("detect", "embed", "tag")

ENTRIES: dict[tuple[str, str], Callable[[dict, dict], "Slot"]] = {}
HOSTED: set[tuple[str, str]] = set()


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


def entry(slot: str, name: str, hosted: bool = False):
    """Register a factory ``(settings, shared) -> Slot`` as entry ``name`` of ``slot``;
    ``shared`` is one dict per machine build, for state entries of several slots share.
    ``hosted``: the units load an engine, so they run in an engine host."""
    if slot not in SLOTS:
        raise ValueError(f"unknown slot {slot!r}")

    def register(factory):
        ENTRIES[(slot, name)] = factory
        if hosted:
            HOSTED.add((slot, name))
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


def hosted(engines: dict) -> dict:
    """``engines`` with the slots naming a ``hosted`` entry moved to the ``local`` host, the VLM
    up before the others load."""
    local = {
        slot: engines[slot]
        for slot in SLOTS
        if (slot, (engines.get(slot) or {}).get("entry")) in HOSTED
    }
    if not local:
        return engines
    hosts = dict(engines.get("hosts") or {})
    if "local" in hosts:
        raise ValueError("engines.hosts: 'local' is the host of slots that name an engine")
    hosts["local"] = {"engines": local, "order": [["vlm"], [s for s in local if s != "vlm"]]}
    remote = {slot: {"entry": "remote", "hosts": ["local"]} for slot in local}
    return {**engines, **remote, "hosts": hosts}


def build(engines: dict) -> dict[str, Slot]:
    """Every slot's ``Slot`` from the engine configuration; every problem in one error.
    ``engines["hosts"]`` (name -> ``{"env", "engines"}``): engine host processes started
    first, all at once, for the slots' ``remote`` entries to name."""
    engines = hosted(engines)
    shared: dict = {}
    if engines.get("hosts"):
        shared["hosts"] = spawn.launch_all(engines["hosts"])
    problems, out = [], {}
    for slot in SLOTS:
        if slot in OPTIONAL and engines.get(slot) is None:
            continue
        try:
            out[slot] = build_slot(slot, engines.get(slot), shared)
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError("engine configuration:\n  " + "\n  ".join(problems))
    return out
