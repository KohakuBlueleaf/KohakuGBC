"""Engines in other processes: the engine host serves slots' units (``gbc host``); the
``remote`` entry makes each a unit of the machine's lane (``RemoteUnit``).

``remote`` settings: ``hosts``, per item ``"host:port/unit"`` (a running ``gbc host``) or the
name of an engine configuration host (``kohakugbc.engines.spawn``: every unit it serves for this
slot, with its slot's adapter, former and lane options); ``transport`` (``inband``, or ``shm``
on one node: spawned hosts' default); ``start_timeout_s`` (1800). Addresses also take
``authkey`` or ``authkey_env``, and ``runs`` (the hosted entry's settings, buildable here) or
``adapter`` (``ADAPTERS``), ``kind``, ``protocol``, ``max_batch``, ``wait_ms``.
"""

import os
from dataclasses import dataclass, field
from functools import partial

from togeari.policies import SizeOrWait

from kohakugbc.engines import placement, registry
from kohakugbc.engines.adapters import ADAPTERS
from kohakugbc.engines.host.host import EngineHost
from kohakugbc.engines.host.remote import RemoteUnit
from kohakugbc.engines.spawn import HostProcess


def authkey(settings: dict) -> bytes:
    if "authkey" in settings:
        return str(settings["authkey"]).encode()
    env = settings.get("authkey_env", "KOHAKUGBC_HOST_KEY")
    if env not in os.environ:
        raise ValueError(f"no authkey: set {env} or give authkey")
    return os.environ[env].encode()


@dataclass
class Served:
    """What a host serves: its units, each unit's slot shape, unit names and shape per slot."""

    units: list = field(default_factory=list)
    shapes: dict = field(default_factory=dict)
    by_slot: dict = field(default_factory=dict)
    slot_shapes: dict = field(default_factory=dict)


def serve(engines: dict) -> Served:
    """The units of ``engines`` (slot -> entry settings) for a host, built as the machine builds
    them (one ``shared`` for all), their placement checked on this process's devices."""
    shared: dict = {}
    out, needs = Served(), []
    for slot, settings in engines.items():
        built = registry.build_slot(slot, settings, shared)
        shape = {
            "adapter": built.adapter,
            "former": built.former,
            "select": built.select,
            "lane": built.lane,
            "call": built.call,
        }
        out.units += built.units
        out.by_slot[slot] = [u.name for u in built.units]
        out.slot_shapes[slot] = shape
        for u in built.units:
            out.shapes[u.name] = shape
        needs += built.needs
    placement.check(needs)
    return out


def _like(settings: dict, slot: str, shared: dict) -> tuple[registry.Slot, str, int, str]:
    """What the lane needs of the hosted units, from the entry they run or by name: the slot
    (adapter, former, lane options), the units' kind, batch size and protocol."""
    if "runs" in settings:
        runs = registry.build_slot(slot, settings["runs"], shared)
        model = runs.units[0]
        return runs, model.kind, getattr(model, "max_batch", 1), model.protocol
    kind, max_batch = settings["kind"], int(settings.get("max_batch", 1))
    former = None
    if kind == "batch":
        former = SizeOrWait(max_batch, float(settings.get("wait_ms", 2.0)))
    adapter = ADAPTERS[settings["adapter"]]()
    protocol = settings.get("protocol", next(iter(adapter.protocols)))
    like = registry.Slot(
        units=[], former=former, adapter=adapter, lane=dict(settings.get("lane") or {})
    )
    return like, kind, max_batch, protocol


def _started(host: HostProcess):
    def on_start(unit: str, error: str | None) -> None:
        if error is None:
            host.unit_up(unit)
        else:
            host.unit_failed(unit, error)

    return on_start


def _spawned_units(slot: str, name: str, host: HostProcess, settings: dict) -> list:
    """The remote units of a spawned host's units of ``slot``, each described by the host."""
    names = host.units.get(slot)
    if not names:
        raise ValueError(f"host {name!r} serves no {slot} units (it serves {sorted(host.units)})")
    out = []
    for i, unit in enumerate(names):
        probe = RemoteUnit(None, host.authkey, kind="?", unit=unit, locate=host.locate)
        info = probe.describe()
        out.append(
            RemoteUnit(
                None,
                host.authkey,
                kind=info["kind"],
                max_batch=info["max_batch"],
                protocol=info["protocol"],
                name=f"{slot}@{name}" if len(names) == 1 else f"{slot}{i}@{name}",
                transport=settings.get("transport", "shm"),
                unit=unit,
                locate=host.locate,
                manage=True,
                start_timeout_s=float(settings.get("start_timeout_s", 1800.0)),
                before_start=partial(host.wait_after, slot=slot),
                on_start=_started(host),
            )
        )
    return out


def _remote(slot: str):
    def factory(settings: dict, shared: dict) -> registry.Slot:
        spawned = shared.get("hosts", {})
        refs = list(settings["hosts"])
        units, shape = [], None
        for i, ref in enumerate(refs):
            if ref in spawned:
                units += _spawned_units(slot, ref, spawned[ref], settings)
                shape = shape or spawned[ref].shapes[slot]
                continue
            if ":" not in ref:
                raise ValueError(f"hosts: {ref!r} is neither host:port/unit nor a spawned host")
            like, kind, max_batch, protocol = _like(settings, slot, shared)
            units.append(
                RemoteUnit(
                    ref,
                    authkey(settings),
                    kind=kind,
                    max_batch=max_batch,
                    protocol=protocol,
                    name=f"remote-{slot}-{i}",
                    transport=settings.get("transport", "inband"),
                )
            )
            shape = shape or {
                "adapter": like.adapter,
                "former": like.former,
                "select": like.select,
                "lane": like.lane,
                "call": like.call,
            }
        return registry.Slot(
            units=units,
            former=shape["former"],
            select=shape["select"],
            adapter=shape["adapter"],
            call=shape["call"],
            lane=dict(shape["lane"]),
        )

    return factory


for _slot in registry.SLOTS:
    registry.entry(_slot, "remote")(_remote(_slot))

__all__ = ["EngineHost", "HostProcess", "RemoteUnit", "Served", "authkey", "serve"]
