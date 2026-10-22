"""Engines on other nodes: the engine host serves a slot's units over the network; the
``remote`` entry makes each a unit of the machine's lane (``RemoteUnit``).

Settings of the ``remote`` entry: ``hosts`` (``"host:port/unit"`` per unit), ``authkey`` or
``authkey_env`` (the environment variable that holds it), and either ``runs`` (the entry the
hosts run, with its settings: the lane takes its adapter, former and lane options; it must be
buildable here) or, for units only the host's node can build, ``adapter`` (a name in
``kohakugbc.engines.adapters.ADAPTERS``), ``kind`` (``stream``, ``batch`` or ``call``),
``protocol``, ``max_batch`` and ``wait_ms`` (batch lanes).
"""

import os

from togeari.policies import SizeOrWait

from kohakugbc.engines import registry
from kohakugbc.engines.adapters import ADAPTERS
from kohakugbc.engines.host.host import EngineHost
from kohakugbc.engines.host.remote import RemoteUnit


def authkey(settings: dict) -> bytes:
    if "authkey" in settings:
        return str(settings["authkey"]).encode()
    env = settings.get("authkey_env", "KOHAKUGBC_HOST_KEY")
    if env not in os.environ:
        raise ValueError(f"no authkey: set {env} or give authkey")
    return os.environ[env].encode()


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


def _remote(slot: str):
    def factory(settings: dict, shared: dict) -> registry.Slot:
        like, kind, max_batch, protocol = _like(settings, slot, shared)
        lane = dict(like.lane)
        key = authkey(settings)
        units = [
            RemoteUnit(
                address,
                key,
                kind=kind,
                max_batch=max_batch,
                protocol=protocol,
                name=f"remote-{slot}-{i}",
            )
            for i, address in enumerate(settings["hosts"])
        ]
        return registry.Slot(
            units=units, former=like.former, select=like.select, adapter=like.adapter, lane=lane
        )

    return factory


for _slot in ("vlm", "detect", "embed", "cpu"):
    registry.entry(_slot, "remote")(_remote(_slot))

__all__ = ["EngineHost", "RemoteUnit", "authkey"]
