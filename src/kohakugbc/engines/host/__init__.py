"""Engines on other nodes: the engine host serves a slot's units over the network; the
``remote`` entry makes each a unit of the machine's lane (``RemoteUnit``).

Settings of the ``remote`` entry: ``hosts`` (``"host:port/unit"`` per unit), ``authkey`` or
``authkey_env`` (the environment variable that holds it), ``runs`` (the entry the hosts run,
with its settings: its adapter, former and lane options are the lane's).
"""

import os

from kohakugbc.engines import registry
from kohakugbc.engines.host.host import EngineHost
from kohakugbc.engines.host.remote import RemoteUnit


def authkey(settings: dict) -> bytes:
    if "authkey" in settings:
        return str(settings["authkey"]).encode()
    env = settings.get("authkey_env", "KOHAKUGBC_HOST_KEY")
    if env not in os.environ:
        raise ValueError(f"no authkey: set {env} or give authkey")
    return os.environ[env].encode()


def _remote(slot: str):
    def factory(settings: dict, shared: dict) -> registry.Slot:
        runs = registry.build_slot(slot, settings["runs"], shared)
        model = runs.units[0]
        key = authkey(settings)
        units = [
            RemoteUnit(
                address,
                key,
                kind=model.kind,
                max_batch=getattr(model, "max_batch", 1),
                protocol=model.protocol,
                name=f"remote-{slot}-{i}",
            )
            for i, address in enumerate(settings["hosts"])
        ]
        return registry.Slot(
            units=units,
            former=runs.former,
            select=runs.select,
            adapter=runs.adapter,
            lane=runs.lane,
        )

    return factory


for _slot in ("vlm", "detect", "embed", "cpu"):
    registry.entry(_slot, "remote")(_remote(_slot))

__all__ = ["EngineHost", "RemoteUnit", "authkey"]
