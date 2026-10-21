"""The ``python`` entry of the ``cpu`` slot: Togeari's Python workers.

Settings: ``workers`` (8), ``mode`` (``"thread"`` or ``"process"``), ``units`` (1).
"""

from togeari.engines.python import PythonCallEngine

from kohakugbc.engines.registry import Slot, entry


@entry("cpu", "python")
def python_workers(settings: dict, shared: dict) -> Slot:
    mode = settings.get("mode", "thread")
    imports = ("kohakugbc.nodes.library",) if mode == "process" else ()
    units = [
        PythonCallEngine(
            workers=int(settings.get("workers", 8)), mode=mode, imports=imports, name=f"cpu-{i}"
        )
        for i in range(int(settings.get("units", 1)))
    ]
    return Slot(units=units, lane=dict(settings.get("lane") or {}))
