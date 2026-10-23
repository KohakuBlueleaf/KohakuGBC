"""Every builtin entry, registered on import; entries whose engine library is not installed
(vLLM, torch, the qwen35 engine's kernels) are absent, and naming one is a configuration error.
Installed packages add entries through the ``kohakugbc.entries`` entry-point group (each names a
module that registers its entries on import); one that fails to import is listed in
``PLUGIN_ERRORS``."""

from importlib.metadata import entry_points

from kohakugbc.engines import host, python
from kohakugbc.engines.replay import entries as replay
from kohakugbc.engines.vlm import api

try:
    from kohakugbc.engines.vlm import inprocess
except ImportError:
    inprocess = None
try:
    from kohakugbc.engines.vlm import qwen35
except ImportError:
    qwen35 = None
try:
    from kohakugbc.engines.torch import entries as torch_entries
except ImportError:
    torch_entries = None

MODULES = tuple(
    m for m in (api, host, python, replay, inprocess, qwen35, torch_entries) if m is not None
)

PLUGINS, PLUGIN_ERRORS = [], {}
for _point in entry_points(group="kohakugbc.entries"):
    try:
        PLUGINS.append(_point.load())
    except Exception as _exc:  # noqa: BLE001
        PLUGIN_ERRORS[_point.name] = repr(_exc)
