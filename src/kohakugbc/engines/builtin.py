"""Every builtin entry, registered on import; entries whose engine library is not installed
(vLLM, torch) are absent, and naming one is a configuration error."""

from kohakugbc.engines import host, python
from kohakugbc.engines.replay import entries as replay
from kohakugbc.engines.vlm import api

try:
    from kohakugbc.engines.vlm import inprocess
except ImportError:
    inprocess = None
try:
    from kohakugbc.engines.torch import entries as torch_entries
except ImportError:
    torch_entries = None

MODULES = tuple(m for m in (api, host, python, replay, inprocess, torch_entries) if m is not None)
