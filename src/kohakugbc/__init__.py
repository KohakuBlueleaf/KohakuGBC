"""KohakuGBC: Graph-Based Captioning as a KIR program on the Togeari dataflow machine.

Layers, by what they may import (``scripts/check_imports.py``):

- ``kohakugbc.program``, ``kohakugbc.nodes``, ``kohakugbc.graph``: the GBC program, its node
  library and controls, the records and their resolution into GBC graphs (``kohakunode`` only).
- ``kohakugbc.machine``, ``kohakugbc.io``: the program on Togeari (bindings, lanes, policies,
  sources, sinks).
- ``kohakugbc.engines``: the engines that fill Togeari's contracts (VLM, torch, remote, replay).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kohakugbc")
except PackageNotFoundError:
    __version__ = "0+unknown"
