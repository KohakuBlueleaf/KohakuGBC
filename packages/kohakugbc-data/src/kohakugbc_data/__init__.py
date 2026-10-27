"""kohakugbc-data: GBC graphs and GBC datasets, without the captioning machine.

- ``GbcGraph``: one GBC graph held natively, from / to JSON and msgpack (+zstd), rendered as
  prompt text (``GbcGraph.prompt``); ``PromptDag``: prompt text parsed back; ``resolve``: the
  caption records of one image resolved into its graph.
- ``kohakugbc_data.prompt``: the prompt renderers in Python (``from_gbc``, ``render``,
  ``parse``, ``to_moss``, detail levels in ``prompt.detail``).
- ``kohakugbc_data.dataset``: the GBC dataset format (``GbcDataset``, ``DatasetWriter``).
"""

from importlib.metadata import PackageNotFoundError, version

from kohakugbc_data._rs import GbcGraph, PromptDag, resolve

try:
    __version__ = version("kohakugbc-data")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["GbcGraph", "PromptDag", "__version__", "resolve"]
