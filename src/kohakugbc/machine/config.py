"""Machine configurations as composable layers.

A layer is a dict, or a Python file whose module-level names (lower case, not starting with
``_``) form one: ``engines``, ``machine``, ``source``, ``sink``, ``controls``. Layers merge in
order; dicts merge key by key, other values are replaced. ``gbc run base.py replay.py`` is
``base.py`` with ``replay.py`` on top.
"""

import runpy
from pathlib import Path

DEFAULTS = {
    "engines": {"cpu": {"entry": "python", "workers": 8, "mode": "thread"}},
    "machine": {
        # Admission: jobs are admitted while the VLM lane reports fewer than this many waiting
        "vlm_waiting": 64,
        "max_jobs": 512,
        "window": 64,
        # A unit's start: loading a model's weights and capturing its graphs can take minutes
        "start_timeout_s": 600,
        "record": None,
        "store": None,
        "report": None,
        "knobs": {},
    },
    "source": None,
    "sink": {"kind": "memory"},
    "controls": {},
}


def merge(base: dict, top: dict) -> dict:
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def layer(item) -> dict:
    if isinstance(item, dict):
        return item
    names = runpy.run_path(str(Path(item)))
    return {k: v for k, v in names.items() if k.islower() and not k.startswith("_")}


def load(*layers) -> dict:
    """The defaults with every layer merged on top, in order."""
    out = DEFAULTS
    for item in layers:
        out = merge(out, layer(item))
    return out
