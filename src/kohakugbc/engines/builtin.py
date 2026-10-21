"""Every builtin entry, registered on import."""

from kohakugbc.engines import host, python
from kohakugbc.engines.replay import entries as replay
from kohakugbc.engines.vlm import api

MODULES = (api, host, python, replay)
