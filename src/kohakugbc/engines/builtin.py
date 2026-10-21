"""Every builtin entry, registered on import."""

from kohakugbc.engines import host, python
from kohakugbc.engines.replay import entries as replay

MODULES = (host, python, replay)
