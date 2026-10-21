"""Every builtin entry, registered on import."""

from kohakugbc.engines import python
from kohakugbc.engines.replay import entries as replay

MODULES = (python, replay)
