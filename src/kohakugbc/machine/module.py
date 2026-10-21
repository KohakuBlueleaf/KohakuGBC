"""The GBC machine as a Togeari machine module: ``togeari run|check|units <this file>``.

Configuration layers come from ``KOHAKUGBC_CONFIG`` (paths separated by the OS path separator).
"""

import os

from kohakugbc.machine.build import build as build_machine


def build():
    layers = [p for p in os.environ.get("KOHAKUGBC_CONFIG", "").split(os.pathsep) if p]
    return build_machine(*layers)
