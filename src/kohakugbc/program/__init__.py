"""The GBC programs (one image job each) and their controls.

- ``gbc`` (``gbc.kir``): GBC's procedure; the call tree is the graph.
- ``gbc_nest`` (``gbc_nest.kir``): GBC-Nest; GBC's rounds with grounded roots and joins, then
  parents from box containment, each nesting and each group checked by one question.
"""

from pathlib import Path

from kohakugbc.program.controls import CHOICES, DEFAULTS, PROGRAM_DEFAULTS, complete

HERE = Path(__file__).parent
# name -> (KIR file, entry @def)
PROGRAMS = {"gbc": ("gbc.kir", "gbc"), "gbc_nest": ("gbc_nest.kir", "gbc_nest")}


def program(name: str = "gbc") -> tuple[Path, str]:
    """The program's KIR file and entry."""
    if name not in PROGRAMS:
        raise ValueError(f"program {name!r}: one of {sorted(PROGRAMS)}")
    file, entry = PROGRAMS[name]
    return HERE / file, entry


def text(name: str = "gbc") -> str:
    return program(name)[0].read_text(encoding="utf-8")


__all__ = [
    "CHOICES",
    "DEFAULTS",
    "PROGRAMS",
    "PROGRAM_DEFAULTS",
    "complete",
    "program",
    "text",
]
