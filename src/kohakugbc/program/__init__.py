"""The GBC program: ``gbc.kir`` (one image job) and its controls."""

from pathlib import Path

from kohakugbc.program.controls import CHOICES, DEFAULTS, complete

PROGRAM = Path(__file__).with_name("gbc.kir")
ENTRY = "gbc"


def text() -> str:
    return PROGRAM.read_text(encoding="utf-8")


__all__ = ["CHOICES", "DEFAULTS", "ENTRY", "PROGRAM", "complete", "text"]
