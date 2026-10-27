"""Box coordinates as written in a prompt, and the header line that declares them.

A box is stored as ``(left, top, right, bottom)`` in [0, 1] and written as four rounded ints:
``abs_centralized`` is pixels on a ``width`` x ``height`` canvas with the origin at its centre
(MOSS-Image's layout space); ``relative`` is ``[0, scale]`` from the top-left. ``yxyx`` writes
``[y_min, x_min, y_max, x_max]`` (MOSS-Image's order), ``xyxy`` ``[x_min, y_min, x_max, y_max]``.
"""

import re
from dataclasses import dataclass

SPACES = ("abs_centralized", "relative")
ORDERS = ("yxyx", "xyxy")
HEADER = re.compile(r"<gbc( [a-z_]+=[^ >]+)*>")
FIELD = re.compile(r" ([a-z_]+)=([^ >]+)")


@dataclass(frozen=True)
class Coords:
    width: int = 1024
    height: int = 1024
    space: str = "abs_centralized"
    order: str = "yxyx"
    scale: int = 1000

    def __post_init__(self):
        if self.space not in SPACES or self.order not in ORDERS:
            raise ValueError(f"space in {SPACES}, order in {ORDERS}: {self}")
        if min(self.width, self.height, self.scale) <= 0:
            raise ValueError(f"width, height and scale are positive: {self}")

    def _axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """(size, offset) of x, then of y: written = round(stored * size - offset)."""
        if self.space == "relative":
            return (self.scale, 0.0), (self.scale, 0.0)
        return (self.width, self.width / 2), (self.height, self.height / 2)

    def write(self, box: tuple) -> list[int]:
        (sx, ox), (sy, oy) = self._axes()
        left, top, right, bottom = box
        x0, y0, x1, y1 = (
            round(left * sx - ox),
            round(top * sy - oy),
            round(right * sx - ox),
            round(bottom * sy - oy),
        )
        return [y0, x0, y1, x1] if self.order == "yxyx" else [x0, y0, x1, y1]

    def read(self, ints: list[int]) -> tuple[float, float, float, float]:
        (sx, ox), (sy, oy) = self._axes()
        a, b, c, d = ints
        y0, x0, y1, x1 = (a, b, c, d) if self.order == "yxyx" else (b, a, d, c)

        # An odd side's edge rounds half a pixel out (+-771.5 -> +-772): clamp that back to
        # [0, 1]; anything further out is off the canvas
        def at(v: int, s: float, o: float) -> float:
            if not -0.5 <= v + o <= s + 0.5:
                raise ValueError(f"{list(ints)}: off the canvas of {self.header()}")
            return min(1.0, max(0.0, (v + o) / s))

        return (at(x0, sx, ox), at(y0, sy, oy), at(x1, sx, ox), at(y1, sy, oy))

    def snap(self, box: tuple) -> tuple[float, float, float, float]:
        """``box`` as it reads back from text."""
        return self.read(self.write(box))

    def header(self) -> str:
        size = f" canvas={self.width}x{self.height}" if self.space == "abs_centralized" else ""
        scale = f" scale={self.scale}" if self.space == "relative" else ""
        return f"<gbc bbox={self.order} space={self.space}{size}{scale}>"

    @classmethod
    def from_header(cls, line: str) -> "Coords":
        if HEADER.fullmatch(line) is None:
            raise ValueError(f"not a header: {line!r}")
        f = dict(FIELD.findall(line))
        w, h = (int(x) for x in f.get("canvas", "1024x1024").split("x"))
        space, order = f.get("space", "abs_centralized"), f.get("bbox", "yxyx")
        return cls(w, h, space, order, int(f.get("scale", 1000)))


MOSS = Coords()  # MOSS-Image's own: pixels from the centre of a 1024x1024 canvas, yxyx
