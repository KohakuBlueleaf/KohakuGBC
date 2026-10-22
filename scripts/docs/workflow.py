"""The README's workflow figure: the GBC procedure on one image, as a conceptual pipeline.

The root describes the image and finds what it names; the regions wait in a frontier. A round
takes the frontier and runs every region as its own job (seen before? describe, find parts); the
parts join the frontier. A decision then runs another round (depth), asks what the image still
holds (breadth, new objects join the frontier), or ends; relation questions run over groups of
nodes, and the graph is resolved. Every step marked ``×?`` spawns a number of jobs its reply
decides. Each step is coloured by the engine it needs.

    python scripts/docs/workflow.py docs/assets/workflow.jpg
"""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib import patches

INK, MUTED, PAPER, LINE, SKY = "#1f2328", "#57606a", "#ffffff", "#c4cbd3", "#dbe9f7"
VLM, DET, EMB, CPU = "#3b6fd8", "#e6730d", "#2f9e44", "#8a8f98"
SPAWN, CALLOUT = "#1a7f37", "#cf222e"
H_STEP = 0.6


class Draw:
    def __init__(self, ax, size):
        self.ax, self.s = ax, size

    def step(self, x, y, w, text, colour, stack=1, ink=PAPER):
        """A step: a filled card centred on ``y``; ``stack`` cards for many jobs of it."""
        for k in range(stack - 1, -1, -1):
            self.ax.add_patch(patches.FancyBboxPatch(
                (x + k * 0.1, y - H_STEP / 2 + k * 0.1), w, H_STEP,
                boxstyle="round,pad=0,rounding_size=0.08", facecolor=colour, edgecolor=PAPER,
                lw=1.5, zorder=5 - k * 0.1,
            ))  # fmt: skip
        self.ax.text(x + w / 2, y, text, fontsize=self.s - 2, color=ink, ha="center",
                     va="center", fontweight="bold", zorder=7)  # fmt: skip

    def spawn(self, x, y):
        """The mark of a step whose reply decides how many jobs follow it."""
        self.ax.add_patch(patches.Circle((x, y), 0.3, facecolor=SPAWN, edgecolor=PAPER, lw=1.5,
                                         zorder=8))  # fmt: skip
        self.ax.text(x, y, "×?", fontsize=self.s - 2, color=PAPER, ha="center", va="center",
                     fontweight="bold", zorder=9)  # fmt: skip

    def arrow(self, a, b, colour=MUTED, lw=1.8):
        self.ax.annotate("", xy=b, xytext=a, zorder=3, arrowprops={
            "arrowstyle": "-|>,head_length=0.5,head_width=0.25", "color": colour, "lw": lw,
            "shrinkA": 0, "shrinkB": 0,
        })  # fmt: skip

    def path(self, points, colour=SPAWN, lw=2.2):
        """An orthogonal path through ``points``, its arrowhead on the last one."""
        xs, ys = zip(*points[:-1])
        self.ax.plot(xs, ys, color=colour, lw=lw, zorder=3, solid_capstyle="butt")
        self.arrow(points[-2], points[-1], colour, lw)

    def label(self, x, y, text, colour=MUTED, ha="left", va="center", size=-3, bold=False):
        self.ax.text(x, y, text, fontsize=self.s + size, color=colour, ha=ha, va=va, zorder=9,
                     linespacing=1.3, fontweight="bold" if bold else "normal")  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--font", type=float, default=16.0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    s = args.font

    W, H = 22.0, 10.2
    fig = plt.figure(figsize=(W, H), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    d = Draw(ax, s)

    ax.text(0.3, H - 0.25, "GBC on one image: a workflow that grows while it runs",
            fontsize=s + 4, fontweight="bold", color=INK, va="top")  # fmt: skip
    ax.text(0.3, H - 0.78, "each reply decides how many jobs come next, and how many rounds run",
            fontsize=s, color=MUTED, va="top")  # fmt: skip
    legend = [patches.Patch(color=c, label=n) for c, n in
              ((VLM, "VLM"), (DET, "detector"), (EMB, "embedder"), (CPU, "CPU"),
               (SPAWN, "×? spawns jobs"))]  # fmt: skip
    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.995, 0.99), ncol=5,
              frameon=False, fontsize=s - 2.5)  # fmt: skip
    for x, title in ((0.3, "1 · root"), (5.6, "2 · a round: every region its own job"),
                     (15.1, "3 · decide"), (17.4, "4 · relations")):  # fmt: skip
        ax.text(x, H - 1.4, title, fontsize=s, color=INK, fontweight="bold", va="top")

    # ---- 1 · root: describe the image, find what it names --------------------------------------
    ax.add_patch(patches.FancyBboxPatch((0.3, 6.0), 1.5, 1.3, boxstyle="round,pad=0,"
                                        "rounding_size=0.08", facecolor=SKY, edgecolor=INK,
                                        lw=1.6, zorder=4))  # fmt: skip
    ax.add_patch(patches.Circle((1.4, 6.95), 0.17, facecolor="#f2b632", zorder=5))
    ax.add_patch(patches.Polygon([(0.38, 6.08), (0.85, 6.75), (1.3, 6.08)], facecolor=CPU,
                                 zorder=5))  # fmt: skip
    ax.add_patch(patches.Polygon([(0.95, 6.08), (1.35, 6.5), (1.72, 6.08)], facecolor=MUTED,
                                 zorder=5))  # fmt: skip
    d.label(2.0, 6.65, "image\n(+ captions, tags)", INK, size=-1.5, bold=True)
    d.step(0.3, 5.0, 4.2, "describe it, name objects", VLM)
    d.step(0.3, 3.9, 4.2, "find each name", DET)
    d.arrow((1.05, 6.0), (1.05, 5.3))
    d.arrow((2.4, 4.7), (2.4, 4.2))
    d.spawn(4.85, 3.9)
    d.label(0.3, 3.3, "how many objects, how many\nregions: known only when\nthe replies "
            "return", CALLOUT, va="top", bold=True)  # fmt: skip

    # ---- the frontier: regions waiting for a round ---------------------------------------------
    fx, fy0, fy1 = 5.6, 3.0, 6.4
    ax.add_patch(patches.FancyBboxPatch((fx, fy0), 1.3, fy1 - fy0, boxstyle="round,pad=0,"
                                        "rounding_size=0.1", facecolor=PAPER, edgecolor=INK,
                                        lw=1.6, zorder=4))  # fmt: skip
    d.label(fx + 0.65, fy1 - 0.3, "frontier", INK, ha="center", size=-2.5, bold=True)
    for k in range(4):
        ax.add_patch(patches.FancyBboxPatch((fx + 0.3, fy0 + 0.25 + k * 0.65), 0.7, 0.45,
                                            boxstyle="round,pad=0,rounding_size=0.05",
                                            facecolor=LINE, edgecolor=MUTED, lw=1.0,
                                            zorder=5))  # fmt: skip
    d.arrow((5.15, 3.9), (fx, 4.3), SPAWN, 2.2)

    # ---- 2 · a round: each region a job, side by side ------------------------------------------
    lanes = (6.6, 4.6, 2.6)
    for y in lanes:
        d.arrow((fx + 1.3, 4.7), (7.6, y), SPAWN, 2.0)
        d.step(7.6, y, 1.5, "seen?", EMB)
    for y in (5.6, 3.6):
        d.label(8.35, y, "⋮", INK, ha="center", size=2, bold=True)
    for y in lanes[:2]:
        d.arrow((9.1, y), (9.6, y))
        d.label(9.35, y + 0.42, "new", MUTED, ha="center", size=-4.5)
        d.step(9.6, y, 1.8, "describe", VLM)
        d.arrow((11.4, y), (11.8, y))
        d.step(11.8, y, 1.8, "find parts", DET)
        d.spawn(13.95, y)
    d.arrow((9.1, 2.6), (9.6, 2.6))
    d.label(9.35, 3.02, "seen", MUTED, ha="center", size=-4.5)
    d.step(9.6, 2.6, 4.0, "merged into it: branch ends", LINE, ink=INK)
    d.label(7.6, 7.4, "waves of jobs until the round's node budget is met", CALLOUT, bold=True)

    # Parts go back to the frontier through the decision
    ax.plot([14.25, 14.6], [6.6, 6.6], color=SPAWN, lw=2.2, zorder=3)
    ax.plot([14.25, 14.6], [4.6, 4.6], color=SPAWN, lw=2.2, zorder=3)
    ax.plot([14.6, 14.6], [6.6, 4.6], color=SPAWN, lw=2.2, zorder=3)
    dx, dy, r = 15.9, 4.6, 0.7
    d.arrow((14.6, 4.6), (dx - r, dy), SPAWN, 2.2)
    d.label(14.7, 5.6, "parts", SPAWN, size=-4, bold=True)

    # ---- 3 · decide: another round, more objects, or done --------------------------------------
    ax.add_patch(patches.Polygon([(dx - r, dy), (dx, dy + r), (dx + r, dy), (dx, dy - r)],
                                 facecolor=CPU, edgecolor=CPU, zorder=5))  # fmt: skip
    d.label(dx, dy, "next?", PAPER, ha="center", size=-2, bold=True)
    d.path([(dx, dy + r), (dx, 7.9), (fx + 0.65, 7.9), (fx + 0.65, fy1)])
    d.label(dx + 0.12, 6.3, "depth", SPAWN, size=-3, bold=True)
    d.label(10.2, 8.0, "depth: another round over the frontier, one level deeper", SPAWN,
            ha="center", va="bottom", size=-2.5, bold=True)  # fmt: skip

    d.arrow((dx, dy - r), (dx, 2.9))
    d.label(dx + 0.12, 3.5, "breadth", SPAWN, size=-3, bold=True)
    d.step(14.95, 2.6, 2.1, "name the rest", VLM)
    d.arrow((dx, 2.3), (dx, 1.9))
    d.step(14.95, 1.6, 2.1, "find them", DET)
    d.spawn(14.5, 1.6)
    d.path([(14.2, 1.6), (fx + 0.65, 1.6), (fx + 0.65, fy0)])
    d.label(10.2, 1.45, "breadth: new objects join the frontier", SPAWN, ha="center", va="top",
            size=-2.5, bold=True)  # fmt: skip

    # ---- 4 · relations over groups of the nodes found ------------------------------------------
    d.arrow((dx + r, dy), (17.4, dy))
    d.label(dx + r + 0.1, dy + 0.3, "done", MUTED, size=-3, bold=True)
    d.step(17.4, 4.6, 3.8, "group the nodes", CPU)
    d.spawn(19.3, 3.95)
    d.step(17.4, 3.2, 3.8, "ask how they relate", VLM, stack=3)
    d.arrow((19.3, 2.9), (19.3, 2.45))
    d.step(17.4, 2.15, 3.8, "GBC graph", INK)
    d.label(17.4, 1.65, "they wait for every branch", CALLOUT, va="top", bold=True)

    d.label(0.3, 0.5, "×?  a reply decides how many jobs follow it, and the decision how many "
            "rounds and breadth steps run: the job's shape exists only while it runs", INK,
            size=-1.5)  # fmt: skip
    fig.savefig(args.out, dpi=args.dpi, facecolor=PAPER, pil_kwargs={"quality": 90})
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
