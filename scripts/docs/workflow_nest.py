"""The GBC-Nest workflow figure, beside GBC's (``workflow.py``): the same rounds, with the root
grounded, joins by head noun, relations among the root's elements only, and a last stage that
shapes the graph: one question per group (separate things, or pieces of one area), identities
merged, and parents from box containment, one question per proposed nesting.

    python scripts/docs/workflow_nest.py docs/assets/workflow_nest.jpg
"""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib import patches
from workflow import CALLOUT, CPU, DET, EMB, INK, MUTED, PAPER, SPAWN, VLM, Draw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--font", type=float, default=16.0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    s = args.font

    W, H = 27.0, 10.2
    fig = plt.figure(figsize=(W, H), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    d = Draw(ax, s)

    ax.text(
        0.3,
        H - 0.25,
        "GBC-Nest on one image: GBC's rounds, a grounded root, a shaped graph",
        fontsize=s + 4,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.3,
        H - 0.78,
        "parents come from where the boxes are, each nesting and each group checked by one "
        "question",
        fontsize=s,
        color=MUTED,
        va="top",
    )
    legend = [
        patches.Patch(color=c, label=n)
        for c, n in (
            (VLM, "VLM"),
            (DET, "detector"),
            (EMB, "embedder"),
            (CPU, "CPU"),
            (SPAWN, "×? spawns jobs"),
        )
    ]
    ax.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.99),
        ncol=5,
        frameon=False,
        fontsize=s - 2.5,
    )
    for x, title in (
        (0.3, "1 · grounded root"),
        (6.3, "2 · a round, as GBC's"),
        (15.2, "3 · decide"),
        (17.2, "4 · relations"),
        (21.4, "5 · shape"),
    ):
        ax.text(x, H - 1.4, title, fontsize=s, color=INK, fontweight="bold", va="top")

    # ---- 1 · root -------------------------------------------------------------------------------
    d.step(0.3, 7.2, 5.0, "describe it, name objects", VLM)
    d.arrow((2.8, 6.9), (2.8, 6.45))
    d.step(0.3, 6.15, 5.0, "+ every listed element, + probes", CPU)
    d.arrow((2.8, 5.85), (2.8, 5.4))
    d.step(0.3, 5.1, 5.0, "find each name and probe", DET)
    d.arrow((2.8, 4.8), (2.8, 4.35))
    d.step(0.3, 4.05, 5.0, "one box per head noun", CPU)
    d.spawn(5.65, 4.05)
    d.label(
        0.3,
        3.4,
        "a single element takes the largest\nof its near-tied boxes; a probe box\n"
        "(words, logo, panel, person) only\nwhere no named element is",
        CALLOUT,
        va="top",
        bold=True,
    )

    # ---- 2 · a round ----------------------------------------------------------------------------
    fx = 6.3
    d.arrow((5.95, 4.05), (fx + 0.4, 4.05), SPAWN, 2.2)
    lanes = (6.6, 4.6, 2.6)
    for y in lanes:
        d.step(fx + 0.5, y, 2.4, "seen? (head noun)", EMB)
        d.arrow((fx + 0.4, 4.05), (fx + 0.5, y), SPAWN, 1.6)
    for y in lanes[:2]:
        d.arrow((fx + 2.9, y), (fx + 3.3, y))
        d.step(fx + 3.3, y, 2.4, "describe", VLM)
        d.arrow((fx + 5.7, y), (fx + 6.1, y))
        d.step(fx + 6.1, y, 2.0, "find parts", DET)
        d.spawn(fx + 8.45, y)
    d.arrow((fx + 2.9, 2.6), (fx + 3.3, 2.6))
    d.step(fx + 3.3, 2.6, 4.8, "tiny: a leaf, no question", "#c4cbd3", ink=INK)
    d.label(
        fx + 0.5,
        7.5,
        "parts: what a person would mention first;\nprobe regions first, outside the budget",
        CALLOUT,
        bold=True,
    )

    # ---- 3 · decide -----------------------------------------------------------------------------
    dx, dy, r = 15.9, 4.6, 0.7
    ax.plot([fx + 8.75, 15.0], [6.6, 6.6], color=SPAWN, lw=2.2, zorder=3)
    ax.plot([fx + 8.75, 15.0], [4.6, 4.6], color=SPAWN, lw=2.2, zorder=3)
    ax.plot([15.0, 15.0], [6.6, 4.6], color=SPAWN, lw=2.2, zorder=3)
    d.arrow((15.0, 4.6), (dx - r, dy), SPAWN, 2.2)
    ax.add_patch(
        patches.Polygon(
            [(dx - r, dy), (dx, dy + r), (dx + r, dy), (dx, dy - r)],
            facecolor=CPU,
            edgecolor=CPU,
            zorder=5,
        )
    )
    d.label(dx, dy, "next?", PAPER, ha="center", size=-2, bold=True)
    d.path([(dx, dy + r), (dx, 8.1), (fx + 0.4, 8.1), (fx + 0.4, 7.9)])
    d.label(dx + 0.12, 6.3, "depth", SPAWN, size=-3, bold=True)

    # ---- 4 · relations --------------------------------------------------------------------------
    d.path([(dx + r, dy), (16.95, dy), (16.95, 5.6), (17.2, 5.6)], MUTED, 1.8)
    d.step(17.2, 5.6, 3.7, "group same-name boxes", CPU)
    d.arrow((19.05, 5.3), (19.05, 4.9))
    d.step(17.2, 4.6, 3.7, "describe each group", VLM, stack=2)
    d.arrow((19.05, 4.3), (19.05, 3.9))
    d.step(17.2, 3.6, 3.7, "interactions (root)", VLM)
    d.label(17.2, 2.9, "positions and restated\nhierarchy not asked", CALLOUT, va="top", bold=True)

    # ---- 5 · shape ------------------------------------------------------------------------------
    d.path([(20.9, 3.6), (21.15, 3.6), (21.15, 6.6), (21.4, 6.6)], MUTED, 1.8)
    d.step(21.4, 6.6, 4.8, "separate or pieces? (per group)", VLM, stack=2)
    d.spawn(26.6, 6.6)
    d.arrow((23.8, 6.3), (23.8, 5.9))
    d.step(21.4, 5.6, 4.8, "collapse pieces, merge same", CPU)
    d.arrow((23.8, 5.3), (23.8, 4.9))
    d.step(21.4, 4.6, 4.8, "inside its holder? (per move)", VLM, stack=2)
    d.spawn(26.6, 4.6)
    d.arrow((23.8, 4.3), (23.8, 3.9))
    d.step(21.4, 3.6, 4.8, "parents from containment", CPU)
    d.arrow((23.8, 3.3), (23.8, 2.9))
    d.step(21.4, 2.6, 4.8, "GBC graph (a DAG)", INK)
    d.label(
        21.4,
        2.0,
        "a vertex may hang from several\nholders; a group moves whole",
        CALLOUT,
        va="top",
        bold=True,
    )

    d.label(
        0.3,
        0.5,
        "GBC's call tree is its graph; GBC-Nest keeps GBC's rounds for what to describe and "
        "takes the hierarchy from the boxes, checked by short yes/no questions",
        INK,
        size=-1.5,
    )
    fig.savefig(args.out, dpi=args.dpi, facecolor=PAPER, pil_kwargs={"quality": 90})
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
