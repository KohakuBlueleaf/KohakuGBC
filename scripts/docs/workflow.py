"""The README's workflow figure: the GBC program as a node graph, and why it needs a machine.

Top, the program's top level (``gbc`` in gbc.kir) as nodes and control flow: the root, the
decision loop (a round of regions, or one more root sentence, until the node budget is met), the
relational questions, the graph. Bottom, what one round runs per region, side by side. Nodes
are coloured by the lane they run on; callouts mark where the path depends on a reply.

    python scripts/docs/workflow.py docs/assets/workflow.jpg
"""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib import patches

INK, MUTED, PAPER, CARD, LINE = "#1f2328", "#57606a", "#ffffff", "#f6f8fa", "#d0d7de"
LANE = {
    "vlm": ("#3b6fd8", "VLM"),
    "detect": ("#e6730d", "detect"),
    "embed": ("#2f9e44", "embed"),
    "cpu": ("#8a8f98", "CPU"),
    "inline": ("#1f2328", "inline"),
}
CALLOUT = "#cf222e"


class Fig:
    def __init__(self, ax, size):
        self.ax, self.s = ax, size
        self.at: dict[str, tuple[float, float, float, float]] = {}

    def node(self, key, x, y, title, ports, lane, w=2.0, h=1.0):
        """A node: a title bar in its lane's colour, its name, its ports below."""
        colour = LANE[lane][0]
        ax = self.ax
        style = "round,pad=0,rounding_size=0.08"
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=PAPER,
                                            edgecolor=colour, lw=2.0, zorder=3))  # fmt: skip
        bar = patches.FancyBboxPatch((x, y + h - 0.36), w, 0.36, boxstyle=style,
                                     facecolor=colour, edgecolor=colour, lw=2.0,
                                     zorder=3)  # fmt: skip
        ax.add_patch(bar)
        ax.text(x + 0.1, y + h - 0.18, title, fontsize=self.s - 0.5, color=PAPER,
                fontweight="bold", va="center", family="monospace", zorder=4)  # fmt: skip
        ax.text(x + 0.1, y + (h - 0.36) / 2, ports, fontsize=self.s - 4.5, color=MUTED,
                va="center", family="monospace", zorder=4, linespacing=1.25)  # fmt: skip
        # Control ports: in on the left, out on the right
        for px in (x, x + w):
            port = patches.RegularPolygon((px, y + h - 0.18), 3, radius=0.09,
                                          orientation=-1.5708, facecolor=INK, zorder=5)  # fmt: skip
            ax.add_patch(port)
        self.at[key] = (x, y, w, h)

    def side(self, key, where):
        x, y, w, h = self.at[key]
        return {
            "out": (x + w + 0.09, y + h - 0.18),
            "in": (x - 0.09, y + h - 0.18),
            "top": (x + w / 2, y + h),
            "bottom": (x + w / 2, y),
            "right": (x + w, y + h / 2),
            "left": (x, y + h / 2),
        }[where]

    def flow(self, a, b, rad=0.0, colour=INK, lw=2.4, label=None, label_at=0.5, dashed=False):
        self.ax.annotate("", xy=b, xytext=a, zorder=2, arrowprops={
            "arrowstyle": "-|>,head_length=0.6,head_width=0.3", "color": colour, "lw": lw,
            "connectionstyle": f"arc3,rad={rad}", "shrinkA": 0, "shrinkB": 0,
            "linestyle": (0, (5, 3)) if dashed else "-",
        })  # fmt: skip
        if label:
            mx = a[0] + (b[0] - a[0]) * label_at
            my = a[1] + (b[1] - a[1]) * label_at
            self.ax.text(mx, my, label, fontsize=self.s - 3, color=colour, ha="center",
                         va="center", zorder=6, family="monospace",
                         bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1.0})  # fmt: skip

    def note(self, x, y, text, colour=CALLOUT, ha="left"):
        self.ax.text(x, y, text, fontsize=self.s - 2.5, color=colour, ha=ha, va="top",
                     zorder=6, linespacing=1.3, fontweight="bold")  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--font", type=float, default=15.0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    s = args.font

    fig = plt.figure(figsize=(20, 9.4), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    f = Fig(ax, s)

    ax.text(0.25, 9.15, "the GBC program as a node graph", fontsize=s + 4, fontweight="bold",
            color=INK, va="top")  # fmt: skip
    ax.text(0.25, 8.68, "gbc.kir, the top level: one image in, one graph out; every image runs "
            "this same program", fontsize=s, color=MUTED, va="top")  # fmt: skip

    # ---- the top level -------------------------------------------------------------------------
    y = 6.2
    f.node("in", 0.25, y, "inputs", "image, captions,\ntags, controls", "inline", w=1.9)
    f.node("root", 2.75, y, "root", "→ records, found\nimage query", "vlm", w=1.9)
    f.node("push", 5.25, y, "push", "frontier, found\n→ frontier", "inline", w=1.7)
    f.node("next", 7.55, y, "next_step", "records, frontier,\nrounds → step", "inline", w=2.1)
    f.node("switch", 10.25, y, "switch", "step:\ndepth|breadth|done", "inline", w=2.1)
    f.node("depth", 13.1, 7.55, "depth_round", "frontier, records\n→ found, records", "vlm",
           w=2.35)  # fmt: skip
    f.node("cover", 13.1, 4.85, "coverage", "records → found,\ngrew", "vlm", w=2.35)
    f.node("relate", 13.1, y, "relate", "records\n→ records", "vlm", w=2.35)
    f.node("gbc", 16.05, y, "to_gbc", "records\n→ graph, stats", "cpu", w=1.85)
    f.node("out", 18.3, y, "outputs", "graph,\nstats", "inline", w=1.45)

    f.flow(f.side("in", "out"), f.side("root", "in"))
    f.flow(f.side("root", "out"), f.side("push", "in"))
    f.flow(f.side("push", "out"), f.side("next", "in"))
    f.flow(f.side("next", "out"), f.side("switch", "in"))
    f.flow(f.side("switch", "out"), f.side("depth", "in"), rad=0.15, label='"depth"', label_at=0.45)
    f.flow(f.side("switch", "out"), f.side("cover", "in"), rad=-0.15, label='"breadth"',
           label_at=0.45)  # fmt: skip
    f.flow(f.side("switch", "out"), f.side("relate", "in"), label='"done"', label_at=0.5)
    f.flow(f.side("relate", "out"), f.side("gbc", "in"))
    f.flow(f.side("gbc", "out"), f.side("out", "in"))
    # The loop: each round goes back to the decision
    f.flow(f.side("depth", "top"), f.side("next", "top"), rad=0.25, colour=INK, lw=2.0,
           label="jump decide", label_at=0.5)  # fmt: skip
    f.flow(f.side("cover", "bottom"), f.side("next", "bottom"), rad=-0.25, colour=INK, lw=2.0,
           label="jump decide", label_at=0.5)  # fmt: skip

    f.note(0.25, 5.85, "the root's reply names the objects:\nhow many regions an image has\n"
           "is known only at run time")  # fmt: skip
    f.note(4.3, 5.85, "rounds until the node budget:\nanother image stops earlier\nor goes "
           "deeper")  # fmt: skip

    # ---- one round, per region ------------------------------------------------------------------
    by = 0.35
    panel = patches.FancyBboxPatch((0.2, by), 19.6, 3.15, boxstyle="round,pad=0,rounding_size=0.1",
                                   facecolor=CARD, edgecolor=LINE, lw=1.4, zorder=0)  # fmt: skip
    ax.add_patch(panel)
    ax.text(0.4, by + 2.95, "inside depth_round: every region of the round, side by side "
            "(each → expand)", fontsize=s, fontweight="bold", color=INK, va="top")  # fmt: skip
    # depth_round opens into the panel below: down the gap between relate and to_gbc
    dx, dy = f.side("depth", "right")
    ax.plot([dx, 15.75, 15.75], [dy, dy, by + 3.4], color=MUTED, lw=1.6, ls=(0, (5, 3)),
            zorder=2)  # fmt: skip
    f.flow((15.75, by + 3.4), (15.75, by + 3.15), colour=MUTED, lw=1.6)
    ry = by + 1.0
    f.node("each", 0.4, ry, "each", "regions\n→ region", "inline", w=1.55)
    f.node("emb", 2.35, ry, "embedder", "name → vec", "embed", w=1.9)
    f.node("reg", 4.65, ry, "register", "seen, vec\n→ is_new", "inline", w=1.8)
    f.node("br", 6.85, ry, "branch", "is_new:\nnew | known", "inline", w=1.8)
    f.node("ask", 9.1, ry, "ask · chat", "region request\n→ answer", "vlm", w=2.1)
    f.node("parts", 11.6, ry, "parts", "answer\n→ its parts", "inline", w=1.7)
    f.node("crop", 13.7, ry, "crop", "image, box\n→ patch", "cpu", w=1.6)
    f.node("det", 15.7, ry, "detector", "patch, parts\n→ boxes", "detect", w=1.85)
    f.node("push2", 17.95, ry, "push", "boxes →\nnext round", "inline", w=1.65)
    for a, b in (("each", "emb"), ("emb", "reg"), ("reg", "br"), ("ask", "parts"),
                 ("parts", "crop"), ("crop", "det"), ("det", "push2")):  # fmt: skip
        f.flow(f.side(a, "out"), f.side(b, "in"))
    f.flow(f.side("br", "out"), f.side("ask", "in"), label='"new"', label_at=0.5)
    ax.text(7.75, ry - 0.12, '"known": merged into the vertex\nseen before; nothing asked',
            fontsize=s - 4, color=MUTED, ha="center", va="top")  # fmt: skip
    f.note(9.1, ry - 0.12, "each describe waits for its\nregion's same-object check;\nits "
           "parts wait for its reply")  # fmt: skip

    ax.text(12.9, ry - 0.1, "why a machine: each image takes its own path, a step at a time;\n"
            "batching stage by stage waits for the slowest image. Togeari runs\nevery image's "
            "path and batches whatever calls are ready, across images.", fontsize=s - 3.5,
            color=INK, va="top", linespacing=1.3, zorder=6)  # fmt: skip

    handles = [patches.Patch(color=c, label=n) for c, n in LANE.values()]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=5,
              frameon=False, fontsize=s - 2, title="node runs on",
              title_fontsize=s - 2)  # fmt: skip
    fig.savefig(args.out, dpi=args.dpi, facecolor=PAPER, pil_kwargs={"quality": 90})
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
