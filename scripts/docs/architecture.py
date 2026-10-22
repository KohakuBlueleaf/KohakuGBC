"""The README's architecture view: KohakuGBC as one picture.

Images enter through admission; each becomes a job running the one GBC program (its steps in
order, each marked with the lanes it calls); the calls of every job meet on four lanes, which
batch them across images onto the engine units behind them; each job ends as a GBC graph, and a
recorded job replays through KIR's reference interpreter to the same graph.

    python scripts/docs/architecture.py IMAGE [IMAGE ...] docs/assets/architecture.jpg
"""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps

INK, MUTED, PAPER, CARD, LINE = "#1f2328", "#57606a", "#ffffff", "#f6f8fa", "#d0d7de"
LANES = {
    "vlm": ("#3b6fd8", "VLM"),
    "detect": ("#e6730d", "detect"),
    "embed": ("#2f9e44", "embed"),
    "cpu": ("#8a8f98", "CPU"),
}
JOBS = ["#d6616b", "#9e5fd1", "#17a2b8"]
STEPS = [
    ("root", "short and detailed captions naming the objects", ["vlm"]),
    ("detect", "boxes for the named objects in the parent's box", ["cpu", "detect"]),
    ("expand", "per region: merge with a seen vertex, or describe it", ["embed", "vlm", "detect"]),
    ("coverage", "one more root sentence for objects missed", ["vlm"]),
    ("relate", "relation and composition questions", ["vlm"]),
    ("to_gbc", "records resolved to the GBC graph", ["cpu"]),
]
ENGINES = {
    "vlm": "OpenAI-compatible / Anthropic API\nvLLM in-process · qwen35 engine",
    "detect": "OWLv2 · SAM 3\ncrops of many images per forward",
    "embed": "EmbeddingGemma\nnames of many images per forward",
    "cpu": "Python workers\nthreads or processes",
}


def box(ax, x, y, w, h, face=CARD, edge=LINE, lw=1.6, radius=0.12, z=1):
    style = f"round,pad=0,rounding_size={radius}"
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=face,
                                        edgecolor=edge, lw=lw, zorder=z))  # fmt: skip


def arrow(ax, a, b, colour=MUTED, lw=2.0, style="-|>", rad=0.0, dashed=False, z=2):
    ax.annotate("", xy=b, xytext=a, zorder=z, arrowprops={
        "arrowstyle": f"{style},head_length=0.6,head_width=0.3", "color": colour, "lw": lw,
        "connectionstyle": f"arc3,rad={rad}", "linestyle": (0, (5, 3)) if dashed else "-",
        "shrinkA": 0, "shrinkB": 0,
    })  # fmt: skip


def thumb(path: str, side: int = 220) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return ImageOps.fit(img, (side, int(side * 1.25)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="a few images for the source, then the output")
    parser.add_argument("--font", type=float, default=15.0)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    *images, out = args.images
    size = args.font

    fig = plt.figure(figsize=(18, 8.6), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    # ---- images and admission ---------------------------------------------------------------
    ax.text(0.35, 8.2, "images", fontsize=size + 1, fontweight="bold", color=INK, va="center")
    for k, path in enumerate(images[:3]):
        y = 6.55 - k * 1.75
        ab = AnnotationBbox(OffsetImage(thumb(path), zoom=0.33), (1.05, y), frameon=True,
                            pad=0.08, bboxprops={"edgecolor": JOBS[k % 3], "lw": 3})  # fmt: skip
        ax.add_artist(ab)
        arrow(ax, (1.85, y), (2.7, 4.75 - (k - 1) * 0.25), colour=JOBS[k % 3], lw=2.2)
    ax.text(1.05, 1.15, "folder · jsonl · trace\nmillions of them", fontsize=size - 2,
            color=MUTED, ha="center", va="center")  # fmt: skip
    box(ax, 2.7, 3.95, 1.25, 1.6, face=PAPER, edge=INK, lw=1.8)
    ax.text(3.32, 4.75, "admit\nwhile the\nVLM lane\nhas room", fontsize=size - 2.5, ha="center",
            va="center", color=INK)  # fmt: skip

    # ---- the job: one GBC program ------------------------------------------------------------
    jx, jy, jw, jh = 4.45, 1.45, 5.85, 6.55
    for k in (2, 1):
        box(ax, jx + 0.16 * k, jy + 0.16 * k, jw, jh, face=PAPER, edge=JOBS[k], lw=1.8, z=0)
    box(ax, jx, jy, jw, jh, face=PAPER, edge=JOBS[0], lw=2.4, z=1)
    arrow(ax, (3.95, 4.75), (jx, 4.75), colour=INK, lw=2.2)
    ax.text(jx + 0.25, jy + jh - 0.32, "one image job: the GBC program (gbc.kir)",
            fontsize=size + 1, fontweight="bold", color=INK, va="center", zorder=3)  # fmt: skip
    step_h, gap = 0.72, 0.16
    top = jy + jh - 0.75
    anchors = {}
    step_w = jw - 1.25
    for i, (name, what, lanes) in enumerate(STEPS):
        y = top - i * (step_h + gap) - step_h
        box(ax, jx + 0.3, y, step_w, step_h, face=CARD, edge=LINE, lw=1.4, radius=0.1, z=3)
        ax.text(jx + 0.47, y + step_h / 2 + 0.14, name, fontsize=size, fontweight="bold",
                color=INK, va="center", zorder=4, family="monospace")  # fmt: skip
        ax.text(jx + 0.47, y + step_h / 2 - 0.17, what, fontsize=size - 4, color=MUTED,
                va="center", zorder=4)  # fmt: skip
        # The lanes this step calls: dots in a row beside the step
        for k, lane in enumerate(lanes):
            cx = jx + 0.3 + step_w + 0.2 + k * 0.27
            cy = y + step_h / 2
            ax.scatter([cx], [cy], s=170, color=LANES[lane][0], zorder=5, edgecolors=PAPER)
            anchors.setdefault(lane, []).append((cx, cy))
    # The rounds: detect and expand repeat until the node budget is met
    y_detect = top - 1 * (step_h + gap) - step_h / 2
    y_expand = top - 2 * (step_h + gap) - step_h / 2
    arrow(ax, (jx + 0.3, y_expand), (jx + 0.3, y_detect), colour=INK, lw=1.8, rad=-0.9)
    ax.text(jx + 0.06, (y_detect + y_expand) / 2, "rounds", fontsize=size - 5, color=INK,
            rotation=90, ha="center", va="center")  # fmt: skip
    ax.text(jx + 0.3, jy + 0.42, "calls issue when their inputs exist; a pending branch runs\n"
            "ahead; a round's regions run side by side; retired in order", fontsize=size - 4.5,
            color=MUTED, ha="left", va="center", zorder=4, linespacing=1.35)  # fmt: skip

    # ---- lanes and engines --------------------------------------------------------------------
    lx, lw_, ex, ew = 11.25, 2.55, 14.35, 3.4
    ax.text(lx, 8.2, "lanes", fontsize=size + 1, fontweight="bold", color=INK, va="center")
    ax.text(lx, 7.75, "every job's calls of one kind,\nbatched across images", fontsize=size - 4,
            color=MUTED, va="center", linespacing=1.3)  # fmt: skip
    ax.text(ex, 8.2, "engine units", fontsize=size + 1, fontweight="bold", color=INK,
            va="center")  # fmt: skip
    ax.text(ex, 7.75, "several per lane, in this process\nor on other nodes", fontsize=size - 4,
            color=MUTED, va="center", linespacing=1.3)  # fmt: skip
    lane_y = {}
    for i, (lane, (colour, name)) in enumerate(LANES.items()):
        y = 6.6 - i * 1.5
        lane_y[lane] = y
        box(ax, lx, y - 0.5, lw_, 1.0, face=PAPER, edge=colour, lw=2.4, radius=0.1)
        ax.text(lx + 0.15, y + 0.25, name, fontsize=size, fontweight="bold", color=colour,
                va="center")  # fmt: skip
        # A queue of calls from several images
        for q in range(7):
            ax.add_patch(patches.Rectangle((lx + 0.18 + q * 0.31, y - 0.35), 0.24, 0.3,
                                           facecolor=JOBS[(q * 2 + i) % 3], edgecolor=PAPER,
                                           lw=0.8, zorder=3))  # fmt: skip
        box(ax, ex, y - 0.5, ew, 1.0, face=CARD, edge=colour, lw=1.6, radius=0.1)
        ax.text(ex + 0.18, y, ENGINES[lane], fontsize=size - 3, color=INK, va="center",
                linespacing=1.3)  # fmt: skip
        arrow(ax, (lx + lw_, y), (ex, y), colour=colour, lw=2.6)
        for cx, cy in anchors.get(lane, []):
            arrow(ax, (cx + 0.1, cy), (lx, y), colour=colour, lw=1.5, rad=0.0)

    # ---- the graph and the proof ---------------------------------------------------------------
    gx, gy = 11.25, 0.35
    box(ax, gx, gy, 6.5, 0.85, face=CARD, edge=INK, lw=1.8, radius=0.1)
    ax.text(gx + 0.2, gy + 0.6, "a GBC graph per image → graphs.jsonl", fontsize=size,
            fontweight="bold", color=INK, va="center")  # fmt: skip
    ax.text(gx + 0.2, gy + 0.25, "recorded jobs replay through KIR's reference interpreter to the "
            "same graph", fontsize=size - 4, color=MUTED, va="center")  # fmt: skip
    arrow(ax, (jx + jw * 0.5, jy), (gx, gy + 0.45), colour=INK, lw=2.2, rad=0.25)
    fig.savefig(out, dpi=args.dpi, facecolor=PAPER, pil_kwargs={"quality": 90})
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
