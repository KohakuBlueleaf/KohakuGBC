"""The README's architecture teaser: GBC on the Togeari machine, in Togeari's own shape.

Five columns, left to right: the one GBC program (gbc.kir); the image jobs (code memory, the
source, admission, the job pool with real jobs of a recorded run, the sink); the dispatcher, an
out-of-order core (fetch per strand, rename, a job's window of GBC statements, issue, writeback,
retire in program order); the lanes (inline, the VLM stream lane, the detect and embed batch
lanes, the CPU call lane); the engines behind them. Completions return to the dispatcher's inbox.

    python scripts/docs/architecture.py JOBS.json docs/assets/architecture.jpg

``JOBS.json``: ``[{"image": path, "key": ..., "calls": ..., "strands": ..., "window": ...,
"vlm": ...}]``, three jobs of a recorded run (``job.end`` events' stats).
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps

INK, MUTED, PAPER, CARD, LINE = "#1f2328", "#57606a", "#ffffff", "#f6f8fa", "#d0d7de"
BLUE, VIOLET, GOLD, GREEN, GREY = "#0969da", "#8250df", "#bf8700", "#1a7f37", "#6e7781"
LANE = {"vlm": "#3b6fd8", "detect": "#e6730d", "embed": "#2f9e44", "cpu": "#8a8f98"}
JOB = ["#d6616b", "#9e5fd1", "#17a2b8"]

CODE = [
    "@def (image, ..., controls)gbc(graph):",
    "  (image, ...)root(records, found)",
    "  decide:",
    "    (records, ...)next_step(step)",
    "    (step)switch(...)",
    "    depth:",
    "      (...)depth_round(...)",
    "      ()jump(`decide`)",
    "  (image, records)relate(records)",
    "  (image, records)to_gbc(graph)",
    "",
    "@def (image, region, ...)expand(...):",
    '  @hint lane="embed"',
    "  (text)embedder(vec)",
    "  (seen, region, vec)register(...)",
    "  (is_new)branch(`new`, `known`)",
    "  new:",
    "    (request)ask(answer)  # chat",
    "    (image, record)detect(found)",
]
WINDOW = [
    ("chat", "request#3 → reply#4", "done"),
    ("detector", "patch#9, req#8 → raw#10", "issued"),
    ("embedder", "text#14 → vec#15", "issued"),
    ("chat", "request#21 → answer#22", "issued"),
    ("register", "seen#12, vec#15", "waits vec#15"),
    ("relate", "image#1, records#30 …", "waits records"),
]


def box(ax, x, y, w, h, face=CARD, edge=LINE, lw=1.4, r=0.1, z=1):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=face,
            edgecolor=edge,
            lw=lw,
            zorder=z,
        )
    )


def arrow(ax, a, b, colour=MUTED, lw=1.8, rad=0.0, dashed=False):
    ax.annotate(
        "",
        xy=b,
        xytext=a,
        zorder=6,
        arrowprops={
            "arrowstyle": "-|>,head_length=0.6,head_width=0.3",
            "color": colour,
            "lw": lw,
            "connectionstyle": f"arc3,rad={rad}",
            "linestyle": (0, (5, 3)) if dashed else "-",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def text(ax, x, y, s, size, colour=INK, bold=False, mono=False, **kw):
    weight = "bold" if bold else "normal"
    family = "monospace" if mono else None
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=colour,
        fontweight=weight,
        family=family,
        va=kw.pop("va", "top"),
        zorder=7,
        **kw,
    )


def thumb(ax, path, x, y, zoom, edge):
    img = ImageOps.fit(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), (160, 160))
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            (x, y),
            frameon=True,
            pad=0.06,
            bboxprops={"edgecolor": edge, "lw": 2.4},
            zorder=8,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs")
    parser.add_argument("out")
    parser.add_argument("--font", type=float, default=14.0)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    with open(args.jobs, encoding="utf-8") as f:
        jobs = json.load(f)[:3]
    s = args.font

    fig = plt.figure(figsize=(21, 9.6), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    top = 9.35

    # ---- 1. one program ---------------------------------------------------------------------
    text(ax, 0.25, top, "one GBC program", s + 4, bold=True)
    text(ax, 0.25, top - 0.38, "written sequentially: gbc.kir", s)
    box(ax, 0.2, 1.15, 4.75, 7.55, face=CARD)
    for i, line in enumerate(CODE):
        colour = (
            LANE["embed"]
            if "embedder" in line or "lane=" in line
            else (LANE["vlm"] if "ask" in line else INK)
        )
        text(ax, 0.35, 8.55 - i * 0.37, line, s - 1.5, colour=colour, mono=True)
    text(
        ax,
        0.25,
        0.95,
        "one image in, one GBC graph out;\nevery choice in `controls`",
        s - 1,
        colour=MUTED,
    )

    # ---- 2. many image jobs -------------------------------------------------------------------
    x2 = 5.35
    text(ax, x2, top, "many image jobs", s + 4, bold=True)
    text(ax, x2, top - 0.38, "each a thread of the program", s)
    box(ax, x2, 7.55, 3.6, 1.1)
    text(ax, x2 + 0.15, 8.55, "Code memory", s, bold=True)
    text(
        ax,
        x2 + 0.15,
        8.18,
        "gbc.kir decoded once, shared;\na job holds only addresses",
        s - 2,
        colour=MUTED,
    )
    box(ax, x2, 6.2, 3.6, 1.15)
    text(ax, x2 + 0.15, 7.25, "Source", s, bold=True)
    text(ax, x2 + 0.15, 6.88, "folder · jsonl · trace", s - 2, colour=MUTED)
    for k, job in enumerate(jobs):
        thumb(ax, job["image"], x2 + 2.35 + k * 0.48, 6.7, 0.21, JOB[k])
    box(ax, x2, 5.15, 3.6, 0.85)
    text(ax, x2 + 0.15, 5.9, "Admission", s, bold=True)
    text(ax, x2 + 0.15, 5.53, "while the VLM lane's load is low", s - 2, colour=MUTED)
    box(ax, x2, 1.95, 3.6, 3.0, face=PAPER, edge=INK)
    text(ax, x2 + 0.15, 4.85, "Job pool", s, bold=True)
    for k, job in enumerate(jobs):
        y = 4.35 - k * 0.8
        box(ax, x2 + 0.15, y - 0.68, 3.3, 0.7, face=CARD, edge=JOB[k], lw=1.8, z=2)
        thumb(ax, job["image"], x2 + 0.5, y - 0.33, 0.19, JOB[k])
        text(ax, x2 + 0.9, y - 0.08, f"job {job['key']}", s - 1.5, bold=True)
        text(
            ax,
            x2 + 0.9,
            y - 0.36,
            f"strands {job['strands']} · window {job['window']}",
            s - 3,
            colour=MUTED,
        )
    box(ax, x2, 0.25, 3.6, 1.5, face=PAPER, edge=GREEN, lw=1.8)
    text(ax, x2 + 0.15, 1.65, "Sink", s, colour=GREEN, bold=True)
    text(
        ax,
        x2 + 0.15,
        1.28,
        "a GBC graph per image → graphs.jsonl;\nrecorded jobs replay "
        "through KIR's\nreference interpreter to the same graph",
        s - 2.5,
        colour=INK,
    )
    arrow(ax, (4.95, 8.1), (x2, 8.1), colour=INK)

    # ---- 3. the dispatcher ---------------------------------------------------------------------
    x3, w3 = 9.4, 5.0
    text(ax, x3, top, "Dispatcher", s + 4, colour=BLUE, bold=True)
    text(ax, x3, top - 0.38, "one thread, out of order, like a CPU core", s)
    box(ax, x3, 0.25, w3, 8.45, face=PAPER, edge=BLUE, lw=2.0, z=0)
    rows = [
        ("Fetch, per strand", "every region of a round, code past a pending branch", 8.55),
        ("Rename", "a read → its last write's slot; a write → a fresh slot", 7.7),
    ]
    for title, body, y in rows:
        text(ax, x3 + 0.2, y, title, s, bold=True)
        text(ax, x3 + 0.2, y - 0.36, body, s - 2.5, colour=MUTED)
    text(ax, x3 + 0.2, 6.85, f"Window (job {jobs[0]['key']})", s, bold=True)
    for i, (node, slots, state) in enumerate(WINDOW):
        y = 6.4 - i * 0.42
        box(ax, x3 + 0.2, y - 0.34, w3 - 0.4, 0.36, face=CARD, edge=LINE, lw=1.0, r=0.05, z=2)
        lane = {"chat": "vlm", "detector": "detect", "embedder": "embed"}.get(node)
        text(
            ax,
            x3 + 0.3,
            y - 0.06,
            node,
            s - 2.5,
            colour=LANE[lane] if lane else INK,
            mono=True,
            bold=bool(lane),
        )
        text(ax, x3 + 1.35, y - 0.06, slots, s - 3, mono=True)
        text(
            ax,
            x3 + w3 - 0.3,
            y - 0.06,
            state,
            s - 3,
            colour=GREEN if state == "done" else (BLUE if state == "issued" else MUTED),
            ha="right",
        )
    rows = [
        ("Ready set + issue", "across every job: oldest job first", 3.65),
        ("Writeback", "fill the slot, wake its readers, resolve branches", 2.8),
        ("Retire, in program order per job", "records → to_gbc → the graph → the sink", 1.95),
    ]
    for title, body, y in rows:
        text(ax, x3 + 0.2, y, title, s, bold=True)
        text(ax, x3 + 0.2, y - 0.36, body, s - 2.5, colour=MUTED)
    text(
        ax,
        x3 + 0.2,
        0.95,
        "a statement issues once its inputs are filled, in any order;\n"
        "its results still retire in program order",
        s - 2.5,
        colour=INK,
    )
    arrow(ax, (x2 + 3.6, 3.5), (x3, 3.5), colour=INK)
    arrow(ax, (x3, 1.4), (x2 + 3.6, 1.0), colour=GREEN)

    # ---- 4. lanes --------------------------------------------------------------------------------
    x4, w4 = 14.8, 2.75
    text(ax, x4, top, "lanes", s + 4, bold=True)
    text(ax, x4, top - 0.38, "queue, batch, pick a unit", s)
    lanes = [
        (
            "INLINE · on the dispatcher",
            "kir.std, next_step, take, push:\nno handoff",
            INK,
            8.6,
            1.0,
        ),
        (
            "STREAM LANE · vlm",
            "chat: paces only; the engine\nbatches; load → admission",
            LANE["vlm"],
            7.35,
            1.15,
        ),
        (
            "BATCH LANE · detect",
            "detector: former size · wait;\ncrops of several images",
            LANE["detect"],
            5.95,
            1.15,
        ),
        (
            "BATCH LANE · embed",
            "embedder: names of several\nimages in one forward",
            LANE["embed"],
            4.55,
            1.15,
        ),
        (
            "CALL LANE · cpu",
            "crops, pictures, parsing,\nresolution: a unit per item",
            LANE["cpu"],
            3.15,
            1.15,
        ),
    ]
    for title, body, colour, y, h in lanes:
        box(ax, x4, y - h, w4, h, face=PAPER, edge=colour, lw=1.8)
        text(ax, x4 + 0.12, y - 0.1, title, s - 2, colour=colour, bold=True)
        text(ax, x4 + 0.12, y - 0.45, body, s - 3.2, colour=INK)
        if title.startswith(("STREAM", "BATCH", "CALL")):
            for q in range(5):
                ax.add_patch(
                    patches.Rectangle(
                        (x4 + w4 - 0.23 - q * 0.21, y - h + 0.1),
                        0.16,
                        0.2,
                        facecolor=JOB[q % 3],
                        edgecolor=PAPER,
                        zorder=4,
                    )
                )
    text(
        ax, x4, 1.9, "calls of every job meet here;\nseveral units per lane", s - 2.5, colour=MUTED
    )
    arrow(ax, (x3 + w3, 4.2), (x4, 4.2), colour=INK)

    # ---- 5. engines ------------------------------------------------------------------------------
    x5, w5 = 17.95, 2.85
    text(ax, x5, top, "engines", s + 4, bold=True)
    text(ax, x5, top - 0.38, "behind one contract", s)
    engines = [
        (
            "Stream engine",
            "API (OpenAI, Anthropic),\nvLLM in-process, qwen35",
            VIOLET,
            7.35,
            1.15,
            6.8,
        ),
        (
            "Batch engine",
            "TorchEngine: OWLv2 or SAM 3;\none batch mixes images",
            GOLD,
            5.95,
            1.15,
            5.4,
        ),
        ("Batch engine", "TorchEngine: EmbeddingGemma", GOLD, 4.55, 1.15, 4.0),
        ("Call engine", "Python workers:\nthreads or processes", GREEN, 3.15, 1.15, 2.6),
    ]
    for title, body, colour, y, h, mid in engines:
        box(ax, x5, y - h, w5, h, face=CARD, edge=colour, lw=1.6)
        text(ax, x5 + 0.12, y - 0.1, title, s - 1.5, colour=colour, bold=True)
        text(ax, x5 + 0.12, y - 0.45, body, s - 3.2)
        arrow(ax, (x4 + w4, mid), (x5, mid), colour=colour, lw=2.0)
    text(ax, x5, 1.9, "remote units: engines\non other nodes", s - 2.5, colour=MUTED)
    # Completions come back to the dispatcher's inbox, along the bottom
    xs = [x5 + w5 - 0.25, x5 + w5 - 0.25, x3 + w3]
    ys = [2.0, 0.45, 0.45]
    ax.plot(xs[:2], ys[:2], color=BLUE, lw=2.0, zorder=6, solid_capstyle="round")
    arrow(ax, (xs[1], ys[1]), (xs[2], ys[2]), colour=BLUE, lw=2.0)
    text(
        ax,
        x4 + 0.1,
        0.92,
        "results from every engine → the dispatcher's inbox → writeback",
        s - 2,
        colour=BLUE,
    )
    fig.savefig(args.out, dpi=args.dpi, facecolor=PAPER, pil_kwargs={"quality": 90})
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
