"""The README's machine view: one run of the GBC machine from its event log (``events = "all"``).

Left, every image job as a row and every engine call it made as a bar on its lane's colour, so
the run reads as images overlapping on shared engines. Right, the calls in flight on each lane
over time, and one image job alone: its calls stacked where they overlap.

The run behind the README's figure: ``configs/machine/replay_text.py`` (30 recorded images on
replay engines at their recorded latencies, VLM capacity 32) with ``machine.knobs.events = "all"``
and a run report.

    python scripts/docs/machine_view.py runs/machine/view/report/events.jsonl \
        docs/assets/machine.jpg
"""

import argparse
import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

LANES = {
    "vlm": ("#3b6fd8", "VLM"),
    "detect": ("#e6730d", "detector"),
    "embed": ("#2f9e44", "embedder"),
    "cpu": ("#8a8f98", "CPU"),
}
INK, MUTED, PAPER = "#1f2328", "#6e7781", "#ffffff"


def load(path: str) -> tuple[dict, list[dict], dict]:
    """(admit time and key per job, calls with lane, node, job, start, end, batch sizes)."""
    jobs, calls, batches = {}, [], defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            kind = e["kind"]
            if kind == "job.admit":
                jobs[e["job"]] = {"key": e["key"], "admit": e["t"]}
            elif kind == "job.end" and e["job"] in jobs:
                jobs[e["job"]]["end"] = e["t"]
            elif kind == "lane.result":
                calls.append(
                    {
                        "lane": e["lane"],
                        "node": e.get("node", ""),
                        "job": e["job"],
                        "start": e["t"] - e["service_s"],
                        "end": e["t"],
                    }
                )
            elif kind == "lane.submit":
                batches[e["lane"]].append(e["items"])
    t0 = min(j["admit"] for j in jobs.values())
    for j in jobs.values():
        j["admit"] -= t0
        j["end"] = j.get("end", t0) - t0
    for c in calls:
        c["start"] -= t0
        c["end"] -= t0
    return jobs, calls, batches


def gantt(ax, jobs, calls, size) -> None:
    order = sorted(jobs, key=lambda j: jobs[j]["admit"])
    row = {j: i for i, j in enumerate(order)}
    for j in order:
        a, e = jobs[j]["admit"], jobs[j]["end"]
        ax.plot([a, e], [row[j], row[j]], color="#d0d7de", lw=1.2, zorder=0)
    for c in calls:
        if c["job"] not in row:
            continue
        colour = LANES[c["lane"]][0]
        height = 0.72 if c["lane"] == "vlm" else 0.5
        ax.add_patch(
            patches.Rectangle(
                (c["start"], row[c["job"]] - height / 2),
                max(c["end"] - c["start"], 0.05),
                height,
                facecolor=colour,
                edgecolor="none",
                alpha=0.85 if c["lane"] == "vlm" else 0.95,
            )
        )
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlim(0, max(j["end"] for j in jobs.values()) * 1.01)
    ax.set_yticks([])
    ax.set_xlabel("seconds", fontsize=size)
    ax.set_ylabel(f"{len(order)} images, in admission order", fontsize=size)
    ax.tick_params(labelsize=size - 2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [patches.Patch(color=c, label=name) for c, name in LANES.values()]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=4,
        fontsize=size - 1,
        frameon=False,
        bbox_to_anchor=(1.0, 1.07),
    )


def in_flight(ax, calls, end, size, capacity: int | None) -> None:
    t = np.linspace(0, end, 600)
    if capacity:
        ax.axhline(capacity, color=LANES["vlm"][0], lw=1.0, ls=(0, (4, 3)), alpha=0.6)
        ax.text(
            end * 0.02,
            capacity * 1.03,
            f"VLM lane capacity {capacity}",
            fontsize=size - 2,
            color=LANES["vlm"][0],
            va="bottom",
        )
    for lane, (colour, name) in LANES.items():
        spans = [(c["start"], c["end"]) for c in calls if c["lane"] == lane]
        n = np.zeros_like(t)
        for s, e in spans:
            n += (t >= s) & (t < e)
        ax.plot(t, n, color=colour, lw=2.4, label=name)
    ax.set_xlim(0, end)
    ax.set_ylim(0, (capacity or 0) * 1.25 or None)
    ax.set_title("calls in flight per lane", fontsize=size, loc="left", color=INK)
    ax.tick_params(labelsize=size - 2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def one_job(ax, jobs, calls, size) -> None:
    """The job with the most VLM calls in flight at once: its calls on stacked tracks."""

    def peak(j):
        spans = sorted(
            (c["start"], c["end"]) for c in calls if c["job"] == j and c["lane"] == "vlm"
        )
        best = 0
        for s, _ in spans:
            best = max(best, sum(1 for a, b in spans if a <= s < b))
        return best

    j = max(jobs, key=peak)
    mine = sorted((c for c in calls if c["job"] == j), key=lambda c: c["start"])
    tracks: list[float] = []
    for c in mine:
        k = next((i for i, free in enumerate(tracks) if free <= c["start"]), None)
        if k is None:
            tracks.append(0.0)
            k = len(tracks) - 1
        tracks[k] = c["end"]
        c["track"] = k
    a = jobs[j]["admit"]
    for c in mine:
        colour = LANES[c["lane"]][0]
        ax.add_patch(
            patches.Rectangle(
                (c["start"] - a, c["track"] - 0.38),
                max(c["end"] - c["start"], 0.03),
                0.76,
                facecolor=colour,
                edgecolor=PAPER,
                lw=0.6,
            )
        )
    ax.set_xlim(0, jobs[j]["end"] - a)
    ax.set_ylim(len(tracks) - 0.4, -0.6)
    ax.set_yticks([])
    ax.set_title(
        f"one image: up to {peak(j)} of its VLM requests at once",
        fontsize=size,
        loc="left",
        color=INK,
    )
    ax.set_ylabel("calls at once", fontsize=size - 1)
    ax.set_xlabel("seconds since admission", fontsize=size)
    ax.tick_params(labelsize=size - 2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events")
    parser.add_argument("out")
    parser.add_argument("--font", type=float, default=15.0)
    parser.add_argument("--capacity", type=int, default=32, help="the VLM lane's capacity")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    jobs, calls, batches = load(args.events)
    end = max(j["end"] for j in jobs.values())
    size = args.font

    fig = plt.figure(figsize=(18, 8.4), facecolor=PAPER)
    vlm = [c for c in calls if c["lane"] == "vlm"]
    det = batches.get("detect", [])
    fig.suptitle(
        f"The GBC machine: {len(jobs)} images, {len(calls)} engine calls in {end:.0f} s "
        f"({len(vlm)} VLM requests; detector batches of up to {max(det, default=1)})",
        fontsize=size + 3,
        x=0.01,
        ha="left",
        y=0.985,
        color=INK,
        fontweight="bold",
    )
    gantt(fig.add_axes([0.04, 0.09, 0.55, 0.8]), jobs, calls, size)
    in_flight(fig.add_axes([0.64, 0.58, 0.345, 0.31]), calls, end, size, args.capacity)
    one_job(fig.add_axes([0.64, 0.09, 0.345, 0.36]), jobs, calls, size)
    fig.savefig(args.out, dpi=args.dpi, facecolor=PAPER, pil_kwargs={"quality": 88})
    print(f"{args.out}: {len(jobs)} jobs, {len(calls)} calls, {end:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
