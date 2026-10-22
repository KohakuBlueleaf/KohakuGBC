"""The README teaser, laid out like the GBC paper's figure: one image's GBC graph.

Left, the image with the boxes of the entities the figure shows (parts dashed). Right, the graph
around the image vertex: the objects the image names on a ring, their parts and instances
outside it, and relation vertices (diamonds) between the image and the objects they relate.
Below, captions of four vertices; an arrow runs from a phrase of a parent's caption to the card
of the child that phrase grounds (the child's caption describes what the phrase names). An
object, its parts and its instances share one colour; every grounded phrase is in its child's.

    python scripts/docs/teaser.py GRAPHS.jsonl IMAGE docs/assets/teaser.jpg [--index N]
"""

import argparse
import json
import math
import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches
from PIL import Image

PALETTE = [
    "#e6550d", "#3182bd", "#31a354", "#9e5fd1", "#d6616b", "#17a2b8", "#b8a000", "#8c564b",
    "#e377c2", "#2f4b7c", "#637939", "#7f7f7f",
]  # fmt: skip
INK, MUTED, PAPER = "#1f2328", "#6e7781", "#ffffff"
RELATION = "#57606a"
KIND_TAG = {"short": "Short", "detail": "Detailed", "composition": "Composition",
            "relation": "Relation"}  # fmt: skip


# ---- the graph's facts ------------------------------------------------------------------------


def index(graph: dict) -> dict[str, dict]:
    return {v["vertex_id"]: v for v in graph["vertices"]}


def parents(by_id: dict) -> dict[str, list[str]]:
    out = {vid: [] for vid in by_id}
    for vid, v in by_id.items():
        for e in v["out_edges"]:
            if e["target"] != vid:
                out[e["target"]].append(vid)
    return out


def below(by_id: dict, u: str) -> list[str]:
    """``u``'s parts or instances (its children that are not relations)."""
    return list(dict.fromkeys(
        e["target"] for e in by_id[u]["out_edges"]
        if e["target"] != u and by_id[e["target"]]["label"] != "relation"
    ))  # fmt: skip


def colours(by_id: dict) -> dict[str, str]:
    """One colour per object the image names; its parts and instances share it."""
    out, n = {}, 0
    for e in by_id[""]["out_edges"]:
        t = e["target"]
        if by_id[t]["label"] != "relation" and t not in out:
            out[t] = PALETTE[n % len(PALETTE)]
            n += 1
    todo = deque(out)
    while todo:
        u = todo.popleft()
        for t in below(by_id, u):
            if t not in out:
                out[t] = out[u]
                todo.append(t)
    for vid, v in by_id.items():
        out.setdefault(vid, INK if v["label"] == "image" else RELATION)
    return out


def names(by_id: dict) -> dict[str, str]:
    """A vertex's name: the phrase of the first edge into it; a relation: its objects."""
    out = {"": "image"}
    for v in by_id.values():
        for e in v["out_edges"]:
            if by_id[e["target"]]["label"] != "relation":
                out.setdefault(e["target"], e["text"])
    for vid, v in by_id.items():
        if v["label"] == "relation":
            out[vid] = " / ".join(e["text"] for e in v["out_edges"])
        out.setdefault(vid, vid)
    return out


# ---- what the figure shows --------------------------------------------------------------------


def pick(by_id: dict) -> tuple[list[str], list[tuple[str, str | None]]]:
    """(vertices in the graph, captioned vertices with the parent whose caption feeds each).

    Captioned: the image; the object with the most parts; one of its parts (an entity with a
    caption); the composition with the most instances. Shown: every object the image names,
    the parts and instances of the captioned ones, and the relations among what is shown."""
    root = list(dict.fromkeys(e["target"] for e in by_id[""]["out_edges"]))
    objects = [t for t in root if by_id[t]["label"] != "relation"]
    relations = [t for t in root if by_id[t]["label"] == "relation"]
    entities = [u for u in objects if by_id[u]["label"] == "entity"]
    compositions = [u for u in objects if by_id[u]["label"] == "composition"]
    entity = max(entities, key=lambda u: len(below(by_id, u)), default=None)
    composition = max(compositions, key=lambda u: len(below(by_id, u)), default=None)
    part = None
    if entity is not None:
        kids = [t for t in below(by_id, entity) if by_id[t]["descs"]]
        kids.sort(key=lambda t: by_id[t]["label"] != "entity")
        part = kids[0] if kids else None
    captioned = [("", None)]
    captioned += [(u, p) for u, p in ((entity, ""), (composition, ""), (part, entity))
                  if u is not None]  # fmt: skip

    shown = [""] + objects
    for u in (entity, composition):
        if u is not None:
            shown += below(by_id, u)
    shown += [r for r in relations
              if all(e["target"] in shown for e in by_id[r]["out_edges"])]  # fmt: skip
    return list(dict.fromkeys(shown)), captioned


# ---- the image --------------------------------------------------------------------------------


def draw_image(ax, image, by_id, shown, colour, called, parent, size) -> None:
    w, h = image.size
    ax.imshow(image)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    for vid in shown:
        v = by_id[vid]
        if v["label"] != "entity" or v["bbox"] is None:
            continue
        b = v["bbox"]
        x0, y0, x1, y1 = b["left"] * w, b["top"] * h, b["right"] * w, b["bottom"] * h
        part = any(by_id[p]["label"] == "entity" for p in parent[vid])
        c = colour[vid]
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=c,
            linewidth=2.4 if part else 4.0, linestyle="--" if part else "-",
        ))  # fmt: skip
        # Instances of a composition stay unlabelled: their labels would pile on each other
        if "" in parent[vid] or part:
            tag = {"facecolor": c, "edgecolor": "none", "pad": 2.5, "alpha": 0.95}
            ax.text(x0 + 10, y0 + 10, called[vid], color=PAPER, fontsize=size, va="top",
                    fontweight="bold", bbox=tag)  # fmt: skip


# ---- the graph --------------------------------------------------------------------------------


def wrap(name: str, width: int = 13) -> str:
    """A long name over two lines, broken at a space."""
    if len(name) <= width:
        return name
    words, best = name.split(" "), None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if best is None or max(len(a), len(b)) < max(len(best[0]), len(best[1])):
            best = (a, b)
    return "\n".join(best) if best else name


def clips(p, q, rect) -> bool:
    """Whether the segment p-q passes through the rectangle (x0, y0, x1, y1) (Liang-Barsky)."""
    x0, y0, x1, y1 = rect
    dx, dy = q[0] - p[0], q[1] - p[1]
    lo, hi = 0.0, 1.0
    for d, a, b in ((dx, p[0] - x0, x1 - p[0]), (dy, p[1] - y0, y1 - p[1])):
        if d == 0:
            if a < 0 or b < 0:
                return False
            continue
        t0, t1 = -a / d, b / d
        if t0 > t1:
            t0, t1 = t1, t0
        lo, hi = max(lo, t0), min(hi, t1)
        if lo > hi:
            return False
    return hi - lo > 1e-9


SIDES = ("below", "above", "right", "left")
VIOLATION = 100.0
BEND = 0.15


def curve(p, q, aspect: float, bend: float = BEND, n: int = 8) -> list[tuple[float, float]]:
    """The arc an edge from ``p`` to ``q`` is drawn along (matplotlib's ``arc3,rad=bend``: a
    quadratic curve whose control point sits ``bend`` times the chord off its middle), as
    ``n`` + 1 points in panel units."""
    (x1, y1), (x2, y2) = (p[0] * aspect, p[1]), (q[0] * aspect, q[1])
    cx, cy = (x1 + x2) / 2 + bend * (y2 - y1), (y1 + y2) / 2 - bend * (x2 - x1)
    out = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
        y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2
        out.append((x / aspect, y))
    return out


def hits(points, rect) -> bool:
    """Whether the polyline through ``points`` passes through ``rect``."""
    return any(clips(a, b, rect) for a, b in zip(points, points[1:]))


def overlap(r, s) -> bool:
    return r[0] < s[2] and s[0] < r[2] and r[1] < s[3] and s[1] < r[3]


class Layout:
    """Node positions in the unit square and the side of each node its label sits on. The start
    is a Kamada-Kawai layout (every edge near its graph distance); simulated annealing (seeded;
    the best of several runs) then moves nodes a little and picks label sides. A violation: an
    edge (drawn as its arc) passing under a node other than its ends, or under any label (its
    ends' included), two boxes overlapping, a box leaving the panel. With none left, shorter
    edges and nodes kept ``gap`` apart decide. Edges may cross each other.

    ``node_r``: half width and height of each node's marker; ``label``: half width and height
    of its label; ``aspect``: panel width over height (distances are measured on the panel)."""

    def __init__(self, nodes, edges, node_r, label, aspect, gap, weight):
        """``weight``: per edge, how much its length counts (children stay near parents)."""
        self.nodes, self.edges, self.weight = nodes, edges, weight
        self.node_r, self.label, self.aspect, self.gap = node_r, label, aspect, gap
        self.incident = {u: [e for e in edges if u in e] for u in nodes}
        # During the repair: the positions nodes are held to (instead of short edges)
        self.anchor: dict | None = None

    def rects(self, u, p, side):
        x, y = p
        rx, ry = self.node_r[u]
        lw, lh = self.label[u]
        node = (x - rx, y - ry, x + rx, y + ry)
        label = {
            "below": (x - lw, y - ry - 2 * lh, x + lw, y - ry),
            "above": (x - lw, y + ry, x + lw, y + ry + 2 * lh),
            "right": (x + rx, y - lh, x + rx + 2 * lw, y + lh),
            "left": (x - rx - 2 * lw, y - lh, x - rx, y + lh),
        }[side]
        return node, label

    def local(self, u, pos, side, box) -> tuple[int, float]:
        """(violations, soft cost) of every term that involves ``u``."""
        bad, soft = 0, 0.0
        node, label = box[u]
        for r in (node, label):
            if r[0] < 0 or r[1] < 0 or r[2] > 1 or r[3] > 1:
                bad += 1
        arcs = {}
        for a, b in self.incident[u]:
            p, q = pos[a], pos[b]
            arc = arcs[(a, b)] = curve(p, q, self.aspect)
            for w in self.nodes:
                if w not in (a, b) and hits(arc, box[w][0]):
                    bad += 1
                if hits(arc, box[w][1]):
                    bad += 1
            if self.anchor is None:
                soft += self.weight[(a, b)] * math.hypot((q[0] - p[0]) * self.aspect, q[1] - p[1])
        if self.anchor is not None:
            ax_, ay_ = self.anchor[u]
            soft += 20.0 * math.hypot((pos[u][0] - ax_) * self.aspect, pos[u][1] - ay_)
        for a, b in self.edges:
            if u in (a, b):
                continue
            arc = curve(pos[a], pos[b], self.aspect)
            if hits(arc, node) or hits(arc, label):
                bad += 1
        for w in self.nodes:
            if w == u:
                continue
            for r in box[w]:
                if overlap(node, r) or overlap(label, r):
                    bad += 1
            d = math.hypot((pos[w][0] - pos[u][0]) * self.aspect, pos[w][1] - pos[u][1])
            if d < self.gap:
                soft += 3.0 * (self.gap - d)
        return bad, soft

    def start(self) -> dict:
        """Kamada-Kawai over the undirected graph (parts and instances closer to their parent
        than the rest), scaled into the panel with room for labels."""
        g = nx.Graph()
        g.add_nodes_from(self.nodes)
        for a, b in self.edges:
            g.add_edge(a, b, weight=1.0 / self.weight[(a, b)] ** 0.5)
        pos = nx.kamada_kawai_layout(g, weight="weight")
        xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
        return {
            u: (0.1 + 0.8 * (x - min(xs)) / (max(xs) - min(xs)),
                0.1 + 0.8 * (y - min(ys)) / (max(ys) - min(ys)))
            for u, (x, y) in pos.items()
        }  # fmt: skip

    def total(self, pos, side, box) -> tuple[int, float]:
        bad, soft = 0, 0.0
        for u in self.nodes:
            b, s = self.local(u, pos, side, box)
            bad, soft = bad + b, soft + s
        return bad, soft  # every term counted twice; only compared with itself

    def run(self, steps: int = 20000, restarts: int = 6) -> tuple[dict, dict, int]:
        rng = random.Random(0)
        first = self.start()
        best = None
        for _ in range(restarts):
            pos = dict(first)
            side = {u: rng.choice(SIDES) for u in self.nodes}
            box = {u: self.rects(u, pos[u], side[u]) for u in self.nodes}
            for step in range(steps):
                frac = step / steps
                temp = 10.0 * (0.01 / 10.0) ** frac
                u = rng.choice(self.nodes)
                old = (pos[u], side[u], box[u])
                before = self.local(u, pos, side, box)
                if rng.random() < 0.3:
                    side[u] = rng.choice(SIDES)
                else:
                    # Small moves only: the start's shape (short edges) stays
                    sigma = 0.08 * (1 - frac) + 0.004
                    pos[u] = (pos[u][0] + rng.gauss(0, sigma / self.aspect),
                              pos[u][1] + rng.gauss(0, sigma))  # fmt: skip
                box[u] = self.rects(u, pos[u], side[u])
                after = self.local(u, pos, side, box)
                delta = VIOLATION * (after[0] - before[0]) + after[1] - before[1]
                if delta > 0 and rng.random() >= math.exp(-delta / temp):
                    pos[u], side[u], box[u] = old
            bad, soft = self.total(pos, side, box)
            if best is None or (bad, soft) < best[0]:
                best = ((bad, soft), dict(pos), dict(side))
        (bad, _), pos, side = best
        # Stretched to span the panel, then repaired: small moves and label sides until the
        # stretch's violations are gone, the stretched shape kept
        pos = self.stretch(pos, side)
        self.anchor = dict(pos)
        for _ in range(3):
            box = {u: self.rects(u, pos[u], side[u]) for u in self.nodes}
            if self.total(pos, side, box)[0] == 0:
                break
            pos, side = self.anneal(rng, pos, side, steps, sigma=0.03, temp=4.0, gap=False)
        self.anchor = None
        box = {u: self.rects(u, pos[u], side[u]) for u in self.nodes}
        bad = self.total(pos, side, box)[0]
        return pos, side, bad // 2

    def anneal(self, rng, pos, side, steps, sigma, temp, gap=True) -> tuple[dict, dict]:
        """Annealing from ``pos`` and ``side``: label sides and gaussian moves of ``sigma``,
        shrinking; returns the state with the fewest violations, then the lowest soft cost."""
        pos, side = dict(pos), dict(side)
        box = {u: self.rects(u, pos[u], side[u]) for u in self.nodes}
        saved_gap = self.gap
        if not gap:
            self.gap = 0.0
        best = (self.total(pos, side, box), dict(pos), dict(side))
        for step in range(steps):
            frac = step / steps
            t = temp * (0.01 / temp) ** frac
            u = rng.choice(self.nodes)
            old = (pos[u], side[u], box[u])
            before = self.local(u, pos, side, box)
            if rng.random() < 0.3:
                side[u] = rng.choice(SIDES)
            else:
                s = sigma * (1 - frac) + 0.002
                pos[u] = (pos[u][0] + rng.gauss(0, s / self.aspect), pos[u][1] + rng.gauss(0, s))
            box[u] = self.rects(u, pos[u], side[u])
            after = self.local(u, pos, side, box)
            delta = VIOLATION * (after[0] - before[0]) + after[1] - before[1]
            if delta > 0 and rng.random() >= math.exp(-delta / t):
                pos[u], side[u], box[u] = old
            elif step % 200 == 0:
                now = self.total(pos, side, box)
                if now < best[0]:
                    best = (now, dict(pos), dict(side))
        now = self.total(pos, side, box)
        if now < best[0]:
            best = (now, dict(pos), dict(side))
        self.gap = saved_gap
        return best[1], best[2]

    def stretch(self, pos, side, margin: float = 0.004) -> dict:
        """``pos`` under the per-axis map ``a * x + b`` with the largest ``a`` that keeps every
        box (its size unchanged) inside the panel: the layout spans the panel on both axes."""
        reach = {}
        for u in self.nodes:
            rs = self.rects(u, pos[u], side[u])
            x, y = pos[u]
            reach[u] = (x - min(r[0] for r in rs), max(r[2] for r in rs) - x,
                        y - min(r[1] for r in rs), max(r[3] for r in rs) - y)  # fmt: skip

        def fit(axis: int) -> tuple[float, float]:
            lo, hi = (0, 1) if axis == 0 else (2, 3)
            a = math.inf
            for u in self.nodes:
                for v in self.nodes:
                    d = pos[v][axis] - pos[u][axis]
                    if d > 1e-9:
                        room = 1 - 2 * margin - reach[v][hi] - reach[u][lo]
                        a = min(a, room / d)
            a = 1.0 if a is math.inf else max(a, 0.0)
            b = max(margin + reach[u][lo] - a * pos[u][axis] for u in self.nodes)
            return a, b

        (ax, bx), (ay, by) = fit(0), fit(1)
        return {u: (ax * x + bx, ay * y + by) for u, (x, y) in pos.items()}


def node_pt(kind: str, size: float) -> float:
    return {"image": 2.8, "relation": 1.7}.get(kind, 2.0) * size


def draw_graph(ax, by_id, shown, colour, called, size) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    renderer = ax.figure.canvas.get_renderer()
    px_w, px_h = ax.bbox.width, ax.bbox.height
    text = {u: wrap(called[u]) for u in shown}
    # Sizes in panel units: each marker's half size, each label's half width and height
    node_r, label = {}, {}
    for u in shown:
        t = ax.text(0, 0, text[u], fontsize=size, fontweight="bold", linespacing=1.0)
        ext = t.get_window_extent(renderer)
        t.remove()
        r = node_pt(by_id[u]["label"], size) * ax.figure.dpi / 72 / 2 + 3
        node_r[u] = (r / px_w, r / px_h)
        label[u] = ((ext.width / 2 + 4) / px_w, (ext.height / 2 + 3) / px_h)
    edges = []
    for u in shown:
        for e in by_id[u]["out_edges"]:
            t = e["target"]
            if t in shown and t != u and (u, t) not in edges:
                edges.append((u, t))
    nodes = [""] + [u for u in shown if u != ""]
    # Nodes as far apart as the panel allows for this many
    gap = 0.9 * math.sqrt(px_w / px_h / len(nodes))
    # A part or an instance near its parent; every edge kept short
    weight = {
        (a, b): 2.5 if "relation" in (by_id[a]["label"], by_id[b]["label"])
        else 3.5 if a == "" else 6.0
        for a, b in edges
    }  # fmt: skip
    centre, side, bad = Layout(nodes, edges, node_r, label, px_w / px_h, gap, weight).run()
    print(f"graph layout: {bad} violation(s) left (an edge under a node or label, boxes "
          f"overlapping, or off the panel)")  # fmt: skip

    for a, b in edges:
        relation = by_id[b]["label"] == "relation" or by_id[a]["label"] == "relation"
        c = RELATION if relation else colour[b]
        ax.annotate("", xy=centre[b], xytext=centre[a], zorder=1, arrowprops={
            "arrowstyle": "-|>,head_length=0.7,head_width=0.35", "color": c,
            "lw": 1.8 if by_id[b]["label"] == "relation" else 2.4, "alpha": 0.9,
            "linestyle": (0, (4, 2)) if by_id[b]["label"] == "relation" else "-",
            "connectionstyle": f"arc3,rad={BEND}",
            "shrinkA": node_pt(by_id[a]["label"], size) * 0.55,
            "shrinkB": node_pt(by_id[b]["label"], size) * 0.55,
        })  # fmt: skip
    marker = {"image": "o", "entity": "o", "composition": "s", "relation": "D"}
    for u in shown:
        kind = by_id[u]["label"]
        x, y = centre[u]
        ax.scatter([x], [y], s=node_pt(kind, size) ** 2, marker=marker[kind], color=colour[u],
                   edgecolors=PAPER, linewidths=2.0, zorder=3)  # fmt: skip
        rx, ry = node_r[u]
        lx, ly, ha, va = {
            "below": (x, y - ry, "center", "top"),
            "above": (x, y + ry, "center", "bottom"),
            "right": (x + rx, y, "left", "center"),
            "left": (x - rx, y, "right", "center"),
        }[side[u]]
        ax.text(lx, ly, text[u], fontsize=size, fontweight="bold", ha=ha, va=va,
                linespacing=1.0, zorder=4, multialignment="center",
                color=colour[u] if kind != "image" else INK)  # fmt: skip


# ---- captions ---------------------------------------------------------------------------------


def runs(text: str, spans: list[tuple[str, str, str]]) -> list[tuple]:
    """``text`` as (word, colour, space after, target): each grounded phrase one piece."""
    low, marks = text.lower(), []
    for phrase, c, target in spans:
        at = low.find(phrase.lower())
        while at >= 0 and any(s <= at < e for s, e, _, _ in marks):
            at = low.find(phrase.lower(), at + 1)
        if at >= 0:
            marks.append((at, at + len(phrase), c, target))
    pieces, i = [], 0
    for s, e, c, target in sorted(marks):
        before = text[i:s]
        pieces += [(w, None, True, None) for w in before.split()]
        if pieces and before and not before[-1].isspace():
            pieces[-1] = (pieces[-1][0], None, False, None)
        pieces.append((text[s:e], c, e < len(text) and text[e].isspace(), target))
        i = e
    return pieces + [(w, None, True, None) for w in text[i:].split()]


def flow(ax, x0, y0, x1, floor, pieces, size, found: dict) -> float:
    """``pieces`` wrapped between x0 and x1 from y0 down (axes units), never under ``floor``
    (cut with an ellipsis). The window extent of each target's first phrase goes to ``found``.
    Returns the y under the last line."""
    renderer = ax.figure.canvas.get_renderer()
    line = size / 72 * ax.figure.dpi * 1.4 / ax.bbox.height
    x, y = x0, y0
    for word, c, space, target in pieces:
        style = {"fontsize": size, "color": c or INK, "fontweight": "bold" if c else "normal",
                 "va": "top", "transform": ax.transAxes}  # fmt: skip
        piece = word + (" " if space else "")
        t = ax.text(x, y, piece, **style)
        dx = t.get_window_extent(renderer).width / ax.bbox.width
        if x + dx > x1 and x > x0:
            t.remove()
            if y - 2 * line < floor:
                ax.text(x, y, "…", fontsize=size, color=INK, va="top", transform=ax.transAxes)
                return y - line
            x, y = x0, y - line
            t = ax.text(x, y, piece, **style)
        if target is not None and target not in found:
            word_only = ax.text(x, y, word, **{**style, "alpha": 0.0})
            found[target] = word_only.get_window_extent(renderer)
        x += dx
    return y - line


def descs(v: dict) -> list[tuple[str, str]]:
    want = {
        "image": ("short", "detail"), "entity": ("detail",),
        "composition": ("composition", "short"), "relation": ("relation",),
    }[v["label"]]  # fmt: skip
    out = []
    for label in want:
        out += [(label, d["text"]) for d in v["descs"] if d["label"] == label][:1]
    return out


def draw_cards(fig, rect, by_id, captioned, colour, called, size):
    """Caption cards in a 2 x 2 grid inside ``rect`` (figure units). Returns each card's
    figure box and, per card, the window extents of the phrases that name other cards."""
    left, bottom, width, height = rect
    cols, gap_x, gap_y = 2, 0.03, 0.045
    rows = -(-len(captioned) // cols)
    cw = (width - gap_x * (cols - 1)) / cols
    ch = (height - gap_y * (rows - 1)) / rows
    wanted = {u for u, _ in captioned}
    parents_of = {p for _, p in captioned if p is not None}
    boxes, phrases = {}, {}
    for i, (vid, _) in enumerate(captioned):
        r, c = divmod(i, cols)
        x, y = left + c * (cw + gap_x), bottom + (rows - 1 - r) * (ch + gap_y)
        ax = fig.add_axes([x, y, cw, ch])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        col = colour[vid]
        bar = size * 1.9 / 72 * fig.dpi / ax.bbox.height
        # Opaque: the links run under the cards and show only between them
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=PAPER, edgecolor="none"))
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=col, alpha=0.08, edgecolor=col,
                                       linewidth=2.0))  # fmt: skip
        ax.add_patch(patches.Rectangle((0, 1 - bar), 1, bar, facecolor=col, edgecolor=col,
                                       linewidth=2.0))  # fmt: skip
        title = "Image captions" if vid == "" else f'Captions for the node "{called[vid]}"'
        t = ax.text(0.5, 1 - bar / 2, title, fontsize=size + 1, fontweight="bold", color=PAPER,
                    ha="center", va="center")  # fmt: skip
        room = 0.96 * ax.bbox.width
        wide = t.get_window_extent(fig.canvas.get_renderer()).width
        if wide > room:
            t.set_fontsize((size + 1) * room / wide)
        spans = [(e["text"], colour[e["target"]], e["target"] if e["target"] in wanted else None)
                 for e in by_id[vid]["out_edges"]]  # fmt: skip
        yy, found = 1 - bar - 0.045, {}
        shown = descs(by_id[vid])
        line = size / 72 * fig.dpi * 1.4 / ax.bbox.height
        feeds = vid in parents_of
        for k, (tag, text) in enumerate(shown):
            # A card whose phrases feed other cards shows its first caption whole; otherwise a
            # caption before the last leaves the last two lines
            last = k == len(shown) - 1
            floor = 0.03 if last else 0.03 + (1.2 if feeds else 2.6) * line
            pieces = [(f"[{KIND_TAG.get(tag, tag)}]", MUTED, True, None)] + runs(text, spans)
            yy = flow(ax, 0.025, yy, 0.975, floor, pieces, size, found) - 0.02
        boxes[vid] = (x, y, cw, ch)
        phrases[vid] = found
    return boxes, phrases


def link(fig, extent, source, target, colour) -> None:
    """The phrase at ``extent`` (window units) in the card ``source`` ringed, and an arrow from
    it to the card ``target`` (both figure boxes). The line runs under the cards, so it shows
    only between them: it leaves the source card straight above the phrase (a card on the
    right) or straight below it (a card below), crosses the gap, and lands on the target's
    title."""
    to_fig = fig.transFigure.inverted()
    (x0, y0), (x1, y1) = to_fig.transform((extent.x0, extent.y0)), to_fig.transform(
        (extent.x1, extent.y1)
    )
    pad = 0.003
    fig.add_artist(patches.FancyBboxPatch(
        (x0 - pad, y0 - pad), x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.004", fill=False, edgecolor=colour, lw=2.4,
        zorder=6,
    ))  # fmt: skip
    sx, sy, sw, sh = source
    tx, ty, tw, th = target
    mid = (x0 + x1) / 2
    land = tx + tw * 0.12
    # The last leg drops straight onto the title, longer than the arrowhead
    if tx > sx + sw:
        lane = sy + sh + 0.03
        points = [(mid, (y0 + y1) / 2), (mid, lane), (land, lane), (land, ty + th)]
    else:
        lane = ty + th + 0.028
        points = [(mid, (y0 + y1) / 2), (mid, lane), (land, lane), (land, ty + th)]
    xs, ys = zip(*points[:-1])
    fig.add_artist(plt.Line2D(xs, ys, color=colour, lw=2.8, transform=fig.transFigure,
                              zorder=-1, solid_joinstyle="round"))  # fmt: skip
    fig.add_artist(patches.FancyArrowPatch(
        points[-2], points[-1], transform=fig.transFigure, color=colour, lw=2.8, zorder=-1,
        arrowstyle="-|>,head_length=0.7,head_width=0.4", mutation_scale=14, shrinkA=0,
        shrinkB=0,
    ))  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs")
    parser.add_argument("image")
    parser.add_argument("out")
    parser.add_argument("--index", type=int, default=0)
    # Sized for a README about 1000 px wide: 18 inches across at 15-16 point type keeps the
    # smallest text near 11 px there
    parser.add_argument("--width", type=float, default=18.0, help="inches")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--font", type=float, default=15.0)
    args = parser.parse_args()
    with open(args.graphs, encoding="utf-8") as f:
        graph = json.loads(f.readlines()[args.index])
    image = Image.open(args.image).convert("RGB")
    by_id = index(graph)
    colour, called, parent = colours(by_id), names(by_id), parents(by_id)
    shown, captioned = pick(by_id)

    w, h = image.size
    width = args.width
    height = width * 0.56
    image_w = min(height * w / h, width * 0.42)
    image_h = image_w * h / w
    fig = plt.figure(figsize=(width, height), facecolor=PAPER)
    a = image_w / width
    img_ax = fig.add_axes([0.004, (1 - image_h / height) / 2, a - 0.008, image_h / height])
    draw_image(img_ax, image, by_id, shown, colour, called, parent, args.font - 1)
    draw_graph(fig.add_axes([a + 0.02, 0.5, 1 - a - 0.03, 0.49]), by_id, shown, colour, called,
               args.font + 1)  # fmt: skip
    boxes, phrases = draw_cards(fig, [a + 0.02, 0.012, 1 - a - 0.03, 0.45], by_id, captioned,
                                colour, called, args.font)  # fmt: skip
    for child, par in captioned:
        if par is not None and child in phrases.get(par, {}):
            link(fig, phrases[par][child], boxes[par], boxes[child], colour[child])
        elif par is not None:
            print(f"warning: the caption of {par!r} shows no phrase naming {child!r}")
    fig.savefig(args.out, dpi=args.dpi, facecolor=PAPER, pil_kwargs={"quality": 88})
    print(f"{args.out}: {len(shown)} of {len(by_id)} vertices; captions {captioned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
