"""GBC-Nest's shaping of the resolved graph: groups checked, identities merged, parents from
containment.

1. ``group_check``: each composition's members, drawn numbered on the picture, are asked about
   once: separate things, or pieces of one area. A composition of pieces becomes one entity
   region; its members and their subtrees go (a vertex another kept vertex also holds stays).
2. Identity: two region vertices, neither an ancestor of the other, with the same label and the
   same head noun whose boxes overlap both ways by SAME_HEAD are one vertex.
3. ``nest_check``: a top-level vertex, or a member of a top-level composition, whose box lies
   within (HOLD of its area) a top-level entity at least HOLD_AREA times larger and covering less
   than WHOLE of the image, is asked about once per smallest such holder, both boxes drawn: is it
   part of the holder, held or worn by it, or within it as a place or frame. Where the answer is
   yes it moves under the holder; a composition whose members all fit one holder moves whole; a
   composition left with fewer than two members dissolves.

The graph moves between nodes as GBC graph JSON text.
"""

import io
import json

from PIL import Image, ImageDraw, ImageFont

from kohakugbc.graph.orders import stats
from kohakugbc.graph.reference import link_and_mask
from kohakugbc.nodes.images import pixels
from kohakugbc.nodes.nest import head
from kohakugbc.nodes.registry import node
from kohakugbc.nodes.specs import chat as specs
from kohakugbc.nodes.specs import parse

SAME_HEAD = 0.5
HOLD = 0.8
HOLD_AREA = 2.0
WHOLE = 0.9
# The checks' pictures: the image's longer side at most this, JPEG at this quality
CHECK_SIDE, CHECK_QUALITY = 1280, 90
RED, BLUE = (230, 20, 20), (20, 90, 240)


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def both_ways(a, b, t: float) -> bool:
    inter = _inter(a, b)
    return area(a) > 0 and area(b) > 0 and inter / area(a) >= t and inter / area(b) >= t


def within(a, b) -> bool:
    """``a`` lies within ``b``: at least HOLD of its area."""
    return area(a) > 0 and _inter(a, b) / area(a) >= HOLD


class Graph:
    """A GBC graph as parent -> [(child, edge text)]; a vertex's name is its first in-edge's
    text."""

    def __init__(self, g: dict):
        self.raw = g
        self.v = {x["vertex_id"]: x for x in g["vertices"]}
        self.root = next(x for x in g["vertices"] if x["label"] == "image")["vertex_id"]
        self.kids = {k: [] for k in self.v}
        for x in g["vertices"]:
            for e in x["out_edges"]:
                if e["target"] in self.v:
                    self.kids[x["vertex_id"]].append((e["target"], e["text"]))
        self.name = {}
        for cs in self.kids.values():
            for c, t in cs:
                self.name.setdefault(c, t)

    def box(self, k) -> list:
        b = self.v[k]["bbox"]
        return [b["left"], b["top"], b["right"], b["bottom"]]

    def regions(self) -> list:
        return [k for k, x in self.v.items() if x["label"] in ("entity", "composition")]

    def parents(self, k) -> list:
        """Region or image parents; relation vertices pointing at ``k`` are not parents."""
        return [
            p
            for p, cs in self.kids.items()
            if self.v[p]["label"] != "relation" and any(c == k for c, _ in cs)
        ]

    def ancestors(self, k) -> set:
        seen, todo = set(), self.parents(k)
        while todo:
            p = todo.pop()
            if p not in seen:
                seen.add(p)
                todo.extend(self.parents(p))
        return seen

    def tops(self) -> list:
        return [k for k in self.regions() if set(self.parents(k)) <= {self.root}]

    def link(self, p, c, text) -> None:
        if all(x != c for x, _ in self.kids[p]):
            self.kids[p].append((c, text))

    def unlink(self, p, c) -> None:
        self.kids[p] = [(x, t) for x, t in self.kids[p] if x != c]

    def relations_of(self, k) -> list:
        """The relation vertices naming ``k`` among their participants."""
        return [
            r
            for r, x in self.v.items()
            if x["label"] == "relation" and any(c == k for c, _ in self.kids.get(r, []))
        ]

    def drop(self, k) -> None:
        """``k`` goes, with every relation naming it."""
        broken = self.relations_of(k)
        for p in list(self.kids):
            self.unlink(p, k)
        self.kids.pop(k, None)
        self.v.pop(k, None)
        for r in broken:
            if r in self.v:
                self.drop(r)

    def merge(self, keep, gone) -> None:
        """``gone`` becomes ``keep``: its children, parents and relations move over, the box
        covers both; a relation naming both goes, one naming ``keep`` covers its new box."""
        for r in self.relations_of(gone):
            if any(c == keep for c, _ in self.kids[r]):
                self.drop(r)
        for c, t in self.kids.pop(gone):
            if c != keep:
                self.link(keep, c, t)
        for p in list(self.kids):
            for c, t in list(self.kids[p]):
                if c == gone:
                    self.unlink(p, gone)
                    if p != keep:
                        self.link(p, keep, t)
        a, b = self.v[keep]["bbox"], self.v[gone]["bbox"]
        for side, f in (("left", min), ("top", min), ("right", max), ("bottom", max)):
            a[side] = f(a[side], b[side])
        self.v.pop(gone)
        for r in self.relations_of(keep):
            rb = self.v[r]["bbox"] = dict(self.v[r]["bbox"])
            for side, f in (("left", min), ("top", min), ("right", max), ("bottom", max)):
                rb[side] = f(rb[side], a[side])

    def dump(self) -> dict:
        """The graph as GBC JSON; in-edges and masks derived from what is left, as resolution
        derives them."""
        verts = []
        for k, x in self.v.items():
            x = dict(x)
            x["out_edges"] = [
                {"source": k, "text": t, "target": c} for c, t in self.kids[k] if c in self.v
            ]
            x["in_edges"] = []
            verts.append(x)
        link_and_mask(verts, self.raw["mask_inside_threshold"])
        return {**self.raw, "vertices": verts}


def loads(graph: str | None) -> Graph | None:
    return None if graph is None else Graph(json.loads(graph))


def dumps(g: Graph) -> str:
    return json.dumps(g.dump(), ensure_ascii=False, separators=(",", ":"))


# ---- the steps -----------------------------------------------------------------------------------


def collapse(g: Graph, separate: dict) -> None:
    """Compositions whose members are pieces (``separate[vertex] is False``) become one entity
    region."""
    for k in [k for k, v in g.v.items() if v["label"] == "composition"]:
        # an earlier collapse may have removed this composition with its members
        if k not in g.v or separate.get(k) is not False:
            continue
        members = {c for c, _ in g.kids.get(k, [])}
        todo, gone = list(members), set()
        while todo:
            c = todo.pop()
            if c in gone or c not in g.v:
                continue
            if c not in members and any(p not in gone and p != k for p in g.parents(c)):
                continue
            gone.add(c)
            todo.extend(x for x, _ in g.kids.get(c, []))
        for c in gone:
            g.drop(c)
        # the arrangement caption ("roof 1 is left of roof 2") describes the removed members
        descs = [d for d in g.v[k]["descs"] if d["label"] != "composition"] or g.v[k]["descs"]
        g.v[k] = {**g.v[k], "label": "entity", "descs": descs}


def identity(g: Graph) -> None:
    changed = True
    while changed:
        changed = False
        regs = g.regions()
        for i, a in enumerate(regs):
            for b in regs[i + 1 :]:
                if a not in g.v or b not in g.v:
                    continue
                if a in g.ancestors(b) or b in g.ancestors(a):
                    continue
                ha, hb = head(g.name.get(a, a)), head(g.name.get(b, b))
                same = ha and ha == hb and both_ways(g.box(a), g.box(b), SAME_HEAD)
                if same and g.v[a]["label"] == g.v[b]["label"]:
                    g.merge(a, b)
                    changed = True


def holders(g: Graph, k: str, tops: list, answers: dict) -> list:
    """The smallest top-level entity vertices whose box holds ``k``'s, less those a nesting
    check answered no for."""
    if k not in g.v:
        return []
    bk = g.box(k)
    hs = [
        t
        for t in tops
        if t != k
        and t in g.v
        and g.v[t]["label"] == "entity"
        and area(g.box(t)) < WHOLE
        and area(g.box(t)) >= HOLD_AREA * area(bk)
        and within(bk, g.box(t))
        and k not in g.ancestors(t)
        and t not in g.ancestors(k)
    ]
    hs = [t for t in hs if not any(o != t and within(g.box(o), g.box(t)) for o in hs)]
    return [t for t in hs if answers.get((k, t), True)]


def candidates(g: Graph) -> list:
    """What may move: top-level vertices, a top-level composition's members in its place."""
    out = []
    for k in g.tops():
        out.extend([c for c, _ in g.kids[k]] if g.v[k]["label"] == "composition" else [k])
    return out


def contain(g: Graph, answers: dict) -> None:
    tops = g.tops()
    for k in tops:
        if g.v[k]["label"] == "composition":
            whole = holders(g, k, tops, answers)
            if whole:
                for h in whole:
                    g.link(h, k, g.name.get(k, k))
                g.unlink(g.root, k)
                continue
            for c, t in list(g.kids[k]):
                hs = holders(g, c, tops, answers)
                if hs:
                    for h in hs:
                        g.link(h, c, t)
                    g.unlink(k, c)
            left = [c for c, _ in g.kids[k] if c in g.v]
            if len(left) < 2:
                for c in left:
                    g.link(g.root, c, g.name.get(c, c))
                g.drop(k)
            continue
        hs = holders(g, k, tops, answers)
        if hs:
            for h in hs:
                g.link(h, k, g.name.get(k, k))
            g.unlink(g.root, k)


# ---- the checks' pictures ------------------------------------------------------------------------


def _canvas(image: dict) -> Image.Image:
    img = Image.fromarray(pixels(image["path"]))
    img.thumbnail((CHECK_SIDE, CHECK_SIDE))
    return img


def _jpeg(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=CHECK_QUALITY)
    return {"jpeg": buf.getvalue(), "size": [img.width, img.height]}


def numbered(image: dict, boxes: list) -> dict:
    """The picture with ``boxes`` drawn red and numbered from 1."""
    img = _canvas(image)
    w, h = img.size
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(16, w // 45))
    for k, b in enumerate(boxes, start=1):
        r = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
        d.rectangle(r, outline=(255, 40, 40), width=4)
        d.rectangle(
            [r[0], r[1], r[0] + font.size * 1.2, r[1] + font.size * 1.3], fill=(255, 40, 40)
        )
        d.text((r[0] + 3, r[1] + 1), str(k), fill=(255, 255, 255), font=font)
    return _jpeg(img)


def paired(image: dict, red: list, blue: list) -> dict:
    """The picture with the holder's box red and the child's blue."""
    img = _canvas(image)
    w, h = img.size
    d = ImageDraw.Draw(img)
    for b, c in ((red, RED), (blue, BLUE)):
        d.rectangle([b[0] * w, b[1] * h, b[2] * w, b[3] * h], outline=c, width=5)
    return _jpeg(img)


# ---- nodes ---------------------------------------------------------------------------------------


@node(outputs=["queries"])
def group_queries(graph, controls: dict) -> list:
    """One question per composition of at least two members."""
    g = loads(graph)
    if g is None or not controls["group_check"]:
        return []
    out = []
    for k in [k for k, v in g.v.items() if v["label"] == "composition"]:
        boxes = [g.box(c) for c, _ in g.kids.get(k, []) if c in g.v]
        if len(boxes) >= 2:
            out.append({"vertex": k, "name": g.name.get(k, k), "boxes": boxes})
    return out


@node(outputs=["request"])
def group_request(image: dict, q: dict) -> dict:
    picture = numbered(image, q["boxes"])
    return specs.group_check(image["key"], q["vertex"], q["name"], len(q["boxes"]), picture)


@node(outputs=["answer"])
def group_answer(q: dict, reply: str) -> list:
    """``[vertex, separate]``; separate is None for a reply that cannot be read."""
    sep = (parse.load_json(reply) or {}).get("separate")
    return [q["vertex"], sep if isinstance(sep, bool) else None]


@node(outputs=["graph"])
def regroup(graph, answers: list):
    """Compositions of pieces collapsed, then identity merges."""
    g = loads(graph)
    if g is None:
        return None
    collapse(g, {k: sep for k, sep in answers})
    identity(g)
    return dumps(g)


@node(outputs=["queries"])
def nest_queries(graph, controls: dict) -> list:
    """One question per proposed nesting: each candidate and each of its smallest holders."""
    g = loads(graph)
    if g is None or not controls["nest_check"]:
        return []
    tops = g.tops()
    return [
        {
            "child": [c, g.name.get(c, c)],
            "holder": [h, g.name.get(h, h)],
            "child_box": g.box(c),
            "holder_box": g.box(h),
        }
        for c in candidates(g)
        for h in holders(g, c, tops, {})
    ]


@node(outputs=["request"])
def nest_request(image: dict, q: dict) -> dict:
    picture = paired(image, q["holder_box"], q["child_box"])
    return specs.nest_check(image["key"], q["child"], q["holder"], picture)


@node(outputs=["answer"])
def nest_answer(q: dict, reply: str) -> list:
    """``[child, holder, inside]``; a reply that cannot be read is a yes (the boxes say so)."""
    inside = (parse.load_json(reply) or {}).get("inside")
    return [q["child"][0], q["holder"][0], inside if isinstance(inside, bool) else True]


@node(outputs=["graph", "stats"])
def reparent(graph, answers: list) -> tuple:
    """Parents from containment, where the checks did not say no; the graph and its counts."""
    g = loads(graph)
    if g is None:
        return None, stats(None)
    contain(g, {(c, h): yes for c, h, yes in answers})
    out = g.dump()
    return json.dumps(out, ensure_ascii=False, separators=(",", ":")), stats(out)
