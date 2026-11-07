"""A GBC graph as the metrics see it.

Vertex ids and the order of vertices are dropped: a vertex is its index into ``boxes``, the
image is ``IMAGE``. Kept: each non-relation vertex's box, type (``entity``/``composition``) and
box confidence (1 when missing); the edges from the image or a vertex to a vertex; each relation
as the set of vertices its out-edges name (a set named twice is one relation). Text, used only
by the full track: a vertex's names are the texts of the edges into it (and its ``name`` field,
when a prediction gives one), its descriptions its ``descs``; a relation's descriptions; the
image's descriptions. A box is GBC's ``bbox`` dict or a list ``[l, t, r, b(, score)]``.

``read`` loads a JSON-lines file of graphs, one per line as the graph or ``{"key", "graph"}``
(the graph or its JSON text), keyed by ``key``, else the graph's ``img_path``.
"""

import json
from dataclasses import dataclass, field

import numpy as np

IMAGE = -1


@dataclass
class Spec:
    boxes: np.ndarray
    types: list
    scores: np.ndarray
    edges: set
    relations: list
    names: list = field(default_factory=list)
    descs: list = field(default_factory=list)
    relation_descs: list = field(default_factory=list)
    image_descs: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.types)


def empty() -> Spec:
    return Spec(np.zeros((0, 4)), [], np.zeros(0), set(), [])


def _box(b) -> tuple[list, float]:
    if isinstance(b, dict):
        c = b.get("confidence")
        return [b["left"], b["top"], b["right"], b["bottom"]], 1.0 if c is None else float(c)
    c = b[4] if len(b) > 4 and b[4] is not None else 1.0
    return [float(x) for x in b[:4]], float(c)


def _texts(v: dict) -> list:
    return [d["text"] if isinstance(d, dict) else str(d) for d in v.get("descs") or []]


def load(graph) -> Spec:
    if isinstance(graph, str):
        graph = json.loads(graph)
    vertices = graph.get("vertices") or []
    image = [v for v in vertices if v.get("label") == "image"]
    image_ids = {v["vertex_id"] for v in image}
    kept = [v for v in vertices if v["vertex_id"] not in image_ids and v.get("label") != "relation"]
    index = {v["vertex_id"]: k for k, v in enumerate(kept)}
    index.update(dict.fromkeys(image_ids, IMAGE))

    boxes, scores = [], []
    for v in kept:
        box, score = _box(v["bbox"])
        boxes.append(box)
        scores.append(score)
    names = [{v["name"]} if v.get("name") else set() for v in kept]
    edges, relations = set(), {}
    for v in vertices:
        targets = [index.get(e["target"]) for e in v.get("out_edges") or []]
        if v.get("label") == "relation":
            members = frozenset(t for t in targets if t is not None and t != IMAGE)
            if members:
                relations.setdefault(members, []).extend(_texts(v))
            continue
        source = index[v["vertex_id"]]
        for e, t in zip(v.get("out_edges") or [], targets):
            if t is None or t in (IMAGE, source):
                continue
            edges.add((source, t))
            if e.get("text"):
                names[t].add(e["text"])
    return Spec(
        np.asarray(boxes, float).reshape(-1, 4),
        [v.get("label") for v in kept],
        np.asarray(scores, float),
        edges,
        list(relations),
        [sorted(n) for n in names],
        [_texts(v) for v in kept],
        list(relations.values()),
        [t for v in image for t in _texts(v)],
    )


def read(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            graph = obj if "vertices" in obj else obj["graph"]
            if isinstance(graph, str):
                graph = json.loads(graph)
            key = obj.get("key") or graph.get("img_path")
            if key in out:
                raise ValueError(f"{path}: image {key!r} appears twice")
            out[key] = load(graph)
    return out
