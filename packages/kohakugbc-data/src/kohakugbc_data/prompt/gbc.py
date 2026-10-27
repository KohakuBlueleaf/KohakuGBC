"""A resolved GBC graph as the prompt DAG.

The image vertex is the root ``[image]``. Entity and composition vertices are vertices, their
edges the children; each relation vertex is one relation line per relation caption, under its
parent, naming the vertices it points to. A vertex's handle is the text of the edge a
depth-first walk first reaches it by (``woman 1``, ``hair``), numbered on a clash (``hair 2``).
"""

import re

from kohakugbc_data.prompt.dag import Dag, Relation, Vertex, check

CAPTIONS = ("detail", "composition", "short", "hardcode")
SPACE = re.compile(r"\s+")


def _clean(text: str | None) -> str | None:
    text = None if text is None else SPACE.sub(" ", text).strip()
    return text or None


def _descs(v: dict) -> list[tuple[str, str]]:
    return [(d["label"], d["text"]) if isinstance(d, dict) else (d[0], d[1]) for d in v["descs"]]


def _kids(by_id: dict, v: dict, relation: bool) -> list[dict]:
    """``v``'s out-edges to relation vertices, or to the others; one per target."""
    out = {e["target"]: e for e in v["out_edges"]}
    return [e for t, e in out.items() if (by_id[t]["label"] == "relation") == relation]


def handles(graph: dict, root: str = "image") -> dict[str, str]:
    """Each non-relation vertex id's handle."""
    by_id = {v["vertex_id"]: v for v in graph["vertices"]}
    image = next(v for v in graph["vertices"] if v["label"] == "image")
    handle = {image["vertex_id"]: root}
    used = {root}

    def name(e: dict) -> None:
        base = _clean(e["text"]) or _clean(e["target"]) or "vertex"
        base = base.replace("[", "(").replace("]", ")")
        h, k = base, 1
        while h in used:
            k += 1
            h = f"{base} {k}"
        used.add(h)
        handle[e["target"]] = h

    def walk(v: dict) -> None:
        for e in _kids(by_id, v, False):
            if e["target"] not in handle:
                name(e)
                walk(by_id[e["target"]])

    walk(image)
    return handle


def from_gbc(
    graph: dict,
    captions: tuple[str, ...] = CAPTIONS,
    boxes: bool = True,
    tag_threshold: float | None = 0.35,
    tag_spaces: bool = True,
    root: str = "image",
) -> Dag:
    """The DAG of ``graph`` (a resolved GBC graph dict), boxes in [0, 1] of the image.

    ``captions``: desc labels in preference; a vertex takes the first it has (none: no text).
    ``tag_threshold``: tags at or above it, most confident first (None: no tags).
    """
    by_id = {v["vertex_id"]: v for v in graph["vertices"]}
    image = next(v for v in graph["vertices"] if v["label"] == "image")
    handle = handles(graph, root)

    def caption(v: dict) -> str | None:
        descs = dict(reversed(_descs(v)))  # the first desc of each label wins
        return next((_clean(descs[c]) for c in captions if c in descs), None)

    def box(v: dict) -> tuple | None:
        b = v.get("bbox")
        if not boxes or b is None or v is image:
            return None
        x0, y0, x1, y1 = (
            min(1.0, max(0.0, float(b[k]))) for k in ("left", "top", "right", "bottom")
        )
        return (x0, y0, max(x0, x1), max(y0, y1))

    def tags(v: dict) -> list[str] | None:
        scores = v.get("tags") or {}
        if tag_threshold is None or not scores:
            return None
        kept = sorted((t for t in scores.items() if t[1] >= tag_threshold), key=lambda t: -t[1])
        return [t.replace("_", " ") if tag_spaces else t for t, _ in kept] or None

    vertices = {}
    for vid, h in handle.items():
        v = by_id[vid]
        relations = []
        for e in _kids(by_id, v, True):
            rel = by_id[e["target"]]
            who = list(dict.fromkeys(handle[x["target"]] for x in rel["out_edges"]))
            for label, text in _descs(rel):
                if label == "relation" and (text := _clean(text)):
                    text = text.replace("[", "(").replace("]", ")")
                    relations.append(Relation(who, text))
        vertices[h] = Vertex(
            text=caption(v),
            box=box(v),
            tags=tags(v),
            children=[handle[e["target"]] for e in _kids(by_id, v, False)],
            relations=relations,
        )
    return check(Dag(root, vertices))
