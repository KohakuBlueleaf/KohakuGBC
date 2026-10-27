"""The prompt DAG: what a graph prompt says, as plain data.

A vertex is named by its handle. Its ``children`` are the vertices nested under it (each one a
real parent edge); its ``relations`` are relation lines under it, each naming its participants
by handle. A box is ``(left, top, right, bottom)`` in [0, 1] of the image. Every part of a
vertex is optional: no text, no box, no tags is a bare handle.
"""

import re
from dataclasses import dataclass, field

MENTION = re.compile(r"\[([^\[\]\n]+)\]")
BOX_TEXT = r"\[(-?[0-9]+),(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)\]"
BOX_LIKE = re.compile(BOX_TEXT + r"( |$)")


@dataclass
class Relation:
    participants: list[str]
    text: str = ""


@dataclass
class Vertex:
    text: str | None = None
    box: tuple[float, float, float, float] | None = None
    tags: list[str] | None = None
    children: list[str] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


@dataclass
class Dag:
    root: str
    vertices: dict[str, Vertex]


class DagError(ValueError):
    pass


def check(dag: Dag) -> Dag:
    """``dag`` when it is one DAG under its root that text can carry; else ``DagError``."""
    vs = dag.vertices
    if dag.root not in vs:
        raise DagError(f"root [{dag.root}] is not a vertex")
    for h, v in vs.items():
        if not h or h != h.strip() or MENTION.fullmatch(f"[{h}]") is None:
            raise DagError(f"handle {h!r}: empty, padded, or holds brackets or a newline")
        if BOX_LIKE.match(f"[{h}]"):
            raise DagError(f"handle {h!r}: reads as a box")
        for s in (v.text, *(v.tags or ())):
            if s is not None and ("\n" in s or s != s.strip() or not s):
                raise DagError(f"[{h}]: text or tag {s!r} is empty, padded or multi-line")
        if any("," in t for t in v.tags or ()):
            raise DagError(f"[{h}]: a tag holds a comma")
        if v.box is None and v.text is not None and BOX_LIKE.match(v.text):
            raise DagError(f"[{h}]: text opens like a box but the vertex has none")
        if v.box is not None:
            left, top, right, bottom = v.box
            if not (0 <= left <= right <= 1 and 0 <= top <= bottom <= 1):
                raise DagError(f"[{h}]: box {v.box} is not (left, top, right, bottom) in [0, 1]")
        if len(set(v.children)) != len(v.children):
            raise DagError(f"[{h}]: a child listed twice")
        for c in v.children:
            if c not in vs or c == dag.root:
                raise DagError(f"[{h}]: child [{c}] is not a vertex below the root")
        for r in v.relations:
            if "\n" in r.text or r.text != r.text.strip():
                raise DagError(f"[{h}]: relation text {r.text!r} is padded or multi-line")
            if not r.participants or len(set(r.participants)) != len(r.participants):
                raise DagError(f"[{h}]: a relation without participants, or one named twice")
            if any(p not in vs for p in r.participants):
                raise DagError(f"[{h}]: relation participant not a vertex: {r.participants}")
            if any(m not in r.participants for m in MENTION.findall(r.text)):
                raise DagError(f"[{h}]: relation text names a handle outside its participants")

    # Every vertex reachable from the root, and no cycle (colour DFS)
    state: dict[str, int] = {}
    stack = [(dag.root, iter(vs[dag.root].children))]
    state[dag.root] = 1
    while stack:
        h, it = stack[-1]
        c = next(it, None)
        if c is None:
            state[h] = 2
            stack.pop()
        elif state.get(c) == 1:
            raise DagError(f"cycle through [{c}]")
        elif c not in state:
            state[c] = 1
            stack.append((c, iter(vs[c].children)))
    if len(state) != len(vs):
        missing = sorted(set(vs) - set(state))
        raise DagError(f"vertices with no parent path from the root: {missing}")
    return dag
