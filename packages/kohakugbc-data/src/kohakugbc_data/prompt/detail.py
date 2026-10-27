"""Detail levels: each vertex of a prompt DAG shown at a sampled level of detail, a child never
above its parents (users describe deeper vertices less).

    6  long caption + tags       3  short caption
    5  short caption + tags      2  handle + box
    4  tags | long caption       1  handle
                                 (every level from 2 up keeps the box)

The root has no box; its level picks its caption and tags the same way. ``sample_levels`` walks
the DAG parents first and steps each vertex down from its parents' lowest level, one level per
failed ``keep`` draw; ``apply_levels`` builds the DAG those levels show.
"""

import random
import re
from dataclasses import replace

from kohakugbc_data.prompt.dag import Dag, check

# Level -> its variants: (caption: "long" | "short" | None, tags, box)
LEVELS: dict[int, list[tuple[str | None, bool, bool]]] = {
    6: [("long", True, True)],
    5: [("short", True, True)],
    4: [(None, True, True), ("long", False, True)],
    3: [("short", False, True)],
    2: [(None, False, True)],
    1: [(None, False, False)],
}
TOP = max(LEVELS)
SENTENCE = re.compile(r"(?<=[.!?])\s")


def first_sentence(text: str | None) -> str | None:
    """The text up to its first sentence end (the whole text when it has none)."""
    return None if text is None else SENTENCE.split(text, maxsplit=1)[0].strip() or None


def parents_first(dag: Dag) -> list[str]:
    """Handles with every vertex after all its parents (Kahn's order from the root)."""
    indeg = {h: 0 for h in dag.vertices}
    for v in dag.vertices.values():
        for c in v.children:
            indeg[c] += 1
    order, ready = [], [dag.root]
    while ready:
        h = ready.pop(0)
        order.append(h)
        for c in dag.vertices[h].children:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    return order


def sample_levels(
    dag: Dag, rng: random.Random, keep: float = 0.6, root: int | None = TOP
) -> dict[str, tuple[int, tuple]]:
    """``{handle: (level, variant)}``; ``root``: the root's level (None: sampled from ``TOP``)."""
    parents: dict[str, list[str]] = {h: [] for h in dag.vertices}
    for h, v in dag.vertices.items():
        for c in v.children:
            parents[c].append(h)
    levels: dict[str, tuple[int, tuple]] = {}
    for h in parents_first(dag):
        if h == dag.root and root is not None:
            level = root
        else:
            level = min((levels[p][0] for p in parents[h]), default=TOP)
            while level > 1 and rng.random() >= keep:
                level -= 1
        levels[h] = (level, rng.choice(LEVELS[level]))
    return levels


def apply_levels(
    dag: Dag, levels: dict[str, tuple[int, tuple]], short: dict[str, str | None]
) -> Dag:
    """The DAG at those levels: ``dag`` holds the long captions, tags and boxes, ``short`` each
    handle's short caption (missing: the long caption's first sentence)."""
    out = {}
    for h, v in dag.vertices.items():
        caption, tags, box = levels[h][1]
        text = {
            "long": v.text,
            "short": short.get(h) or first_sentence(v.text),
            None: None,
        }[caption]
        out[h] = replace(
            v,
            text=text,
            tags=v.tags if tags else None,
            box=v.box if box else None,
            children=list(v.children),
            relations=list(v.relations),
        )
    return check(Dag(dag.root, out))


def levels_text(levels: dict[str, tuple[int, tuple]]) -> str:
    """One line per vertex: ``level caption+tags+box handle``, for reading a sample."""
    rows = []
    for h, (level, (caption, tags, box)) in levels.items():
        parts = [caption or "-", "tags" if tags else "-", "box" if box else "-"]
        rows.append(f"{level} {'+'.join(parts):16s} [{h}]")
    return "\n".join(rows)
