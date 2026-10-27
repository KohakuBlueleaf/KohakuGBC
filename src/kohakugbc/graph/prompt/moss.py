"""The prompt DAG as MOSS-Image's layout JSON: a summary caption and a flat item list.

Every boxed vertex but the root is one item, in the order the text walks them; MOSS-Image's
layout has no hierarchy, relations or tags, so those are dropped unless asked into the summary.
"""

from kohakugbc.graph.prompt.coords import MOSS, Coords
from kohakugbc.graph.prompt.dag import Dag, check
from kohakugbc.graph.prompt.text import hosts


def to_moss(
    dag: Dag,
    coords: Coords = MOSS,
    label: str = "object",
    meta: list[str] | None = None,
    relations: bool = False,
) -> dict:
    """``coords`` must be MOSS-Image's ``abs_centralized`` space; ``relations``: append the
    relation texts to the summary caption."""
    check(dag)
    if coords.space != "abs_centralized":
        raise ValueError("MOSS-Image layouts take bbox_coord_space abs_centralized")
    host = hosts(dag, "first")
    order: list[str] = []
    rel_texts: list[str] = []

    def walk(h: str) -> None:
        rel_texts.extend(r.text for r in dag.vertices[h].relations if r.text)
        for c in dag.vertices[h].children:
            if host[c] == h:
                order.append(c)
                walk(c)

    walk(dag.root)
    caption = dag.vertices[dag.root].text or ""
    if relations and rel_texts:
        caption = " ".join([caption, *rel_texts]).strip()
    summary = {"caption": caption}
    if meta:
        summary = {"meta": meta, "canvas": "single", **summary}
    items = [
        {"label": label, "caption": dag.vertices[h].text or "", "bbox": coords.write(v.box)}
        for h in order
        if (v := dag.vertices[h]).box is not None
    ]
    return {
        "summary": summary,
        "items": items,
        "bbox_coord_space": coords.space,
        "bbox_format": coords.order,
    }
