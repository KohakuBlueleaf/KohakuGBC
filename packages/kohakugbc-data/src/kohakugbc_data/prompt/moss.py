"""The prompt DAG as a MOSS-Image Layout draft: the readable JSON that MOSS-Image's own
``skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`` compiles into the one-line
model prompt (centred ``yxyx`` boxes, edges enclosed).

Every boxed vertex but the root is one flat item, in the order the text walks them, with its
caption (the handle when it has none) and its tags; MOSS-Image's layout has no hierarchy or
relations, so relations go into the summary caption only when asked.
"""

from kohakugbc_data.prompt.dag import Dag, check
from kohakugbc_data.prompt.text import hosts


def to_moss(
    dag: Dag,
    width: int = 1024,
    height: int = 1024,
    label: str = "object",
    meta: list[str] | None = None,
    canvas: str | None = "single",
    tags: bool = True,
    relations: bool = False,
) -> dict:
    """The Layout draft on a ``width`` x ``height`` canvas (multiples of 16): boxes as
    ``[x_min, y_min, x_max, y_max]`` pixels from the top-left, ``bbox_coord_space`` ``abs``."""
    check(dag)
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
    caption = dag.vertices[dag.root].text or dag.root
    if relations and rel_texts:
        caption = " ".join([caption, *rel_texts])
    summary: dict = {}
    if meta:
        summary["meta"] = meta
    if canvas:
        summary["canvas"] = canvas
    summary["caption"] = caption
    items = []
    for h in order:
        v = dag.vertices[h]
        if v.box is None:
            continue
        left, top, right, bottom = v.box
        if not (left < right and top < bottom):
            continue
        item = {"label": label, "caption": v.text or h}
        if tags and v.tags:
            item["tags"] = list(v.tags)
        item["bbox"] = [left * width, top * height, right * width, bottom * height]
        items.append(item)
    return {
        "width": width,
        "height": height,
        "bbox_coord_space": "abs",
        "bbox_format": "xyxy",
        "summary": summary,
        "items": items,
    }
