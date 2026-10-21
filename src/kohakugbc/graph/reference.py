"""Resolution of one image's records into a GBC graph: the Python reference.

A literal port of ml-gbc's ``node_infos_to_gbc_graph`` (``parse_node_infos.py``,
``to_gbc_graph.py``, ``GbcGraphFull``'s edge and mask update) on plain records. The graph is
the dict ``GbcGraphFull.model_dump()`` gives, key for key, so ml-gbc loads it as written.
"""

from kohakugbc.graph.records import DESC_LABELS, VERTEX_LABELS

FULL_BOX = [0.0, 0.0, 1.0, 1.0, None]
CLIP_SCORES = {"scores": None, "truncation": False}


class ResolutionError(ValueError):
    """Records that ml-gbc's conversion cannot turn into a graph."""


def resolve(
    records: list[dict],
    img_path: str | None,
    img_size: list | None = None,
    threshold: float = 0.85,
) -> dict | None:
    """The GBC graph of one image, or None when no record has a description."""
    nodes, mapping = _parse(records)
    if not nodes:
        return None
    vertices = [_vertex(node, mapping) for node in nodes.values()]
    _link_and_mask(vertices, threshold)
    graph = {"vertices": vertices, "img_url": None, "img_path": img_path}
    graph.update(_captions(vertices))
    graph["img_size"] = None if img_size is None else [int(img_size[0]), int(img_size[1])]
    graph["mask_inside_threshold"] = float(threshold)
    return graph


# ---- ids, numbered instances, relation split (parse_node_infos.py) ----------------------------


def _parse(records: list[dict]) -> tuple[dict, dict]:
    # Vertex id -> its record, in first-seen order; entity id -> the vertex ids it points to
    nodes: dict[str, dict] = {}
    mapping: dict[str, list[str]] = {}
    numbered: dict[str, list[str]] = {}
    relational = []

    for rec in records:
        if not rec["descs"]:
            continue
        ids = rec["ids"]
        if not ids:
            raise ResolutionError("a record without ids")
        if ids[0][1] == "relation":
            relational.append(rec)
            continue

        # Every joined id maps to the first; numbered ids (cup_1) gather under their stem
        first, label = ids[0][0], ids[0][1]
        nodes[first] = rec
        for entity_id, entity_label, _ in ids:
            if entity_id in mapping:
                raise ResolutionError(f"entity id {entity_id!r} appears twice")
            if entity_label != label:
                raise ResolutionError(
                    f"joined ids of {first!r} have labels {label}, {entity_label}"
                )
            mapping[entity_id] = [first]
            if "_" in entity_id:
                stem, suffix = entity_id.rsplit("_", 1)
                if suffix.isdigit():
                    numbered.setdefault(stem, []).append(first)

    # A stem with no composition vertex points to its instances
    for stem, firsts in numbered.items():
        if stem not in mapping:
            mapping[stem] = firsts

    for rec in relational:
        split = _split_relation(rec, nodes, mapping)
        nodes.update(split)
        for vertex_id in split:
            if vertex_id not in mapping:
                mapping[vertex_id] = [vertex_id]
    return nodes, mapping


def _node(entity_id: str, nodes: dict, mapping: dict) -> tuple[dict, str]:
    if entity_id not in mapping:
        raise ResolutionError(f"a relation names {entity_id!r}, which has no vertex")
    vertex_id = mapping[entity_id][0]
    return nodes[vertex_id], vertex_id


def _split_relation(rec: dict, nodes: dict, mapping: dict) -> dict:
    """One vertex per set of entities the answers mention, linked from the asked vertex."""
    if len(rec["ids"]) != 1:
        raise ResolutionError("a relation record names exactly one asked vertex")
    asked, asked_id = _node(rec["ids"][0][0], nodes, mapping)
    parent = {**asked, "children": list(asked["children"])}

    split: dict[str, dict] = {}
    for text, desc_label, assoc in rec["descs"]:
        assoc = sorted(assoc)
        vertex_id = "[" + "|".join(assoc) + "]"
        if vertex_id not in split:
            children, boxes = [], []
            for entity_id in assoc:
                child, _ = _node(entity_id, nodes, mapping)
                # The edge label is the id's last part, by ml-gbc's design
                children.append([entity_id, child["ids"][0][1], entity_id.rsplit("_", 1)[-1]])
                boxes.append(child["box"])
            split[vertex_id] = {
                "ids": [[vertex_id, "relation", None]],
                "box": _union(boxes, vertex_id),
                "descs": [],
                "children": children,
            }
            parent["children"] += [[vertex_id, "relation", c[2]] for c in children]
            split[asked_id] = parent
        split[vertex_id]["descs"].append([text, desc_label, [vertex_id]])
    return split


def _union(boxes: list, vertex_id: str) -> list:
    if not boxes:
        raise ResolutionError(f"relation vertex {vertex_id} has no entities")
    if any(b is None for b in boxes):
        raise ResolutionError(f"relation vertex {vertex_id} includes a vertex without a box")
    left, top, right, bottom = boxes[0][:4]
    for b in boxes[1:]:
        left, top = min(left, b[0]), min(top, b[1])
        right, bottom = max(right, b[2]), max(bottom, b[3])
    return [left, top, right, bottom, None]


# ---- vertices (to_gbc_graph.py) ----------------------------------------------------------------


def _vertex(node: dict, mapping: dict) -> dict:
    vertex_id, label, _ = node["ids"][0]
    if mapping.get(vertex_id) != [vertex_id]:
        raise ResolutionError(f"vertex id {vertex_id!r} does not map to itself")
    if label not in VERTEX_LABELS:
        label = "entity"

    descs = []
    for text, desc_label, _ in node["descs"]:
        if desc_label not in DESC_LABELS:
            raise ResolutionError(f"description label {desc_label!r}")
        descs.append(
            {
                "text": text,
                "label": desc_label,
                "full_label": f"{desc_label}-{label}",
                "statistics": None,
                "clip_scores": dict(CLIP_SCORES),
                "toxicity_scores": None,
            }
        )

    # One out-edge per named phrase to every vertex its id maps to
    out_edges = []
    for entity_id, _, text in node["children"]:
        if text is None:
            raise ResolutionError(f"vertex {vertex_id!r} names a child without text")
        if text.strip() == "":
            continue
        out_edges += [
            {"source": vertex_id, "text": text, "target": target}
            for target in mapping.get(entity_id, ())
        ]

    box = node["box"] or FULL_BOX
    return {
        "vertex_id": vertex_id,
        "bbox": {
            "left": float(box[0]),
            "top": float(box[1]),
            "right": float(box[2]),
            "bottom": float(box[3]),
            "confidence": None if box[4] is None else float(box[4]),
        },
        "label": label,
        "descs": descs,
        "in_edges": [],
        "out_edges": out_edges,
        "sub_masks": [],
        "super_masks": [],
    }


# ---- in-edges and masks (GbcGraphFull.update_edges_and_masks) --------------------------------


def _link_and_mask(vertices: list[dict], threshold: float) -> None:
    by_id = {v["vertex_id"]: v for v in vertices}
    ordered = sorted(vertices, key=lambda v: v["vertex_id"])
    for v in ordered:
        kept = []
        for edge in v["out_edges"]:
            if edge["target"] not in by_id:
                continue
            child = by_id[edge["target"]]
            if edge not in child["in_edges"]:
                child["in_edges"].append(edge)
            kept.append(edge)
        v["out_edges"] = kept

        kept = []
        for edge in v["in_edges"]:
            if edge["source"] not in by_id:
                continue
            parent = by_id[edge["source"]]
            if edge not in parent["out_edges"]:
                parent["out_edges"].append(edge)
            kept.append(edge)
        v["in_edges"] = kept

        v["sub_masks"] = []
        v["super_masks"] = []
        for alt in ordered:
            if alt["vertex_id"] == v["vertex_id"]:
                continue
            if _mostly_inside(v["bbox"], alt["bbox"], threshold):
                _add(alt["sub_masks"], v["vertex_id"])
                _add(v["super_masks"], alt["vertex_id"])
            if _mostly_inside(alt["bbox"], v["bbox"], threshold):
                _add(v["sub_masks"], alt["vertex_id"])
                _add(alt["super_masks"], v["vertex_id"])


def _add(ids: list, vertex_id: str) -> None:
    if vertex_id not in ids:
        ids.append(vertex_id)


def _area(left: float, top: float, right: float, bottom: float) -> float:
    return max(0, right - left) * max(0, bottom - top)


def _mostly_inside(inner: dict, outer: dict, threshold: float) -> bool:
    area = _area(inner["left"], inner["top"], inner["right"], inner["bottom"])
    overlap = _area(
        max(inner["left"], outer["left"]),
        max(inner["top"], outer["top"]),
        min(inner["right"], outer["right"]),
        min(inner["bottom"], outer["bottom"]),
    )
    return (overlap / area if area > 0 else 1) > threshold


# ---- graph captions (GbcGraph.model_post_init) -------------------------------------------------


def _captions(vertices: list[dict]) -> dict:
    found = {"original": None, "short": None, "detail": None}
    for v in vertices:
        if v["label"] == "image":
            for desc in v["descs"]:
                if desc["label"] in found and found[desc["label"]] is None:
                    found[desc["label"]] = desc["text"]
            break
    return {
        "original_caption": found["original"],
        "short_caption": found["short"],
        "detail_caption": found["detail"],
    }
