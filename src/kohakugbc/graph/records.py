"""Records: what one image job collects, as plain data.

A record is one query result of the job, the plain-data form of ml-gbc's ``NodeInfo``:

``{"ids": [ref, ...], "box": box | None, "descs": [desc, ...], "children": [ref, ...]}``

- ``ids``: the entities the result answers for, ``[entity_id, label, text]``. The first is the
  vertex; the others are regions that joined it as the same object. The label is the vertex
  label (``image``, ``entity``, ``composition``, ``relation``).
- ``box``: ``[left, top, right, bottom, confidence]`` relative to the image; ``None`` is the
  whole image.
- ``descs``: ``[text, desc_label, [entity_id, ...]]``; the ids name the entities a relational
  answer is about (empty otherwise).
- ``children``: ``[entity_id, label, text]`` for every phrase of the result that names an
  entity; ``text`` becomes the edge label.

A relation record (``ids[0]`` labelled ``relation``) names the vertex the question was asked on;
resolution splits it into one vertex per set of entities its answers mention. A record without
descs (an absent entity, a failed query) gives no vertex.

Records hold lists, never tuples, so a record equals its JSON round trip.
"""

DESC_LABELS = ("short", "relation", "composition", "detail", "original", "hardcode", "bagofwords")
VERTEX_LABELS = ("image", "entity", "composition", "relation")


def ref(entity_id: str, label: str, text: str | None) -> list:
    return [entity_id, label, text]


def record(ids: list, box: list | None, descs: list, children: list) -> dict:
    return {"ids": ids, "box": box, "descs": descs, "children": children}


def box_of(bbox: dict | None) -> list | None:
    """An ml-gbc ``Bbox`` dump as a record box (coordinates as floats)."""
    if bbox is None:
        return None
    conf = bbox.get("confidence")
    return [
        float(bbox["left"]),
        float(bbox["top"]),
        float(bbox["right"]),
        float(bbox["bottom"]),
        None if conf is None else float(conf),
    ]


def from_node_info(dump: dict) -> dict:
    """The record of one ml-gbc ``NodeInfo.model_dump()`` (a line of ``node_infos.jsonl``)."""
    inputs = dump["input_dict"]
    result = dump["query_result"]

    # entity_info is one EntityInfo or a list of joined ones
    infos = inputs["entity_info"]
    if isinstance(infos, dict):
        infos = [infos]
    ids = [ref(i.get("entity_id", ""), i["label"], i.get("text")) for i in infos]

    descs = [[d["text"], d["label"], list(assoc)] for d, assoc in result["descs"]]
    children = [
        ref(e.get("entity_id", ""), e["label"], e.get("text")) for e, _ in result["entities"]
    ]
    return record(ids, box_of(inputs.get("bbox")), descs, children)
