"""Region nodes: detect regions, join a region to a vertex seen before, describe it.

A region is ``{"id", "text", "box", "depth", "leaf"}``: an entity detection found, its box in
image coordinates, its depth (the image is 0), and the caption its parent's reply wrote for it
(``leaf_caption = "parent"``, regions at the maximum depth only).

Ports of ml-gbc's ``DetectionAction`` and ``BaseEntityQuery`` merge, and the PoC's
``RoundEngine`` (``_run_detection``, ``_merge_existing``, ``_prefill_batch``).
"""

import json

from kohakugbc.graph.records import record
from kohakugbc.nodes import boxes as bx
from kohakugbc.nodes.registry import node
from kohakugbc.nodes.specs import chat as specs
from kohakugbc.nodes.specs import detect as det
from kohakugbc.nodes.specs import parse
from kohakugbc.nodes.text import detectable, same_object

# ---- detection ---------------------------------------------------------------------------------


@node(outputs=["entities", "depth", "cap"])
def root_entities(root: dict, controls: dict) -> tuple[list, int, int]:
    """The root's elements to detect, capped (``max_top_level``), one level down."""
    return root["children"], 1, controls["max_top_level"]


@node(outputs=["request"])
def detect_request(image: dict, parent: dict, entities: list, cap) -> dict:
    """The detector request for the detectable entities (at most ``cap``) in the parent's box."""
    texts = [e[2] for e in entities if detectable(e[2])]
    if cap is not None:
        texts = texts[:cap]
    return det.request(image["key"], parent["box"], texts)


@node(outputs=["go"])
def has_texts(request: dict) -> bool:
    return bool(request["texts"])


@node(outputs=["found"])
def detected(raw: dict, patch, request: dict, parent: dict, entities: list, depth, controls):
    """The regions the detector found, one level below the parent; at the maximum depth with
    ``leaf_caption = "parent"``, each takes the caption the parent's reply wrote for it."""
    named = {e[2]: e for e in entities}
    asked = [named[t] for t in request["texts"]]
    size = [int(patch.shape[1]), int(patch.shape[0])]
    found = det.children(raw, size, request, asked, parent["box"], controls)

    captions = {}
    leaf = controls["leaf_caption"] == "parent" and depth >= controls["max_depth"]
    if leaf and parent["box"] is not None:
        captions = parse.feature_captions(parent.get("raw"))
    return [{**r, "depth": depth, "leaf": captions.get(r["text"])} for r in found]


# ---- joining a vertex seen before (BaseEntityQuery.query) ----------------------------------------


@node(outputs=["seen"])
def empty_seen() -> list:
    return []


@node(outputs=["seen", "is_new", "target"])
def register(seen: list, region: dict, vec, controls: dict) -> tuple[list, bool, str | None]:
    """A region joins the first vertex seen before whose box overlaps its own both ways and one of
    whose names is the same object; the vertex's box widens to cover it. Otherwise the region is
    a new vertex."""
    threshold = controls["overlap_threshold"]
    for k, entry in enumerate(seen):
        if not bx.overlapped(region["box"], entry["box"], threshold):
            continue
        for text, other in zip(entry["texts"], entry["vecs"]):
            if same_object(region["text"], text, vec, other, controls["same_threshold"]):
                joined = {
                    **entry,
                    "box": bx.union([entry["box"], region["box"]]),
                    "texts": [*entry["texts"], region["text"]],
                    "vecs": [*entry["vecs"], vec],
                }
                return [*seen[:k], joined, *seen[k + 1 :]], False, entry["vertex"]
    entry = {"vertex": region["id"], "box": region["box"], "texts": [region["text"]], "vecs": [vec]}
    return [*seen, entry], True, None


# ---- describing a region -------------------------------------------------------------------------


@node(outputs=["how"])
def answer_from(region: dict, replies) -> str:
    """Where the region's reply comes from: its parent's reply, a batched reply, or its own."""
    if region.get("leaf"):
        return "parent"
    if replies and region["id"] in replies:
        return "batch"
    return "own"


@node(outputs=["reply"])
def parent_reply(region: dict) -> str:
    return json.dumps({"present": True, "detailed_caption": region["leaf"], "features": []})


@node(outputs=["reply"])
def batch_reply(region: dict, replies: dict) -> str:
    return replies[region["id"]]


@node(outputs=["request"])
def entity_request(image: dict, region: dict, patch, controls: dict) -> dict:
    return specs.entity(image["key"], region, patch, controls)


@node(outputs=["reply"])
def model_reply(answer: str) -> str:
    """A model's reply as GBC parses it: markdown bold removed."""
    return answer.replace("**", "")


@node(outputs=["record"])
def entity_record(region: dict, reply: str) -> dict:
    """The region's record: its captions and the features they name; no descriptions when the
    reply says the entity is absent or cannot be read."""
    out = parse.entity(reply, region["id"], region["text"])
    ids = [[region["id"], "entity", region["text"]]]
    return {**record(ids, region["box"], out["descs"], out["entities"]), "raw": reply}


@node(outputs=["entities", "depth", "cap", "look"])
def parts(record: dict, region: dict, controls: dict) -> tuple[list, int, int, bool]:
    """The features to detect below the region, if it is not at the maximum depth."""
    depth = region["depth"] + 1
    entities = [e for e in record["children"] if detectable(e[2])]
    return (
        entities,
        depth,
        controls["max_features_per_entity"],
        bool(entities and depth <= controls["max_depth"]),
    )


@node(outputs=["records"])
def add_result(records: list, region: dict, record, target) -> list:
    """The region's new record, or its join into ``target``: the target's ids gain the region,
    its box widens to cover it."""
    if record is not None:
        return [*records, record]
    out = []
    for rec in records:
        if rec["ids"][0][0] == target and rec["box"] is not None:
            rec = {
                **rec,
                "ids": [*rec["ids"], [region["id"], "entity", region["text"]]],
                "box": bx.union([rec["box"], region["box"]]),
            }
        out.append(rec)
    return out


# ---- several regions in one request (RoundEngine._prefill_batch) ---------------------------------


@node(outputs=["regions", "several"])
def batch_regions(picked: list, controls: dict) -> tuple[list, bool]:
    """The regions one batched request asks about: those without a parent's caption, when
    ``entity_batch`` is above 1 and at least two remain."""
    if controls["entity_batch"] <= 1:
        return [], False
    regions = [r for r in picked if not r.get("leaf")]
    return regions, len(regions) >= 2


@node(outputs=["request"])
def entity_batch_request(image: dict, regions: list, patches: list) -> dict:
    return specs.entity_batch(image["key"], regions, patches)


@node(outputs=["replies"])
def batch_replies(regions: list, reply: str) -> dict:
    """Each region's own reply, cut from the batched one; regions the reply leaves out get none."""
    numbered = {}
    for i, item in enumerate((parse.load_json(reply) or {}).get("regions") or []):
        if isinstance(item, dict):
            number = item.get("region")
            numbered[number if isinstance(number, int) else i + 1] = item
    replies = {}
    for k, region in enumerate(regions, start=1):
        item = numbered.get(k)
        if item is None:
            continue
        single = {
            "present": bool(item.get("present")),
            "detailed_caption": item.get("detailed_caption") or "",
            "features": item.get("features") or [],
        }
        replies[region["id"]] = json.dumps(single, ensure_ascii=False)
    return replies


@node(outputs=["text"])
def name_of(region: dict) -> str:
    return region["text"]
