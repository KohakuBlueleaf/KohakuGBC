"""Coverage, relation and composition nodes, and the graph out.

Ports of the PoC's ``RoundEngine.breadth_round`` and ``finalize``, ml-gbc's
``get_relational_queries``, ``RelationQuery`` and ``CompositionQuery``. ml-gbc's unseeded
choices (the phrase naming a set of children, the relation count, the composition hints) are
drawn from ``draw.rng`` keyed by what is asked.
"""

import re

from kohakugbc.graph import to_text
from kohakugbc.graph.reference import resolve
from kohakugbc.nodes import boxes as bx
from kohakugbc.nodes.images import cut, encode
from kohakugbc.nodes.registry import node
from kohakugbc.nodes.specs import chat as specs
from kohakugbc.nodes.specs import composition as comp
from kohakugbc.nodes.specs import draw, parse
from kohakugbc.nodes.text import detectable, singular

CONCISE = r"\[([^\]]+)\]\s*\[([^\]]+)\]"
RELATIONAL = ("relation", "composition")

# ---- coverage (breadth rounds) -----------------------------------------------------------------


@node(outputs=["request"])
def coverage_request(image: dict, records: list, patch, controls: dict) -> dict:
    labels = [c[2] for c in records[0]["children"]]
    return specs.coverage(image["key"], labels, patch, controls)


@node(outputs=["records", "entities", "grew"])
def coverage_root(records: list, reply: str, controls: dict) -> tuple[list, list, bool]:
    """One more root sentence naming objects no element covers yet, and those objects."""
    reply = reply.replace("**", "")
    sentence = (
        reply.split("Sentence:", 1)[-1].strip().split("\n")[0] if "Sentence:" in reply else ""
    )
    if not sentence or not re.search(CONCISE, sentence):
        return records, [], False
    root = records[0]
    seen = {singular(c[2].lower()) for c in root["children"]}
    out = parse.concise(sentence, "", set())
    new = [e for e in out["entities"] if singular(e[2]) not in seen and detectable(e[2])]
    new = new[: controls["coverage_max_new"]]
    if not new:
        return records, [], False
    root = {
        **root,
        "descs": [*root["descs"], out["descs"][0]],
        "children": [*root["children"], *new],
    }
    return [root, *records[1:]], new, True


@node(outputs=["root"])
def root_of(records: list) -> dict:
    return records[0]


# ---- relational queries (get_relational_queries) ------------------------------------------------


@node(outputs=["records"])
def drop_relational(records: list) -> list:
    return [r for r in records if r["ids"][0][1] not in RELATIONAL]


def _queries_for(vertex: dict, by_id: dict, image_key: str) -> tuple[list, list]:
    """The composition queries and the relation query (if any) a vertex's children pose."""
    vid = vertex["vertex_id"]
    has_relation = False
    text_to_ids: dict[str, list[str]] = {}
    for edge in vertex["out_edges"]:
        target = by_id[edge["target"]]
        if target["label"] == "image":
            raise ValueError("an image vertex is a child of another vertex")
        if target["label"] == "relation":
            has_relation = True
        else:
            text_to_ids.setdefault(edge["text"], []).append(edge["target"])
    ids_to_texts: dict[tuple, list[str]] = {}
    for text, ids in text_to_ids.items():
        ids_to_texts.setdefault(tuple(sorted(ids)), []).append(text)

    to_relate, compositions = [], []
    for ids, texts in ids_to_texts.items():
        text = draw.rng("relational-text", image_key, vid, *texts).choice(texts)
        entity_id = f"{vid}_{text}" if vid else text
        if len(ids) == 1:
            to_relate.append([entity_id, text])
            continue
        to_relate.append([entity_id, text])
        names = [[entity_id, "composition", text]]
        names += [[f"{vid}_{t}" if vid else t, "composition", t] for t in texts if t != text]
        parts = [[i, "entity", f"{singular(text)} {k}"] for k, i in enumerate(sorted(ids), start=1)]
        boxes = [by_id[i]["bbox"] for i in sorted(ids)]
        compositions.append({"kind": "composition", "names": names, "parts": parts, "boxes": boxes})

    relation = []
    if not has_relation and len(to_relate) > 1:
        relation = [
            {"kind": "relation", "vertex": vid, "box": vertex["bbox"], "entities": to_relate}
        ]
    return compositions, relation


def _box(b: dict) -> list:
    return [b["left"], b["top"], b["right"], b["bottom"], b["confidence"]]


@node(outputs=["queries"])
def relational_queries(image: dict, records: list, controls: dict) -> list:
    """The relation and composition questions the current records pose: compositions of every
    vertex, then relations, as ml-gbc lists them."""
    if controls["relational"] == "none":
        return []
    graph = resolve(records, image["path"], image["size"])
    if graph is None:
        return []
    by_id = {v["vertex_id"]: v for v in graph["vertices"]}
    for v in graph["vertices"]:
        v["bbox"] = _box(v["bbox"])
    compositions, relations = [], []
    for v in graph["vertices"]:
        if v["label"] in RELATIONAL:
            continue
        c, r = _queries_for(v, by_id, image["key"])
        compositions += c
        relations += r
    match controls["relational"]:
        case "relation":
            return relations
        case "composition":
            return compositions
        case _:
            return compositions + relations


@node(outputs=["request"])
def relational_request(image: dict, q: dict, controls: dict) -> dict:
    """The request of one question: relations on the asked vertex's crop; a composition on the
    crop of its parts' union, its parts boxed and numbered, with tree hints."""
    if q["kind"] == "relation":
        count = draw.rng("relation-count", image["key"], q["vertex"]).randint(2, len(q["entities"]))
        patch = encode(cut(image, q["box"]), controls["max_side"])
        return specs.relation(
            image["key"], q["vertex"], q["box"], q["entities"], count, patch, controls
        )

    union = bx.union(q["boxes"])
    rel = [bx.to_relative(union, b) for b in q["boxes"]]
    first, _, text = q["names"][0]
    rng = draw.rng("composition", image["key"], first)
    texts = [p[2] for p in q["parts"]]
    hints, node_texts = comp.hints(texts, q["boxes"], rel, rng)
    patch = encode(comp.annotate(cut(image, union), rel), controls["max_side"])
    parts = [[p[0], p[2]] for p in q["parts"]]
    request = specs.composition(
        image["key"],
        [n[0] for n in q["names"]],
        text,
        parts,
        q["boxes"],
        node_texts,
        hints,
        patch,
        controls,
    )
    return {**request, "hints": hints}


@node(outputs=["record"])
def relational_record(q: dict, request: dict, reply: str) -> dict:
    """The answer's record: relations between at least two of the entities; or the composition
    caption, its general descriptions and the tree hints."""
    reply = reply.replace("**", "")
    if q["kind"] == "relation":
        out = parse.relation(reply, q["entities"])
        return {
            "ids": [[q["vertex"], "relation", None]],
            "box": q["box"],
            "descs": out["descs"],
            "children": [],
        }
    first = q["names"][0][0]
    out = parse.composition(reply, first, q["parts"])
    descs = out["descs"] + [[h, "hardcode", [first]] for h in request["hints"]]
    return {"ids": q["names"], "box": bx.union(q["boxes"]), "descs": descs, "children": q["parts"]}


@node(outputs=["records"])
def extend(records: list, answered: list) -> list:
    return [*records, *answered]


# ---- the graph -----------------------------------------------------------------------------------


@node(outputs=["graph", "stats"])
def to_gbc(image: dict, records: list, controls: dict) -> tuple:
    """The GBC graph of the records (JSON text) and its counts."""
    return to_text(records, image["path"], image["size"])
