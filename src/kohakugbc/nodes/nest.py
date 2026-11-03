"""GBC-Nest's root, detection and join nodes (``gbc_nest.kir``).

- Root: every element the image query lists is kept, a bracket that only modifies the next one
  ("[anime-style] [girl]") is not an element, and the fixed ``probes`` phrases are detected
  beside the named elements.
- Detection: single elements sharing a head noun get one box each, one-to-one; a single
  element takes the largest of its near-tied boxes; a probe box is kept only where no named
  element is; a region under ``small_area`` of the image is a leaf captioned by its name; a probe
  region is free: described first, outside the node budget, never a leaf.
- Joins: a region never joins one of its ancestors, and joins a vertex seen before with the same
  head noun whose box overlaps its own both ways by ``head_overlap``; otherwise GBC's join.
"""

import re

from kohakugbc.graph.records import record
from kohakugbc.nodes import boxes as bx
from kohakugbc.nodes import regions
from kohakugbc.nodes.registry import node
from kohakugbc.nodes.specs import parse
from kohakugbc.nodes.text import detectable, singular
from kohakugbc.program.controls import complete

LINKS = {"in", "with", "of", "on", "at", "wearing", "holding", "near", "under", "from"}
MODIFIER = re.compile(r"\[([^\]]+)\]\s+\[([^\]]+)\]")
# One-to-one assignment: candidate boxes from this score, NMS at ASSIGN_NMS between candidates,
# a name scores a candidate by its own boxes overlapping it by ASSIGN_MATCH
ASSIGN_MIN, ASSIGN_NMS, ASSIGN_MATCH = 0.1, 0.5, 0.7


def head(name: str) -> str:
    """The noun a name is about: the last word before its first linking word, singular
    ("silver-haired character in coat" -> "character", "apple 2" -> "apple")."""
    words = re.sub(r"[^a-z0-9 \-']", " ", name.lower()).split()
    words = [w for w in words if not w.isdigit()]
    if not words:
        return ""
    cut = next((k for k, w in enumerate(words) if w in LINKS and k > 0), len(words))
    return singular(words[cut - 1])


def iou(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score(box) -> float:
    return 1.0 if len(box) < 5 or box[4] is None else float(box[4])


@node(outputs=["controls"])
def nest_defaults(controls) -> dict:
    return complete(controls, "gbc_nest")


# ---- root --------------------------------------------------------------------------------------


@node(outputs=["root"])
def nest_image_root(image: dict, reply: str) -> dict:
    """The image vertex from GBC's image query, every listed element kept (one the captions do
    not name gets a bag-of-words caption naming it), modifier brackets dropped."""
    text = reply.replace("**", "")
    out = parse.image(text)
    obj = parse.load_json(text) or {}
    listed = [e for e in obj.get("elements") or [] if isinstance(e, dict)]
    entities, descs = list(out["entities"]), list(out["descs"])

    # Elements the concise caption does not tag
    seen = {e[2] for e in entities}
    added = []
    for el in listed:
        name = str(el.get("name") or "").strip().lower()
        if not name or name in seen or not detectable(name):
            continue
        label = "multiple" if el.get("multiplicity") == "multiple" else "single"
        entity = parse.name_entity(name, label, "", set(seen))
        if entity is not None:
            seen.add(entity[2])
            added.append(entity)
    if added:
        entities += added
        descs.append([", ".join(e[2] for e in added), "bagofwords", [""]])

    # "[anime-style] [girl]": the first bracket modifies the second
    names = {str(e.get("name") or "").strip().lower() for e in listed}
    caption = str(obj.get("concise_caption") or "")
    modifiers = {m.group(1).strip().lower() for m in MODIFIER.finditer(caption)} - names
    entities = [e for e in entities if e[2] not in modifiers]
    return {**record([["", "image", None]], None, descs, entities), "raw": reply}


@node(outputs=["root"])
def with_probes(root: dict, controls: dict) -> dict:
    """The root with the ``probes`` phrases among its elements, named by a bag-of-words caption."""
    seen = {e[2] for e in root["children"]}
    added = []
    for phrase in controls["probes"]:
        entity = parse.name_entity(phrase, "multiple", "", set(seen))
        if entity is not None and phrase not in seen:
            seen.add(phrase)
            added.append(entity)
    if not added:
        return root
    desc = [", ".join(e[2] for e in added), "bagofwords", [""]]
    return {**root, "children": [*root["children"], *added], "descs": [*root["descs"], desc]}


# ---- detection ---------------------------------------------------------------------------------


def assign(raw: dict, texts: list, group: list, floor: float) -> dict:
    """One box per name of ``group``, one name per box: candidates are the group's boxes from
    ASSIGN_MIN (NMS between them), a name scores a candidate by its best box overlapping it, the
    best pairs are taken greedily, an assigned box's score raised to ``floor``."""
    idx = {texts.index(t) for t in group}
    triples = zip(raw["boxes"], raw["scores"], raw["labels"])
    mine = [(b, s, lab) for b, s, lab in triples if lab in idx]
    cands = []
    for b, s, _ in sorted(mine, key=lambda x: -x[1]):
        if s >= ASSIGN_MIN and all(iou(b, c) < ASSIGN_NMS for c in cands):
            cands.append(b)
    pairs = []
    for label in idx:
        for k, c in enumerate(cands):
            best = [s for b, s, lab in mine if lab == label and iou(b, c) >= ASSIGN_MATCH]
            s = max(best, default=0.0)
            if s >= ASSIGN_MIN:
                pairs.append((s, label, k))
    used_l, used_c, chosen = set(), set(), []
    for s, label, k in sorted(pairs, reverse=True):
        if label in used_l or k in used_c:
            continue
        used_l.add(label)
        used_c.add(k)
        chosen.append((cands[k], max(s, floor), label))
    triples = zip(raw["boxes"], raw["scores"], raw["labels"])
    rest = [(b, s, lab) for b, s, lab in triples if lab not in idx] + chosen
    return {
        **raw,
        "boxes": [r[0] for r in rest],
        "scores": [r[1] for r in rest],
        "labels": [r[2] for r in rest],
    }


def prefer_large(raw: dict, texts: list, labels: dict, tie: float) -> dict:
    """A single element keeps one box: the largest of those scoring at least ``tie`` of its best,
    its score lifted just above the best (selection would drop a box that holds another)."""
    boxes, scores = raw["boxes"], list(raw["scores"])
    drop = set()
    for k, t in enumerate(texts):
        if labels.get(t) != "single":
            continue
        mine = [i for i, lab in enumerate(raw["labels"]) if lab == k]
        if len(mine) < 2:
            continue
        best = max(scores[i] for i in mine)
        ties = [i for i in mine if scores[i] >= tie * best]
        big = max(ties, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        scores[big] = min(1.0, best + 1e-3)
        drop.update(i for i in mine if i != big)
    keep = [i for i in range(len(scores)) if i not in drop]
    return {
        **raw,
        "boxes": [boxes[i] for i in keep],
        "scores": [scores[i] for i in keep],
        "labels": [raw["labels"][i] for i in keep],
    }


@node(outputs=["found"])
def nest_detected(raw: dict, patch, request: dict, parent: dict, entities: list, depth, controls):
    """GBC's detected regions after GBC-Nest's box choices (module docstring)."""
    texts = request["texts"]
    labels = {e[2]: e[1] for e in entities}

    # Single names sharing a head noun: one box each
    heads: dict = {}
    for t in texts:
        if labels.get(t) == "single":
            heads.setdefault(head(t), []).append(t)
    for group in heads.values():
        if len(group) >= 2:
            raw = assign(raw, texts, group, controls["box_score"] + 1e-4)
    raw = prefer_large(raw, texts, labels, controls["single_tie"])
    found = regions.detected(raw, patch, request, parent, entities, depth, controls)

    # Probes at the root: sure boxes only, and only where no named element or kept probe is
    probes = set(controls["probes"])
    if depth == 1:
        found = [
            r
            for r in found
            if r["text"] not in probes or score(r["box"]) >= controls["probe_score"]
        ]
        named = [r["box"] for r in found if r["text"] not in probes]
        kept, keep_ids = [], set()
        for r in sorted((r for r in found if r["text"] in probes), key=lambda r: -score(r["box"])):
            if any(iou(r["box"], b) >= controls["probe_gap"] for b in named + kept):
                continue
            kept.append(r["box"])
            keep_ids.add(r["id"])
        found = [r for r in found if r["text"] not in probes or r["id"] in keep_ids]

    out = []
    for r in found:
        if r["text"] in probes:
            out.append({**r, "leaf": None, "free": True})
            continue
        small = bx.area(r["box"]) < controls["small_area"]
        out.append({**r, "leaf": r["leaf"] or (r["text"] if small else None)})
    return out


# ---- joins ---------------------------------------------------------------------------------------


@node(outputs=["seen", "is_new", "target"])
def nest_register(
    seen: list, region: dict, texts: list, vecs: list, controls: dict
) -> tuple[list, bool, str | None]:
    """A region joins the best-overlapping vertex seen before (not an ancestor) that shares its
    head noun, its box overlapping the region's both ways by ``head_overlap``; else GBC's
    ``register`` over the vertices that are not its ancestors."""
    parts = region["id"].split("_")
    ancestors = {"_".join(parts[:k]) for k in range(1, len(parts))}
    mine = [e for e in seen if e["vertex"] not in ancestors]
    theirs = [e for e in seen if e["vertex"] in ancestors]

    name = head(region["text"])
    best, best_k = 0.0, None
    for k, e in enumerate(mine):
        if name not in {head(t) for t in e["texts"]}:
            continue
        if not bx.overlapped(region["box"], e["box"], controls["head_overlap"]):
            continue
        o = iou(region["box"], e["box"])
        if o > best:
            best, best_k = o, k
    if best_k is None:
        out, is_new, target = regions.register(mine, region, texts, vecs, controls)
        return [*out, *theirs], is_new, target

    known = dict(zip(texts, vecs))
    e = mine[best_k]
    joined = {
        **e,
        "box": bx.union([e["box"], region["box"]]),
        "texts": [*e["texts"], region["text"]],
        "vecs": [*e["vecs"], known.get(region["text"])],
    }
    return [*mine[:best_k], joined, *mine[best_k + 1 :], *theirs], False, e["vertex"]
