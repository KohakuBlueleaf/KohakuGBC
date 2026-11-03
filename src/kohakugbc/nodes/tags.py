"""Tag nodes: the tagger's request for a finished graph, and the tags placed on its vertices.

``tagging`` (controls): ``none``; ``image`` (the image's tags on the image vertex); ``regions``
(each tag placed on the vertices it describes). The image's tag set is the tagger's own
(``tag_source = "tagger"``), the job's ``tags`` (``"given"``, e.g. danbooru's) or both.

``tag_placement``: ``top`` (``place_top``: the image vertex holds the whole set, each vertex whose
only parent is the image its own tags, nothing moved) or ``place`` (below). ``tag_regions``:
``crop`` (each target vertex's box tagged as its own picture) or ``pool`` (the box's tokens
pooled from the whole picture's pass, which every token sees through global attention).

``tag_select = "important"`` (``regions`` tagging): every entity and composition is pooled in
the whole picture's pass; the vertices that are the best region of at least ``tag_min_tags`` of
the image's tags (``held``) are the important ones, at most ``tag_max_regions``, none within or
around another; with ``crop`` they get a second pass of their own crops. The image vertex holds
the whole set, each important vertex its own tags, every other vertex none (``place_focus``).

Placement (``place``), per tag of the set:

- image scope (counts such as 1girl / 2girls / multiple_girls, solo, framing, backgrounds, and
  the rating / meta / style / copyright categories): the image vertex only;
- otherwise the entity and composition vertices whose region score reaches the tag's category
  threshold (times ``tag_region_scale``) and lies within ``tag_region_margin`` (logits) of the
  best region's (every region sees the whole picture through the ViT's global attention, so a
  tag scores on regions without it too, only lower); none: the image vertex;
- identity (character tags): the topmost of those (a part of the character scores too);
- attributes: the deepest of those (``pink hair`` on the hair, not on the girl as well);
- a plural tag on several vertices (``boots`` on boot 1 and boot 2) moves to their lowest common
  ancestor (the composition ``2 boots``, else the wearer, else the image), never split.

A vertex gets ``tags`` (tag -> score) and one ``bagofwords`` caption (the tags, spaces for
underscores); the graph gets ``tags`` (the image's whole set with categories, and every region's
scores of it).
"""

import json
import math
import re

import numpy as np
from PIL import Image

from kohakugbc.nodes import images
from kohakugbc.nodes.registry import node
from kohakugbc.nodes.text import singular

# The picture sent to the tagger: the longer side at most this (the PixAI canvas, which the
# engine fits the picture to exactly); a cap on what crosses to the engine, never an upscale
TAG_SIDE = 1008
IMAGE_CATEGORIES = {"rating", "meta", "style", "copyright"}
# Important vertices (``held``): two regions are one place when this share of the smaller lies
# inside the other; a larger region holding a tag within LIFT of the best takes it; a region
# covering WHOLE of the image is the image; a candidate within (WITHIN) one taken is skipped
SAME_PLACE, LIFT, WHOLE, WITHIN = 0.5, 0.05, 0.9, 0.8
IDENTITY_CATEGORIES = {"character"}
TAGGED_LABELS = ("entity", "composition")
# General tags about the whole picture: people counts, framing, view, background, medium
IMAGE_GENERAL = re.compile(
    r"^(\d+\+?(girl|boy|other)s?|multiple_(girls|boys|others)|solo|solo_focus|no_humans"
    r"|everyone|.*_background|background|cowboy_shot|full_body|upper_body|lower_body|portrait"
    r"|close-up|wide_shot|very_wide_shot|from_(above|below|side|behind|outside)|dutch_angle"
    r"|pov|.*out_of_frame|scenery|outdoors|indoors|day|night|evening|sunset|depth_of_field"
    r"|blurry|bokeh|lens_flare|chromatic_aberration|light_rays|border|letterboxed|greyscale"
    r"|monochrome|sketch|lineart|traditional_media|comic|4koma|multiple_views|official_art"
    r"|signature|watermark|artist_name|dated|.*_username|text_focus|english_text"
    r"|speech_bubble|realistic|photorealistic|3d|pixel_art)$"
)


def scope(tag: str, category: str) -> str:
    """``image``, ``identity`` or ``attribute``."""
    if category in IMAGE_CATEGORIES or IMAGE_GENERAL.match(tag):
        return "image"
    return "identity" if category in IDENTITY_CATEGORIES else "attribute"


def plural(tag: str) -> bool:
    words = tag.replace("_", " ")
    return singular(words) != words.lower()


def given_tags(tags) -> list[str]:
    """The job's tags as tagger names (spaces to underscores, lower case)."""
    if not tags:
        return []
    names = tags.keys() if isinstance(tags, dict) else tags
    return [str(t).strip().lower().replace(" ", "_") for t in names if str(t).strip()]


@node(outputs=["tagging"])
def tag_mode(controls: dict) -> bool:
    return controls["tagging"] != "none"


def _capped(picture: np.ndarray) -> np.ndarray:
    scale = TAG_SIDE / max(picture.shape[:2])
    if scale >= 1:
        return picture
    size = (max(1, round(picture.shape[1] * scale)), max(1, round(picture.shape[0] * scale)))
    img = Image.fromarray(picture).convert("RGB")
    return np.asarray(img.resize(size, Image.BILINEAR))


def targets(vertices: list, placement: str) -> list[dict]:
    """The vertices tagged on their own: ``top``, the entities and compositions whose only
    parent is the image (a relation naming them is no parent); ``place``, every entity and
    composition."""
    root = vertices[0]["vertex_id"]
    labels = {v["vertex_id"]: v["label"] for v in vertices}
    out = []
    for v in vertices:
        if v["label"] not in TAGGED_LABELS or not v.get("bbox"):
            continue
        parents = {
            e["source"] for e in v.get("in_edges", []) if labels.get(e["source"]) != "relation"
        }
        if placement == "place" or parents == {root}:
            out.append(v)
    return out


def _box(v: dict) -> list:
    b = v["bbox"]
    return [b["left"], b["top"], b["right"], b["bottom"]]


@node(outputs=["request"])
def tag_request(image: dict, graph: str, tags, controls: dict) -> dict:
    """The tagger's request: the whole image fitted to 1008 px; ``regions`` tagging, the target
    vertices (``targets``) as ``crops`` (``tag_regions = "crop"``, each its own picture) or as
    boxes pooled from the whole image's tokens (``"pool"``); with ``tag_select = "important"``,
    every entity and composition pooled (``tag_focus`` picks from them); the job's tags to
    score."""
    regions, crops = {}, {}
    candidates = given_tags(tags) if controls["tag_source"] != "tagger" else []
    if controls["tagging"] == "regions" and controls["tag_select"] == "important":
        for v in targets(json.loads(graph)["vertices"], "place"):
            regions[v["vertex_id"]] = _box(v)
        picture = _capped(images.cut(image, None))
        return {"picture": picture, "regions": regions, "crops": {}, "candidates": candidates}
    if controls["tagging"] == "regions":
        for v in targets(json.loads(graph)["vertices"], controls["tag_placement"]):
            b = v["bbox"]
            box = [b["left"], b["top"], b["right"], b["bottom"]]
            if controls["tag_regions"] == "crop":
                crop = images.cut(image, box)
                if min(crop.shape[:2]) > 0:
                    crops[v["vertex_id"]] = _capped(crop)
            else:
                regions[v["vertex_id"]] = box
    picture = _capped(images.cut(image, None))
    return {"picture": picture, "regions": regions, "crops": crops, "candidates": candidates}


def _overlap(a: list, b: list) -> float:
    """The share of the smaller box inside the other."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min(bx_area(a), bx_area(b))
    return w * h / small if small > 0 else 0.0


def _within(a: list, b: list) -> bool:
    """The smaller of two boxes lies mostly inside the other."""
    return _overlap(a, b) >= WITHIN


def _words(text: str) -> set:
    return {singular(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3}


def _texts(graph: dict) -> dict:
    """Per vertex, the words of its captions and of its descendants' captions and names."""
    verts = {v["vertex_id"]: v for v in graph["vertices"]}
    out = {}
    for vid in verts:
        words, todo, seen = set(), [vid], set()
        while todo:
            u = todo.pop()
            if u in seen or u not in verts or verts[u]["label"] == "relation":
                continue
            seen.add(u)
            for d in verts[u]["descs"]:
                words |= _words(d["text"])
            for e in verts[u]["out_edges"]:
                words |= _words(e["text"])
                todo.append(e["target"])
        out[vid] = words
    return out


def held(graph: dict, answer: dict, controls: dict) -> dict:
    """Per vertex, the image's tags it owns by the pooled scores (image-scope tags left out).

    Every region sees the whole picture through global attention, so pooled scores sit near 1
    on most regions and the best one alone says nothing. A tag is owned at a place only when its
    best region reaches the tag's threshold (times ``tag_region_scale``) and leads every region
    elsewhere (not overlapping it by SAME_PLACE) by ``tag_local_margin``; at that place, the
    largest region holding it within LIFT of the best owns it (a part does not take its
    subject's tags); and a word of the tag must appear in the owner's captions or its
    descendants' captions and names (a door does not own the skirt of a girl in front of it).
    Regions covering WHOLE of the image or more are the image itself."""
    verts = {v["vertex_id"]: v for v in graph["vertices"]}
    boxes = {
        vid: _box(v)
        for vid, v in verts.items()
        if v["label"] in TAGGED_LABELS and v.get("bbox") and bx_area(_box(v)) < WHOLE
    }
    texts = _texts(graph)
    out: dict = {}
    scale, margin = controls["tag_region_scale"], controls["tag_local_margin"]
    for tag in answer["image"]:
        if scope(tag, answer["categories"].get(tag, "general")) == "image":
            continue
        threshold = answer["thresholds"].get(tag)
        if threshold is None:
            continue
        region = {v: s.get(tag, 0.0) for v, s in answer["regions"].items() if v in boxes}
        if not region:
            continue
        best = max(region, key=lambda v: (region[v], v))
        if region[best] < threshold * scale:
            continue
        same = [v for v in region if _overlap(boxes[v], boxes[best]) >= SAME_PLACE]
        elsewhere = max((region[v] for v in region if v not in same), default=0.0)
        if region[best] - elsewhere < margin:
            continue
        holders = [v for v in same if region[v] >= region[best] - LIFT]
        owner = max(holders, key=lambda v: (bx_area(boxes[v]), v))
        if _words(re.sub(r"\(.*?\)", "", tag).replace("_", " ")) & texts[owner]:
            out.setdefault(owner, []).append(tag)
    return out


@node(outputs=["focus"])
def tag_focus(graph: str, scores: dict, controls: dict) -> list:
    """With ``tag_select = "important"``: the vertices owning at least ``tag_min_tags`` of the
    image's tags (``held``), most first (larger first on ties), a vertex within or around one
    taken already skipped, at most ``tag_max_regions``."""
    if controls["tagging"] != "regions" or controls["tag_select"] != "important":
        return []
    g = json.loads(graph)
    boxes = {v["vertex_id"]: _box(v) for v in g["vertices"] if v.get("bbox")}
    counts = {v: len(t) for v, t in held(g, scores, controls).items() if v in boxes}
    ranked = [v for v, n in counts.items() if n >= controls["tag_min_tags"]]
    ranked.sort(key=lambda v: (-counts[v], -bx_area(boxes[v]), v))
    chosen: list = []
    for v in ranked:
        if len(chosen) >= controls["tag_max_regions"]:
            break
        if not any(_within(boxes[v], boxes[c]) for c in chosen):
            chosen.append(v)
    return chosen


@node(outputs=["go"])
def focus_crops(focus: list, controls: dict) -> bool:
    """Whether the chosen vertices get a crop pass of their own (``tag_regions = "crop"``)."""
    return bool(focus) and controls["tag_regions"] == "crop"


@node(outputs=["request"])
def focus_request(image: dict, graph: str, focus: list, tags, controls: dict) -> dict:
    """The tagger's second request: the chosen vertices' crops."""
    by_id = {v["vertex_id"]: v for v in json.loads(graph)["vertices"]}
    crops = {}
    for vid in focus:
        crop = images.cut(image, _box(by_id[vid]))
        if min(crop.shape[:2]) > 0:
            crops[vid] = _capped(crop)
    candidates = given_tags(tags) if controls["tag_source"] != "tagger" else []
    picture = _capped(images.cut(image, None))
    return {"picture": picture, "regions": {}, "crops": crops, "candidates": candidates}


@node(outputs=["scores"])
def with_focus(scores: dict, focus: list, focused) -> dict:
    """The first pass's scores with the chosen vertices and, from a crop pass, their crops' own
    tags."""
    out = {**scores, "focus": list(focus)}
    if focused:
        out["crops"] = focused.get("crops", {})
        out["categories"] = {**scores["categories"], **focused["categories"]}
        out["thresholds"] = {**scores["thresholds"], **focused["thresholds"]}
    return out


def bx_area(b: list) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def place_focus(graph: dict, answer: dict, given: list[str], controls: dict) -> dict:
    """``tag_select = "important"``: the image vertex holds the whole set; each chosen vertex its
    crop's own tags (a crop pass), else the tags it holds by the pooled scores (``held``);
    image-scope tags left out; every other vertex none."""
    vertices = graph["vertices"]
    root = vertices[0]["vertex_id"]
    out: dict = {v["vertex_id"]: {} for v in vertices}
    out[root] = dict(_tagset(answer, given, controls))
    pooled = held(graph, answer, controls)
    crops = answer.get("crops", {})
    for vid in answer.get("focus", []):
        if vid in crops:
            own = crops[vid]
        else:
            own = {t: answer["regions"][vid][t] for t in pooled.get(vid, [])}
        for tag, p in own.items():
            if scope(tag, answer["categories"].get(tag, "general")) != "image":
                out[vid][tag] = p
    return out


def _structure(vertices: list) -> tuple[dict, dict]:
    """Each vertex's depth (from the image) and its ancestors (itself included); a relation
    hangs under the vertex holding what it relates, so its edges keep that containment. An
    edge to a vertex the graph does not hold is skipped."""
    held = {v["vertex_id"] for v in vertices}
    parents = {
        v["vertex_id"]: [e["source"] for e in v.get("in_edges", []) if e["source"] in held]
        for v in vertices
    }
    root = vertices[0]["vertex_id"]
    depth, frontier = {root: 0}, [root]
    children = {v["vertex_id"]: [e["target"] for e in v.get("out_edges", [])] for v in vertices}
    while frontier:
        nxt = []
        for u in frontier:
            for c in children.get(u, []):
                if c not in depth:
                    depth[c] = depth[u] + 1
                    nxt.append(c)
        frontier = nxt
    ancestors: dict = {}

    def up(v: str) -> set:
        if v not in ancestors:
            ancestors[v] = {v}
            for p in parents.get(v, []):
                ancestors[v] |= up(p)
        return ancestors[v]

    for v in parents:
        up(v)
    return depth, ancestors


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _tagset(answer: dict, given: list[str], controls: dict) -> dict:
    source = controls["tag_source"]
    tagset = {}
    if source in ("tagger", "both"):
        tagset.update(answer["image"])
    if source in ("given", "both"):
        tagset.update({t: answer["given"].get(t) for t in given})
    return tagset


def place(graph: dict, answer: dict, given: list[str], controls: dict) -> dict:
    """``{vertex_id: {tag: score}}`` (see the module's rules)."""
    if controls["tag_select"] == "important":
        return place_focus(graph, answer, given, controls)
    if controls["tag_placement"] == "top":
        return place_top(graph, answer, given, controls)
    vertices = graph["vertices"]
    root = vertices[0]["vertex_id"]
    labels = {v["vertex_id"]: v["label"] for v in vertices}
    depth, ancestors = _structure(vertices)
    tagset = _tagset(answer, given, controls)
    out: dict = {v["vertex_id"]: {} for v in vertices}
    scale, margin = controls["tag_region_scale"], controls["tag_region_margin"]
    for tag, score in tagset.items():
        category = answer["categories"].get(tag, "general")
        kind = scope(tag, category)
        threshold = answer["thresholds"].get(tag)
        found = []
        if kind != "image" and threshold is not None:
            region = {
                v: scores.get(tag, 0.0)
                for v, scores in answer["regions"].items()
                if labels.get(v) in TAGGED_LABELS
            }
            passing = {v: p for v, p in region.items() if p >= threshold * scale}
            if passing:
                floor = _logit(max(passing.values())) - margin
                found = [v for v, p in passing.items() if _logit(p) >= floor]
        if not found:
            out[root][tag] = score
            continue
        found_set = set(found)
        if kind == "identity":
            keep = [v for v in found if not (ancestors[v] - {v}) & found_set]
        else:
            keep = [v for v in found if not any(v in ancestors[o] for o in found_set - {v})]
        if plural(tag) and len(keep) > 1:
            common = {
                v
                for v in set.intersection(*(ancestors[v] for v in keep))
                if labels[v] in TAGGED_LABELS
            } or {root}
            keep = [max(common, key=lambda v: (depth.get(v, 0), labels[v] == "composition"))]
        for v in keep:
            out[v][tag] = answer["regions"].get(v, {}).get(tag, score)
    return out


def place_top(graph: dict, answer: dict, given: list[str], controls: dict) -> dict:
    """The default placement: the image vertex holds the whole set; each vertex whose only
    parent is the image (``targets``) its own tags: a crop's own tag set (``crops``), else the
    image's tags its pooled region scores at or above the threshold (times
    ``tag_region_scale``); image-scope tags left out. Nothing moves: vertices overlap as their
    boxes do."""
    vertices = graph["vertices"]
    root = vertices[0]["vertex_id"]
    tagset = _tagset(answer, given, controls)
    out: dict = {v["vertex_id"]: {} for v in vertices}
    out[root] = dict(tagset)
    scale = controls["tag_region_scale"]
    crops = answer.get("crops", {})
    for v in targets(vertices, "top"):
        vid = v["vertex_id"]
        if vid in crops:
            own = crops[vid]
        else:
            scores = answer["regions"].get(vid, {})
            own = {
                t: scores[t]
                for t in tagset
                if t in answer["thresholds"]
                and scores.get(t, 0.0) >= answer["thresholds"][t] * scale
            }
        for tag, p in own.items():
            if scope(tag, answer["categories"].get(tag, "general")) != "image":
                out[vid][tag] = p
    return out


@node(outputs=["graph", "stats"])
def place_tags(graph: str, stats: dict, scores: dict, tags, controls: dict) -> tuple:
    """The graph with its tags placed (``tagging``: ``image`` keeps them on the image)."""
    g = json.loads(graph)
    if controls["tagging"] == "image":
        source = controls["tag_source"]
        placed = {g["vertices"][0]["vertex_id"]: {}}
        if source in ("tagger", "both"):
            placed[g["vertices"][0]["vertex_id"]].update(scores["image"])
        if source in ("given", "both"):
            for t in given_tags(tags):
                placed[g["vertices"][0]["vertex_id"]][t] = scores["given"].get(t)
    else:
        placed = place(g, scores, given_tags(tags), controls)
    on_image = 0
    for v in g["vertices"]:
        found = placed.get(v["vertex_id"]) or {}
        v["tags"] = found
        if found:
            text = ", ".join(t.replace("_", " ") for t in found)
            v["descs"].append({"text": text, "label": "bagofwords"})
        if v["label"] == "image":
            on_image = len(found)
    g["tags"] = {
        "image": scores["image"],
        "categories": scores["categories"],
        "regions": scores["regions"],
        "crops": scores.get("crops", {}),
        "focus": scores.get("focus", []),
    }
    placed_n = sum(len(t) for t in placed.values())
    stats = {**stats, "tags": placed_n, "tags_on_image": on_image}
    return json.dumps(g, ensure_ascii=False), stats
