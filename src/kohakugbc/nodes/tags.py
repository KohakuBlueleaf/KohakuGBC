"""Tag nodes: the tagger's request for a finished graph, and the tags placed on its vertices.

``tagging`` (controls): ``none``; ``image`` (the image's tags on the image vertex); ``regions``
(each tag placed on the vertices it describes). The image's tag set is the tagger's own
(``tag_source = "tagger"``), the job's ``tags`` (``"given"``, e.g. danbooru's) or both.

``tag_placement``: ``top`` (``place_top``: the image vertex holds the whole set, each vertex whose
only parent is the image its own tags, nothing moved) or ``place`` (below). ``tag_regions``:
``crop`` (each target vertex's box tagged as its own picture) or ``pool`` (the box's tokens
pooled from the whole picture's pass, which every token sees through global attention).

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


@node(outputs=["request"])
def tag_request(image: dict, graph: str, tags, controls: dict) -> dict:
    """The tagger's request: the whole image fitted to 1008 px; ``regions`` tagging, the target
    vertices (``targets``) as ``crops`` (``tag_regions = "crop"``, each its own picture) or as
    boxes pooled from the whole image's tokens (``"pool"``); the job's tags to score."""
    regions, crops = {}, {}
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
    candidates = given_tags(tags) if controls["tag_source"] != "tagger" else []
    picture = _capped(images.cut(image, None))
    return {"picture": picture, "regions": regions, "crops": crops, "candidates": candidates}


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
    }
    placed_n = sum(len(t) for t in placed.values())
    stats = {**stats, "tags": placed_n, "tags_on_image": on_image}
    return json.dumps(g, ensure_ascii=False), stats
