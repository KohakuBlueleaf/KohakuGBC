"""Root nodes: the image vertex, from GBC's image query or from existing captions and tags.

Ports of ml-gbc's image query (``ImageQuery`` with the PoC's JSON prompts) and the PoC's seeded
root (``gbcpoc.seeded``).
"""

import re

from kohakugbc.graph.records import record
from kohakugbc.nodes.registry import node
from kohakugbc.nodes.specs import chat as specs
from kohakugbc.nodes.specs import parse
from kohakugbc.nodes.text import detectable, ref_count, singular
from kohakugbc.program.controls import complete

COUNT_TAG = re.compile(r"^(\d+)\+?(girl|boy|other)s?$")
# Tags that name no object (scene, framing, meta), and head nouns of part tags
NON_OBJECT_TAGS = {
    "solo", "night", "day", "outdoors", "indoors", "scenery", "realistic", "comic", "monochrome",
    "greyscale", "border", "cover", "text", "english text", "watermark", "signature",
    "artist name", "character name", "multiple views", "close-up", "full body", "upper body",
    "cowboy shot", "portrait", "no humans", "animal focus", "male focus", "female focus", "sky",
    "cloud", "blurry", "depth of field", "pun",
}  # fmt: skip
PART_HEADS = {
    "hair", "eyes", "ears", "bangs", "mouth", "skin", "ponytail", "twintails", "braid",
    "sidelocks", "ahoge", "background", "lips", "teeth", "fangs", "eyebrows", "eyelashes",
    "pupils", "nails", "hairband", "shadows", "shadow",
}  # fmt: skip


@node(outputs=["controls"])
def with_defaults(controls) -> dict:
    return complete(controls)


@node(outputs=["mode"])
def root_mode(controls: dict, captions, tags) -> str:
    """The root mode; a seeded mode without its inputs fails the job with the reason."""
    mode = controls["root"]
    if mode != "query" and not (captions and captions.get("detail")):
        raise ValueError(f"root={mode!r} needs captions with a 'detail' caption")
    if mode == "tags" and not tags:
        raise ValueError("root='tags' needs tags")
    return mode


# ---- image query --------------------------------------------------------------------------------


@node(outputs=["request"])
def image_query_request(image: dict, patch, controls: dict) -> dict:
    return specs.image_query(image["key"], patch, controls)


@node(outputs=["root"])
def image_root(image: dict, reply: str) -> dict:
    """The image vertex from GBC's image query: detailed and concise captions, their elements."""
    out = parse.image(reply.replace("**", ""))
    return {**record([["", "image", None]], None, out["descs"], out["entities"]), "raw": reply}


# ---- seeded root -------------------------------------------------------------------------------


def general_tags(tags, threshold: float) -> list[str]:
    general = (tags or {}).get("general") or {}
    return [t for t, conf in sorted(general.items(), key=lambda kv: -kv[1]) if conf >= threshold]


def tag_element(tag: str) -> tuple[str, str] | None:
    """A whole-object tag as an element name and multiplicity; None for other tags."""
    match = COUNT_TAG.match(tag)
    if match:
        count, noun = int(match.group(1)), match.group(2)
        noun = "person" if noun == "other" else noun
        return (noun + "s", "multiple") if count > 1 else (noun, "single")
    text = re.sub(r"\s*\(.*?\)", "", tag.replace("_", " ")).strip()
    head = text.split()[-1] if text else ""
    if not text or text in NON_OBJECT_TAGS or head.endswith("ing") or head in PART_HEADS:
        return None
    return text, "multiple" if singular(text) != text else "single"


def seed_descs(captions: dict) -> list:
    descs = [[captions["detail"].strip(), "detail", [""]]]
    if captions.get("short"):
        descs.append([captions["short"].strip(), "short", [""]])
    return descs


def grounded(name: str, texts: list[str]) -> str | None:
    """The name as the captions spell it: exact, else its singular or plural."""
    name = name.strip().lower()
    for candidate in (name, singular(name), name + "s"):
        if candidate and ref_count(texts, candidate) > 0:
            return candidate
    return None


@node(outputs=["request"])
def elements_request(image: dict, captions: dict, tags, patch, controls: dict) -> dict:
    """The element query over existing captions; with a patch, only objects visible in it."""
    hint = general_tags(tags, controls["tag_threshold"]) if controls["tag_hint"] else []
    return specs.seed_elements(image["key"], captions, hint, patch, controls)


def _proposed(reply: str) -> list[tuple[str, str]]:
    items = []
    for item in (parse.load_json(reply) or {}).get("elements") or []:
        if isinstance(item, dict) and str(item.get("name", "")).strip():
            mult = str(item.get("multiplicity", "single")).lower()
            items.append((str(item["name"]), mult if mult in parse.MULTIPLICITIES else "single"))
    return items


@node(outputs=["root"])
def seeded_root(image: dict, captions: dict, tags, reply: str, controls: dict) -> dict:
    """The image vertex from existing captions; its elements from the element reply, or from the
    tags (``root = "tags"``), each grounded in the captions."""
    descs = seed_descs(captions)
    texts = [d[0] for d in descs]
    if controls["root"] == "tags":
        proposed = [e for e in map(tag_element, general_tags(tags, controls["tag_threshold"])) if e]
    else:
        proposed = _proposed(reply)

    children, seen = [], set()
    for name, mult in proposed:
        text = grounded(name, texts)
        if text is None or not detectable(text):
            continue
        if singular(text) in seen:
            continue
        seen.add(singular(text))
        children.append([text, mult, text])
        if len(children) >= controls["max_top_level"]:
            break
    return {**record([["", "image", None]], None, descs, children), "raw": reply}


@node(outputs=["records"])
def first_record(root: dict) -> list:
    return [root]
