"""VLM requests: what each model node asks, as a protocol-neutral request.

A request is ``{"key", "system", "query", "images", "schema", "max_tokens", "temperature"}``:
``images`` is a list of ``[label, picture]`` (a label text before the image, or None; the
picture ``{"jpeg", "size"}`` as ``kohakugbc.nodes.images.encode`` makes it, None for an empty
crop), ``schema`` the name of a JSON schema in ``specs/schemas``. Protocols turn it into a
server's request (``kohakugbc.engines.vlm``). The prompts are the PoC's JSON prompt sets
(``specs/prompts``): ``json``, ``json_leafcap`` (entity replies that caption their features),
``json_batch``, ``coverage``, ``seeded``.
"""

import json
from functools import cache
from pathlib import Path

from kohakugbc.nodes.specs import trace
from kohakugbc.nodes.text import singular

HERE = Path(__file__).parent
TEMPERATURE = 0.1
MAX_TOKENS = {"image_query": 1024, "entity": 512, "relation": 1024, "composition": 1024}


@cache
def prompt(spec: str, name: str) -> str:
    return (HERE / "prompts" / spec / f"{name}.txt").read_text(encoding="utf-8")


@cache
def schema(name: str) -> dict:
    return json.loads((HERE / "schemas" / name).read_text(encoding="utf-8"))


def request(key: str, system, query: str, images: list, schema_name, max_tokens: int) -> dict:
    return {
        "key": key,
        "system": system,
        "query": query,
        "images": images,
        "schema": schema_name,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }


def prompt_set(controls: dict) -> str:
    """The prompt set of the four GBC queries: leaf captions from the parent need entity replies
    that caption their features."""
    return "json_leafcap" if controls["leaf_caption"] == "parent" else "json"


# ---- root --------------------------------------------------------------------------------------


def image_query(image_key: str, patch, controls: dict) -> dict:
    spec = prompt_set(controls)
    return request(
        trace.vlm_key("image_query", spec, image_key),
        prompt(spec, "system_image"),
        prompt(spec, "query_image"),
        [[None, patch]],
        "image.json",
        MAX_TOKENS["image_query"],
    )


def seed_elements(image_key: str, captions: dict, tags: list[str], patch, controls: dict) -> dict:
    """The objects named by existing captions; with ``patch``, only those visible in it."""
    source = "text" if patch is None else "image"
    tag_line = f"\nTags: {', '.join(t.replace('_', ' ') for t in tags)}\n" if tags else ""
    image_line = (
        "\nOnly list objects that are visible in the attached image.\n" if patch is not None else ""
    )
    query = (
        prompt("seeded", "query_elements")
        .replace("{description}", (captions.get("detail") or "").strip())
        .replace("{brief}", (captions.get("short") or "").strip())
        .replace("{tag_line}", tag_line)
        .replace("{image_line}", image_line)
    )
    system = prompt("seeded", "system_elements").replace(
        "{max_elements}", str(controls["max_top_level"])
    )
    key = trace.vlm_key(
        "seed_elements", "seeded", image_key, source=source, tag_hint=controls["tag_hint"]
    )
    images = [] if patch is None else [[None, patch]]
    return request(key, system, query, images, "seed_elements.json", 400)


def coverage(image_key: str, labels: list[str], patch, controls: dict) -> dict:
    query = prompt("coverage", "query_coverage").format(
        ", ".join(f"[{t}]" for t in labels), controls["coverage_max_new"]
    )
    return request(
        trace.vlm_key("coverage", "coverage", image_key, labels=list(labels)),
        prompt("coverage", "system_coverage"),
        query,
        [[None, patch]],
        None,
        400,
    )


# ---- regions -----------------------------------------------------------------------------------


def entity(image_key: str, region: dict, patch, controls: dict) -> dict:
    spec = prompt_set(controls)
    text = singular(region["text"])
    return request(
        trace.vlm_key(
            "entity", spec, image_key, entity=region["id"], box=trace.box_key(region["box"])
        ),
        prompt(spec, "system_entity"),
        prompt(spec, "query_entity").format(text, text),
        [[None, patch]],
        "entity_leafcap.json" if spec == "json_leafcap" else "entity.json",
        MAX_TOKENS["entity"],
    )


def entity_batch(image_key: str, regions: list[dict], patches: list) -> dict:
    """Several regions in one request, each image after its label."""
    images = [
        [f"Region {k}: '{singular(r['text'])}'", patch]
        for k, (r, patch) in enumerate(zip(regions, patches), start=1)
    ]
    key = trace.vlm_key(
        "entity_batch",
        "json_batch",
        image_key,
        regions=[[r["id"], trace.box_key(r["box"])] for r in regions],
    )
    return request(
        key,
        prompt("json_batch", "system_entity_batch"),
        f"Analyze each of the {len(regions)} regions.",
        images,
        "entity_batch.json",
        400 * len(regions),
    )


# ---- relations ---------------------------------------------------------------------------------


def relation(
    image_key: str, vertex_id: str, box, entities: list, count: int, patch, controls: dict
) -> dict:
    """Relations among ``[[id, text], ...]`` asked on the crop of the vertex's box (boxes widen
    as regions join, so the box is part of what is asked)."""
    spec = prompt_set(controls)
    to_relate = ", ".join(f"[{text}]" for _, text in entities)
    key = trace.vlm_key(
        "relation", spec, image_key, vertex=vertex_id, box=trace.box_key(box), entities=entities
    )
    return request(
        key,
        prompt(spec, "system_relation"),
        prompt(spec, "query_relation").format(count, to_relate),
        [[None, patch]],
        "relation.json",
        MAX_TOKENS["relation"],
    )


def composition(
    image_key: str,
    names: list,
    text: str,
    parts: list,
    boxes: list,
    node_texts: list,
    hints: list,
    patch,
    controls: dict,
) -> dict:
    """The composition of ``parts`` ``[[id, text], ...]`` with their ``boxes`` on the annotated
    crop of their union; ``names`` are the composition's ids, the asked one first."""
    spec = prompt_set(controls)
    query = prompt(spec, "query_composition").format(
        f"{len(parts)} {text}",
        text,
        ", ".join(node_texts),
        "\n".join("- " + h for h in hints),
    )
    return request(
        trace.vlm_key(
            "composition",
            spec,
            image_key,
            entity=names,
            parts=parts,
            boxes=[trace.box_key(b) for b in boxes],
        ),
        prompt(spec, "system_composition"),
        query,
        [[None, patch]],
        "composition.json",
        MAX_TOKENS["composition"],
    )
