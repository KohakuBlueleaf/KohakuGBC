"""The important-vertex rule (``tag_select = "important"``) on hand-made pooled scores: each of
the ways pooled scores mislead it."""

import json

from kohakugbc.nodes import tags
from kohakugbc.program import complete

CONTROLS = complete({"tagging": "regions", "tag_select": "important"})


def vertex(vid: str, label: str, box, caption: str = "", kids=()) -> dict:
    bbox = None if box is None else dict(zip(("left", "top", "right", "bottom"), box))
    return {
        "vertex_id": vid,
        "label": label,
        "bbox": bbox,
        "descs": [{"text": caption, "label": "detail"}] if caption else [],
        "out_edges": [{"source": vid, "text": k, "target": k} for k in kids],
    }


def answer_of(image: dict, regions: dict) -> dict:
    return {
        "image": image,
        "categories": {t: "general" for t in image},
        "thresholds": {t: 0.35 for t in image},
        "regions": regions,
    }


GRAPH = {
    "vertices": [
        vertex("", "image", None),
        vertex(
            "girl",
            "entity",
            [0.1, 0.1, 0.5, 0.95],
            "A girl with pink hair and a pleated skirt, with a smile.",
            ["girl_hair"],
        ),
        vertex("girl_hair", "entity", [0.15, 0.1, 0.35, 0.3], "Long pink hair with a ribbon."),
        vertex(
            "door",
            "entity",
            [0.3, 0.2, 0.7, 0.95],
            "A wooden door; a girl in a skirt stands in front of it.",
        ),
        vertex("lamp", "entity", [0.8, 0.1, 0.95, 0.3], "A brass lamp with a green shade."),
    ]
}
IMAGE = {
    "pink_hair": 0.9,
    "hair_ribbon": 0.6,
    "smile": 0.8,
    "pleated_skirt": 0.7,
    "long_hair": 0.9,
    "highres": 1.0,
    "lamp": 0.5,
}


def test_a_tag_every_region_scores_belongs_to_no_region():
    flat = {v: {"long_hair": 1.0} for v in ("girl", "girl_hair", "door", "lamp")}
    assert tags.held(GRAPH, answer_of({"long_hair": 0.9}, flat), CONTROLS) == {}


def test_a_part_does_not_take_its_subjects_tags():
    regions = {
        "girl": {"pink_hair": 0.97, "hair_ribbon": 0.95, "smile": 0.9},
        "girl_hair": {"pink_hair": 1.0, "hair_ribbon": 0.99, "smile": 0.93},
        "door": {"pink_hair": 0.3, "hair_ribbon": 0.2, "smile": 0.2},
        "lamp": {"pink_hair": 0.1, "hair_ribbon": 0.1, "smile": 0.1},
    }
    owned = tags.held(GRAPH, answer_of(IMAGE, regions), CONTROLS)
    assert owned == {"girl": ["pink_hair", "hair_ribbon", "smile"]}


def test_a_tag_its_owner_never_mentions_is_not_its():
    regions = {
        "girl": {"pleated_skirt": 0.5, "lamp": 0.1},
        "girl_hair": {"pleated_skirt": 0.1, "lamp": 0.1},
        "door": {"pleated_skirt": 0.95, "lamp": 0.1},
        "lamp": {"pleated_skirt": 0.1, "lamp": 0.9},
    }
    owned = tags.held(GRAPH, answer_of(IMAGE, regions), CONTROLS)
    # The door's caption names the skirt of the girl in front of it: it owns the skirt; the lamp
    # owns its own name
    assert owned == {"door": ["pleated_skirt"], "lamp": ["lamp"]}
    regions["door"] = {"pleated_skirt": 0.95, "lamp": 0.1}
    graph = json.loads(json.dumps(GRAPH))
    graph["vertices"][3]["descs"] = [{"text": "A wooden door.", "label": "detail"}]
    assert tags.held(graph, answer_of(IMAGE, regions), CONTROLS) == {"lamp": ["lamp"]}


def test_the_important_vertices_and_their_tags():
    regions = {
        "girl": {"pink_hair": 0.97, "hair_ribbon": 0.95, "smile": 0.9, "lamp": 0.1},
        "girl_hair": {"pink_hair": 1.0, "hair_ribbon": 0.99, "smile": 0.93, "lamp": 0.1},
        "door": {"pink_hair": 0.3, "hair_ribbon": 0.2, "smile": 0.2, "lamp": 0.1},
        "lamp": {"pink_hair": 0.1, "hair_ribbon": 0.1, "smile": 0.1, "lamp": 0.9},
    }
    answer = answer_of(IMAGE, regions)
    assert tags.tag_focus(json.dumps(GRAPH), answer, CONTROLS) == ["girl"]
    placed = tags.place(GRAPH, {**answer, "focus": ["girl"]}, [], CONTROLS)
    assert set(placed["girl"]) == {"pink_hair", "hair_ribbon", "smile"}
    assert placed["lamp"] == {} and "highres" in placed[""]
