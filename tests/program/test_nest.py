"""GBC-Nest's grounding rules on hand-made inputs: what each rule decides where the recorded runs
may not reach it."""

import numpy as np

from kohakugbc.nodes import nest
from kohakugbc.program import complete

CONTROLS = complete({}, "gbc_nest")


def entry(vertex: str, text: str, box: list) -> dict:
    return {"vertex": vertex, "box": box, "texts": [text], "vecs": [None]}


def test_head_noun():
    assert nest.head("silver-haired character in coat") == "character"
    assert nest.head("apple 2") == "apple"
    assert nest.head("red apples") == "apple"


def test_two_names_never_take_one_box():
    # Both names score best on box A; B is the second name's only other box
    a, b = [0, 0, 10, 10], [20, 0, 30, 10]
    raw = {"boxes": [a, a, b], "scores": [0.9, 0.8, 0.5], "labels": [0, 1, 1]}
    out = nest.assign(
        raw, ["blonde girl", "red-haired girl"], ["blonde girl", "red-haired girl"], 0.31
    )
    chosen = {lab: box for box, lab in zip(out["boxes"], out["labels"])}
    assert chosen == {0: a, 1: b}


def test_a_single_name_takes_its_largest_near_tie():
    small, large = [0, 0, 5, 5], [0, 0, 50, 50]
    raw = {"boxes": [small, large], "scores": [0.94, 0.91], "labels": [0, 0]}
    out = nest.prefer_large(raw, ["girl"], {"girl": "single"}, 0.9)
    assert out["boxes"] == [large] and out["scores"][0] > 0.94


def test_a_region_never_joins_its_ancestor():
    # "plants" found inside "plants": same head noun, same box, but its parent
    seen = [entry("plants", "plants", [0.1, 0.1, 0.9, 0.9, 0.9])]
    region = {"id": "plants_plants", "text": "plants", "box": [0.1, 0.1, 0.9, 0.9, 0.8]}
    _, is_new, target = nest.nest_register(seen, region, [], [], CONTROLS)
    assert is_new and target is None


def test_same_head_noun_joins_by_box():
    seen = [entry("blonde character", "blonde character", [0.1, 0.1, 0.5, 0.9, 0.9])]
    region = {"id": "embrace_characters_0", "text": "characters", "box": [0.12, 0.1, 0.5, 0.88]}
    out, is_new, target = nest.nest_register(seen, region, [], [], CONTROLS)
    assert not is_new and target == "blonde character"
    assert out[0]["texts"] == ["blonde character", "characters"]
    # A different head noun never joins by box (its names' embeddings unlike)
    hair = {"id": "x_hair", "text": "hair", "box": [0.12, 0.1, 0.5, 0.88]}
    vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    _, is_new, _ = nest.nest_register(seen, hair, ["hair", "blonde character"], vecs, CONTROLS)
    assert is_new


def test_a_region_never_joins_an_ancestor_reached_through_a_join():
    # The lights wrap the tree: "christmas tree_lights_0" joins "lights_0", so the tree is above
    # lights_0; a tree found inside lights_0 must not join the tree (lights_0 -> tree -> lights_0)
    tree, lights = [0.2, 0.1, 0.8, 0.9, 0.9], [0.2, 0.1, 0.8, 0.9, 0.9]
    seen = [
        entry("christmas tree", "christmas tree", tree),
        entry("lights", "lights", [0.0, 0.0, 1.0, 1.0, 0.9]),
        entry("lights_0", "lights", lights),
    ]
    joining = {"id": "christmas tree_lights_0", "text": "lights", "box": lights}
    seen, is_new, target = nest.nest_register(seen, joining, [], [], CONTROLS)
    assert not is_new and target == "lights_0"
    assert "christmas tree" in next(e for e in seen if e["vertex"] == "lights_0")["above"]
    inner = {"id": "lights_0_christmas tree", "text": "christmas tree", "box": tree}
    _, is_new, target = nest.nest_register(seen, inner, [], [], CONTROLS)
    assert is_new and target is None


def test_a_join_never_closes_a_cycle_the_other_way_round():
    # The tree inside lights_0 joined the tree first; lights found in the tree must not join
    # lights_0, which is now above the tree
    tree, lights = [0.2, 0.1, 0.8, 0.9, 0.9], [0.2, 0.1, 0.8, 0.9, 0.9]
    seen = [
        entry("christmas tree", "christmas tree", tree),
        entry("lights", "lights", [0.0, 0.0, 1.0, 1.0, 0.9]),
        entry("lights_0", "lights", lights),
    ]
    inner = {"id": "lights_0_christmas tree", "text": "christmas tree", "box": tree}
    seen, is_new, target = nest.nest_register(seen, inner, [], [], CONTROLS)
    assert not is_new and target == "christmas tree"
    joining = {"id": "christmas tree_lights_0", "text": "lights", "box": lights}
    _, is_new, target = nest.nest_register(seen, joining, [], [], CONTROLS)
    assert is_new and target is None
    assert nest.ancestors(seen, "christmas tree_lights_0") >= {"christmas tree", "lights_0"}
