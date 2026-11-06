"""GBC-Nest's shaping keeps the graph sound: a vertex that goes takes the relations naming it, a
merge never leaves a relation naming one vertex twice and widens the relations that name the
merged vertex, and the masks and in-edges written are those resolution derives for the final
vertices."""

import copy

from kohakugbc.graph.records import record, ref
from kohakugbc.graph.reference import link_and_mask, resolve
from kohakugbc.nodes import shape

BOXES = {
    "cars": [0.0, 0.5, 0.6, 1.0, 0.9],
    "cars_0": [0.0, 0.5, 0.3, 1.0, 0.9],
    "cars_1": [0.3, 0.5, 0.6, 1.0, 0.9],
    "building": [0.5, 0.0, 1.0, 0.6, 0.9],
    "man": [0.6, 0.6, 0.8, 1.0, 0.9],
    "man_2": [0.62, 0.55, 0.85, 1.0, 0.9],
}


def graph() -> dict:
    def entity(k, label="entity", children=()):
        return record([ref(k, label, k)], BOXES[k], [[f"a {k}", "detail", []]], list(children))

    records = [
        record(
            [ref("", "image", None)],
            None,
            [["a street", "detail", []]],
            [
                ref("cars", "composition", "cars"),
                ref("building", "entity", "building"),
                ref("man", "entity", "man"),
                ref("man_2", "entity", "man"),
            ],
        ),
        entity(
            "cars",
            "composition",
            [ref("cars_0", "entity", "car 1"), ref("cars_1", "entity", "car 2")],
        ),
        entity("cars_0"),
        entity("cars_1"),
        entity("building"),
        entity("man"),
        entity("man_2"),
        record(
            [ref("", "relation", None)],
            None,
            [
                ["car 1 is left of the building", "relation", ["cars_0", "building"]],
                ["car 1 is beside car 2", "relation", ["cars_0", "cars_1"]],
                ["the cars are below the building", "relation", ["building", "cars"]],
                ["the man stands by the other man", "relation", ["man", "man_2"]],
                ["the man is right of the building", "relation", ["building", "man_2"]],
            ],
            [],
        ),
    ]
    return resolve(records, "x.png")


def relations(g: dict) -> set:
    return {v["vertex_id"] for v in g["vertices"] if v["label"] == "relation"}


def assert_sound(g: dict) -> None:
    ids = {v["vertex_id"] for v in g["vertices"]}
    for v in g["vertices"]:
        assert set(v["sub_masks"]) | set(v["super_masks"]) <= ids, v["vertex_id"]
        assert {e["target"] for e in v["out_edges"]} <= ids
        if v["label"] == "relation":
            assert v["out_edges"], v["vertex_id"]
    # what resolution derives for these vertices, derived again from scratch
    fresh = copy.deepcopy(g["vertices"])
    for v in fresh:
        v["in_edges"] = []
    link_and_mask(fresh, g["mask_inside_threshold"])
    assert fresh == g["vertices"]


def test_resolved_graph_has_every_relation():
    g = graph()
    assert relations(g) == {
        "[building|cars_0]",
        "[cars_0|cars_1]",
        "[building|cars]",
        "[man|man_2]",
        "[building|man_2]",
    }


def test_collapsed_members_take_their_relations():
    g = shape.Graph(graph())
    shape.collapse(g, {"cars": False})
    out = g.dump()
    assert relations(out) == {"[building|cars]", "[man|man_2]", "[building|man_2]"}
    assert_sound(out)


def test_a_dissolved_composition_takes_its_relations():
    g = shape.Graph(graph())
    g.drop("cars")
    out = g.dump()
    assert "[building|cars]" not in relations(out)
    assert {"cars_0", "cars_1"} <= {v["vertex_id"] for v in out["vertices"]}
    assert_sound(out)


def test_a_merge_drops_the_relation_between_the_two_and_widens_the_others():
    g = shape.Graph(graph())
    g.merge("man", "man_2")
    out = g.dump()
    by_id = {v["vertex_id"]: v for v in out["vertices"]}
    assert "[man|man_2]" not in by_id
    rel = by_id["[building|man_2]"]
    assert {e["target"] for e in rel["out_edges"]} == {"building", "man"}
    man, b = by_id["man"]["bbox"], rel["bbox"]
    assert b["left"] <= man["left"] and b["top"] <= man["top"]
    assert b["right"] >= man["right"] and b["bottom"] >= man["bottom"]
    assert_sound(out)


def test_an_untouched_graph_comes_out_as_resolved():
    g = graph()
    assert shape.Graph(copy.deepcopy(g)).dump() == g
