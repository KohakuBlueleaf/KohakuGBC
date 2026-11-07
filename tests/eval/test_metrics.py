"""The two tracks against graphs broken on purpose, and the correspondence against brute force:
each break moves the part it should, ids, names and order never move anything, and the
program finds the best correspondence there is."""

import copy
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pytest

from kohakugbc.eval import Options, correspond, empty, evaluate, iou, load, measure
from kohakugbc.eval.score import full_score, structure_score
from kohakugbc.eval.text import Embedder, Sims

JOBS = Path(__file__).parents[1] / "fixtures" / "traces" / "nest" / "jobs.jsonl"
GRAPHS = [json.loads(json.loads(line)["graph"]) for line in JOBS.read_text().splitlines()]
EXACT = Options(taus=(0.5,))
FULL = Options()


def report(pairs, opt=FULL, sims=None):
    out, _ = evaluate([(k, load(p), load(g)) for k, (p, g) in enumerate(pairs)], opt, sims)
    return out


def at(rep, part, q="f1", tau=0.5, track="structure"):
    return rep[track]["taus"][str(tau)][part][q]


def box(x0, t, r, b, c=None):
    return {"left": x0, "top": t, "right": r, "bottom": b, "confidence": c}


def vertex(vid, label, bbox, targets=(), descs=()):
    edges = [{"source": vid, "text": t, "target": t} for t in targets]
    descs = [{"text": d} for d in descs]
    return {"vertex_id": vid, "label": label, "bbox": bbox, "out_edges": edges, "descs": descs}


def graph(*vertices, top=()):
    return {"vertices": [vertex("", "image", box(0, 0, 1, 1), top), *vertices]}


def non_image(g):
    return [v for v in g["vertices"] if v["label"] not in ("image", "relation")]


def anonymize(g, seed=1, strip_text=True):
    """The same graph with every id replaced, the vertices shuffled and (``strip_text``) no text."""
    rng = random.Random(seed)
    ids = {v["vertex_id"]: f"v{rng.random():.12f}" for v in g["vertices"]}
    out = []
    for v in g["vertices"]:
        w = dict(v, vertex_id=ids[v["vertex_id"]])
        if strip_text:
            w["descs"] = []
        w["out_edges"] = [
            {
                "source": ids[v["vertex_id"]],
                "text": "" if strip_text else e["text"],
                "target": ids.get(e["target"], "?"),
            }
            for e in v["out_edges"]
        ]
        out.append(w)
    rng.shuffle(out)
    return {"vertices": out}


# ---- the structure track -------------------------------------------------------------------


def test_fixtures_have_every_part():
    specs = [load(g) for g in GRAPHS]
    assert all(len(s) and s.edges for s in specs)
    assert any(s.relations for s in specs)


def test_a_graph_against_itself_scores_one():
    rep = report([(g, g) for g in GRAPHS])
    assert rep["structure"]["score"] == 1.0
    assert rep["structure"]["taus"]["0.5"]["vertex_ap"] == 1.0


def test_ids_order_and_text_never_matter():
    rep = report([(anonymize(g, seed), g) for seed, g in enumerate(GRAPHS)])
    assert rep["structure"]["score"] == 1.0


def test_twins_are_told_apart_by_structure_whatever_the_order():
    # Two people with one box, each with a child of its own; the prediction lists them in
    # shuffled order under other ids
    gt = graph(
        vertex("a", "entity", box(0, 0, 0.5, 1), ["a_hat"]),
        vertex("b", "entity", box(0, 0, 0.5, 1), ["b_bag"]),
        vertex("a_hat", "entity", box(0, 0, 0.2, 0.2)),
        vertex("b_bag", "entity", box(0.3, 0.7, 0.5, 1)),
        top=["a", "b"],
    )
    for seed in range(8):
        assert report([(anonymize(gt, seed), gt)])["structure"]["score"] == 1.0


def test_an_empty_prediction_scores_zero():
    rep, rows = evaluate([(k, empty(), load(g)) for k, g in enumerate(GRAPHS)])
    assert rep["structure"]["score"] == 0.0
    assert all(r["structure"]["score"] == 0.0 for r in rows)


def test_a_part_empty_on_both_sides_is_left_out():
    gt = graph(vertex("a", "entity", box(0, 0, 0.5, 0.5)), top=["a"])
    assert report([(gt, gt)], EXACT)["structure"]["score"] == 1.0
    flat = copy.deepcopy(gt)
    flat["vertices"][0]["out_edges"] = []
    # vertex 1, edge 0, relation inactive
    assert report([(flat, gt)], EXACT)["structure"]["score"] == 0.5


def jitter(g, scale, rng):
    g = copy.deepcopy(g)
    for v in non_image(g):
        b = v["bbox"]
        w, h = b["right"] - b["left"], b["bottom"] - b["top"]
        dx, dy = rng.normal(0, scale, 2)
        b["left"], b["right"] = b["left"] + dx * w, b["right"] + dx * w
        b["top"], b["bottom"] = b["top"] + dy * h, b["bottom"] + dy * h
    return g


def test_box_noise_lowers_the_score_monotonically():
    scores = []
    for scale in (0.0, 0.03, 0.1, 0.3):
        rng = np.random.default_rng(0)
        scores.append(report([(jitter(g, scale, rng), g) for g in GRAPHS])["structure"]["score"])
    assert scores == sorted(scores, reverse=True) and scores[0] == 1.0 and scores[-1] < 0.6


def test_dropped_vertices_cost_recall_exactly():
    pairs, kept, total = [], 0, 0
    for g in GRAPHS:
        p = copy.deepcopy(g)
        gone = {v["vertex_id"] for v in non_image(p)[::3]}
        p["vertices"] = [v for v in p["vertices"] if v["vertex_id"] not in gone]
        pairs.append((p, g))
        kept += len(load(p))
        total += len(load(g))
    rep = report(pairs, EXACT)
    assert at(rep, "vertex", "precision") == 1.0
    assert at(rep, "vertex", "recall") == kept / total
    assert at(rep, "edge", "recall") < 1.0


def test_rewired_edges_cost_edges_not_vertices():
    pairs = []
    for g in GRAPHS:
        p = copy.deepcopy(g)
        ids = [v["vertex_id"] for v in non_image(p)]
        for v in non_image(p):
            for e in v["out_edges"]:
                if e["target"] in ids:
                    e["target"] = ids[(ids.index(e["target"]) + 1) % len(ids)]
        pairs.append((p, g))
    rep = report(pairs, EXACT)
    assert at(rep, "vertex") == 1.0
    assert at(rep, "edge") < 1.0


def test_a_lost_relation_costs_relation_recall_exactly():
    pairs, total, lost = [], 0, 0
    for g in GRAPHS:
        p = copy.deepcopy(g)
        rels = [v for v in p["vertices"] if v["label"] == "relation"]
        total += len(load(g).relations)
        if rels:
            p["vertices"].remove(rels[0])
            lost += 1
        pairs.append((p, g))
    rep = report(pairs, EXACT)
    assert at(rep, "relation", "recall") == (total - lost) / total
    assert at(rep, "relation", "precision") == 1.0
    assert at(rep, "vertex") == 1.0


def test_structure_tells_a_parent_from_its_child():
    # Each predicted box is closer to the other vertex's box; only the edges say which is which
    gt = graph(
        vertex("a", "entity", box(0, 0, 1, 1), ["b"]),
        vertex("b", "entity", box(0, 0, 0.9, 1)),
        top=["a"],
    )
    pred = graph(
        vertex("p", "entity", box(0, 0, 0.92, 1), ["q"]),
        vertex("q", "entity", box(0, 0, 0.98, 1)),
        top=["p"],
    )
    rep = report([(pred, gt)], EXACT)
    assert at(rep, "vertex") == at(rep, "edge") == 1.0


def test_types_must_agree():
    gt = graph(vertex("a", "composition", box(0, 0, 0.5, 0.5)))
    pred = graph(vertex("a", "entity", box(0, 0, 0.5, 0.5)))
    assert at(report([(pred, gt)], EXACT), "vertex") == 0.0


def test_confidence_orders_vertex_ap():
    gt = graph(vertex("a", "entity", box(0, 0, 0.5, 0.5)))
    good = graph(
        vertex("a", "entity", box(0, 0, 0.5, 0.5, 0.9)),
        vertex("x", "entity", box(0.6, 0.6, 1, 1, 0.1)),
    )
    bad = graph(
        vertex("a", "entity", box(0, 0, 0.5, 0.5, 0.1)),
        vertex("x", "entity", box(0.6, 0.6, 1, 1, 0.9)),
    )
    assert report([(good, gt)], EXACT)["structure"]["taus"]["0.5"]["vertex_ap"] == 1.0
    ap_bad = report([(bad, gt)], EXACT)["structure"]["taus"]["0.5"]["vertex_ap"]
    assert ap_bad == pytest.approx(0.5, abs=0.01)


def test_vertex_ap_interpolates_precision():
    gt = graph(
        vertex("a", "entity", box(0, 0, 0.4, 0.4)),
        vertex("b", "entity", box(0.5, 0.5, 0.9, 0.9)),
    )
    pred = graph(
        vertex("x", "entity", box(0.0, 0.6, 0.3, 1.0, 0.9)),
        vertex("a", "entity", box(0, 0, 0.4, 0.4, 0.8)),
        vertex("b", "entity", box(0.5, 0.5, 0.9, 0.9, 0.7)),
    )
    ap = report([(pred, gt)], EXACT)["structure"]["taus"]["0.5"]["vertex_ap"]
    assert ap == pytest.approx(2 / 3)


def test_list_boxes_load_as_dict_boxes():
    g = GRAPHS[0]
    p = copy.deepcopy(g)
    for v in p["vertices"]:
        b = v["bbox"]
        v["bbox"] = [b["left"], b["top"], b["right"], b["bottom"]]
    a, b = load(p), load(g)
    assert np.array_equal(a.boxes, b.boxes) and a.edges == b.edges


# ---- the correspondence is the best there is -----------------------------------------------


def random_graph(rng, n, types=("entity", "composition")):
    """n vertices on a few shared boxes (so many pairs tie on IoU), random edges and relations."""
    cells = [box(0, 0, 0.5, 0.5), box(0, 0, 0.55, 0.5), box(0.4, 0.4, 1, 1), box(0, 0, 1, 1)]
    ids = [f"n{k}" for k in range(n)]
    vs = [vertex(i, rng.choice(types), copy.deepcopy(rng.choice(cells))) for i in ids]
    for k, v in enumerate(vs):
        v["out_edges"] = [
            {"source": v["vertex_id"], "text": "", "target": ids[j]}
            for j in range(k + 1, n)
            if rng.random() < 0.4
        ]
    top = [i for i in ids if rng.random() < 0.5]
    rels = []
    for _ in range(rng.randint(0, 2)):
        members = rng.sample(ids, rng.randint(1, min(3, n)))
        rels.append(vertex("[" + "|".join(members) + "]", "relation", box(0, 0, 1, 1), members))
    return graph(*vs, *rels, top=top)


def brute(pred, gt, ious, tau, score, sims=None):
    """The best score over every allowed one-to-one correspondence."""
    best = 0.0
    for chosen in itertools.product(range(-1, len(gt)), repeat=len(pred)):
        used = [v for v in chosen if v >= 0]
        if len(used) != len(set(used)):
            continue
        if any(
            v >= 0 and (ious[u, v] < tau or pred.types[u] != gt.types[v])
            for u, v in enumerate(chosen)
        ):
            continue
        best = max(best, score(measure(pred, gt, np.array(chosen), sims)))
    return best


def random_pair(seed):
    rng = random.Random(seed)
    pred = load(random_graph(rng, rng.randint(1, 5)))
    gt = load(random_graph(rng, rng.randint(1, 5)))
    return pred, gt, iou(pred.boxes, gt.boxes)


@pytest.mark.parametrize("seed", range(40))
def test_the_program_finds_the_best_structure_correspondence(seed):
    pred, gt, ious = random_pair(seed)
    for tau in (0.5, 0.85):
        got = structure_score(measure(pred, gt, correspond(pred, gt, ious, tau), None))
        assert got == pytest.approx(brute(pred, gt, ious, tau, structure_score), abs=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_the_program_finds_the_best_full_correspondence(seed):
    pred, gt, ious = random_pair(seed)
    r = np.random.default_rng(seed)
    sims = {
        "names": r.random((len(pred), len(gt))),
        "descs": r.random((len(pred), len(gt))),
        "relation_descs": r.random((len(pred.relations), len(gt.relations))),
        "image_descs": 0.5,
        "image_sides": [1, 1],
    }
    got = full_score(measure(pred, gt, correspond(pred, gt, ious, 0.5, sims), sims))
    assert got == pytest.approx(brute(pred, gt, ious, 0.5, full_score, sims), abs=1e-6)


# ---- the full track ------------------------------------------------------------------------

# Words a vision-language model uses interchangeably share a meaning; others have their own
MEANINGS = {"girl": "girl", "young woman": "girl", "hat": "hat", "cap": "hat"}
NO_BASELINE = dict.fromkeys(("names", "descs", "relation_descs", "image_descs"), 0.0)


class Model:
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.random.default_rng(sum(map(ord, MEANINGS.get(t, t)))).normal(size=64)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


SIMS = Sims(Embedder(Model()), NO_BASELINE)


def named(person, hat):
    g = graph(
        vertex("p", "entity", box(0, 0, 0.5, 1), ["h"], ["a person"]),
        vertex("h", "entity", box(0, 0, 0.3, 0.2), [], ["a hat"]),
        top=["p"],
    )
    g["vertices"][0]["out_edges"][0]["text"] = person
    g["vertices"][1]["out_edges"][0]["text"] = hat
    return g


def test_the_full_track_reads_names_by_meaning_not_by_equality():
    gt = named("girl", "hat")
    same = report([(gt, gt)], EXACT, SIMS)
    synonyms = report([(named("young woman", "cap"), gt)], EXACT, SIMS)
    unrelated = report([(named("dog", "car"), gt)], EXACT, SIMS)
    assert same["full"]["score"] == pytest.approx(1.0)
    assert synonyms["full"]["score"] == pytest.approx(1.0)
    assert unrelated["full"]["score"] < synonyms["full"]["score"]
    # the structure track never reads text
    for rep in (same, synonyms, unrelated):
        assert rep["structure"]["score"] == 1.0


def test_the_full_track_is_order_and_id_free_too():
    pairs = [(anonymize(g, seed, strip_text=False), g) for seed, g in enumerate(GRAPHS)]
    assert report(pairs, EXACT, SIMS)["full"]["score"] == pytest.approx(1.0)
