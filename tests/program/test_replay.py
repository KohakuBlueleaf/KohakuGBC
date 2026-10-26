"""The GBC program reproduces the PoC: recorded PoC jobs (``tests/fixtures/traces``, cut from
``scripts/trace/poc_arms.py`` runs) replayed through KIR's reference interpreter give the PoC's
graph, by digest; every model request the program makes is one the PoC made, in the same
words."""

import json
from pathlib import Path

import pytest

from kohakugbc.graph import digest
from kohakugbc.io.sources import TraceJobs
from kohakugbc.nodes import regions, rounds
from kohakugbc.nodes.specs import detect as det
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace
from kohakugbc.nodes.text import same_object
from kohakugbc.program.reference import run

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "traces"
ARMS = sorted(p.name for p in FIXTURES.iterdir() if (p / "meta.json").exists())


def inputs_of(arm: str) -> dict:
    return {item.key: item.inputs for item in TraceJobs(FIXTURES / arm)}


def normalized(graph) -> dict | None:
    if graph is None:
        return None
    g = json.loads(graph) if isinstance(graph, str) else json.loads(json.dumps(graph))
    g["img_path"] = Path(g["img_path"]).name
    return g


def replay(arm: str) -> list[tuple[str, bool, ReplayModels]]:
    trace = Trace(FIXTURES / arm)
    inputs = inputs_of(arm)
    out = []
    for key, job in sorted(trace.jobs.items()):
        models = ReplayModels(trace)
        graph = run(inputs[key], models)["graph"]
        out.append((key, digest(normalized(graph)) == digest(normalized(job["graph"])), models))
    return out


def test_fixtures_cover_the_arms():
    assert {"text", "query", "tags", "batch4", "leafcap"} <= set(ARMS)
    # Every key names its whole input: no key was answered twice in different words
    assert all(not Trace(FIXTURES / arm).conflicts for arm in ARMS)


@pytest.mark.parametrize("arm", ARMS)
def test_program_reproduces_the_poc(arm):
    results = replay(arm)
    assert results
    for key, equal, models in results:
        assert not models.misses, (key, models.misses[:3])
        # The PoC read prompt files in the Windows locale (cp950): a prompt it sent may be the
        # UTF-8 file decoded as cp950, and differ in nothing else
        for m in models.mismatches:
            assert m["asked"].encode("utf-8").decode("cp950", "replace") == m["recorded"], (
                key,
                m["key"],
                m["field"],
            )
        assert equal, key


def test_replay_catches_a_broken_node(monkeypatch):
    # Frontier order by insertion only (no priority): some graph must change
    monkeypatch.setattr(rounds, "_order", lambda region, controls: (region["order"],))
    changed = sum(not equal for arm in ("text", "query") for _, equal, _ in replay(arm))
    assert changed > 0


def test_detection_selection_equals_poc():
    """Every recorded detector call: the boxes the program keeps from the raw output are the
    ones ml-gbc kept (relative to the crop, with confidences)."""
    checked = 0
    for arm in ARMS:
        trace = Trace(FIXTURES / arm)
        for rec in trace.detect.values():
            key = json.loads(rec["key"])
            raw = rec["raw"]
            for label, (text, params) in enumerate(zip(key["texts"], rec["params"])):
                mine = [k for k, lab in enumerate(raw["labels"]) if lab == label]
                if not mine:
                    continue
                single = params["nms"] == det.NMS_SINGLE and params["high_conf"] is not None
                kept = det.select(
                    [raw["boxes"][k] for k in mine],
                    [raw["scores"][k] for k in mine],
                    rec["size"][0],
                    rec["size"][1],
                    single,
                    params["max_rel_area"],
                    params["topk"],
                )
                assert kept == rec["selected"].get(text, []), (rec["key"], text)
                checked += 1
    assert checked > 20


def test_same_object_decisions_equal_poc():
    checked = 0
    for arm in ARMS:
        trace = Trace(FIXTURES / arm)
        for d in trace.same:
            va, vb = trace.embed.get(d["a"]), trace.embed.get(d["b"])
            if va is None or vb is None:
                continue
            assert same_object(d["a"], d["b"], va, vb, 0.725) == d["same"], d
            checked += 1
    assert checked > 0


def test_register_joins_in_program_order():
    # Two overlapping regions with the same name: the second joins the first, whose box widens
    controls = {"overlap_threshold": 0.85, "same_threshold": 0.725}
    a = {"id": "cup", "text": "cup", "box": [0.1, 0.1, 0.5, 0.5, 0.9]}
    b = {"id": "x_cup", "text": "cups", "box": [0.1, 0.1, 0.52, 0.5, 0.8]}
    assert regions.embed_needs([], a, controls) == []
    seen, new_a, _ = regions.register([], a, [], [], controls)
    assert regions.embed_needs(seen, b, controls) == []
    seen, new_b, target = regions.register(seen, b, [], [], controls)
    assert (new_a, new_b, target) == (True, False, "cup")
    assert seen[0]["box"][:4] == [0.1, 0.1, 0.52, 0.5] and seen[0]["texts"] == ["cup", "cups"]
