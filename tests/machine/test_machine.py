"""The GBC machine on pseudo engines: the PoC's graphs, conformance to KIR's reference
interpreter, configuration errors, regions of a wave side by side, a full VLM lane."""

import json
import time
from pathlib import Path

import pytest
from togeari.conformance import conform

from kohakugbc.graph import digest
from kohakugbc.io.sources import TraceJobs
from kohakugbc.machine import build
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace
from kohakugbc.program.reference import run as reference

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "traces"
ARMS = sorted(p.name for p in FIXTURES.iterdir() if (p / "meta.json").exists())


def engines(trace: Path, **vlm) -> dict:
    out = {
        s: {"entry": "replay", "trace": str(trace), "latency_ms": 1.0}
        for s in ("vlm", "detect", "embed")
    }
    out["vlm"].update(vlm)
    return out


def run(layers: dict):
    machine, source, sink = build(layers)
    with machine:
        machine.run(source, sink)
        job_program = machine.code.entries["gbc"].region.job
    return sink.records, {i.key: i.inputs for i in source}, job_program


def named(graph) -> str:
    g = json.loads(graph) if isinstance(graph, str) else json.loads(json.dumps(graph))
    g["img_path"] = Path(g["img_path"]).name
    return digest(g)


@pytest.mark.parametrize("arm", ARMS)
def test_machine_gives_the_poc_graphs_and_conforms(arm):
    trace = FIXTURES / arm
    records, inputs, job_program = run(
        {
            "engines": engines(trace),
            "source": {"kind": "trace", "path": str(trace)},
            "machine": {"record": 1.0},
        }
    )
    expected = Trace(trace).jobs
    assert len(records) == len(expected)
    for rec in records:
        assert rec.ok, (rec.key, rec.failure)
        assert named(rec.outputs["graph"]) == named(expected[rec.key]["graph"]), rec.key
        verdict = conform(job_program, rec, inputs[rec.key])
        assert verdict.ok, (rec.key, verdict.reason)


def test_configuration_errors_are_listed_together():
    with pytest.raises(ValueError) as err:
        build({"engines": {"vlm": {"entry": "nope"}, "detect": {}}})
    text = str(err.value)
    assert "engines.vlm: unknown entry 'nope'" in text
    assert "engines.detect: no entry" in text and "engines.embed: no entry" in text


@pytest.mark.serial
def test_regions_of_a_wave_run_side_by_side():
    # One job, every VLM request 50 ms, room for all of them: the job takes far less than its
    # requests one after another
    trace = FIXTURES / "text"
    asked, job = 0, None
    for item in TraceJobs(trace):
        models = ReplayModels(Trace(trace))
        reference(item.inputs, models)
        if len(models.asked) > asked:
            asked, job = len(models.asked), item
    layers = {
        "engines": engines(trace, latency_ms=50.0, capacity=64),
        "source": {"kind": "trace", "path": str(trace)},
    }
    machine, _, sink = build(layers)
    with machine:
        start = time.time()
        handle = machine.submit(job.inputs, key=job.key)
        assert handle.result(timeout=60).ok
        wall = time.time() - start
    assert asked >= 8
    assert wall < 0.5 * asked * 0.05, (wall, asked)


@pytest.mark.serial
def test_vlm_lane_is_kept_full():
    # 30 jobs on a VLM lane of 8 at 20 ms per request: the run takes at most 1.4x the lane's own
    # floor (every request over the capacity)
    trace = FIXTURES / "text"
    models = ReplayModels(Trace(trace))
    for item in TraceJobs(trace):
        reference(item.inputs, models)
    floor = len(models.asked) * 6 * 0.02 / 8
    layers = {
        "engines": engines(trace, latency_ms=20.0, capacity=8),
        "source": {"kind": "trace", "path": str(trace), "repeat": 6},
        "machine": {"vlm_waiting": 16},
    }
    start = time.time()
    records, _, _ = run(layers)
    wall = time.time() - start
    assert all(r.ok for r in records) and len(records) == 30
    assert wall < 1.4 * floor + 1.0, (wall, floor)
