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


def test_waits_are_sampled_beside_the_report(tmp_path):
    # Slow VLM replies: jobs wait on the VLM lane, and the samples say so
    trace = FIXTURES / "text"
    records, _, _ = run(
        {
            "engines": engines(trace, latency_ms=30.0, capacity=4),
            "source": {"kind": "trace", "path": str(trace), "repeat": 4},
            "machine": {"report": str(tmp_path / "report"), "waits": {"every_s": 0.05}},
        }
    )
    assert all(r.ok for r in records)
    rows = [json.loads(line) for line in (tmp_path / "report" / "waits.jsonl").open()]
    assert len(rows) >= 3
    assert all(r["cost_ms"] >= 0 and r["entities"] + r["relations"] <= 10_000 for r in rows)
    assert any(r["lanes"].get("vlm", {}).get("lane_capacity", 0) > 0 for r in rows), rows
    assert all(sum(r["reasons"].values()) == r["relations"] for r in rows)


def test_an_expired_vlm_attempt_answers_empty_and_is_counted():
    # Every VLM reply takes longer than the lane's deadline: each request expires, the program's
    # ask answers empty, and the job carries the deadline failures
    trace = FIXTURES / "text"
    slow = engines(trace, latency_ms=200.0)
    slow["vlm"]["lane"] = {"timeout_s": 0.02}
    records, _, _ = run({"engines": slow, "source": {"kind": "trace", "path": str(trace)}})
    assert records and all(r.ok for r in records)
    failures = [f for r in records for f in r.stats.get("policy_failures", [])]
    assert failures and all(f["reason"] == "deadline" and f["scope"] == "attempt" for f in failures)
    assert {f["node"] for f in failures} == {"chat"}


def test_configuration_errors_are_listed_together():
    with pytest.raises(ValueError) as err:
        build({"engines": {"vlm": {"entry": "nope"}, "detect": {}}})
    text = str(err.value)
    assert "engines.vlm: unknown entry 'nope'" in text
    assert "engines.detect: no entry" in text and "engines.embed: no entry" in text


def one_job(trace: Path, job, latency_ms: float) -> float:
    layers = {
        "engines": engines(trace, latency_ms=latency_ms, capacity=64),
        "source": {"kind": "trace", "path": str(trace)},
    }
    machine, _, _ = build(layers)
    with machine:
        start = time.time()
        handle = machine.submit(job.inputs, key=job.key)
        assert handle.result(timeout=60).ok
        return time.time() - start


def full_lane(trace: Path, latency_ms: float) -> float:
    layers = {
        "engines": engines(trace, latency_ms=latency_ms, capacity=8),
        "source": {"kind": "trace", "path": str(trace), "repeat": 6},
        "machine": {"vlm_waiting": 16},
    }
    start = time.time()
    records, _, _ = run(layers)
    wall = time.time() - start
    assert all(r.ok for r in records) and len(records) == 30
    return wall


@pytest.mark.serial
def test_regions_of_a_wave_run_side_by_side():
    # One job, room for all of its VLM requests: what 50 ms per request adds to the job's time
    # (beside the same job at 1 ms, the runner's own overhead) is far less than the requests one
    # after another
    trace = FIXTURES / "text"
    asked, job = 0, None
    for item in TraceJobs(trace):
        models = ReplayModels(Trace(trace))
        reference(item.inputs, models)
        if len(models.asked) > asked:
            asked, job = len(models.asked), item
    base = one_job(trace, job, 1.0)
    wall = one_job(trace, job, 50.0)
    assert asked >= 8
    assert wall - base < 0.5 * asked * 0.05, (wall, base, asked)


@pytest.mark.serial
def test_vlm_lane_is_kept_full():
    # 30 jobs on a VLM lane of 8: what 50 ms per request adds to the run (beside the same run at
    # 1 ms, the runner's own overhead) is at most 1.4x the lane's floor, every request over the
    # capacity
    trace = FIXTURES / "text"
    models = ReplayModels(Trace(trace))
    for item in TraceJobs(trace):
        reference(item.inputs, models)
    floor = len(models.asked) * 6 * 0.05 / 8
    base = full_lane(trace, 1.0)
    wall = full_lane(trace, 50.0)
    assert wall - base < 1.4 * floor, (wall, base, floor)
