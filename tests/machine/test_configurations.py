"""Engine configurations on pseudo units: several units per lane, units behind an engine host
(in this process and in another), units failing, slowing, joining and leaving mid-run, jobs
resumed from snapshots. Every job that ends ok gives the PoC's graph."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from togeari.policies import Throughput
from togeari.stores import MemoryStore
from togeari.testing import SimStream

from kohakugbc.engines import registry
from kohakugbc.engines.host import EngineHost
from kohakugbc.engines.replay.entries import ReplayAdapter
from kohakugbc.graph import digest
from kohakugbc.machine import build_parts
from kohakugbc.nodes.specs.trace import Trace

REPO = Path(__file__).resolve().parents[2]
TRACE = REPO / "tests" / "fixtures" / "traces" / "text"
EXPECTED = Trace(TRACE).jobs
KEY = b"kohakugbc-test"


def replay(**settings) -> dict:
    return {"entry": "replay", "trace": str(TRACE), "latency_ms": 2.0, **settings}


def layers(vlm: dict | None = None, repeat: int = 1, **machine) -> dict:
    return {
        "engines": {"vlm": vlm or replay(), "detect": replay(), "embed": replay()},
        "source": {"kind": "trace", "path": str(TRACE), "repeat": repeat},
        "machine": machine,
    }


def named(graph) -> str:
    g = json.loads(graph) if isinstance(graph, str) else json.loads(json.dumps(graph))
    g["img_path"] = Path(g["img_path"]).name
    return digest(g)


def check(records, n: int) -> None:
    assert len(records) == n
    for rec in records:
        assert rec.ok, (rec.key, rec.failure)
        assert named(rec.outputs["graph"]) == named(EXPECTED[rec.key.split("#")[0]]["graph"])


def run(config: dict):
    parts = build_parts(config)
    with parts.machine:
        parts.machine.run(parts.source, parts.sink)
    return parts.sink.records, parts


def test_several_units_share_the_lane():
    records, parts = run(layers(replay(units=3, capacity=4), repeat=4))
    check(records, 20)
    assert all(u.counts["items"] > 0 for u in parts.slots["vlm"].units)


def test_units_behind_an_engine_host():
    slot = registry.build_slot("vlm", replay(units=2, capacity=8), {})
    host = EngineHost(slot.units, ("127.0.0.1", 0), KEY)
    address = host.start()
    try:
        hosts = [f"{address[0]}:{address[1]}/{u.name}" for u in slot.units]
        remote = {"entry": "remote", "hosts": hosts, "authkey": KEY.decode(), "runs": replay()}
        records, _ = run(layers(remote, repeat=2))
        check(records, 10)
        assert all(u.counts["items"] > 0 for u in slot.units)
    finally:
        host.stop()


@pytest.mark.serial
def test_units_behind_a_host_process(tmp_path):
    config = tmp_path / "host.py"
    config.write_text(f"engines = {{'detect': {replay()!r}}}\n", encoding="utf-8")
    env = {**os.environ, "KOHAKUGBC_HOST_KEY": KEY.decode()}
    command = [sys.executable, "-m", "kohakugbc.cli.main", "host", str(config)]
    command += ["--slot", "detect", "--listen", "127.0.0.1:7591"]
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert "serving" in proc.stdout.readline()
        cfg = layers()
        cfg["engines"]["detect"] = {
            "entry": "remote",
            "hosts": ["127.0.0.1:7591/replay-detect-0"],
            "authkey": KEY.decode(),
            "runs": replay(),
        }
        records, _ = run(cfg)
        check(records, 5)
    finally:
        proc.terminate()
        proc.wait(10)


def test_a_unit_down_mid_run_hands_its_work_to_the_others():
    # Unit 0 is down from 0.05 s to 1 s after start: its work goes to unit 1
    vlm = replay(units=2, capacity=4, latency_ms=10.0, outages=[[[0.05, 1.0]]])
    records, parts = run(layers(vlm, repeat=4))
    check(records, 20)
    assert parts.slots["vlm"].units[0].counts["outage_rejects"] > 0


def test_vlm_item_failures_are_empty_replies():
    # A failed VLM request answers with an empty reply: every job ends, some with smaller graphs
    # (off the recorded path, names the PoC never compared get hashed vectors)
    for seed in range(4):
        cfg = layers(replay(item_error_rate=0.3, seed=seed))
        cfg["engines"]["embed"] = replay(unknown="hash")
        records, _ = run(cfg)
        assert len(records) == 5 and all(r.ok for r in records), [r.failure for r in records]
        differ = sum(named(r.outputs["graph"]) != named(EXPECTED[r.key]["graph"]) for r in records)
        assert differ > 0


def test_detector_failures_fail_their_job_with_the_reason():
    cfg = layers()
    cfg["engines"]["detect"] = replay(item_error_rate=1.0)
    records, _ = run(cfg)
    assert len(records) == 5
    assert all(not r.ok and r.failure["node"].endswith("detector") for r in records)


@pytest.mark.serial
def test_a_slow_unit_takes_less_work():
    vlm = replay(units=2, capacity=4, latency_ms=20.0, speeds=[1.0, 0.2])
    vlm["lane"] = {"select": Throughput()}
    records, parts = run(layers(vlm, repeat=6))
    check(records, 30)
    fast, slow = parts.slots["vlm"].units
    assert fast.counts["items"] > 2 * slow.counts["items"], (fast.counts, slow.counts)


def test_units_join_and_leave_mid_run():
    parts = build_parts(layers(replay(latency_ms=5.0), repeat=6))
    joined = SimStream(name="joined-vlm", latency_ms=5.0, capacity=16, tail=0.0)
    with parts.machine:
        runner = threading.Thread(target=parts.machine.run, args=(parts.source, parts.sink))
        runner.start()
        time.sleep(0.3)
        parts.machine.attach(parts.lanes["vlm"], joined)
        time.sleep(0.2)
        parts.machine.detach(parts.slots["vlm"].units[0], drain=True)
        runner.join(120)
    check(parts.sink.records, 30)
    assert joined.counts["items"] > 0
    assert isinstance(parts.slots["vlm"].adapter, ReplayAdapter)


def test_snapshots_without_run_ahead_only():
    with pytest.raises(ValueError, match="run_ahead=False"):
        build_parts(layers(store=MemoryStore(), knobs={"snapshot_every": 5}))


def test_jobs_resume_from_snapshots():
    store = MemoryStore()
    knobs = {"snapshot_every": 5, "run_ahead": False}
    cfg = layers(replay(latency_ms=20.0), store=store, knobs=knobs)
    parts = build_parts(cfg)
    with parts.machine:
        runner = threading.Thread(target=parts.machine.run, args=(parts.source, parts.sink))
        runner.start()
        time.sleep(0.4)
        parts.machine.stop(drain=False)
        runner.join(60)
    assert sum(r.ok for r in parts.sink.records) < 5
    assert len(store) > 0

    # The unfinished jobs continue from their snapshots and end with the PoC's graphs
    records, _ = run(cfg)
    check(records, 5)
    assert any(r.stats.get("resumed") for r in records)
