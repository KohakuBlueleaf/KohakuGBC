"""Remote units keep the engine contract (stream and batch units behind an engine host, the host
killed mid-check), and the machine runs real engines on other processes: the torch detector
behind ``gbc host``, and (``KOHAKUGBC_REMOTE_VLM``) a VLM host on another node."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from togeari import WorkItem
from togeari.testing import check_engine

from kohakugbc.engines import registry
from kohakugbc.engines.adapters import VllmChatAdapter
from kohakugbc.engines.host import EngineHost, RemoteUnit
from kohakugbc.graph import digest
from kohakugbc.io.sources import TraceJobs
from kohakugbc.machine import build_parts
from kohakugbc.nodes.images import cut
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace
from kohakugbc.program.reference import run as reference

TRACE = Path(__file__).resolve().parents[1] / "fixtures" / "traces" / "text"
KEY = b"kohakugbc-remote"


class Capture(ReplayModels):
    def __init__(self, trace):
        super().__init__(trace)
        self.requests: list[dict] = []

    def chat(self, request: dict) -> str:
        self.requests.append(request)
        return super().chat(request)


def chat_requests() -> list[dict]:
    models = Capture(Trace(TRACE))
    for item in TraceJobs(TRACE):
        reference(item.inputs, models)
    return models.requests


def detector_inputs() -> list[dict]:
    trace = Trace(TRACE)
    out = []
    for rec in trace.detect.values():
        key = json.loads(rec["key"])
        image = {"path": str(trace.root / trace.meta["image_root"] / key["image"])}
        request = {"key": rec["key"], "texts": key["texts"], "box": key["box"]}
        out.append({"patch": cut(image, key["box"]), "request": request})
    return out


@pytest.mark.parametrize("slot", ["vlm", "detect"])
def test_remote_units_keep_the_engine_contract(slot):
    settings = {"entry": "replay", "trace": str(TRACE), "latency_ms": 5.0}
    local = registry.build_slot(slot, settings, {})
    host = EngineHost(local.units, ("127.0.0.1", 0), KEY)
    address = host.start()
    try:
        where = f"{address[0]}:{address[1]}/{local.units[0].name}"
        unit = RemoteUnit(where, KEY, kind=local.units[0].kind, max_batch=4, name=f"r-{slot}")
        if slot == "vlm":
            samples = [
                WorkItem("chat", {"request": r}, local.adapter) for r in chat_requests()[:12]
            ]
            bad = {**samples[0].inputs["request"], "key": "never asked"}
            failing = [WorkItem("chat", {"request": bad}, local.adapter)]
        else:
            samples = [WorkItem("detector", d, local.adapter) for d in detector_inputs()[:8]]
            bad = {**samples[0].inputs, "request": {"key": "never asked", "texts": ["x"]}}
            failing = [WorkItem("detector", bad, local.adapter)]
        report = check_engine(
            lambda: unit,
            samples,
            failing_items=failing,
            kill=host.stop,
            rounds=120,
            timeout_s=60,
            # Over 100 submits, so q0.99 is held to 5 ms; the max leaves room for one stall of a
            # shared CI runner (85 ms seen on Windows) beside the pickling sender thread
            submit_bound_ms=5.0,
            submit_max_ms=250.0,
        )
        report.raise_if_failed()
    finally:
        host.stop()


def named(graph) -> str:
    g = json.loads(graph) if isinstance(graph, str) else json.loads(json.dumps(graph))
    g["img_path"] = Path(g["img_path"]).name
    return digest(g)


@pytest.mark.serial
@pytest.mark.weights
@pytest.mark.gpu
def test_live_detector_behind_a_host_process(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    # The detector as the PoC ran it, in its own process: the machine gives the PoC's graphs
    detect = {"entry": "torch", "devices": ["cuda:0"], "dtype": "float16", "max_batch": 1}
    config = tmp_path / "host.py"
    config.write_text(f"engines = {{'detect': {detect!r}}}\n", encoding="utf-8")
    env = {**os.environ, "KOHAKUGBC_HOST_KEY": KEY.decode()}
    command = [sys.executable, "-m", "kohakugbc.cli.main", "host", str(config)]
    command += ["--slot", "detect", "--listen", "127.0.0.1:7592"]
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        assert "serving" in line, line
        unit = line.split("[")[1].split("]")[0].strip("'")
        cfg = {
            "engines": {
                "vlm": {"entry": "replay", "trace": str(TRACE), "latency_ms": 1.0},
                "detect": {
                    "entry": "remote",
                    "hosts": [f"127.0.0.1:7592/{unit}"],
                    "authkey": KEY.decode(),
                    "runs": {**detect, "devices": ["cpu"]},
                },
                "embed": {"entry": "replay", "trace": str(TRACE)},
            },
            "source": {"kind": "trace", "path": str(TRACE)},
        }
        parts = build_parts(cfg)
        with parts.machine:
            parts.machine.run(parts.source, parts.sink)
        expected = Trace(TRACE).jobs
        for rec in parts.sink.records:
            assert rec.ok, (rec.key, rec.failure)
            assert named(rec.outputs["graph"]) == named(expected[rec.key]["graph"]), rec.key
    finally:
        proc.terminate()
        proc.wait(30)


@pytest.mark.skipif(not os.environ.get("KOHAKUGBC_REMOTE_VLM"), reason="no remote VLM host")
def test_vlm_on_another_node():
    """``KOHAKUGBC_REMOTE_VLM``: ``host:port/unit`` of a ``gbc host --slot vlm`` serving the
    ``vllm`` entry (``KOHAKUGBC_HOST_KEY`` its key): the remote unit keeps the contract, and the
    machine on it ends every job with a described image vertex."""
    where = os.environ["KOHAKUGBC_REMOTE_VLM"]
    key = os.environ["KOHAKUGBC_HOST_KEY"].encode()
    unit = RemoteUnit(where, key, kind="stream", protocol="vllm", name="remote-vllm")
    adapter = VllmChatAdapter()
    samples = [WorkItem("chat", {"request": r}, adapter) for r in chat_requests()[:12]]
    report = check_engine(
        lambda: unit, samples, rounds=24, timeout_s=300, tolerance=lambda a, b: bool(a) and bool(b)
    )
    report.raise_if_failed()

    vlm = {"entry": "remote", "hosts": [where], "adapter": "vllm", "kind": "stream"}
    cfg = {
        "engines": {
            "vlm": vlm,
            "detect": {"entry": "replay", "trace": str(TRACE), "latency_ms": 1.0},
            "embed": {"entry": "replay", "trace": str(TRACE), "unknown": "hash"},
        },
        "source": {"kind": "trace", "path": str(TRACE)},
        "controls": {"max_top_level": 0, "max_breadth_rounds": 0, "relational": "none"},
    }
    parts = build_parts(cfg)
    with parts.machine:
        parts.machine.run(parts.source, parts.sink)
    assert all(r.ok for r in parts.sink.records), [r.failure for r in parts.sink.records]
    assert all(json.loads(r.outputs["graph"])["detail_caption"] for r in parts.sink.records)
