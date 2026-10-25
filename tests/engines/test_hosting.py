"""Engines in engine host processes the machine starts (``engines["hosts"]``): the machine on
spawned hosts gives the PoC's graphs over both transports; a spawned host's unit keeps the
engine contract with its process killed mid-check (the next connection starts it again); a
payload of 4 MiB crosses in shared memory and comes back equal, its blocks released."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from togeari import EngineBase, Ok, WorkItem
from togeari.testing import check_engine

from kohakugbc.engines import registry, spawn
from kohakugbc.engines.host import EngineHost, RemoteUnit
from kohakugbc.engines.host.host import PAYLOAD
from kohakugbc.graph import digest
from kohakugbc.io.sources import TraceJobs
from kohakugbc.machine import build_parts
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace
from kohakugbc.program.reference import run as reference

REPO = Path(__file__).resolve().parents[2]
TRACE = REPO / "tests" / "fixtures" / "traces" / "text"
EXPECTED = Trace(TRACE).jobs
KEY = b"kohakugbc-hosting"


def replay(**settings) -> dict:
    return {"entry": "replay", "trace": str(TRACE), "latency_ms": 2.0, **settings}


def named(graph) -> str:
    g = json.loads(graph) if isinstance(graph, str) else json.loads(json.dumps(graph))
    g["img_path"] = Path(g["img_path"]).name
    return digest(g)


def hosted(transport: str) -> dict:
    """VLM on one host, detector and embedder on another, each slot's lane on remote units."""
    return {
        "engines": {
            "hosts": {
                "vlm0": {"engines": {"vlm": replay(units=2, capacity=8)}},
                "aux0": {
                    "engines": {"detect": replay(), "embed": replay()},
                    "after": ["vlm0"],
                    "order": [["detect"], ["embed"]],
                },
            },
            "vlm": {"entry": "remote", "hosts": ["vlm0"], "transport": transport},
            "detect": {"entry": "remote", "hosts": ["aux0"], "transport": transport},
            "embed": {"entry": "remote", "hosts": ["aux0"], "transport": transport},
        },
        "source": {"kind": "trace", "path": str(TRACE), "repeat": 2},
    }


def spawned(parts) -> set:
    """The host processes behind a machine's remote units."""
    units = [u for s in parts.slots.values() for u in s.units]
    return {u.locate.__self__ for u in units if getattr(u, "locate", None)}


def chat_requests() -> list[dict]:
    models = Capture(Trace(TRACE))
    for item in TraceJobs(TRACE):
        reference(item.inputs, models)
    return models.requests


class Capture(ReplayModels):
    def __init__(self, trace):
        super().__init__(trace)
        self.requests: list[dict] = []

    def chat(self, request: dict) -> str:
        self.requests.append(request)
        return super().chat(request)


@pytest.mark.serial
@pytest.mark.parametrize("transport", ["shm", "inband"])
def test_the_machine_on_spawned_hosts(transport):
    parts = build_parts(hosted(transport))
    try:
        units = parts.slots["vlm"].units
        assert [u.name for u in units] == ["vlm0@vlm0", "vlm1@vlm0"]
        assert [u.name for u in parts.slots["detect"].units] == ["detect@aux0"]
        assert type(parts.slots["detect"].adapter).__name__ == "ReplayAdapter"
        with parts.machine:
            parts.machine.run(parts.source, parts.sink)
        assert len(parts.sink.records) == 10
        for rec in parts.sink.records:
            assert rec.ok, (rec.key, rec.failure)
            assert named(rec.outputs["graph"]) == named(EXPECTED[rec.key.split("#")[0]]["graph"])
        assert all(u.state["seen"] > 0 for u in units)
    finally:
        for host in spawned(parts):
            host.close()


@pytest.mark.serial
def test_a_spawned_hosts_unit_keeps_the_contract_through_a_kill():
    cfg = hosted("shm")["engines"]
    hosts = registry.spawn.launch_all({"vlm0": cfg["hosts"]["vlm0"]})
    slot = registry.build_slot("vlm", cfg["vlm"], {"hosts": hosts})
    local = registry.build_slot("vlm", replay(), {})
    unit = slot.units[0]
    try:
        samples = [WorkItem("chat", {"request": r}, local.adapter) for r in chat_requests()[:12]]
        report = check_engine(
            lambda: unit, samples, kill=lambda: hosts["vlm0"].proc.kill(), rounds=80, timeout_s=120
        )
        report.raise_if_failed()
        # The check stopped the unit; starting it finds the host's process (started again by
        # a background reconnect, or by this start) and its unit up on it
        unit.start()
        assert unit.ready() and hosts["vlm0"].proc.poll() is None
        deadline = time.monotonic() + 60
        got = []
        unit.submit(samples[0], lambda _, r: got.extend(r))
        while not got:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert isinstance(got[0], Ok), got
    finally:
        unit.stop()
        hosts["vlm0"].close()


def test_a_host_loads_after_the_hosts_it_waits_for():
    first = spawn.HostProcess("first", {})
    second = spawn.HostProcess("second", {})
    first.units, second.after = {"vlm": ["a", "b"]}, [first]
    done = threading.Event()
    waiter = threading.Thread(target=lambda: (second.wait_after(10), done.set()), daemon=True)
    waiter.start()
    first.unit_up("a")
    assert not done.wait(0.3), "went on with one of the first host's two units up"
    first.unit_up("b")
    assert done.wait(5)

    # Stages within one host: the detector waits for the host's VLM, the VLM for nothing
    card = spawn.HostProcess("card", {}, order=[["vlm"], ["detect", "embed"]])
    card.units = {"vlm": ["v"], "detect": ["d"], "embed": ["e"]}
    card.wait_after(0.1, slot="vlm")
    staged = threading.Event()
    waiter = threading.Thread(
        target=lambda: (card.wait_after(10, slot="detect"), staged.set()), daemon=True
    )
    waiter.start()
    assert not staged.wait(0.3), "the detector went on before the host's VLM was up"
    card.unit_up("v")
    assert staged.wait(5)

    third = spawn.HostProcess("third", {})
    third.units, second.after = {"vlm": ["c"]}, [third]
    third.unit_failed("c", "out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        second.wait_after(1)
    with pytest.raises(ValueError, match="cycle"):
        spawn.launch_all(
            {"x": {"engines": {}, "after": ["y"]}, "y": {"engines": {}, "after": ["x"]}}
        )


class Echo(EngineBase):
    """A batch unit answering each payload with itself."""

    kind = "batch"
    protocol = "echo"
    max_batch = 4

    def capacity(self) -> int:
        return 8

    def submit(self, work, done):
        done(0, [Ok(item.payload()) for item in work])
        return 0


def test_a_large_payload_crosses_in_shared_memory():
    host = EngineHost([Echo("echo")], ("127.0.0.1", 0), KEY)
    address = host.start()
    unit = RemoteUnit(f"{address[0]}:{address[1]}/echo", KEY, kind="batch", transport="shm")
    blocks = []
    encode = unit.codec.encode

    def spy(obj):
        packet, owned = encode(obj)
        blocks.append(len(packet.blocks))
        return packet, owned

    unit.codec.encode = spy
    unit.start()
    try:
        array = np.arange(1 << 20, dtype=np.float32)
        got = []
        unit.submit([WorkItem("echo", {"payload": array}, PAYLOAD)], lambda _, r: got.extend(r))
        deadline = 200
        while not got and deadline:
            deadline -= 1
            unit.changed.acquire()
            unit.changed.wait(0.05)
            unit.changed.release()
        assert blocks == [1], blocks
        assert isinstance(got[0], Ok) and np.array_equal(got[0].value, array)
        # The host read the payload and said so: the unit holds no block any more
        for _ in range(100):
            if not unit.held:
                break
            unit.changed.acquire()
            unit.changed.wait(0.05)
            unit.changed.release()
        assert unit.held == {}
    finally:
        unit.stop()
        host.stop()
