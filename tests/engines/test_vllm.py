"""The in-process vLLM entry: the engine contract on recorded GBC requests, and the GBC machine
on it. Needs vLLM, a CUDA device and ``KOHAKUGBC_VLLM_MODEL`` (a Qwen3.5 checkpoint)."""

import json
import os
from pathlib import Path

import pytest

vllm = pytest.importorskip("vllm")

from togeari import WorkItem  # noqa: E402
from togeari.testing import check_engine  # noqa: E402

from kohakugbc.engines import registry  # noqa: E402
from kohakugbc.io.sources import TraceJobs  # noqa: E402
from kohakugbc.machine import build_parts  # noqa: E402
from kohakugbc.nodes.specs.parse import load_json  # noqa: E402
from kohakugbc.nodes.specs.replay import ReplayModels  # noqa: E402
from kohakugbc.nodes.specs.trace import Trace  # noqa: E402
from kohakugbc.program.reference import run as reference  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.weights,
    pytest.mark.skipif(not os.environ.get("KOHAKUGBC_VLLM_MODEL"), reason="no model"),
]
TRACE = Path(__file__).resolve().parents[1] / "fixtures" / "traces" / "text"


def settings() -> dict:
    return {
        "entry": "vllm",
        "model": os.environ.get("KOHAKUGBC_VLLM_MODEL", ""),
        "gpu_memory_utilization": float(os.environ.get("KOHAKUGBC_VLLM_MEMORY", "0.5")),
        "max_num_seqs": 32,
    }


class Capture(ReplayModels):
    def __init__(self, trace):
        super().__init__(trace)
        self.requests: list[dict] = []

    def chat(self, request: dict) -> str:
        self.requests.append(request)
        return super().chat(request)


def recorded_requests() -> list[dict]:
    models = Capture(Trace(TRACE))
    for item in TraceJobs(TRACE):
        reference(item.inputs, models)
    return models.requests


def alike(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (load_json(a) is None) == (load_json(b) is None)


def test_vllm_unit_keeps_the_engine_contract():
    slot = registry.build_slot("vlm", settings(), {})
    unit = slot.units[0]
    unit.start()
    try:
        samples = [WorkItem("chat", {"request": r}, slot.adapter) for r in recorded_requests()[:16]]
        empty = {**samples[0].inputs["request"], "images": [[None, None]]}
        failing = [WorkItem("chat", {"request": empty}, slot.adapter)]

        # One unit started once (an engine start loads the model): the factory hands it out
        report = check_engine(
            lambda: unit,
            samples,
            failing_items=failing,
            rounds=48,
            timeout_s=300,
            tolerance=alike,
            submit_bound_ms=5.0,
        )
        report.raise_if_failed()
    finally:
        unit.stop()


def test_machine_on_vllm_in_process():
    cfg = {
        "engines": {
            "vlm": settings(),
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
    for r in parts.sink.records:
        graph = json.loads(r.outputs["graph"])
        assert graph["detail_caption"], r.key
