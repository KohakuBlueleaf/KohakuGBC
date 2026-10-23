"""The ``qwen35`` entry: Togeari's engine contract on recorded GBC requests, replies that keep
their schemas, the forced-token jump and the snapshots at work, and the GBC machine on it."""

import json
import os
from pathlib import Path

import pytest
import torch
from togeari import WorkItem
from togeari.testing import check_engine

from kohakugbc.engines import registry
from kohakugbc.engines.builtin import MODULES
from kohakugbc.engines.vlm import qwen35
from kohakugbc.machine import build_parts
from kohakugbc.nodes.specs.parse import load_json

TRACE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "traces" / "text"
MODEL = os.environ.get("QWEN35_MODEL", "")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.weights,
    pytest.mark.skipif(not MODEL, reason="no model"),
]


def settings(**extra) -> dict:
    return {
        "entry": "qwen35",
        "model": MODEL,
        "memory": 0.75,
        "max_seqs": 32,
        "prefill_tokens": 4096,
        **extra,
    }


def alike(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (load_json(a) is None) == (load_json(b) is None)


def test_entry_is_installed():
    assert qwen35 in MODULES
    assert ("vlm", "qwen35") in registry.ENTRIES


CONFIGS = {
    "vllm-schemas": {},
    "compact-exact": {"any_whitespace": False, "jump": "exact"},
    "string-jump": {
        "jump": "string",
        "state_dtype": "bfloat16",
        "boundaries": ["system", "images"],
    },
}


@pytest.mark.parametrize("config", list(CONFIGS))
def test_unit_keeps_the_engine_contract(recorded, config):
    slot = registry.build_slot("vlm", settings(**CONFIGS[config]), {})
    unit = slot.units[0]
    unit.start()
    try:
        samples = [WorkItem("chat", {"request": r}, slot.adapter) for r, _ in recorded[:16]]
        empty = {**recorded[0][0], "images": [[None, None]]}
        failing = [WorkItem("chat", {"request": empty}, slot.adapter)]
        report = check_engine(
            lambda: unit,
            samples,
            failing_items=failing,
            rounds=48,
            timeout_s=60,
            tolerance=alike,
        )
        report.raise_if_failed()
        stats = unit.stats()
        assert not stats["failures"], stats["failures"]
        counts = stats["counts"]
        assert counts["snapshot_hits"] > 0, counts
        assert counts["graph_steps"] > 0, counts
        assert (counts["forced_tokens"] > 0) == (config != "vllm-schemas"), counts
    finally:
        unit.stop()
    assert torch.cuda.memory_allocated() < 2**30


def test_machine_on_qwen35():
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
