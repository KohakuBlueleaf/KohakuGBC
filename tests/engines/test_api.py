"""The ``api`` entry: request shapes per protocol and server flavour, the GBC machine over HTTP
against the replay server (the PoC's graphs, the PoC's requests), and the engine contract."""

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from togeari import WorkItem
from togeari.testing import check_engine

from kohakugbc.engines import registry
from kohakugbc.engines.replay.server import ReplayServer
from kohakugbc.engines.vlm import protocols
from kohakugbc.engines.vlm.api import ChatAdapter
from kohakugbc.graph import digest
from kohakugbc.io.sources import TraceJobs
from kohakugbc.machine import build_parts
from kohakugbc.nodes.images import encode
from kohakugbc.nodes.specs import chat as specs
from kohakugbc.nodes.specs.parse import load_json
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace
from kohakugbc.program.reference import run as reference

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "traces"


def picture(w: int, h: int) -> dict:
    return encode(np.full((h, w, 3), 128, dtype=np.uint8), 1568)


def entity_request() -> dict:
    controls = {"leaf_caption": "query"}
    region = {"id": "cup", "text": "cups", "box": [0.1, 0.1, 0.5, 0.5, 0.9]}
    return specs.entity("x.png", region, picture(64, 48), controls)


# ---- request shapes -----------------------------------------------------------------------------


@pytest.mark.parametrize("flavour", protocols.FLAVOURS)
def test_openai_flavours_place_the_schema_and_thinking_switch(flavour):
    body = protocols.openai_body(entity_request(), "m", flavour, {"top_p": 0.9})
    where = {
        "vllm": "structured_outputs",
        "sglang": "response_format",
        "llama.cpp": "json_schema",
        "openai": "response_format",
        "openrouter": "response_format",
    }[flavour]
    assert where in body and body["top_p"] == 0.9 and body["max_tokens"] == 512
    thinking = body.get("chat_template_kwargs", {}).get("enable_thinking")
    assert thinking is (None if flavour in ("openai", "openrouter") else False)
    user = body["messages"][-1]["content"]
    assert [p["type"] for p in user] == ["image_url", "text"]
    assert user[-1]["text"].startswith("Please assess the image focusing on 'cup'")


def test_kohaku_grammar_sends_the_schema_as_vllm_xargs():
    req = entity_request()
    body = ChatAdapter("openai", "m", structured="kohaku").prepare({"request": req}, "http").json
    assert "structured_outputs" not in body
    sent = json.loads(body["vllm_xargs"]["kohaku_json_schema"])
    assert sent == specs.schema(req["schema"])
    plain = ChatAdapter("openai", "m", structured="prompt").prepare({"request": req}, "http").json
    assert "vllm_xargs" not in plain and "structured_outputs" not in plain
    with pytest.raises(ValueError, match="vllm flavour"):
        protocols.openai_body(req, "m", "sglang", {}, True, True)


def test_batched_request_labels_each_picture():
    regions = [{"id": f"r{i}", "text": "cups", "box": [0, 0, 1, 1, None]} for i in range(3)]
    req = specs.entity_batch("x.png", regions, [picture(32, 32)] * 3)
    user = protocols.openai_body(req, "m", "vllm", {})["messages"][-1]["content"]
    assert [p["type"] for p in user] == ["text", "image_url"] * 3 + ["text"]
    assert user[0]["text"] == "Region 1: 'cup'"


def test_anthropic_body_and_reply():
    body = protocols.anthropic_body(entity_request(), "claude", "", {})
    assert body["system"].startswith("You") or body["system"]
    kinds = [p["type"] for p in body["messages"][0]["content"]]
    assert kinds == ["image", "text"]
    data = body["messages"][0]["content"][0]["source"]["data"]
    assert Image.open(io.BytesIO(base64.b64decode(data))).size == (64, 48)
    assert protocols.anthropic_reply({"content": [{"type": "text", "text": "{}"}]}) == "{}"


def test_an_empty_crop_is_an_item_failure():
    req = {**entity_request(), "images": [[None, None]]}
    with pytest.raises(ValueError, match="empty crop"):
        ChatAdapter("openai", "m").prepare({"request": req}, "http")


# ---- the machine over HTTP ----------------------------------------------------------------------


def named(graph) -> str:
    g = json.loads(graph) if isinstance(graph, str) else json.loads(json.dumps(graph))
    g["img_path"] = Path(g["img_path"]).name
    return digest(g)


@pytest.mark.parametrize(
    ("arm", "protocol", "flavour"),
    [("text", "openai", "vllm"), ("query", "openai", "llama.cpp"), ("batch4", "anthropic", "")],
)
def test_machine_over_http_gives_the_poc_graphs(arm, protocol, flavour):
    trace = FIXTURES / arm
    server = ReplayServer(str(trace), latency_s=0.005)
    url = server.start()
    try:
        vlm = {
            "entry": "api",
            "protocol": protocol,
            "flavour": flavour,
            "model": "replay",
            "endpoints": [{"base_url": url, "max_inflight": 16}],
        }
        cfg = {
            "engines": {
                "vlm": vlm,
                "detect": {"entry": "replay", "trace": str(trace), "latency_ms": 1.0},
                "embed": {"entry": "replay", "trace": str(trace), "latency_ms": 1.0},
            },
            "source": {"kind": "trace", "path": str(trace)},
        }
        parts = build_parts(cfg)
        with parts.machine:
            parts.machine.run(parts.source, parts.sink)
    finally:
        server.stop()
    expected = Trace(trace).jobs
    assert len(parts.sink.records) == len(expected)
    for rec in parts.sink.records:
        assert rec.ok, (rec.key, rec.failure)
        assert named(rec.outputs["graph"]) == named(expected[rec.key]["graph"]), rec.key
    # Every request arrived as the PoC sent it: known, same query, same picture sizes
    assert server.counts["unknown"] == 0
    assert server.mismatches == []


# ---- the engine contract -----------------------------------------------------------------------


class Capture(ReplayModels):
    def __init__(self, trace):
        super().__init__(trace)
        self.requests: list[dict] = []

    def chat(self, request: dict) -> str:
        self.requests.append(request)
        return super().chat(request)


def recorded_requests(arm: str) -> list[dict]:
    trace = Trace(FIXTURES / arm)
    models = Capture(trace)
    for item in TraceJobs(FIXTURES / arm):
        reference(item.inputs, models)
    return models.requests


def test_api_unit_keeps_the_engine_contract():
    server = ReplayServer(str(FIXTURES / "text"), latency_s=0.01)
    url = server.start()
    try:
        settings = {"entry": "api", "model": "replay"}
        settings["endpoints"] = [{"base_url": url, "max_inflight": 8}]
        slot = registry.build_slot("vlm", settings, {})
        requests = recorded_requests("text")[:20]
        samples = [WorkItem("chat", {"request": r}, slot.adapter) for r in requests]
        unknown = {**requests[0], "key": "never asked"}
        failing = [WorkItem("chat", {"request": unknown}, slot.adapter)]
        report = check_engine(
            lambda: registry.build_slot("vlm", settings, {}).units[0],
            samples,
            failing_items=failing,
            rounds=60,
        )
        report.raise_if_failed()
    finally:
        server.stop()


@pytest.mark.skipif(not os.environ.get("KOHAKUGBC_VLM_URL"), reason="no local VLM server")
def test_api_unit_against_a_local_server():
    """``KOHAKUGBC_VLM_URL`` (an OpenAI-compatible base URL), ``KOHAKUGBC_VLM_MODEL``,
    ``KOHAKUGBC_VLM_FLAVOUR``: the recorded requests get JSON replies the parsers read."""
    settings = {
        "entry": "api",
        "model": os.environ.get("KOHAKUGBC_VLM_MODEL", "qwen3.5-4b"),
        "flavour": os.environ.get("KOHAKUGBC_VLM_FLAVOUR", "vllm"),
        "endpoints": [{"base_url": os.environ["KOHAKUGBC_VLM_URL"], "max_inflight": 16}],
    }
    slot = registry.build_slot("vlm", settings, {})
    requests = recorded_requests("text")[:16]
    samples = [WorkItem("chat", {"request": r}, slot.adapter) for r in requests]

    # A sampling model's repeated replies differ in words: both must be non-empty, and JSON the
    # parsers read for schema-constrained requests (both JSON, or both free text)
    def alike(a: str, b: str) -> bool:
        return bool(a) and bool(b) and (load_json(a) is None) == (load_json(b) is None)

    report = check_engine(
        lambda: registry.build_slot("vlm", settings, {}).units[0],
        samples,
        rounds=32,
        timeout_s=300,
        tolerance=alike,
    )
    report.raise_if_failed()

    # The machine on the live model, its root alone (the trace's root mode; no detection, no
    # coverage: the replayed detector holds no boxes for what a live model names): every job
    # ends with a described image vertex
    trace = FIXTURES / "text"
    cfg = {
        "engines": {
            "vlm": settings,
            "detect": {"entry": "replay", "trace": str(trace), "latency_ms": 1.0},
            "embed": {"entry": "replay", "trace": str(trace), "unknown": "hash"},
        },
        "source": {"kind": "trace", "path": str(trace)},
        "controls": {"max_top_level": 0, "max_breadth_rounds": 0, "relational": "none"},
    }
    parts = build_parts(cfg)
    with parts.machine:
        parts.machine.run(parts.source, parts.sink)
    assert all(r.ok for r in parts.sink.records), [r.failure for r in parts.sink.records]
    for r in parts.sink.records:
        graph = json.loads(r.outputs["graph"])
        assert graph["detail_caption"] and graph["short_caption"], r.key
