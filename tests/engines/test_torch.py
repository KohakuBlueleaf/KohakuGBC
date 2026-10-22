"""The torch entries: OWLv2 and EmbeddingGemma give what the PoC's models gave on the recorded
crops and names, batched equal to single, keep the engine contract, run the GBC machine to the
PoC's graphs; SAM 3 batched equals SAM 3 single; the placement plan refuses what does not fit."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from togeari import WorkItem  # noqa: E402
from togeari.testing import check_engine  # noqa: E402

from kohakugbc.engines import registry  # noqa: E402
from kohakugbc.engines.torch.modules import (  # noqa: E402
    Owlv2Detector,
    Sam3Detector,
    SentenceEmbedder,
)
from kohakugbc.graph import digest  # noqa: E402
from kohakugbc.machine import build_parts  # noqa: E402
from kohakugbc.nodes.images import cut  # noqa: E402
from kohakugbc.nodes.specs.detect import SCORE_THRESHOLD  # noqa: E402
from kohakugbc.nodes.specs.trace import Trace  # noqa: E402

pytestmark = [pytest.mark.weights]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "traces"
TRACE = FIXTURES / "text"
CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if CUDA else "cpu"


def crops(trace: Trace) -> list[tuple[dict, dict]]:
    """(payload, recorded raw) per recorded detector call."""
    out = []
    for rec in trace.detect.values():
        key = json.loads(rec["key"])
        image = {"path": str(trace.root / trace.meta["image_root"] / key["image"])}
        patch = cut(image, key["box"])
        out.append(({"patch": patch, "request": {"texts": key["texts"]}}, rec["raw"]))
    return out


def same_raw(a: dict, b: dict, px: float = 1.0, score: float = 5e-3) -> bool:
    """Equal detections: per text, the same count, boxes within ``px``, scores within ``score``."""
    if a["labels"] != b["labels"]:
        return False
    for ba, bb in zip(a["boxes"], b["boxes"]):
        if max(abs(x - y) for x, y in zip(ba, bb)) > px:
            return False
    return all(abs(x - y) <= score for x, y in zip(a["scores"], b["scores"]))


@pytest.fixture(scope="module")
def poc_detector():
    """OWLv2 as the PoC ran it: float16, one crop per forward."""
    model = Owlv2Detector()
    model.load(DEVICE, torch.float16 if CUDA else torch.float32)
    return model


@pytest.fixture(scope="module")
def detector():
    """OWLv2 as the torch entry runs it by default: float32, batched."""
    model = Owlv2Detector()
    model.load(DEVICE, torch.float32)
    return model


@pytest.fixture(scope="module")
def embedder():
    model = SentenceEmbedder()
    model.load("cpu", torch.float32)
    return model


def run(model, payloads):
    with torch.inference_mode():
        return model.split(model.forward(model.prepare(payloads)), len(payloads))


@pytest.mark.gpu
def test_detector_reproduces_the_poc(poc_detector):
    # Every recorded call, to the last digit the PoC logged
    pairs = crops(Trace(TRACE))
    assert len(pairs) > 10
    for p, raw in pairs:
        assert same_raw(run(poc_detector, [p])[0], raw, px=1e-3, score=1e-6), p["request"]


@pytest.mark.gpu
def test_detector_batched_equals_single(detector):
    payloads = [p for p, _ in crops(Trace(TRACE))]
    single = [run(detector, [p])[0] for p in payloads]
    for size in (2, 4, 8):
        batched = []
        for i in range(0, len(payloads), size):
            batched += run(detector, payloads[i : i + size])
        assert all(same_raw(a, b, px=0.05, score=5e-4) for a, b in zip(batched, single)), size


def kept(raw: dict) -> dict:
    """The detections the program's selection can keep (score above its threshold)."""
    rows = [r for r in zip(raw["boxes"], raw["scores"], raw["labels"]) if r[1] > SCORE_THRESHOLD]
    return {"boxes": [r[0] for r in rows], "scores": [r[1] for r in rows],
            "labels": [r[2] for r in rows]}  # fmt: skip


def same_set(a: dict, b: dict, px: float, score: float) -> bool:
    """Equal detections in any order (instances of near-equal score may swap places): the same
    count, and every box of ``a`` matched to its own box of ``b`` of the same label within
    ``px`` and ``score``."""
    if len(a["scores"]) != len(b["scores"]):
        return False
    free = list(zip(b["boxes"], b["scores"], b["labels"]))
    for box, s, label in zip(a["boxes"], a["scores"], a["labels"]):
        match = next((m for m in free if m[2] == label and abs(m[1] - s) <= score
                      and max(abs(x - y) for x, y in zip(m[0], box)) <= px), None)  # fmt: skip
        if match is None:
            return False
        free.remove(match)
    return True


@pytest.mark.gpu
def test_sam3_batched_equals_single():
    """SAM 3 runs its vision encoder once per crop and one decoder pass per (crop, text):
    batching crops and texts gives each crop's detections alone; a crop answered with another
    crop's features is seen."""
    if not CUDA:
        pytest.skip("SAM 3 on the CPU is too slow for the suite")
    model = Sam3Detector()
    model.load(DEVICE, torch.float32)
    payloads = [p for p, _ in crops(Trace(TRACE))][:12]
    single = [kept(run(model, [p])[0]) for p in payloads]
    assert sum(len(s["scores"]) for s in single) > 5
    for size in (2, 4):
        batched = []
        for i in range(0, len(payloads), size):
            batched += [kept(r) for r in run(model, payloads[i : i + size])]
        # Boxes come from thresholded masks: a weak mask's edge pixel may flip between batch
        # shapes and move its box a few pixels; 2% of the crop's longer side
        assert all(
            same_set(a, b, px=0.02 * max(p["patch"].shape[:2]), score=2e-3)
            for a, b, p in zip(batched, single, payloads)
        ), size
    # The check sees crops whose features were swapped
    swapped = run(model, [{**payloads[0], "patch": payloads[1]["patch"]}])[0]
    assert not same_set(kept(swapped), single[0], px=0.02 * 500, score=2e-3)


def test_embedder_reproduces_the_poc_and_batches(embedder):
    trace = Trace(TRACE)
    texts = sorted(trace.embed)[:24]
    assert texts
    batched = run(embedder, texts)
    for text, vec in zip(texts, batched):
        assert np.abs(vec - trace.embed[text]).max() < 1e-4, text
        assert np.abs(vec - run(embedder, [text])[0]).max() < 1e-5, text


def test_units_keep_the_engine_contract():
    trace = Trace(TRACE)
    slots = registry.build(
        {
            "vlm": {"entry": "replay", "trace": str(TRACE)},
            "detect": {"entry": "torch", "devices": [DEVICE], "max_batch": 4},
            "embed": {"entry": "torch", "devices": ["cpu"]},
            "cpu": {"entry": "python"},
        }
    )
    det, emb = slots["detect"], slots["embed"]
    det_samples = [
        WorkItem("detector", {"patch": p["patch"], "request": p["request"]}, det.adapter)
        for p, _ in crops(trace)[:8]
    ]
    emb_samples = [WorkItem("embedder", {"text": t}, emb.adapter) for t in sorted(trace.embed)[:8]]

    def close(a, b):
        if isinstance(a, dict):
            return same_raw(a, b)
        return np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-5

    for unit, samples in ((det.units[0], det_samples), (emb.units[0], emb_samples)):
        report = check_engine(lambda u=unit: u, samples, rounds=40, timeout_s=300, tolerance=close)
        report.raise_if_failed()


def named(graph) -> str:
    g = json.loads(graph) if isinstance(graph, str) else json.loads(json.dumps(graph))
    g["img_path"] = Path(g["img_path"]).name
    return digest(g)


def machine_graphs(detect: dict) -> dict:
    """Graphs of the fixture jobs: the VLM replays the PoC's replies, the detector and the
    embedder run for real (names off the recorded path get hashed vectors)."""
    cfg = {
        "engines": {
            "vlm": {"entry": "replay", "trace": str(TRACE), "latency_ms": 1.0},
            "detect": {"entry": "torch", "devices": [DEVICE], **detect},
            "embed": {"entry": "torch", "devices": ["cpu"]},
        },
        "source": {"kind": "trace", "path": str(TRACE)},
    }
    parts = build_parts(cfg)
    with parts.machine:
        parts.machine.run(parts.source, parts.sink)
    assert all(r.ok for r in parts.sink.records), [r.failure for r in parts.sink.records]
    return {r.key: named(r.outputs["graph"]) for r in parts.sink.records}


@pytest.mark.gpu
def test_machine_on_live_vision_models():
    # As the PoC ran its detector: the PoC's graphs
    poc = machine_graphs({"dtype": "float16", "max_batch": 1})
    expected = Trace(TRACE).jobs
    assert all(poc[k] == named(expected[k]["graph"]) for k in expected)

    # The default detector, batched and alone: the same graphs
    batched = machine_graphs({"max_batch": 8, "wait_ms": 20.0})
    alone = machine_graphs({"max_batch": 1})
    assert batched == alone


@pytest.mark.gpu
def test_one_unit_per_placement():
    # The detector on the GPU and on the CPU: two units of one lane, the same graphs
    cuda_and_cpu = machine_graphs({"devices": [DEVICE, "cpu"], "max_batch": 4})
    alone = machine_graphs({"max_batch": 1})
    assert cuda_and_cpu == alone
    slots = registry.build(
        {
            "vlm": {"entry": "replay", "trace": str(TRACE)},
            "detect": {"entry": "torch", "devices": [DEVICE, "cpu"]},
            "embed": {"entry": "replay", "trace": str(TRACE)},
            "cpu": {"entry": "python"},
        }
    )
    assert [n.device for n in slots["detect"].needs] == [DEVICE, "cpu"]
    assert len(slots["detect"].units) == 2


def test_placement_refuses_what_does_not_fit():
    if not CUDA:
        pytest.skip("no CUDA device to plan")
    cfg = {
        "engines": {
            "vlm": {"entry": "replay", "trace": str(TRACE)},
            "detect": {"entry": "torch", "devices": ["cuda:0"], "memory_gb": 10_000},
            "embed": {"entry": "replay", "trace": str(TRACE)},
        }
    }
    with pytest.raises(ValueError, match="cuda:0: needs"):
        build_parts(cfg)
