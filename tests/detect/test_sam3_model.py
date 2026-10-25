"""GBC's own SAM 3 (``kohakugbc_models.detect.sam3``) against transformers' on recorded crops.

The structural check runs both in fp64: a re-implementation that computes the same thing
differs from transformers at fp64 rounding, far below what fp32 alone moves transformers' own
outputs (the bound is that fp32 difference scaled down, measured on the same pairs). A RoPE
applied on swapped axes must fail it. CPU checks: the text cache and the pass buckets."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kohakugbc.engines.torch.modules import Sam3Detector as Reference  # noqa: E402
from kohakugbc.engines.torch.modules import _rows  # noqa: E402
from kohakugbc.nodes.images import cut  # noqa: E402
from kohakugbc_models.detect.sam3 import vision  # noqa: E402
from kohakugbc_models.detect.sam3.model import Sam3Detector, fp8_recipe  # noqa: E402
from kohakugbc_models.kernels import tuning  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "traces" / "text"
CUDA = torch.cuda.is_available()


def crops(n: int) -> list[tuple]:
    meta = json.loads((FIXTURES / "meta.json").read_text(encoding="utf-8"))
    out = []
    for line in (FIXTURES / "detect.jsonl").read_text(encoding="utf-8").splitlines()[:n]:
        key = json.loads(json.loads(line)["key"])
        image = {"path": str(FIXTURES / meta["image_root"] / key["image"])}
        out.append((cut(image, key["box"]), key["texts"][:2]))
    return out


def transformers_rows(dtype, items) -> list[dict]:
    ref = Reference(max_units=1)
    ref.load("cuda:0", dtype)
    rows = []
    with torch.inference_mode():
        for patch, texts in items:
            batch = ref.prepare([{"patch": patch, "request": {"texts": texts}}])
            vision_out = ref.model.get_vision_features(
                pixel_values=batch["pixels"]["pixel_values"].to("cuda:0", dtype)
            )
            text = batch["text"].to("cuda:0")
            for u in range(len(texts)):
                o = ref.model(
                    vision_embeds=_rows(vision_out, torch.tensor([0], device="cuda:0")),
                    input_ids=text["input_ids"][u : u + 1],
                    attention_mask=text["attention_mask"][u : u + 1],
                )
                rows.append(torch.cat([o.pred_logits[0], o.pred_boxes[0].flatten()]).double())
    del ref
    torch.cuda.empty_cache()
    return rows


def our_rows(dtype, items) -> list[dict]:
    det = Sam3Detector(max_pairs=2, max_crops=1)
    det.load("cuda:0", dtype)
    rows = []
    with torch.inference_mode():
        for patch, texts in items:
            found = det.detect(det.encode_images([patch]), [(0, t) for t in texts])
            rows += [torch.cat([r["logits"], r["boxes"].flatten()]).double() for r in found]
    del det
    torch.cuda.empty_cache()
    return rows


def max_diff(a: list, b: list) -> float:
    return max((x - y).abs().max().item() for x, y in zip(a, b))


@pytest.mark.gpu
@pytest.mark.weights
def test_fp64_matches_transformers_structurally(monkeypatch):
    if not CUDA:
        pytest.skip("needs a GPU")
    items = crops(2)
    ruler = transformers_rows(torch.float64, items)
    fp32_move = max_diff(transformers_rows(torch.float32, items), ruler)
    ours = max_diff(our_rows(torch.float64, items), ruler)
    assert ours < fp32_move * 1e-3, (ours, fp32_move)
    # Negative: the rotary embedding on swapped (h, w) axes is seen
    tables = vision.rope_tables

    def swapped(grid, window, head_dim, theta, device):
        cos, sin = tables(grid, window, head_dim, theta, device)
        half = head_dim // 2
        return tuple(torch.cat([t[..., half:], t[..., :half]], dim=-1) for t in (cos, sin))

    monkeypatch.setattr(vision, "rope_tables", swapped)
    broken = max_diff(our_rows(torch.float64, items), ruler)
    assert broken > fp32_move * 1e-3, (broken, fp32_move)


@pytest.mark.gpu
@pytest.mark.weights
def test_graphed_fusion_equals_the_issued_one():
    """Each replay reads the pass's own inputs (copied into the graph) and runs the same ops:
    bit for bit the issued pass, on every crop after the capturing one."""
    if not CUDA:
        pytest.skip("needs a GPU")
    det = Sam3Detector(max_pairs=2, max_crops=1, fusion_graph=True)
    det.load("cuda:0", torch.bfloat16)
    issued = det.model.fuse
    items = crops(3)
    with torch.inference_mode():
        for patch, texts in items:
            feats = det.encode_images([patch])
            pairs = [(0, t) for t in texts]
            graphed = det.detect(feats, pairs)
            det.fuse, kept = issued, det.fuse
            plain = det.detect(feats, pairs)
            det.fuse = kept
            for g, p in zip(graphed, plain):
                assert all(torch.equal(g[k], p[k]) for k in ("logits", "presence", "boxes"))
    # One graph per pass size, captured once and replayed for every later pass of that size
    assert set(det.fuse.graphs) == {det.bucket(len(t)) for _, t in items}
    assert len(items) > len(det.fuse.graphs)


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.parametrize("recipe", [None, "best"])
def test_graphed_vision_equals_the_issued_one(recipe):
    """Each crop count's replay reads its own pixels and returns features that outlive the next
    replay of the same graph: bit for bit the eager pass on the kernels a graph is tuned for
    (``tuning.metric("graph")``), on every crop (bf16, and the card's fastest fp8 recipe), every
    crop count captured (no silent fallback to eager)."""
    if not CUDA:
        pytest.skip("needs a GPU")
    fp8 = fp8_recipe("cuda:0", "auto") if recipe else None
    if recipe and fp8 is None:
        pytest.skip("this card runs no fp8 recipe")
    det = Sam3Detector(max_pairs=2, max_crops=3, vision_graph=True, fp8=fp8, kernels=True)
    det.load("cuda:0", torch.bfloat16)
    patches = [p for p, _ in crops(7)]
    with torch.inference_mode():
        # Seven crops: passes of 3, 3, 1, so the 3-crop graph replays twice in one call
        graphed = det.encode_images(patches)
        graphed_positions = det.positions.clone()
        kept = det.vision_graphs
        det.vision_graphs = None
        with tuning.metric("graph"):
            plain = det.encode_images(patches)
        det.vision_graphs = kept
    assert len(graphed) == len(plain) == 7
    for i, (g, p) in enumerate(zip(graphed, plain)):
        assert set(g) == set(p) and all(torch.equal(g[k], p[k]) for k in g), i
    assert not torch.equal(graphed[0][2], graphed[3][2])
    assert torch.equal(graphed_positions, det.positions)
    assert set(kept.graphs) == {1, 2, 3} and all(v is not None for v in kept.graphs.values())


def test_vision_graph_and_vision_compile_are_exclusive():
    with pytest.raises(ValueError):
        Sam3Detector(vision_compile="default", vision_graph=True)


def test_text_cache_keeps_the_most_recent():
    det = Sam3Detector(text_cache=2)
    det.texts.update({"a": 1, "b": 2})
    det.encode_texts(["a"])  # cached: no model call, moves "a" to the end
    assert list(det.texts) == ["b", "a"]
    det.texts["c"] = 3
    det.encode_texts(["c"])
    assert list(det.texts) == ["a", "c"]


@pytest.mark.parametrize("pairs, bucket", [(1, 1), (3, 4), (16, 16), (17, 32), (40, 32)])
def test_compiled_passes_run_at_power_of_two_buckets(pairs, bucket):
    det = Sam3Detector(max_pairs=32, fusion_compile="default")
    assert det.bucket(pairs) == bucket
    assert Sam3Detector(max_pairs=32, fusion_graph=True).bucket(pairs) == bucket
    exact = Sam3Detector(max_pairs=32, fusion_graph=True, fusion_buckets="exact")
    assert exact.bucket(pairs) == pairs
    assert Sam3Detector(max_pairs=32).bucket(pairs) == pairs


def test_levels_follow_the_box_source():
    assert Sam3Detector(box_source="pred").levels == (2,)
    assert Sam3Detector(box_source="mask").levels == (0, 1, 2)


def test_crop_fixtures_load():
    items = crops(3)
    assert len(items) == 3 and all(isinstance(p, np.ndarray) for p, _ in items)
