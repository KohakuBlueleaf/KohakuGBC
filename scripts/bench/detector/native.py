"""GBC's own SAM 3 (``kohakugbc_models.detect.sam3``) by stage, in the layout of ``stages.py``, so
its time, peaks and detections set directly against transformers' (``compare.py``).

Per (dtype, ``--max-pairs``, ``--fusion-compile``, ``--ops``, ``--fusion-graph``): each
workload in calls of ``--batch`` requests through
``Sam3Detector.run`` (images encoded once per crop, texts once per unique text, pairs
``max_pairs`` per fusion pass, ``--max-crops`` per vision pass, boxes of ``--box-source``). The
text cache is cleared before every pass, so each pass encodes its texts as a cold engine would
(``--text-cache warm``: kept, as a running engine's); the pixels are made inside the timed call
(``--prep outside``: before it, as the engine's prep threads do).

Writes ``<out>/<dtype>_<box source>_p<max pairs>/stages.json`` and ``<out>/summary.md``.

    python scripts/bench/detector/native.py --graphs G --image I --trace T --out OUT \
        --max-pairs 4 16 64
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from stages import RULES, Stages, environment, summary
from workloads import graph_file_requests, recorded_requests, trace_requests

from kohakugbc_models.detect.sam3.model import Sam3Detector

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
FUSION = ("enc", "dec", "scoring")
VISION = ("vision", "vit", "neck")


def one_pass(
    det: Sam3Detector,
    stages: Stages,
    requests: list,
    batch: int,
    device: str,
    warm_texts: bool = False,
    prep_outside: bool = False,
):
    """``warm_texts``: the text cache kept from the last pass (a running engine's), else cleared;
    ``prep_outside``: the pixels made and pinned before the timer (the engine's prep threads do
    it beside the forward), else inside it."""
    calls, raws = [], []
    if not warm_texts:
        det.texts.clear()
    torch.cuda.reset_peak_memory_stats(device)
    stages.peak_bytes = 0
    for i in range(0, len(requests), batch):
        part = requests[i : i + batch]
        payloads = [{"patch": r["patch"], "request": {"texts": r["texts"]}} for r in part]
        pixels = det.pin(det.preprocess([p["patch"] for p in payloads])) if prep_outside else None
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            out = det.run(payloads, pixels=pixels)
        torch.cuda.synchronize(device)
        calls.append(
            {
                "requests": [r["name"] for r in part],
                "pairs": sum(len(r["texts"]) for r in part),
                "ms": (time.perf_counter() - start) * 1e3,
            }
        )
        raws += out
    peak = {
        "allocated_gib": stages.fold() / 2**30,
        "reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    return calls, stages.take(), peak, raws


def run_config(
    args,
    dtype: str,
    pairs: int,
    compiled: str,
    kernels: str,
    graph: str,
    vision: str,
    text: str,
    recipe: str,
    crops: int,
    workloads: dict,
) -> dict:
    device = args.device
    mode = None if compiled == "none" else compiled
    vision_mode = None if vision in ("none", "graph") else vision
    det = Sam3Detector(
        args.model,
        box_source=args.box_source,
        max_pairs=pairs,
        max_crops=crops,
        vision_graph=vision == "graph",
        fusion_compile=mode,
        kernels=kernels == "kernels",
        fusion_graph=graph != "eager",
        fusion_buckets="exact" if graph == "graph-exact" else "pow2",
        vision_compile=vision_mode,
        text_graph=text == "graph",
        fp8=None if recipe.startswith("none") else recipe.split("+")[0],
        fp8_encoder=recipe.endswith("+enc"),
    )
    start = time.perf_counter()
    det.load(device, DTYPES[dtype])
    load_s = time.perf_counter() - start
    stages = Stages(device)
    # A compiled or graphed pass is timed whole (module hooks inside it would break it)
    whole = mode or graph != "eager"
    vision_whole = vision != "none"
    rules = [
        r
        for r in RULES
        if not (whole and r[1].split(".")[0] in FUSION)
        and not (vision_whole and r[1].split(".")[0] in VISION)
        and not (text == "graph" and r[1] == "text")
    ]
    stages.hook(det.model, rules)
    stages.wrap(det, "encode_texts", "text")
    stages.wrap(det, "fuse", "fusion")
    if vision_whole:
        stages.wrap(det, "encode_pixels", "vision")
    resident = torch.cuda.memory_allocated(device) / 2**30
    tag = (
        f"{args.box_source} p{pairs} c{crops} {compiled} {kernels} {graph} vision-{vision} "
        f"text-{text} fp8-{recipe}"
    )
    result = {
        "dtype": dtype,
        "box_source": tag,
        "load_s": load_s,
        "resident_gib": resident,
        "workloads": {},
    }
    modes = {"warm_texts": args.text_cache == "warm", "prep_outside": args.prep == "outside"}
    for wname, requests in workloads.items():
        # A batch below the crop cap runs the same passes as crops == batch: measured there
        for batch in (b for b in args.batch if b >= crops):
            one_pass(det, stages, requests, batch, device, **modes)  # warm-up
            stages.working.clear()
            passes = [
                one_pass(det, stages, requests, batch, device, **modes) for _ in range(args.repeat)
            ]
            labels = sorted({label for _, sums, _, _ in passes for label in sums})
            result["workloads"][f"{wname}_b{batch}"] = {
                "crops_per_pass": min(crops, batch),
                "requests": len(requests),
                "pairs": sum(len(r["texts"]) for r in requests),
                "wall_ms": statistics.median(sum(c["ms"] for c in p[0]) for p in passes),
                "wall_ms_passes": [sum(c["ms"] for c in p[0]) for p in passes],
                "stages": {
                    label: {
                        "ms": statistics.median(p[1].get(label, {"ms": 0.0})["ms"] for p in passes),
                        "calls": passes[0][1].get(label, {"calls": 0})["calls"],
                        "working_gib": stages.working.get(label),
                    }
                    for label in labels
                },
                "calls": passes[-1][0],
                "peak": max((p[2] for p in passes), key=lambda p: p["allocated_gib"]),
                "detections": [
                    {
                        "request": r["name"],
                        "texts": r["texts"],
                        "boxes": raw["boxes"],
                        "scores": raw["scores"],
                        "labels": raw["labels"],
                    }
                    for r, raw in zip(requests, passes[-1][3])
                ],
            }
    out_dir = (
        Path(args.out) / f"{dtype}_{args.box_source}_p{pairs}_c{crops}_{compiled}_{kernels}_{graph}"
        f"_vision-{vision}_text-{text}_fp8-{recipe}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stages.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    del det, stages
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--graphs", help="w1: a graphs.jsonl (with --image)")
    parser.add_argument("--image")
    parser.add_argument("--trace", help="w2: a recorded trace")
    parser.add_argument("--recorded", help="rec: requests record.py saved from a machine run")
    parser.add_argument("--recorded-limit", type=int, default=0)
    parser.add_argument("--max-crops", type=int, nargs="+", default=[4])
    parser.add_argument("--text-cache", default="cold", choices=["cold", "warm"])
    parser.add_argument("--prep", default="inside", choices=["inside", "outside"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16"])
    parser.add_argument("--box-source", default="pred")
    parser.add_argument("--max-pairs", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument(
        "--fusion-compile",
        nargs="+",
        default=["none"],
        help="none, default, reduce-overhead (one run each)",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["eager"],
        choices=["eager", "kernels"],
        help="eager ops, or the kernel registry's (one run each)",
    )
    parser.add_argument(
        "--fusion-graph",
        nargs="+",
        default=["eager"],
        choices=["eager", "graph", "graph-exact"],
        help="fusion issued per op, or replayed from CUDA graphs per pow2 / exact pass size",
    )
    parser.add_argument(
        "--vision-compile",
        nargs="+",
        default=["none"],
        help="none, graph (CUDA graphs per crop count), or a compile mode: default,"
        " max-autotune-no-cudagraphs (the vision encoder; one run each)",
    )
    parser.add_argument(
        "--text-graph",
        nargs="+",
        default=["eager"],
        choices=["eager", "graph"],
        help="the text encoder issued per op, or replayed from CUDA graphs (one run each)",
    )
    parser.add_argument(
        "--fp8",
        nargs="+",
        default=["none"],
        choices=["none", "row", "mx", "row+enc", "mx+enc"],
        help="the ViT's linears in bf16 or fp8 per recipe; +enc: the DETR encoder's MLPs too",
    )
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    workloads = {}
    if args.graphs:
        workloads["w1"] = graph_file_requests(args.graphs, args.image)
    if args.trace:
        workloads["w2"] = trace_requests(args.trace)
    if args.recorded:
        workloads["rec"] = recorded_requests(args.recorded, args.recorded_limit)
    if not workloads:
        parser.error("no workload: --graphs/--image, --trace or --recorded")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    env = environment(args.device) | {"args": vars(args)}
    (Path(args.out) / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    results = [
        run_config(
            args, dtype, pairs, compiled, kernels, graph, vision, text, recipe, crops, workloads
        )
        for dtype in args.dtypes
        for pairs in args.max_pairs
        for compiled in args.fusion_compile
        for kernels in args.ops
        for graph in args.fusion_graph
        for vision in args.vision_compile
        for text in args.text_graph
        for recipe in args.fp8
        for crops in args.max_crops
        if not (compiled != "none" and graph != "eager")
    ]
    text = summary(results)
    (Path(args.out) / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
