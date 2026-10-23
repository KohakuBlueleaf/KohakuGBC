"""SAM 3 by stage: where a detector call's device time and memory go.

Runs GBC's ``Sam3Detector`` (transformers' SAM 3) per dtype and box source on a workload and
times every stage with CUDA events around its module calls, summed per stage label per pass:

- ``vision`` (``vit.embed``, ``vit.layer.window`` / ``vit.layer.global`` with their
  ``vit.attn.*``, ``vit.mlp``, ``vit.norm``; ``neck``), once per crop;
- ``text``, ``enc`` (``enc.self_attn``, ``enc.cross_attn``, ``enc.mlp``), ``dec``
  (``dec.self_attn``, ``dec.text_cross_attn``, ``dec.vision_cross_attn``, ``dec.mlp``,
  ``dec.rpb``), ``scoring``, ``mask`` (``mask.pixel_decoder``), per (crop, text) pair;
- ``rows`` (the per-pair copies of a crop's features) and ``post`` (``Sam3Detector.boxes``:
  the boxes of the box source).

The top-level stages (``vision``, ``rows``, ``text``, ``enc``, ``dec``, ``scoring``, ``mask``,
``post``) never nest, and each records the device memory it allocated above its start (its
working peak). Workloads: ``w1`` the requests one GBC graph asked, ``w2`` every detector request
of a recorded trace; each runs one request per call and ``--batch`` requests per call.

Writes ``<out>/<dtype>_<box source>/``: ``stages.json`` (per workload and batching: per stage
the median over ``--repeat`` passes of its summed ms and its calls, the stage's working peak,
per call the wall ms, the pass's peak allocated / reserved, and the detections of the last pass),
``ops_<workload>.txt`` (the torch profiler's op table by device time, one pass), ``trace_w1.json``
(chrome trace of one ``w1`` pass); ``<out>/env.json`` and ``<out>/summary.md``.

    python scripts/bench/detector/stages.py --model facebook/sam3 --graphs G.jsonl --image I.jpg \
        --trace tests/fixtures/traces/text --out runs/bench/detector/5090/d0
"""

import argparse
import gc
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from torch.profiler import ProfilerActivity, profile
from workloads import graph_file_requests, trace_requests

from kohakugbc.engines.torch import modules
from kohakugbc.engines.torch.modules import Sam3Detector

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
GLOBAL_LAYERS = (7, 15, 23, 31)
TOP = ("vision", "rows", "text", "enc", "dec", "scoring", "mask", "post", "fusion")
RULES = [
    (r"vision_encoder$", "vision"),
    (r"vision_encoder\.backbone\.embeddings$", "vit.embed"),
    (r"vision_encoder\.backbone\.layers\.(\d+)$", "vit.layer.{kind}"),
    (r"vision_encoder\.backbone\.layers\.(\d+)\.attention$", "vit.attn.{kind}"),
    (r"vision_encoder\.backbone\.layers\.\d+\.mlp$", "vit.mlp"),
    (r"vision_encoder\.backbone\.layers\.\d+\.layer_norm[12]$", "vit.norm"),
    (r"vision_encoder\.neck$", "neck"),
    (r"text_encoder$", "text"),
    (r"detr_encoder$", "enc"),
    (r"detr_encoder\.layers\.\d+\.self_attn$", "enc.self_attn"),
    (r"detr_encoder\.layers\.\d+\.cross_attn$", "enc.cross_attn"),
    (r"detr_encoder\.layers\.\d+\.mlp$", "enc.mlp"),
    (r"detr_decoder$", "dec"),
    (r"detr_decoder\.layers\.\d+\.self_attn$", "dec.self_attn"),
    (r"detr_decoder\.layers\.\d+\.text_cross_attn$", "dec.text_cross_attn"),
    (r"detr_decoder\.layers\.\d+\.vision_cross_attn$", "dec.vision_cross_attn"),
    (r"detr_decoder\.layers\.\d+\.mlp$", "dec.mlp"),
    (r"dot_product_scoring$", "scoring"),
    (r"mask_decoder$", "mask"),
    (r"mask_decoder\.pixel_decoder$", "mask.pixel_decoder"),
]


class Stages:
    """CUDA events around every labelled call; working peaks of the top-level stages."""

    def __init__(self, device: str):
        self.device = device
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.open: list[tuple[str, torch.cuda.Event]] = []
        self.working: dict[str, float] = {}
        self.start_bytes = 0
        self.peak_bytes = 0

    def fold(self) -> int:
        """The allocator's peak since the last reset folded into the pass peak, then reset."""
        self.peak_bytes = max(self.peak_bytes, torch.cuda.max_memory_allocated(self.device))
        torch.cuda.reset_peak_memory_stats(self.device)
        return self.peak_bytes

    def enter(self, label: str) -> None:
        if label in TOP:
            self.fold()
            self.start_bytes = torch.cuda.memory_allocated(self.device)
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.open.append((label, start))

    def leave(self, label: str) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        open_label, start = self.open.pop()
        assert open_label == label, (open_label, label)
        self.events.append((label, start, end))
        if label in TOP:
            peak = torch.cuda.max_memory_allocated(self.device) - self.start_bytes
            self.working[label] = max(self.working.get(label, 0.0), peak / 2**30)

    def take(self) -> dict:
        """Summed ms and calls per label since the last take (synchronises)."""
        torch.cuda.synchronize(self.device)
        out: dict[str, dict] = {}
        for label, start, end in self.events:
            row = out.setdefault(label, {"ms": 0.0, "calls": 0})
            row["ms"] += start.elapsed_time(end)
            row["calls"] += 1
        self.events.clear()
        return out

    def hook(self, model: torch.nn.Module, rules=None) -> None:
        for name, module in model.named_modules():
            for pattern, label in RULES if rules is None else rules:
                match = re.match(pattern, name)
                if not match:
                    continue
                if "{kind}" in label:
                    kind = "global" if int(match.group(1)) in GLOBAL_LAYERS else "window"
                    label = label.format(kind=kind)
                module.register_forward_pre_hook(lambda m, a, label=label: self.enter(label))
                module.register_forward_hook(lambda m, a, o, label=label: self.leave(label))

    def wrap(self, owner, attr: str, label: str) -> None:
        fn = getattr(owner, attr)

        def wrapped(*args, **kwargs):
            self.enter(label)
            try:
                return fn(*args, **kwargs)
            finally:
                self.leave(label)

        setattr(owner, attr, wrapped)


def detections(requests: list[dict], raws: list[dict]) -> list[dict]:
    return [
        {
            "request": r["name"],
            "texts": r["texts"],
            "boxes": raw["boxes"],
            "scores": raw["scores"],
            "labels": raw["labels"],
        }
        for r, raw in zip(requests, raws)
    ]


def one_pass(det: Sam3Detector, stages: Stages, requests: list[dict], batch: int, device: str):
    calls, raws = [], []
    torch.cuda.reset_peak_memory_stats(device)
    stages.peak_bytes = 0
    for i in range(0, len(requests), batch):
        part = requests[i : i + batch]
        payloads = [{"patch": r["patch"], "request": {"texts": r["texts"]}} for r in part]
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            out = det.forward(det.prepare(payloads))
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


def run_dtype(args, dtype: str, source: str, workloads: dict) -> dict:
    device = args.device
    det = Sam3Detector(
        args.model, max_batch=args.batch, max_units=args.max_units, box_source=source
    )
    start = time.perf_counter()
    det.load(device, DTYPES[dtype])
    load_s = time.perf_counter() - start
    stages = Stages(device)
    stages.hook(det.model)
    stages.wrap(modules, "_rows", "rows")
    stages.wrap(det, "boxes", "post")
    resident = torch.cuda.memory_allocated(device) / 2**30
    out_dir = Path(args.out) / f"{dtype}_{source}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dtype": dtype,
        "box_source": source,
        "load_s": load_s,
        "resident_gib": resident,
        "workloads": {},
    }
    for wname, requests in workloads.items():
        for batch in sorted({1, args.batch}):
            one_pass(det, stages, requests, batch, device)  # warm-up
            stages.working.clear()
            passes = [one_pass(det, stages, requests, batch, device) for _ in range(args.repeat)]
            labels = sorted({label for _, sums, _, _ in passes for label in sums})
            summed = {
                label: {
                    "ms": statistics.median(p[1].get(label, {"ms": 0.0})["ms"] for p in passes),
                    "calls": passes[0][1].get(label, {"calls": 0})["calls"],
                    "working_gib": stages.working.get(label),
                }
                for label in labels
            }
            walls = [sum(c["ms"] for c in p[0]) for p in passes]
            result["workloads"][f"{wname}_b{batch}"] = {
                "requests": len(requests),
                "pairs": sum(len(r["texts"]) for r in requests),
                "wall_ms": statistics.median(walls),
                "wall_ms_passes": walls,
                "stages": summed,
                "calls": passes[-1][0],
                "peak": max((p[2] for p in passes), key=lambda p: p["allocated_gib"]),
                "detections": detections(requests, passes[-1][3]),
            }

        # One profiled pass per workload at the largest batching
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            one_pass(det, stages, requests, args.batch, device)
        stages.take()
        table = prof.key_averages().table(sort_by="device_time_total", row_limit=80)
        (out_dir / f"ops_{wname}.txt").write_text(table, encoding="utf-8")
        if wname == "w1":
            prof.export_chrome_trace(str(out_dir / "trace_w1.json"))

    (out_dir / "stages.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    # The wrapped methods close over the detector: a collection frees it before the next run
    del det, stages
    gc.collect()
    torch.cuda.empty_cache()
    return result


def summary(results: list[dict]) -> str:
    lines = ["# SAM 3 stages", ""]
    for key in results[0]["workloads"]:
        w0 = results[0]["workloads"][key]
        lines += [f"## {key}: {w0['requests']} requests, {w0['pairs']} pairs", ""]
        head = "| stage | " + " | ".join(
            f"{r['dtype']} {r['box_source']} ms (calls) | working GiB" for r in results
        )
        lines += [head + " |", "|" + " --- |" * (1 + 2 * len(results))]
        labels = sorted(
            {label for r in results for label in r["workloads"][key]["stages"]},
            key=lambda s: (s.split(".")[0] not in TOP, s),
        )
        for label in labels:
            cells = []
            for r in results:
                row = r["workloads"][key]["stages"].get(label)
                if row is None:
                    cells += ["", ""]
                    continue
                working = "" if row["working_gib"] is None else f"{row['working_gib']:.2f}"
                cells += [f"{row['ms']:.1f} ({row['calls']})", working]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        walls = " | ".join(
            f"{r['workloads'][key]['wall_ms']:.1f} | "
            f"{r['workloads'][key]['peak']['allocated_gib']:.2f} alloc / "
            f"{r['workloads'][key]['peak']['reserved_gib']:.2f} reserved"
            for r in results
        )
        lines += [f"| **wall** | {walls} |", ""]
    lines += ["| dtype | box source | load s | resident GiB |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {r['dtype']} | {r['box_source']} | {r['load_s']:.1f} | {r['resident_gib']:.2f} |"
        for r in results
    ]
    return "\n".join(lines) + "\n"


def environment(device: str) -> dict:
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.max.sm,power.limit",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "nvidia_smi": smi,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--trace", required=True, help="a recorded trace folder (detect.jsonl)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16", "float32"])
    parser.add_argument("--box-sources", nargs="+", default=["pred"])
    parser.add_argument("--batch", type=int, default=4, help="requests per call (besides 1)")
    parser.add_argument("--max-units", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    workloads = {
        "w1": graph_file_requests(args.graphs, args.image),
        "w2": trace_requests(args.trace),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    env = environment(args.device) | {"args": vars(args)}
    (Path(args.out) / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    results = [
        run_dtype(args, dtype, source, workloads)
        for dtype in args.dtypes
        for source in args.box_sources
    ]
    text = summary(results)
    (Path(args.out) / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
