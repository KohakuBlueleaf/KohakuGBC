"""SAM 3's device memory plan on this card (``kohakugbc_models.detect.sam3.memory``): resident
bytes, the vision pass's working bytes per crop count, the fusion pass's per pair count, the bytes
the graph pools and the text cache keep, fitted linearly and stored for the engine. One plan per
``--fp8`` x ``--text-device``; also writes ``<out>/memory-<card>.json`` / ``.md`` with every
measured point.

    python scripts/bench/detector/memory.py --graphs G --image I --out OUT --fp8 none row
"""

import argparse
import gc
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from workloads import graph_file_requests

from kohakugbc_models.detect.sam3 import memory
from kohakugbc_models.detect.sam3.model import Sam3Detector
from kohakugbc_models.kernels.hardware import card_key

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def peak(fn, device) -> int:
    """The peak allocated bytes above what was allocated when ``fn`` started."""
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) - base


def fitted(points: dict) -> tuple:
    xs = np.array(sorted(points))
    ys = np.array([points[x] for x in xs], dtype=np.float64)
    slope, base = np.polyfit(xs, ys, 1)
    return int(max(base, 0)), int(max(slope, 0))


def measure(args, recipe: str, text_device: str, requests: list) -> dict:
    device = args.device
    det = Sam3Detector(
        args.model,
        box_source=args.box_source,
        max_pairs=max(args.pairs),
        max_crops=max(args.crops),
        kernels=True,
        fusion_graph=True,
        fusion_buckets="exact",
        text_graph=True,
        fp8=None if recipe == "none" else recipe,
        text_device=None if text_device == "device" else text_device,
    )
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(device)
    det.load(device, DTYPES[args.dtype])
    resident = torch.cuda.memory_allocated(device) - before
    patches = [requests[i % len(requests)]["patch"] for i in range(max(args.crops))]
    texts = [t for r in requests for t in r["texts"]]
    vision, fusion = {}, {}
    with torch.inference_mode():
        pixels = det.preprocess(patches)
        for crops in args.crops:
            det.encode_pixels(pixels[:crops])  # warm: tuning, builds
            vision[crops] = peak(partial(det.encode_pixels, pixels[:crops]), device)
        crops_feats = det.encode_pixels(pixels[:1])
        kept_before = torch.cuda.memory_allocated(device)
        # Every pass size up to the cap: exact buckets capture one graph each
        for pairs in range(1, max(args.pairs) + 1):
            det.detect(crops_feats, [(0, texts[i % len(texts)]) for i in range(pairs)])
        for pairs in args.pairs:
            part = [(0, texts[i % len(texts)]) for i in range(pairs)]
            fusion[pairs] = peak(partial(det.detect, crops_feats, part), device)
        kept = torch.cuda.memory_allocated(device) - kept_before
        reserve = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    vision_base, per_crop = fitted(vision)
    fusion_base, per_pair = fitted(fusion)
    plan = memory.Plan(
        card=card_key(device),
        knobs={
            "dtype": args.dtype,
            "fp8": recipe,
            "text_device": text_device,
            "box_source": args.box_source,
        },
        resident=resident,
        vision_base=vision_base,
        per_crop=per_crop,
        fusion_base=fusion_base,
        per_pair=per_pair,
        kept=kept,
        reserve=reserve,
    )
    name = memory.plan_name(
        args.dtype,
        None if recipe == "none" else recipe,
        None if text_device == "device" else text_device,
        args.box_source,
    )
    memory.save(plan, device, name)
    row = {
        "plan": name,
        "resident_gib": resident / 2**30,
        "kept_gib": kept / 2**30,
        "reserve_gib": reserve / 2**30,
        "vision_mib": {c: v / 2**20 for c, v in vision.items()},
        "fusion_mib": {p: v / 2**20 for p, v in fusion.items()},
        "fit": {
            "vision_base_mib": vision_base / 2**20,
            "per_crop_mib": per_crop / 2**20,
            "fusion_base_mib": fusion_base / 2**20,
            "per_pair_mib": per_pair / 2**20,
        },
        "estimate_gib": {
            f"{c}x{p}": plan.estimate(c, p) / 2**30 for c in args.crops for p in args.pairs
        },
        "reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    del det
    gc.collect()
    torch.cuda.empty_cache()
    return row


def summary(rows: list, card: str) -> str:
    lines = [
        f"# SAM 3 memory plan: {card}",
        "",
        "| plan | resident GiB | kept GiB | reserve GiB | per crop MiB | per pair MiB "
        "| 4 crops x 16 pairs GiB |",
        "|" + " --- |" * 7,
    ]
    for r in rows:
        f = r["fit"]
        est = r["estimate_gib"].get("4x16", max(r["estimate_gib"].values()))
        lines.append(
            f"| {r['plan']} | {r['resident_gib']:.2f} | {r['kept_gib']:.2f} "
            f"| {r['reserve_gib']:.2f} | {f['per_crop_mib']:.0f} | {f['per_pair_mib']:.0f} "
            f"| {est:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--box-source", default="pred")
    parser.add_argument("--fp8", nargs="+", default=["none"])
    parser.add_argument("--text-device", nargs="+", default=["device", "cpu"])
    parser.add_argument("--crops", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--pairs", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    requests = graph_file_requests(args.graphs, args.image)
    rows = [measure(args, r, t, requests) for r in args.fp8 for t in args.text_device]
    card = card_key(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"memory-{card}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    text = summary(rows, card)
    (out / f"memory-{card}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
