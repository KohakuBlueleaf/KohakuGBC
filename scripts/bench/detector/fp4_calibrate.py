"""Calibrate SAM 3's NVFP4 input scales: every linear input's largest amax (``fp8.Fp4Input``)
over the crops of a folder of images, written as the JSON ``Sam3Detector(fp4_scales=...)`` and
the card config's ``GBC_DETECT_FP4_SCALES`` read.

Each image gives its whole frame and its four quadrants; every crop is asked ``--texts`` (the
DETR encoder's inputs depend on them, the ViT's do not). The model runs uncaptured, the
producers taking each activation's own amax, so no warm-up input enters the maximum. Also
writes ``<out>.stats.json``: per input the amax, the median over the batches and their ratio.

    python scripts/bench/detector/fp4_calibrate.py FOLDER --out sam3_fp4.json [--images 64]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from workloads import pixels

from kohakugbc_models.detect.sam3.model import Sam3Detector
from kohakugbc_models.kernels import fp8

TEXTS = ["person", "dog", "car", "chair", "cup", "tree", "building", "sign"]


def crops(image) -> list:
    h, w = image.shape[:2]
    halves = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h)]
    halves.append((w // 2, h // 2, w, h))
    return [image] + [image[y0:y1, x0:x1] for x0, y0, x1, y1 in halves]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="facebook/sam3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--images", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--texts", nargs="+", default=TEXTS)
    p.add_argument("--encoder", action="store_true", help="the DETR encoder's MLPs in NVFP4 too")
    args = p.parse_args()

    det = Sam3Detector(
        args.model,
        max_crops=args.batch,
        max_pairs=16,
        kernels=True,
        fp8="nvfp4",
        fp8_encoder=args.encoder,
        capture="lazy",
        fp4_record=True,
    )
    det.load(args.device, torch.bfloat16)
    paths = sorted(Path(args.folder).glob("*.jpg"))[: args.images]
    patches = [c for path in paths for c in crops(pixels(path))]
    sites = fp8.fp4_inputs(det.model)
    history: dict[str, list[float]] = {n: [] for n in sites}
    with torch.inference_mode():
        for i in range(0, len(patches), args.batch):
            for site in sites.values():
                site.seen.zero_()
            payloads = [
                {"patch": c, "request": {"texts": args.texts}} for c in patches[i : i + args.batch]
            ]
            det.run(payloads)
            for name, amax in det.fp4_amax().items():
                history[name].append(amax)
    amax = {n: max(h) for n, h in history.items()}
    stats = {
        n: {"amax": max(h), "median": statistics.median(h), "ratio": max(h) / statistics.median(h)}
        for n, h in history.items()
        if h and statistics.median(h) > 0
    }
    out = Path(args.out)
    out.write_text(json.dumps(amax, indent=1), encoding="utf-8")
    out.with_suffix(".stats.json").write_text(
        json.dumps({"images": len(paths), "crops": len(patches), "inputs": stats}, indent=1),
        encoding="utf-8",
    )
    print(f"{len(amax)} inputs over {len(patches)} crops -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
