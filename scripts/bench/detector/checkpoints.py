"""Detector checkpoints measured on the requests one GBC graph asked: the root's object names on
the whole image, then each entity's part names on its crop, as the program sends them.

Models: a transformers checkpoint (OWLv2, SAM 3, SAM3-LiteText: GBC's own detector classes),
``efficientsam3:<efficientvit|repvit|tinyvit>`` (the EfficientSAM3 package) or
``yoloe:<weights>`` (ultralytics YOLOE, e.g. ``yoloe:yoloe-26s-seg.pt``). A backend whose package
is not in this interpreter is unavailable: run the script with the interpreter that has it, into
the same ``<out>``.

Every (model, dtype, threads) runs in its own process, so its peak memory is its own. Per run,
writes ``<out>/<model>_<dtype>_t<threads>/``: ``result.json`` (settings, threads, load time,
first call, every repeat's per-request time, their medians, the process's peak working set, and
every request's detections: text, box in image fractions, score) and ``sheet.jpg`` (each
request's crop with its boxes at or above ``--show``). ``<out>/summary.json`` gathers every run
in ``<out>``. ``--dtypes``: bfloat16 runs a model in bfloat16 (EfficientSAM3: under autocast;
YOLOE runs float32 only).

    python scripts/bench/detector/checkpoints.py GRAPHS.jsonl IMAGE runs/bench/detectors \
        --models google/owlv2-base-patch16-ensemble facebook/sam3 --dtypes bfloat16 float32
"""

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from workloads import graph_file_requests, peak_gib

try:
    from kohakugbc.engines.torch.modules import Owlv2Detector, Sam3Detector
except ImportError:  # an interpreter without GBC: the other backends only
    Owlv2Detector = Sam3Detector = None
try:
    from huggingface_hub import hf_hub_download
    from sam3.model.sam3_image_processor import Sam3Processor as EfficientProcessor
    from sam3.model_builder import build_efficientsam3_image_model
except ImportError:
    build_efficientsam3_image_model = None
try:
    from ultralytics import YOLOE
except ImportError:
    YOLOE = None
try:
    from kohakuamx.torchpatch import patch_torch
except ImportError:
    patch_torch = None

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#9a6324"]
# EfficientSAM3 full models (stage 3 fine-tuned): checkpoint, backbone, its size, text encoder
EFFICIENT = {
    "efficientvit": ("efficientsam3_ft/efficientsam3_efficientvit.pt", "efficientvit", "b1"),
    "repvit": ("efficientsam3_ft/efficientsam3_repvit.pt", "repvit", "m1.1"),
    "tinyvit": ("efficientsam3_ft/efficientsam3_tinyvit.pt", "tinyvit", "11m"),
}
EFFICIENT_TEXT = "MobileCLIP-S0"
# The CPU setups: torch as imported; its OpenMP threads bound one per physical core (the
# environment KohakuAMX's optimize_torch sets, read once when torch loads); bound, with
# F.scaled_dot_product_attention and the norms routed to KohakuAMX's AMX kernels
SETUPS = ("none", "bind", "amx")


class Gbc:
    """GBC's own detector class (OWLv2, SAM 3, SAM3-LiteText) on one request."""

    def __init__(self, model: str, threshold: float):
        cls = Sam3Detector if "sam3" in model.lower() else Owlv2Detector
        self.det = cls(model, box_threshold=threshold)

    def load(self, device: str, dtype: str) -> None:
        self.det.load(device, DTYPES[dtype])

    def detect(self, patch: np.ndarray, texts: list[str]) -> dict:
        batch = self.det.prepare([{"patch": patch, "request": {"texts": texts}}])
        return self.det.forward(batch)[0]


class Efficient:
    """EfficientSAM3: the image encoder once per request, then one grounding pass per text."""

    def __init__(self, variant: str, threshold: float):
        self.variant, self.threshold = variant, threshold
        self.processor = None
        self.autocast = contextlib.nullcontext

    def load(self, device: str, dtype: str) -> None:
        path, backbone, size = EFFICIENT[self.variant]
        model = build_efficientsam3_image_model(
            checkpoint_path=hf_hub_download("Simon7108528/EfficientSAM3", path),
            device=device,
            backbone_type=backbone,
            model_name=size,
            text_encoder_type=EFFICIENT_TEXT,
            text_encoder_context_length=16,
            load_from_HF=False,
        ).to(device)
        self.processor = EfficientProcessor(
            model, device=device, confidence_threshold=self.threshold
        )
        if dtype != "float32":
            kind = device.split(":")[0]
            self.autocast = lambda: torch.autocast(kind, dtype=DTYPES[dtype])

    def detect(self, patch: np.ndarray, texts: list[str]) -> dict:
        raw = {"boxes": [], "scores": [], "labels": []}
        with self.autocast():
            state = self.processor.set_image(Image.fromarray(patch))
            for label, text in enumerate(texts):
                state = self.processor.set_text_prompt(text, state)
                for box, score in zip(state["boxes"].float().tolist(), state["scores"].tolist()):
                    raw["boxes"].append(box)
                    raw["scores"].append(score)
                    raw["labels"].append(label)
        return raw


class Yoloe:
    """Ultralytics YOLOE: the request's texts set as its classes, then one prediction."""

    def __init__(self, weights: str, threshold: float):
        self.weights, self.threshold = weights, threshold
        self.model = self.device = None

    def load(self, device: str, dtype: str) -> None:
        if dtype != "float32":
            raise ValueError("YOLOE runs float32 only")
        self.model, self.device = YOLOE(self.weights), device

    def detect(self, patch: np.ndarray, texts: list[str]) -> dict:
        self.model.set_classes(texts)
        # Ultralytics takes BGR arrays
        result = self.model.predict(
            patch[:, :, ::-1], conf=self.threshold, device=self.device, verbose=False
        )[0]
        return {
            "boxes": result.boxes.xyxy.tolist(),
            "scores": result.boxes.conf.tolist(),
            "labels": [int(c) for c in result.boxes.cls.tolist()],
        }


def detector(model: str, threshold: float):
    kind, _, name = model.partition(":")
    if kind == "efficientsam3" and name:
        if build_efficientsam3_image_model is None:
            raise SystemExit("EfficientSAM3 is not installed in this interpreter")
        return Efficient(name, threshold)
    if kind == "yoloe" and name:
        if YOLOE is None:
            raise SystemExit("ultralytics is not installed in this interpreter")
        return Yoloe(name, threshold)
    if Owlv2Detector is None:
        raise SystemExit("kohakugbc is not installed in this interpreter")
    return Gbc(model, threshold)


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device.startswith("xpu"):
        torch.xpu.synchronize(device)


def device_peak_gib(device: str) -> float | None:
    """Peak memory torch allocated on a CUDA or XPU device (None on the CPU)."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated(device) / 2**30
    if device.startswith("xpu"):
        return torch.xpu.max_memory_allocated(device) / 2**30
    return None


def one(args) -> int:
    """One (model, dtype, threads, setup) run in this process."""
    if args.threads[0]:
        torch.set_num_threads(args.threads[0])
    applied = {}
    if args.setups[0] == "amx":
        if patch_torch is None:
            raise SystemExit("kohakuamx is not installed in this interpreter")
        applied = patch_torch(omp=False)
    if args.setups[0] != "none":
        torch.set_num_interop_threads(1)
    requests = graph_file_requests(args.graphs, args.image)
    det = detector(args.models[0], args.threshold)
    start = time.perf_counter()
    det.load(args.device, args.dtypes[0])
    load_s = time.perf_counter() - start

    def call(r):
        sync(args.device)
        t = time.perf_counter()
        with torch.inference_mode():
            raw = det.detect(r["patch"], r["texts"])
        sync(args.device)
        return time.perf_counter() - t, raw

    first_s, raws = 0.0, []
    for r in requests:
        took, raw = call(r)
        first_s += took
        raws.append(raw)
    repeats = [[call(r)[0] for r in requests] for _ in range(args.repeat)]
    per_request = [statistics.median(col) for col in zip(*repeats)]

    detections = []
    for r, raw in zip(requests, raws):
        h, w = r["patch"].shape[:2]
        x0, y0, x1, y1 = r["box"] or [0.0, 0.0, 1.0, 1.0]
        found = []
        for box, score, label in zip(raw["boxes"], raw["scores"], raw["labels"]):
            frac = [
                x0 + box[0] / w * (x1 - x0),
                y0 + box[1] / h * (y1 - y0),
                x0 + box[2] / w * (x1 - x0),
                y0 + box[3] / h * (y1 - y0),
            ]
            found.append({"text": r["texts"][label], "box": frac, "score": score, "px": box})
        detections.append(
            {"request": r["name"], "box": r["box"], "texts": r["texts"], "found": found}
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.models[0],
        "dtype": args.dtypes[0],
        "device": args.device,
        "threads": torch.get_num_threads(),
        "setup": args.setups[0],
        "patched": applied,
        "omp_env": {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "KMP_", "MKL_"))},
        "torch": torch.__version__,
        "threshold": args.threshold,
        "load_s": load_s,
        "first_s": first_s,
        "repeats": repeats,
        "per_request_s": dict(zip([r["name"] for r in requests], per_request)),
        "workload_s": sum(per_request),
        "units": sum(len(r["texts"]) for r in requests),
        "peak_gib": peak_gib(),
        "device_peak_gib": device_peak_gib(args.device),
        "detections": detections,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    title = f"{args.models[0]}  {args.dtypes[0]}  {args.setups[0]}  (scores >= {args.show})"
    sheet(requests, raws, args.show, title, out / "sheet.jpg")
    return 0


def sheet(requests, raws, show: float, title: str, path: Path) -> None:
    """Each request's crop (at most 520 px tall) with its boxes at or above ``show``."""
    font = ImageFont.load_default(size=18)
    panels = []
    for r, raw in zip(requests, raws):
        img = Image.fromarray(np.ascontiguousarray(r["patch"]))
        scale = min(1.0, 520 / img.height)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
        draw = ImageDraw.Draw(img)
        for box, score, label in zip(raw["boxes"], raw["scores"], raw["labels"]):
            if score < show:
                continue
            colour = PALETTE[label % len(PALETTE)]
            xy = [c * scale for c in box]
            draw.rectangle(xy, outline=colour, width=3)
            text = f"{r['texts'][label]} {score:.2f}"
            left, top = xy[0] + 2, max(0, xy[1] - 22)
            draw.rectangle(draw.textbbox((left, top), text, font=font), fill=colour)
            draw.text((left, top), text, fill="white", font=font)
        panels.append((r["name"], img))
    width = sum(p.width for _, p in panels) + 12 * (len(panels) + 1)
    out = Image.new("RGB", (width, 520 + 70), "white")
    draw = ImageDraw.Draw(out)
    draw.text((12, 8), title, fill="black", font=ImageFont.load_default(size=22))
    x = 12
    for name, p in panels:
        draw.text((x, 40), name, fill="black", font=font)
        out.paste(p, (x, 64))
        x += p.width + 12
    out.save(path, quality=90)


def environment(setup: str, threads: int) -> dict:
    """A run's environment: bound setups pin torch's OpenMP threads before torch loads."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMP_", "KMP_"))}
    if setup != "none":
        env.update(
            OMP_PROC_BIND="TRUE", OMP_PLACES="cores", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE"
        )
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
            env["KOHAKU_AMX_THREADS"] = str(threads)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs")
    parser.add_argument("image")
    parser.add_argument("out")
    parser.add_argument("--models", nargs="+", default=["google/owlv2-base-patch16-ensemble"])
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16"], choices=sorted(DTYPES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[0],
        help="torch threads, one run each (0: its default)",
    )
    parser.add_argument(
        "--setups", nargs="+", default=["none"], choices=SETUPS, help="CPU setups, one run each"
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.05, help="lowest score kept")
    parser.add_argument("--show", type=float, default=0.3, help="sheet: lowest score drawn")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        return one(args)

    failed = []
    for model in args.models:
        for dtype in args.dtypes:
            for threads in args.threads:
                for setup in args.setups:
                    tag = model.split("/")[-1].replace(":", "-")
                    out = Path(args.out) / f"{tag}_{dtype}_t{threads}_{setup}"
                    command = [
                        sys.executable,
                        __file__,
                        args.graphs,
                        args.image,
                        str(out),
                        "--one",
                        "--models",
                        model,
                        "--dtypes",
                        dtype,
                        "--device",
                        args.device,
                        "--threads",
                        str(threads),
                        "--setups",
                        setup,
                        "--repeat",
                        str(args.repeat),
                        "--threshold",
                        str(args.threshold),
                        "--show",
                        str(args.show),
                    ]
                    if subprocess.run(command, env=environment(setup, threads)).returncode:
                        failed.append(
                            {"model": model, "dtype": dtype, "threads": threads, "setup": setup}
                        )

    runs = []
    for path in sorted(Path(args.out).glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        runs.append({k: v for k, v in result.items() if k != "detections"})
    summary = {"runs": runs, "failed_last_call": failed}
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for r in runs:
        print(
            f"{r['model']:40s} {r['dtype']:9s} t{r['threads']:<3d} {r.get('setup', 'none'):4s} "
            f"load {r['load_s']:6.1f} s  "
            f"first {r['first_s']:6.2f} s  workload {r['workload_s']:6.2f} s  "
            f"peak {r['peak_gib'] or 0:5.2f} GiB  device "
            f"{r.get('device_peak_gib') or 0:5.2f} GiB"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
