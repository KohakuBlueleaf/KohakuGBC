"""GBC's own SAM 3 against transformers' on the same (crop, text) pairs, with fp64 as the ruler.

Per pair the outputs GBC reads: the query logits ``[200]``, the presence logit, the boxes
``[200, 4]`` (xyxy, 0..1) and, for ``--box-source mask``, the mask logits on w1. Rulers are
transformers' SAM 3 in fp64. Compared against them:

- ours in fp64 (the re-implementation itself: differences here are structural);
- transformers and ours in each ``--dtypes`` (ours must profile like transformers in the same
  dtype).

Also per dtype: the wall time and device peak of each model on each workload (the batching
each runs: transformers ``--batch`` requests per call, ours ``--max-pairs`` pairs per call).

Writes ``<out>/reference.json`` (per workload, model and output: the merged error profile and
the per-pair profiles) and ``<out>/summary.md``.
"""

import argparse
import gc
import json
import sys
import time
from functools import partial
from pathlib import Path

import torch
from workloads import graph_file_requests, trace_requests

from kohakugbc.engines.torch.modules import Sam3Detector as Transformers
from kohakugbc.engines.torch.modules import _rows
from kohakugbc_models.detect.sam3.model import Sam3Detector as Ours
from kohakugbc_models.numerics.errors import merge, profile

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}
FIELDS = ("logits", "presence", "boxes")


def transformers_pairs(det: Transformers, requests: list[dict], batch: int, masks: bool) -> list:
    out = []
    for start in range(0, len(requests), batch):
        part = requests[start : start + batch]
        payloads = [{"patch": r["patch"], "request": {"texts": r["texts"]}} for r in part]
        prepared = det.prepare(payloads)
        pixel_values = prepared["pixels"]["pixel_values"].to(det.device, det.dtype)
        vision = det.model.get_vision_features(pixel_values=pixel_values)
        text = prepared["text"].to(det.device)
        units = prepared["units"]
        for u in range(0, len(units), det.max_units):
            chunk = units[u : u + det.max_units]
            crop = torch.tensor([i for i, _, _ in chunk], device=det.device)
            o = det.model(
                vision_embeds=_rows(vision, crop),
                input_ids=text["input_ids"][u : u + len(chunk)],
                attention_mask=text["attention_mask"][u : u + len(chunk)],
            )
            for j in range(len(chunk)):
                row = {
                    "logits": o.pred_logits[j],
                    "presence": o.presence_logits[j],
                    "boxes": o.pred_boxes[j],
                }
                if masks:
                    row["masks"] = o.pred_masks[j]
                out.append({k: v.detach().double().cpu() for k, v in row.items()})
    return out


def our_pairs(det: Ours, requests: list[dict], masks: bool) -> list:
    crops = det.encode_images([r["patch"] for r in requests])
    pairs = [(i, t) for i, r in enumerate(requests) for t in r["texts"]]
    rows = det.detect(crops, pairs)
    keep = FIELDS + (("masks",) if masks else ())
    return [{k: row[k].detach().double().cpu() for k in keep} for row in rows]


def timed(fn, device: str) -> tuple:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    return (
        result,
        (time.perf_counter() - start) * 1e3,
        torch.cuda.max_memory_allocated(device) / 2**30,
    )


def compare(rows: list, rulers: list, fields) -> dict:
    """Per field: the profile of every pair's values together (``all``; a per-pair cosine of a
    one-value field such as the presence logit is only ever +-1), the merged per-pair profiles
    and the per-pair profiles."""
    out = {}
    for field in fields:
        got = [r[field].flatten() for r in rows if field in r]
        ref = [r[field].flatten() for r in rulers if field in r]
        per = [profile(g, f) for g, f in zip(got, ref)]
        out[field] = {
            "all": profile(torch.cat(got), torch.cat(ref)),
            "merged": merge(per),
            "pairs": per,
        }
    return out


def load(kind: str, args, dtype: str):
    # fp64 attention runs SDPA's math path, which holds the whole score matrix: one crop and
    # two pairs per call there
    ruler = dtype == "float64"
    if kind == "transformers":
        det = Transformers(
            args.model,
            max_batch=1 if ruler else args.batch,
            max_units=2 if ruler else args.max_units,
        )
    else:
        det = Ours(
            args.model,
            # the mask logits are compared: mask_full keeps them (mask reduces them in-kernel)
            box_source="mask_full" if args.box_source == "mask" else args.box_source,
            max_pairs=2 if ruler else args.max_pairs,
            max_crops=1 if ruler else 4,
            kernels=args.kernels and not ruler,
            fp8=kind.split("fp8-")[1].split("+")[0] if "fp8-" in kind else None,
            fp8_encoder=kind.endswith("+enc"),
        )
    det.load(args.device, DTYPES[dtype])
    return det


def run(kind: str, args, dtype: str, workloads: dict, masks: bool) -> dict:
    det = load(kind, args, dtype)
    batch = 1 if dtype == "float64" else args.batch
    out = {}
    with torch.inference_mode():
        for name, requests in workloads.items():
            want = masks and name == "w1"
            if kind == "transformers":
                fn = partial(transformers_pairs, det, requests, batch, want)
            else:
                fn = partial(our_pairs, det, requests, want)
            fn()  # warm-up
            rows, ms, peak = timed(fn, args.device)
            out[name] = {"rows": rows, "ms": ms, "peak_gib": peak}
    del det, fn
    gc.collect()
    torch.cuda.empty_cache()
    return out


def summary(result: dict) -> str:
    lines = ["# SAM 3: ours against transformers, fp64 ruler", ""]
    for name, w in result["workloads"].items():
        lines += [
            f"## {name}: {w['pairs']} pairs",
            "",
            "| model | dtype | field | cosine (all) | cosine min (pairs) | abs mean "
            "| abs median | abs max | rel median | ms | peak GiB |",
            "|" + " --- |" * 11,
        ]
        for run_key, r in w["runs"].items():
            model, dtype = run_key.split("/")
            for field, c in r["errors"].items():
                a, m = c["all"], c["merged"]
                lines.append(
                    f"| {model} | {dtype} | {field} | {a['cosine']:.9f} | {m['cosine_min']:.9f} "
                    f"| {a['abs']['mean']:.3e} | {a['abs']['median']:.3e} | {a['abs']['max']:.3e} "
                    f"| {a['rel']['median']:.3e} | {r['ms']:.1f} | {r['peak_gib']:.2f} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16", "float32"])
    parser.add_argument("--box-source", default="pred", choices=["pred", "mask"])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-units", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument(
        "--kernels",
        action="store_true",
        help="ours with the registry's ops (its fp64 run stays eager)",
    )
    parser.add_argument(
        "--fp8",
        nargs="*",
        default=[],
        choices=["row", "mx", "row+enc", "mx+enc"],
        help="also ours with the ViT (+enc: and the encoder's MLPs) in fp8 per recipe",
    )
    parser.add_argument(
        "--skip-fp64-ours",
        action="store_true",
        help="leave out ours in fp64 (the structural check)",
    )
    args = parser.parse_args()

    workloads = {
        "w1": graph_file_requests(args.graphs, args.image),
        "w2": trace_requests(args.trace),
    }
    masks = args.box_source == "mask"
    fields = FIELDS + (("masks",) if masks else ())
    rulers = run("transformers", args, "float64", workloads, masks)
    result = {"args": vars(args), "workloads": {}}
    for name, requests in workloads.items():
        result["workloads"][name] = {"pairs": sum(len(r["texts"]) for r in requests), "runs": {}}
    first = [] if args.skip_fp64_ours else [("ours", "float64")]
    kinds = ("transformers", "ours", *(f"ours-fp8-{r}" for r in args.fp8))
    for kind, dtype in first + [(k, d) for d in args.dtypes for k in kinds]:
        got = run(kind, args, dtype, workloads, masks)
        for name in workloads:
            ruler_rows = rulers[name]["rows"]
            result["workloads"][name]["runs"][f"{kind}/{dtype}"] = {
                "ms": got[name]["ms"],
                "peak_gib": got[name]["peak_gib"],
                "errors": compare(got[name]["rows"], ruler_rows, fields),
            }
    for name in workloads:
        result["workloads"][name]["runs"]["transformers/float64"] = {
            "ms": rulers[name]["ms"],
            "peak_gib": rulers[name]["peak_gib"],
            "errors": {},
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    text = summary(result)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
