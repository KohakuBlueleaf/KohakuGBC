"""Detections of two ``stages.py`` runs side by side: per workload, how far each request's boxes
and scores moved.

``--min-score`` keeps the pairs whose first-run score reaches it (the instances GBC's selection
can keep). Instances pair in order within a request (the box sources keep the same instances in
the same order; dtypes may not, so pairs then follow the best IoU among same-label instances).
Writes the
IoU and score-difference distribution per workload, the counts of instances below IoU 0.9 / 0.5,
and the list of the worst pairs, to stdout and ``--out`` (JSON).

    python scripts/bench/detector/compare.py RUN/bfloat16_pred RUN/bfloat16_mask --out cmp.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def iou(a: list, b: list) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 1.0


def pairs(a: dict, b: dict, ordered: bool) -> list[tuple]:
    rows_a = list(zip(a["boxes"], a["scores"], a["labels"]))
    rows_b = list(zip(b["boxes"], b["scores"], b["labels"]))
    if ordered:
        return [(x, y) for x, y in zip(rows_a, rows_b)]
    out, free = [], list(rows_b)
    for x in rows_a:
        same = [y for y in free if y[2] == x[2]]
        if not same:
            continue
        y = max(same, key=lambda y: iou(x[0], y[0]))
        free.remove(y)
        out.append((x, y))
    return out


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {
        "n": len(values),
        "min": values[0],
        "p05": pick(0.05),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": values[-1],
    }


def compare(run_a: Path, run_b: Path, ordered: bool, min_score: float = 0.0) -> dict:
    a = json.loads((run_a / "stages.json").read_text(encoding="utf-8"))
    b = json.loads((run_b / "stages.json").read_text(encoding="utf-8"))
    out = {"a": str(run_a), "b": str(run_b), "workloads": {}}
    for key, wa in a["workloads"].items():
        wb = b["workloads"][key]
        ious, deltas, worst, counts = [], [], [], {"a": 0, "b": 0}
        for da, db in zip(wa["detections"], wb["detections"]):
            counts["a"] += len(da["scores"])
            counts["b"] += len(db["scores"])
            for x, y in pairs(da, db, ordered):
                if x[1] < min_score:
                    continue
                value = iou(x[0], y[0])
                ious.append(value)
                deltas.append(abs(x[1] - y[1]))
                worst.append((value, da["request"], da["texts"][x[2]], x[1], x[0], y[0]))
        worst.sort(key=lambda w: w[0])
        out["workloads"][key] = {
            "instances": counts,
            "iou": quantiles(ious),
            "score_abs_diff": quantiles(deltas),
            "below_0.9": sum(v < 0.9 for v in ious),
            "below_0.5": sum(v < 0.5 for v in ious),
            "worst": [
                {"iou": w[0], "request": w[1], "text": w[2], "score": w[3], "a": w[4], "b": w[5]}
                for w in worst[:10]
            ],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--out")
    parser.add_argument("--unordered", action="store_true", help="pair by best IoU per label")
    parser.add_argument(
        "--min-score", type=float, default=0.0, help="pairs whose a-score is at least"
    )
    args = parser.parse_args()
    result = compare(Path(args.a), Path(args.b), not args.unordered, args.min_score)
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    for key, w in result["workloads"].items():
        print(
            f"{key}: instances {w['instances']}, IoU {w['iou']}, |dscore| max "
            f"{w['score_abs_diff'].get('max')}, below 0.9: {w['below_0.9']}, "
            f"below 0.5: {w['below_0.5']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
