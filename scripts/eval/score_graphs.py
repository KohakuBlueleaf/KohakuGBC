"""Score predicted GBC graphs against the ground truth (``kohakugbc.eval``).

    python scripts/eval/score_graphs.py GT.jsonl PRED.jsonl --out report.json
        [--rows per_image.jsonl] [--text-model google/embeddinggemma-300m]

The structure track always runs; ``--text-model`` (sentence-transformers) adds the full track,
its baselines measured on the ground truth. A ground-truth image without a prediction scores as
an empty prediction; predictions for images not in the ground truth are counted (``unknown``)
and ignored.
"""

import argparse
import json
import sys

from kohakugbc.eval.score import TAUS, Options, evaluate
from kohakugbc.eval.spec import empty, read
from kohakugbc.eval.text import Embedder, Sims, baselines

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("pred")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", help="per-image scores and counts, as JSON lines")
    ap.add_argument("--taus", type=float, nargs="+", default=list(TAUS))
    ap.add_argument("--text-model", help="sentence-transformers model: adds the full track")
    a = ap.parse_args()

    gt, pred = read(a.gt), read(a.pred)
    sims, base = None, None
    if a.text_model:
        if SentenceTransformer is None:
            sys.exit("--text-model needs sentence-transformers")
        embed = Embedder(SentenceTransformer(a.text_model))
        embed.warm([*gt.values(), *pred.values()])
        base = baselines(list(gt.values()), embed)
        sims = Sims(embed, base)

    pairs = ((k, pred.get(k, empty()), g) for k, g in gt.items())
    report, rows = evaluate(pairs, Options(tuple(a.taus)), sims)
    report["missing"] = sum(k not in pred for k in gt)
    report["unknown"] = sum(k not in gt for k in pred)
    if base is not None:
        report["text_model"], report["text_baselines"] = a.text_model, base
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    if a.rows:
        with open(a.rows, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    line = f"images {report['images']}  structure {report['structure']['score']:.4f}"
    if "full" in report:
        line += f"  full {report['full']['score']:.4f}"
    print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
