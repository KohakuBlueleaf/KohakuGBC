"""The detect and embed models on the host CPU: GBC's own SAM 3 (eager ops) against
transformers' on the teaser's requests (4 crops, 12 pairs), and GBC's own EmbeddingGemma against
sentence-transformers on the traces' names; per dtype, median of ``--repeat`` passes after a warm
one. Threads come from the environment (``OMP_NUM_THREADS``, ``OMP_PROC_BIND``, ``OMP_PLACES``
set before the process starts), recorded with the rows. Writes ``<out>/cpu-<threads>.json`` and
``.md``.

    OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores python scripts/bench/cpu.py ...
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "detector"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "embed"))
from embedder import detected, embed_all, recorded
from workloads import graph_file_requests

from kohakugbc.engines.torch.modules import NativeEmbedder, SentenceEmbedder
from kohakugbc.engines.torch.modules import Sam3Detector as Transformers
from kohakugbc_models.detect.sam3.model import Sam3Detector as Ours

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def timed(fn, repeat: int) -> tuple:
    out = fn()
    walls = []
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn()
        walls.append((time.perf_counter() - start) * 1e3)
    return statistics.median(walls), walls, out


def detections(raws: list) -> int:
    return sum(len(r["boxes"]) for r in raws)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--embed-model", default="google/embeddinggemma-300m")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-crops", type=int, default=4, help="ours: crops per vision pass")
    parser.add_argument("--max-pairs", type=int, default=16, help="ours: pairs per fusion pass")
    parser.add_argument("--tag", default="", help="appended to the output file names")
    args = parser.parse_args()

    threads = torch.get_num_threads()
    env = {k: v for k, v in os.environ.items() if k.startswith(("OMP_", "KMP_"))}
    requests = graph_file_requests(args.graphs, args.image)
    payloads = [{"patch": r["patch"], "request": {"texts": r["texts"]}} for r in requests]
    texts = list(dict.fromkeys(list(recorded(args.traces)) + detected(args.traces)))
    rows = []
    with torch.inference_mode():
        for dtype in args.dtypes:
            ours = Ours(
                "facebook/sam3", kernels=True, max_crops=args.max_crops, max_pairs=args.max_pairs
            )
            ours.load("cpu", DTYPES[dtype])
            ms, walls, out = timed(lambda o=ours: o.run(payloads), args.repeat)
            rows.append(
                {
                    "model": f"sam3 ours ({args.max_crops} crops / {args.max_pairs} pairs)",
                    "dtype": dtype,
                    "ms": ms,
                    "walls": walls,
                    "detections": detections(out),
                }
            )
            del ours
            ref = Transformers("facebook/sam3", max_batch=4, max_units=4)
            ref.load("cpu", DTYPES[dtype])
            ms, walls, out = timed(lambda r=ref: r.forward(r.prepare(payloads)), args.repeat)
            rows.append(
                {
                    "model": "sam3 transformers",
                    "dtype": dtype,
                    "ms": ms,
                    "walls": walls,
                    "detections": detections(out),
                }
            )
            del ref
            for name, emb in (
                ("embed ours", NativeEmbedder(args.embed_model, max_batch=64)),
                ("embed sentence-transformers", SentenceEmbedder(args.embed_model, max_batch=64)),
            ):
                emb.load("cpu", DTYPES[dtype])
                ms, walls, _ = timed(lambda e=emb: embed_all(e, texts, 64), args.repeat)
                rows.append(
                    {
                        "model": name,
                        "dtype": dtype,
                        "ms": ms,
                        "walls": walls,
                        "texts": len(texts),
                        "texts_per_s": len(texts) / ms * 1e3,
                    }
                )
                del emb
            for row in rows[-4:]:
                print(row, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"threads": threads, "env": env, "cpu": os.cpu_count(), "rows": rows}
    name = f"cpu-{threads}{args.tag}"
    (out / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    lines = [
        f"# CPU, {threads} threads ({env})",
        "",
        "| model | dtype | ms | note |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        note = (
            f"{r['texts_per_s']:.0f} texts/s"
            if "texts_per_s" in r
            else (f"{r['detections']} detections")
        )
        lines.append(f"| {r['model']} | {r['dtype']} | {r['ms']:.0f} | {note} |")
    text = "\n".join(lines) + "\n"
    (out / f"{name}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
