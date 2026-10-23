"""The embed slot's model, every implementation on the texts GBC embeds: speed and accuracy side
by side.

Texts: every text the traces embedded (``embed.jsonl``, with the recorded vectors) and every
name their detector requests carried (``detect.jsonl``: the names GBC may embed), unique.
Implementations (``--impls``):

- ``st-fp32``, ``st-bf16``: sentence-transformers (``SentenceEmbedder``, the current slot) in
  that dtype, padded batches;
- ``native-eager``: GBC's own EmbeddingGemma, packed stream, eager ops, issued per op;
- ``native-kernels``: the same with the kernel registry's varlen attention / segment mean;
- ``native-graphs``: the same replayed from CUDA graphs per stream bucket;
- a ``-model`` suffix: the residual stream in the model's dtype (else fp32).

Per implementation and ``--batch``: the wall time to embed every text (median of ``--repeat``
passes after a warm pass), texts/s, the device peak; every implementation's vectors profiled
against sentence-transformers in fp64 (``numerics.errors.profile``, plus the per-vector cosine
distribution), and against the vectors recorded in the traces. Writes ``<out>/embed.json`` and
``<out>/embed.md``.

    python scripts/bench/embed/embedder.py --model M --traces runs/traces/poc30 --out OUT
"""

import argparse
import base64
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from kohakugbc.engines.torch.modules import NativeEmbedder, SentenceEmbedder
from kohakugbc_models.numerics.errors import profile

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
ST_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def recorded(traces: str) -> dict[str, np.ndarray]:
    out = {}
    for path in sorted(Path(traces).glob("*/embed.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            out[row["text"]] = np.frombuffer(base64.b64decode(row["vector"]), np.float32)
    return out


def detected(traces: str) -> list[str]:
    """Every text the traces' detector requests named (the names GBC may embed), unique."""
    out = {}
    for path in sorted(Path(traces).glob("*/detect.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            for text in json.loads(json.loads(line)["key"])["texts"]:
                out[text] = None
    return list(out)


def build(impl: str, model: str, batch: int, dtype: str):
    if impl.startswith("st-"):
        return SentenceEmbedder(model, max_batch=batch), ST_DTYPES[impl.split("-")[1]]
    residual = "model" if impl.endswith("-model") else "float32"
    base = impl.removesuffix("-model")
    kernels, graphs = base != "native-eager", base == "native-graphs"
    native = NativeEmbedder(
        model, max_batch=batch, kernels=kernels, graphs=graphs, residual=residual
    )
    return native, DTYPES[dtype]


def embed_all(engine_model, texts: list[str], batch: int) -> np.ndarray:
    out = []
    for lo in range(0, len(texts), batch):
        part = texts[lo : lo + batch]
        with torch.inference_mode():
            raw = engine_model.forward(engine_model.prepare(part))
        out += engine_model.split(raw, len(part))
    return np.stack(out)


def timed(engine_model, texts: list[str], batch: int, device: str, repeat: int) -> tuple:
    embed_all(engine_model, texts, batch)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    walls = []
    for _ in range(repeat):
        start = time.perf_counter()
        vectors = embed_all(engine_model, texts, batch)
        torch.cuda.synchronize(device)
        walls.append((time.perf_counter() - start) * 1e3)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    return vectors, walls, peak


def cosines(got: np.ndarray, ref: np.ndarray) -> dict:
    g = got / np.linalg.norm(got, axis=1, keepdims=True).clip(1e-12)
    r = ref / np.linalg.norm(ref, axis=1, keepdims=True).clip(1e-12)
    c = (g * r).sum(axis=1)
    return {
        "min": float(c.min()),
        "p01": float(np.quantile(c, 0.01)),
        "median": float(np.median(c)),
    }


def ruler(model: str, texts: list[str], device: str) -> np.ndarray:
    emb = SentenceEmbedder(model, max_batch=16)
    emb.load(device, torch.float64)
    out = embed_all(emb, texts, 16)
    del emb
    gc.collect()
    torch.cuda.empty_cache()
    return out


def summary(rows: list[dict]) -> str:
    lines = [
        "# The embed slot's model",
        "",
        "| impl | batch | ms (median) | texts/s | peak GiB | vs fp64: abs mean | abs max "
        "| cos min | vs recorded: cos min |",
        "|" + " --- |" * 9,
    ]
    for r in rows:
        e = r["fp64"]
        lines.append(
            f"| {r['impl']} | {r['batch']} | {r['ms']:.1f} | {r['texts_per_s']:.0f} "
            f"| {r['peak_gib']:.2f} | {e['abs']['mean']:.3e} | {e['abs']['max']:.3e} "
            f"| {r['fp64_cos']['min']:.7f} | {r['recorded_cos']['min']:.7f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/embeddinggemma-300m")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", help="the native impls' dtype")
    parser.add_argument(
        "--impls",
        nargs="+",
        default=[
            "st-fp32",
            "st-bf16",
            "native-eager",
            "native-kernels",
            "native-graphs",
            "native-graphs-model",
        ],
    )
    parser.add_argument("--batch", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    book = recorded(args.traces)
    texts = list(dict.fromkeys(list(book) + detected(args.traces)))
    reference = np.stack([book[t] for t in book])
    with_record = [texts.index(t) for t in book]
    exact = ruler(args.model, texts, args.device)
    rows = []
    for impl in args.impls:
        for batch in args.batch:
            engine_model, dtype = build(impl, args.model, batch, args.dtype)
            engine_model.load(args.device, dtype)
            vectors, walls, peak = timed(engine_model, texts, batch, args.device, args.repeat)
            ms = statistics.median(walls)
            row = {
                "impl": impl,
                "batch": batch,
                "dtype": str(dtype),
                "texts": len(texts),
                "ms": ms,
                "ms_passes": walls,
                "texts_per_s": len(texts) / ms * 1e3,
                "peak_gib": peak,
                "fp64": profile(torch.from_numpy(vectors), torch.from_numpy(exact)),
                "fp64_cos": cosines(vectors, exact),
                "recorded_cos": cosines(vectors[with_record], reference),
            }
            rows.append(row)
            print(
                f"{impl} b{batch}: {ms:.1f} ms, {row['texts_per_s']:.0f} texts/s, "
                f"cos vs fp64 min {row['fp64_cos']['min']:.7f}",
                flush=True,
            )
            del engine_model
            gc.collect()
            torch.cuda.empty_cache()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "embed.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "embed.md").write_text(summary(rows), encoding="utf-8")
    print(summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
