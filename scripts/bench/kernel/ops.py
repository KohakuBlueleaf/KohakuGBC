"""SAM 3's ops (``Capture.kind`` names each call kind), every backend on real activations
captured from GBC's own SAM 3 on the teaser's detector requests (4 crops, their pairs).

Each case runs in each ``--dtypes`` (operands cast) through ``torch`` (the eager reference in
that dtype: the baseline), ``compile``, each kernel entry at its tuned config (the tuning table
holds every config's time) and, with ``--every-tile``, each attention tile on its own. Each
output is profiled against the eager reference in fp64 on the same values. Writes
``<out>/ops-<card>.json`` (rows) and ``<out>/ops-<card>.md`` (summary).

    python scripts/bench/kernel/ops.py --graphs G --image I --out runs/kernel
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "detector"))
from workloads import graph_file_requests

from kohakugbc_models.detect.sam3.model import Sam3Detector
from kohakugbc_models.detect.sam3.ops import EAGER_IMPLS, NAMES, Ops
from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.compile import compiled
from kohakugbc_models.kernels.bench import run_case, write
from kohakugbc_models.kernels.hardware import arch_of, card_key

try:
    from kohakugbc_models.kernels.backends.triton.attention import entry
    from kohakugbc_models.kernels.backends.triton.blocks import entry as blocks_entry
except ImportError:  # no Triton here: the other backends only
    entry = blocks_entry = None

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}
ATTENTION = ("attention", "vit_attention", "attention_rpb")


class Capture(Ops):
    """Eager ops that keep the first call of each case kind (operands cloned)."""

    def __init__(self):
        super().__init__()
        self.cases: dict[str, tuple] = {}
        for name in NAMES:
            setattr(self, name, self.wrap(name, getattr(self, name)))

    def wrap(self, name, fn):
        def run(*args, **kwargs):
            kind = self.kind(name, args)
            if kind not in self.cases:
                self.cases[kind] = (
                    name,
                    [a.clone() if torch.is_tensor(a) else a for a in args],
                    dict(kwargs),
                )
            return fn(*args, **kwargs)

        return run

    @staticmethod
    def stage(x: torch.Tensor) -> str:
        """``vit`` (1024 channels), ``enc`` (the 72 x 72 image tokens) or ``dec`` (queries)."""
        if x.shape[-1] == 1024:
            return "vit"
        return "enc" if x.shape[1] > 1000 else "dec"

    @classmethod
    def kind(cls, name: str, args) -> str:
        if name == "vit_attention":
            return "vit_window" if args[5] < args[0].shape[1] else "vit_global"
        if name == "attention_rpb":
            return "dec_vision_rpb"
        if name == "attention":
            mask = args[3] if len(args) > 3 else None
            if args[0].shape[2] > 1000:
                return "enc_text_cross" if mask is not None else "enc_self"
            return "dec_text_cross" if mask is not None else "dec_self"
        if name == "qkv_rope":
            return "qkv_window" if args[5] < args[0].shape[1] else "qkv_global"
        if name == "box_bias":
            return "dec_box_bias"
        if name == "mask_box":
            return "mask_box"
        if name == "linear_act":
            return f"{cls.stage(args[0])}_fc1_{args[3]}"
        return f"{cls.stage(args[0])}_add_norm"


def capture(
    model: str, requests: list, device: str, dtype: torch.dtype, pairs: int, box_source: str
) -> dict:
    det = Sam3Detector(model, max_pairs=pairs, box_source=box_source)
    det.load(device, dtype)
    cap = Capture()
    det.model.bind_ops(cap)
    payloads = [{"patch": r["patch"], "request": {"texts": r["texts"]}} for r in requests]
    with torch.inference_mode():
        det.run(payloads)
    return cap.cases


def candidates(name: str, every_tile: bool, device: str, dtype) -> dict:
    """Every backend that serves the op for this device and dtype (eager as ``torch``), at its
    tuned config."""
    ref = EAGER_IMPLS[name]
    out = {"torch": ref, "compile": compiled(name, ref)}
    query = registry.Query(name, arch_of(device), dtype)
    for backend, impl in registry.IMPLS.get(name, {}).items():
        if backend in ("triton", "tilelang") and registry.reason(backend, query, device) is None:
            out[backend] = impl.fn
    if every_tile and "triton" in out:
        module = entry if name in ATTENTION else blocks_entry
        tiles = {
            "vit_attention": entry.VIT_TILES,
            "linear_act": blocks_entry.GEMM_TILES,
            "qkv_rope": blocks_entry.GEMM_TILES + [{"path": "split"}],
            "add_layer_norm": blocks_entry.NORM_TILES,
        }.get(name, entry.TILES)
        for tile in tiles:
            label = ",".join(f"{k}={v}" for k, v in tile.items())
            out[f"triton:{label}"] = tile_runner(module, name, tile)
    return out


def tile_runner(module, name: str, tile: dict):
    """The entry's op with every choice bound to ``tile`` (the entry's chooser replaced)."""

    def run(*args, **kwargs):
        saved = dict(entry._BOUND)
        entry._BOUND.clear()
        chooser = "choose" if module is entry else "choose_in"
        original = getattr(module, chooser)
        setattr(module, chooser, lambda *a, **k: tile)
        try:
            return getattr(module, name)(*args, **kwargs)
        finally:
            setattr(module, chooser, original)
            entry._BOUND.update(saved)

    return run


def summary(rows: list) -> str:
    lines = [
        "# SAM 3 ops",
        "",
        "| case | dtype | candidate | ms | output | cosine | abs mean | abs max " "| rel median |",
        "|" + " --- |" * 9,
    ]
    for row in rows:
        for name, r in row["rows"].items():
            if "error" in r:
                lines.append(
                    f"| {row['case']} | {row['dtype']} | {name} | error | {r['error'][:80]} |"
                )
                continue
            for i, e in enumerate(r["errors"]):
                lines.append(
                    f"| {row['case']} | {row['dtype']} | {name} | {r['ms']:.3f} | {i} "
                    f"| {e['cosine']:.7f} | {e['abs']['mean']:.3e} | {e['abs']['max']:.3e} "
                    f"| {e['rel']['median']:.3e} |"
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"])
    parser.add_argument("--ops", nargs="+", default=list(NAMES), choices=NAMES)
    parser.add_argument("--every-tile", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=16, help="pairs per captured pass")
    parser.add_argument("--box-source", default="pred", help="mask captures mask_box")
    parser.add_argument(
        "--capture-dtype",
        default="bfloat16",
        choices=sorted(DTYPES),
        help="the model's dtype while capturing (fp16 before Ampere)",
    )
    args = parser.parse_args()

    registry.load_backends()
    requests = graph_file_requests(args.graphs, args.image)
    cases = capture(
        args.model,
        requests,
        args.device,
        DTYPES[args.capture_dtype],
        args.max_pairs,
        args.box_source,
    )
    rows = []
    for kind, (name, operands, kwargs) in sorted(cases.items()):
        if name not in args.ops:
            continue
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]

            def make(dtype, operands=operands):
                return [
                    (
                        a.to(dtype)
                        if torch.is_tensor(a) and a.is_floating_point() and a.dtype != torch.float32
                        else a
                    )
                    for a in operands
                ]

            def bind(fn, kwargs=kwargs):
                return lambda *call: fn(*call, **kwargs)

            row = run_case(
                name,
                kind,
                make,
                bind(EAGER_IMPLS[name]),
                {
                    n: bind(fn)
                    for n, fn in candidates(name, args.every_tile, args.device, dtype).items()
                },
                args.device,
                dtype,
            )
            row["shapes"] = [list(a.shape) for a in operands if torch.is_tensor(a)]
            rows.append(row)
            print(
                f"{kind} {dtype_name}: "
                + ", ".join(
                    f"{n} {r['ms']:.3f} ms" if "ms" in r else f"{n} error ({r['error'][:300]})"
                    for n, r in row["rows"].items()
                )
            )
    card = card_key(args.device)
    out = Path(args.out)
    write(rows, out / f"ops-{card}.json")
    (out / f"ops-{card}.md").write_text(summary(rows), encoding="utf-8")
    print(json.dumps({"cases": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
