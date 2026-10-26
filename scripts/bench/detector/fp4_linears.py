"""SAM 3's ViT linears one by one on the card: bf16, fp8 ``mx`` and NVFP4, each with the
producer that feeds it (NVFP4: the amax taken per call, or a static ``Fp4Input``), and the NVFP4
GEMM through torch's scaled matmul (bias in its epilogue or not) or FlashInfer's CUTLASS / cuDNN
FP4 GEMM.

Rows are the ViT's tokens at ``--crops`` crops (72^2 each). Each case is one CUDA graph of
``--nodes`` calls; times are microseconds per call.

    python scripts/bench/detector/fp4_linears.py --out fp4_linears.json [--crops 8]
"""

import argparse
import json
import sys
from pathlib import Path

import flashinfer
import torch
import torch.nn.functional as F

from kohakugbc_models.kernels import fp8, registry

HIDDEN, INNER = 1024, 4736
# (name, K, N, the producer that feeds it)
LINEARS = [
    ("qkv", HIDDEN, 3 * HIDDEN, "add_ln_quant"),
    ("o", HIDDEN, HIDDEN, "quant"),
    ("fc1", HIDDEN, INNER, "add_ln_quant"),
    ("fc2", INNER, HIDDEN, "act_quant"),
]


def graph_us(call, nodes: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            call()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) * 1e3 / (5 * nodes), 2)


def producer(name: str, x, r, w, b, recipe: str, site=None, backend: str = "triton"):
    op = registry.resolve(name, x.device, x.dtype, backend=backend)
    if name == "quant":
        return lambda: op(x, recipe, site)
    if name == "act_quant":
        return lambda: op(x, "gelu", recipe, site)
    return lambda: op(x, r, w, b, 1e-6, recipe, site)[1:]


def agreement(ours: tuple, eager: tuple, rows: int) -> dict:
    """The share of codes and of the real rows' scale bytes equal to the eager reference's,
    and the tensor scales' relative difference."""
    q, (s, inv) = ours
    eq, (es, einv) = eager
    real = es.view(torch.uint8).view(-1, 128, es.numel() // (-(-rows // 128) * 128))
    mine = s.view(torch.uint8).view_as(real)
    return {
        "codes": (q == eq).float().mean().item(),
        "scales": (mine == real).float().mean().item(),
        "inv_gs_rel": ((inv - einv).abs() / einv.abs()).item(),
    }


def case(name: str, k: int, n: int, prod: str, rows: int, nodes: int) -> dict:
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    lw = torch.rand(k, device="cuda", dtype=torch.bfloat16) + 0.5
    lb = torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1
    wt = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.1
    row = {"linear": name, "m": rows, "k": k, "n": n, "producer": prod, "us": {}, "tflops": {}}
    us = row["us"]
    us["bf16"] = graph_us(lambda: F.linear(x, wt, bias), nodes)
    eager = producer(prod, x, r, lw, lb, "nvfp4", backend="eager")()
    static = fp8.Fp4Input().cuda()
    static.set(float(eager[1][1]) * fp8.E4M3_MAX * fp8.E2M1_MAX)
    row["match"] = {}
    for tag, recipe, site in (("mx", "mx", None), ("nvfp4", "nvfp4", None)) + (
        ("nvfp4_static", "nvfp4", static),
    ):
        make = producer(prod, x, r, lw, lb, recipe, site)
        xq, xs = make()
        if recipe == "nvfp4":
            row["match"][tag] = agreement((xq, xs), eager, rows)
        wq, ws = fp8.quantize_weight(wt, recipe)
        us[f"{tag}_producer"] = graph_us(make, nodes)
        if tag == "nvfp4_static":
            continue
        operands = (xq, xs, wq, ws, recipe)
        us[f"{recipe}_gemm_bias"] = graph_us(lambda o=operands: fp8.mm(*o, bias), nodes)
        us[f"{recipe}_gemm"] = graph_us(lambda o=operands: fp8.mm(*o), nodes)
    xq, xs = producer(prod, x, r, lw, lb, "nvfp4")()
    wq, ws = fp8.quantize_weight(wt, "nvfp4")
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    us["bias_add"] = graph_us(lambda: out.add_(bias), nodes)
    alpha = (xs[1] * ws[1]).float()
    for backend in ("cutlass", "cudnn"):
        try:
            us[f"nvfp4_flashinfer_{backend}"] = graph_us(
                lambda backend=backend: flashinfer.mm_fp4(
                    xq,
                    wq.t(),
                    xs[0].view(torch.uint8).view(rows + (-rows % 128), -1).view(fp8.E4M3),
                    ws[0].view(torch.uint8).view(n + (-n % 128), -1).view(fp8.E4M3).t(),
                    alpha,
                    torch.bfloat16,
                    backend=backend,
                ),
                nodes,
            )
        except Exception as e:  # noqa: BLE001 - a backend the build lacks is reported, not fatal
            row.setdefault("errors", {})[backend] = repr(e)[:300]
    try:
        F.scaled_mm(
            xq.view(torch.float4_e2m1fn_x2),
            wq.view(torch.float4_e2m1fn_x2).t(),
            [xs[0].flatten(), xs[1]],
            [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            [ws[0].flatten(), ws[1]],
            [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            bias=bias,
            output_dtype=torch.bfloat16,
        )
        row["torch_fp4_bias_epilogue"] = True
    except Exception as e:  # noqa: BLE001 - whether torch takes the bias is the finding
        row["torch_fp4_bias_epilogue"] = repr(e)[:300]
    flops = 2 * rows * k * n
    gemms = ("bf16", "mx_gemm", "nvfp4_gemm", "nvfp4_flashinfer_cutlass", "nvfp4_flashinfer_cudnn")
    for key in gemms:
        if key in us:
            row["tflops"][key] = round(flops / us[key] / 1e6, 1)
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--crops", type=int, nargs="+", default=[1, 8])
    p.add_argument("--nodes", type=int, default=16)
    args = p.parse_args()
    report = {"gpu": torch.cuda.get_device_name(), "rows": []}
    for crops in args.crops:
        for name, k, n, prod in LINEARS:
            row = case(name, k, n, prod, crops * 72 * 72, args.nodes)
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
