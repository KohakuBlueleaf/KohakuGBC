"""The Triton fp8 producers, one row per program (warps tuned per problem against the PyTorch
paths)."""

import torch
import triton

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.compile import compiled
from kohakugbc_models.kernels.backends.eager import quant as eager
from kohakugbc_models.kernels.backends.triton.attention.entry import bucket, choose_in
from kohakugbc_models.kernels.backends.triton.quant import _kernels
from kohakugbc_models.kernels.fp8 import E4M3, MX_BLOCK

TILES = [{"num_warps": w} for w in (4, 8, 16)]
PYTORCH = [{"path": "torch"}, {"path": "compile"}]


def outputs(rows: int, channels: int, recipe: str, device) -> tuple:
    q = torch.empty(rows, channels, dtype=E4M3, device=device)
    if recipe == "row":
        return q, torch.empty(rows, 1, dtype=torch.float32, device=device)
    return q, torch.empty(rows, channels // MX_BLOCK, dtype=torch.uint8, device=device)


def run(op: str, key: str, device, launch, args: tuple):
    def tile(config):
        return lambda: launch(config["num_warps"])

    def path(name):
        fn = getattr(eager, op) if name == "torch" else compiled(op, getattr(eager, op))
        return lambda: fn(*args)

    config = choose_in(op, key, device, tile, path, TILES + PYTORCH)
    return path(config["path"])() if "path" in config else tile(config)()


def quant(x, recipe: str) -> tuple:
    channels = x.shape[-1]
    x2 = x.reshape(-1, channels)
    rows = x2.shape[0]
    q, s = outputs(rows, channels, recipe, x.device)
    block = triton.next_power_of_2(channels)

    def launch(warps):
        _kernels.quant[(rows,)](
            x2, q, s, x2.stride(0), channels, ROW=recipe == "row", BC=block, num_warps=warps
        )
        return q, s

    key = f"{x.dtype}-c{channels}-{recipe}-m{bucket(rows)}"
    return run("quant", key, x.device, launch, (x, recipe))


ACTS = {"gelu": 1, "relu": 2}


def act_quant(h, act: str, recipe: str) -> tuple:
    channels = h.shape[-1]
    h2 = h.reshape(-1, channels)
    rows = h2.shape[0]
    q, s = outputs(rows, channels, recipe, h.device)
    block = triton.next_power_of_2(channels)

    def launch(warps):
        _kernels.act_quant[(rows,)](
            h2,
            q,
            s,
            h2.stride(0),
            channels,
            ACT=ACTS[act],
            ROW=recipe == "row",
            BC=block,
            num_warps=warps,
        )
        return q, s

    key = f"{h.dtype}-c{channels}-{act}-{recipe}-m{bucket(rows)}"
    return run("act_quant", key, h.device, launch, (h, act, recipe))


def add_ln_quant(x, r, w, b, eps: float, recipe: str) -> tuple:
    channels = x.shape[-1]
    xc, rc = x.contiguous(), r.contiguous()
    rows = xc.numel() // channels
    s = torch.empty_like(xc)
    q, scales = outputs(rows, channels, recipe, x.device)
    block = triton.next_power_of_2(channels)

    def launch(warps):
        _kernels.add_ln_quant[(rows,)](
            xc,
            rc,
            w,
            b,
            s,
            q,
            scales,
            channels,
            eps,
            ROW=recipe == "row",
            BC=block,
            num_warps=warps,
        )
        return s, q, scales

    key = f"{x.dtype}-c{channels}-{recipe}-m{bucket(rows)}"
    return run("add_ln_quant", key, x.device, launch, (x, r, w, b, eps, recipe))


registry.register("quant", "triton", quant)
registry.register("act_quant", "triton", act_quant)
registry.register("add_ln_quant", "triton", add_ln_quant)
