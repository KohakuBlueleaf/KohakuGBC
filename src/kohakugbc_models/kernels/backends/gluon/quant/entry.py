"""The sm10x fp8 producers (``quant``, ``act_quant``, ``add_ln_quant``): ``mx`` here, its scales
emitted blocked (``fp8.mm`` takes them as they are), tiles tuned per problem against the Triton
entry (``backends.triton.quant``, natural scales); every other recipe runs the Triton entry."""

import torch
import triton

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.gluon.gemm.entry import choose
from kohakugbc_models.kernels.backends.gluon.quant import _kernels
from kohakugbc_models.kernels.backends.triton.attention.entry import bucket
from kohakugbc_models.kernels.backends.triton.quant import entry as triton_quant
from kohakugbc_models.kernels.fp8 import E4M3, MX_BLOCK

TILES = [
    {"BM": bm, "BC": bc, "num_warps": w}
    for bm, bc, w in (
        (8, 128, 4),
        (16, 128, 4),
        (32, 128, 8),
        (4, 256, 4),
        (8, 256, 8),
        (2, 512, 4),
    )
]
ROWS = [{"BM": bm, "num_warps": w} for bm in (1, 2, 4, 8) for w in (4, 8)]
ACTS = {"none": 0, "gelu": 1, "relu": 2}


def supports(query) -> str | None:
    if query.dtype not in (torch.bfloat16, torch.float16):
        return f"16-bit inputs only ({query.dtype})"
    return None


def outputs(rows: int, channels: int, device) -> tuple:
    """e4m3 values ``[rows, channels]`` and the blocked scale bytes (rows padded to 128, scale
    columns to 4); the number of 4-scale column blocks."""
    nc = -(-channels // MX_BLOCK // 4)
    q = torch.empty(rows, channels, dtype=E4M3, device=device)
    s = torch.empty(-(-rows // 128) * 128 * nc * 4, dtype=torch.uint8, device=device)
    return q, s, nc


def elementwise(x, act: str, op: str, fallback):
    channels = x.shape[-1]
    x2 = x.reshape(-1, channels)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    if channels % MX_BLOCK:
        return fallback()
    q, s, nc = outputs(rows, channels, x.device)

    def run_config(config):
        def run():
            grid = (triton.cdiv(rows, config["BM"]), triton.cdiv(channels, config["BC"]))
            _kernels.quant[grid](
                x2,
                q,
                s,
                rows,
                channels,
                x2.stride(0),
                nc,
                ACT=ACTS[act],
                BM=config["BM"],
                BC=config["BC"],
                num_warps=config["num_warps"],
            )
            return q, s.view(torch.float8_e8m0fnu)

        return run

    key = f"{x.dtype}-c{channels}-{act}-mx-m{bucket(rows)}"
    config = choose(op, key, x.device, run_config, lambda _: fallback, TILES + [{"path": "triton"}])
    return fallback() if "path" in config else run_config(config)()


def quant(x, recipe: str, site=None) -> tuple:
    def fallback():
        return triton_quant.quant(x, recipe, site)

    return elementwise(x, "none", "quant", fallback) if recipe == "mx" else fallback()


def act_quant(h, act: str, recipe: str, site=None) -> tuple:
    def fallback():
        return triton_quant.act_quant(h, act, recipe, site)

    return elementwise(h, act, "act_quant", fallback) if recipe == "mx" else fallback()


def add_ln_quant(x, r, w, b, eps: float, recipe: str, site=None) -> tuple:
    channels = x.shape[-1]

    def fallback():
        return triton_quant.add_ln_quant(x, r, w, b, eps, recipe, site)

    if recipe != "mx" or channels % MX_BLOCK:
        return fallback()
    xc, rc = x.contiguous(), r.contiguous()
    rows = xc.numel() // channels
    out = torch.empty_like(xc)
    q, scales, nc = outputs(rows, channels, x.device)
    block = triton.next_power_of_2(channels)

    def run_config(config):
        def run():
            _kernels.add_ln_quant[(triton.cdiv(rows, config["BM"]), 1)](
                xc,
                rc,
                w,
                b,
                out,
                q,
                scales,
                rows,
                channels,
                nc,
                eps,
                BM=config["BM"],
                BC=block,
                num_warps=config["num_warps"],
            )
            return out, q, scales.view(torch.float8_e8m0fnu)

        return run

    key = f"{x.dtype}-c{channels}-mx-m{bucket(rows)}"
    config = choose(
        "add_ln_quant", key, x.device, run_config, lambda _: fallback, ROWS + [{"path": "triton"}]
    )
    return fallback() if "path" in config else run_config(config)()


registry.register("quant", "gluon", quant, supports)
registry.register("act_quant", "gluon", act_quant, supports)
registry.register("add_ln_quant", "gluon", add_ln_quant, supports)
