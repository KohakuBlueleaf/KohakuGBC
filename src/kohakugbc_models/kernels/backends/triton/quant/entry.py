"""The Triton fp8 producers: ``row`` / ``mx`` one row per program (warps tuned per problem against
the PyTorch paths); ``nvfp4`` tiled (``_fp4``), its tensor scale from an amax launch before the
quantizing one, or from a static ``Fp4Input`` (one launch)."""

import torch
import triton

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends.compile import compiled
from kohakugbc_models.kernels.backends.eager import quant as eager
from kohakugbc_models.kernels.backends.triton.attention.entry import bucket, choose_in
from kohakugbc_models.kernels.backends.triton.quant import _fp4, _kernels
from kohakugbc_models.kernels.fp8 import E4M3, FP4_BLOCK, MX_BLOCK, Fp4Input

TILES = [{"num_warps": w} for w in (4, 8, 16)]
PYTORCH = [{"path": "torch"}, {"path": "compile"}]
# NVFP4 tiles: (rows, columns, warps) for the elementwise producers, (rows, warps) for the norm
FP4_TILES = [
    {"BM": m, "BK": k, "num_warps": w}
    for m, k, w in ((1, 512, 4), (2, 1024, 4), (4, 256, 4), (4, 512, 4), (4, 1024, 8), (8, 256, 4))
]
FP4_ROWS = [{"BM": m, "num_warps": w} for m in (1, 2, 4) for w in (4, 8)]
# The E2M1 conversion: ``ptx`` the hardware instruction (sm100a / sm120a), ``arith`` comparisons
CVT = {"name": "ptx"}
ACTS = {"gelu": 1, "relu": 2}


def outputs(rows: int, channels: int, recipe: str, device) -> tuple:
    """(values, scales, the NVFP4 amax / ``1 / gs`` pair or a placeholder)."""
    if recipe == "nvfp4":
        kbp = -(-channels // FP4_BLOCK // 4) * 4
        q = torch.empty(rows, channels // 2, dtype=torch.uint8, device=device)
        s = torch.empty(-(-rows // 128) * 128 * kbp, dtype=torch.uint8, device=device)
        return q, s, torch.empty(2, dtype=torch.float32, device=device)
    q = torch.empty(rows, channels, dtype=E4M3, device=device)
    if recipe == "row":
        return q, torch.empty(rows, 1, dtype=torch.float32, device=device), None
    return q, torch.empty(rows, channels // MX_BLOCK, dtype=torch.uint8, device=device), None


def result(q, s, g, recipe: str) -> tuple:
    """The values and scales as ``fp8.mm`` reads them."""
    return (q, (s.view(E4M3), g[1:])) if recipe == "nvfp4" else (q, s)


def run(op: str, key: str, device, launch, args: tuple, tiles=None):
    """``tiles`` given: Triton only (the NVFP4 producers have no PyTorch path worth timing)."""

    def tile(config):
        return lambda: launch(config)

    def path(name):
        fn = getattr(eager, op) if name == "torch" else compiled(op, getattr(eager, op))
        return lambda: fn(*args)

    config = choose_in(op, key, device, tile, path, tiles or TILES + PYTORCH)
    return path(config["path"])() if "path" in config else tile(config)()


def fp4_passes(site: Fp4Input | None) -> tuple:
    """(the launches' ``PASS`` values, the ``G`` they share): one launch on a static site's
    constant, else the amax launch and the quantizing one on a fresh ``G``."""
    if site is not None and site.static:
        return (2,), site.g
    return (1, 2), None


def recorded(site: Fp4Input | None, g) -> None:
    if site is not None and site.record and not site.static:
        torch.maximum(site.seen, g[:1], out=site.seen)


def fp4_elementwise(x, act: int, op: str, key: str, args: tuple, site):
    channels = x.shape[-1]
    if channels % FP4_BLOCK:
        raise ValueError(f"nvfp4 needs a width divisible by {FP4_BLOCK} ({channels})")
    x2 = x.reshape(-1, channels)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    q, s, g = outputs(rows, channels, "nvfp4", x.device)
    passes, shared = fp4_passes(site)
    g = shared if shared is not None else g
    padded = -(-rows // 128) * 128

    def launch(config):
        bm, bk = config["BM"], config["BK"]
        if shared is None:
            g.zero_()
        for p in passes:
            _fp4.quant[(padded // bm, triton.cdiv(channels, bk))](
                x2,
                q,
                s,
                g,
                rows,
                channels,
                x2.stride(0),
                ACT=act,
                PASS=p,
                INV=shared is None,
                BM=bm,
                BK=bk,
                PTX=CVT["name"] == "ptx",
                num_warps=config["num_warps"],
            )
        return result(q, s, g, "nvfp4")

    mode = "static" if shared is not None else "dynamic"
    out = run(op, f"{key}-{mode}", x.device, launch, args, FP4_TILES)
    recorded(site, g)
    return out


def quant(x, recipe: str, site: Fp4Input | None = None) -> tuple:
    channels = x.shape[-1]
    key = f"{x.dtype}-c{channels}-{recipe}-m{bucket(x.numel() // channels)}"
    if recipe == "nvfp4":
        return fp4_elementwise(x, 0, "quant", key, (x, recipe, site), site)
    x2 = x.reshape(-1, channels)
    rows = x2.shape[0]
    q, s, g = outputs(rows, channels, recipe, x.device)
    block = triton.next_power_of_2(channels)

    def launch(config):
        _kernels.quant[(rows,)](
            x2,
            q,
            s,
            x2.stride(0),
            channels,
            ROW=recipe == "row",
            BC=block,
            num_warps=config["num_warps"],
        )
        return result(q, s, g, recipe)

    return run("quant", key, x.device, launch, (x, recipe))


def act_quant(h, act: str, recipe: str, site: Fp4Input | None = None) -> tuple:
    channels = h.shape[-1]
    key = f"{h.dtype}-c{channels}-{act}-{recipe}-m{bucket(h.numel() // channels)}"
    if recipe == "nvfp4":
        return fp4_elementwise(h, ACTS[act], "act_quant", key, (h, act, recipe, site), site)
    h2 = h.reshape(-1, channels)
    rows = h2.shape[0]
    q, s, g = outputs(rows, channels, recipe, h.device)
    block = triton.next_power_of_2(channels)

    def launch(config):
        _kernels.act_quant[(rows,)](
            h2,
            q,
            s,
            h2.stride(0),
            channels,
            ACT=ACTS[act],
            ROW=recipe == "row",
            BC=block,
            num_warps=config["num_warps"],
        )
        return result(q, s, g, recipe)

    return run("act_quant", key, h.device, launch, (h, act, recipe))


def fp4_add_ln(xc, rc, w, b, eps: float, key: str, args: tuple, site):
    channels = xc.shape[-1]
    if channels % FP4_BLOCK:
        raise ValueError(f"nvfp4 needs a width divisible by {FP4_BLOCK} ({channels})")
    rows = xc.numel() // channels
    s = torch.empty_like(xc)
    q, scales, g = outputs(rows, channels, "nvfp4", xc.device)
    passes, shared = fp4_passes(site)
    g = shared if shared is not None else g
    stats = torch.empty(rows, 2, dtype=torch.float32, device=xc.device) if shared is None else g
    padded = -(-rows // 128) * 128
    block = max(64, triton.next_power_of_2(channels))

    def launch(config):
        if shared is None:
            g.zero_()
        for p in passes if shared is None else (0,):
            _fp4.add_ln_quant[(padded // config["BM"],)](
                xc,
                rc,
                w,
                b,
                s,
                stats,
                q,
                scales,
                g,
                rows,
                channels,
                eps,
                PASS=p,
                INV=shared is None,
                BM=config["BM"],
                BC=block,
                PTX=CVT["name"] == "ptx",
                num_warps=config["num_warps"],
            )
        return (s, *result(q, scales, g, "nvfp4"))

    mode = "static" if shared is not None else "dynamic"
    out = run("add_ln_quant", f"{key}-{mode}", xc.device, launch, args, FP4_ROWS)
    recorded(site, g)
    return out


def add_ln_quant(x, r, w, b, eps: float, recipe: str, site: Fp4Input | None = None) -> tuple:
    channels = x.shape[-1]
    xc, rc = x.contiguous(), r.contiguous()
    rows = xc.numel() // channels
    key = f"{x.dtype}-c{channels}-{recipe}-m{bucket(rows)}"
    if recipe == "nvfp4":
        return fp4_add_ln(xc, rc, w, b, eps, key, (x, r, w, b, eps, recipe, site), site)
    s = torch.empty_like(xc)
    q, scales, g = outputs(rows, channels, recipe, x.device)
    block = triton.next_power_of_2(channels)

    def launch(config):
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
            num_warps=config["num_warps"],
        )
        return (s, *result(q, scales, g, recipe))

    return run("add_ln_quant", key, x.device, launch, (x, r, w, b, eps, recipe))


registry.register("quant", "triton", quant)
registry.register("act_quant", "triton", act_quant)
registry.register("add_ln_quant", "triton", add_ln_quant)
